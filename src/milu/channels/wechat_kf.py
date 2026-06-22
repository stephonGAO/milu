"""微信客服（经企业微信）接入 —— echo 链路。

目标（第一阶段）：把传输链路在代码层打通，**先不接 milu Agent**：

    微信用户发消息
      → 企业微信 POST 回调（加密 XML，仅含 Token + OpenKfId，不含正文）
      → 解密拿到 Token/OpenKfId
      → 用 Token 调 sync_msg「拉取」真正的消息列表
      → 调 send_msg 把内容**原样回显**（echo）

之后再把 echo 替换为 milu，并抽象成 Gateway 的一个 Channel。`on_text` 钩子
即是预留给「接 milu」的接缝——届时只需传入一个把用户文本跑过 Agent 的回调，
传输层一行不用改。

接口参考：
- 接收消息和事件：https://developer.work.weixin.qq.com/document/path/94670
- 读取消息 sync_msg / 发送消息 send_msg：同上文档站
- 获取 access_token：https://developer.work.weixin.qq.com/document/path/91039

运行（需先配好 4 个环境变量，见 WeChatKfConfig.from_env）：
    python -m milu.channels.wechat_kf
回调地址：http://<公网域名>{callback_path}（必须公网 HTTPS，开发期用内网穿透）。
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from dataclasses import dataclass
from typing import Awaitable, Callable

import httpx
from fastapi import APIRouter, FastAPI, Request, Response

from .wecom_crypto import WeComCrypto, WeComCryptError

logger = logging.getLogger("milu.channels.wechat_kf")

_QYAPI = "https://qyapi.weixin.qq.com/cgi-bin"

# 「接 milu」的接缝：给定（external_userid, 用户文本）返回回复文本。
# echo 阶段不传，默认原样回显。
OnText = Callable[[str, str], Awaitable[str]]


# ── 配置 ──────────────────────────────────────────────────────────────
@dataclass
class WeChatKfConfig:
    """微信客服接入配置（密钥走环境变量，不落配置文件）。"""

    corp_id: str           # 企业 ID
    secret: str            # 「微信客服」的 Secret（换 access_token 用，务必用客服那个）
    token: str             # 回调配置的 Token
    encoding_aes_key: str  # 回调配置的 EncodingAESKey（43 位）
    callback_path: str = "/wechat/kf/callback"
    # msgid 去重窗口大小（防企业微信重试导致重复回复；echo 阶段放内存）
    dedup_capacity: int = 4096

    @classmethod
    def from_env(cls) -> "WeChatKfConfig":
        """从环境变量读取配置（与 milu 约定一致：密钥只走 env）。

        必填：WECHAT_KF_CORP_ID / WECHAT_KF_SECRET / WECHAT_KF_TOKEN / WECHAT_KF_AESKEY
        选填：WECHAT_KF_CALLBACK_PATH（默认 /wechat/kf/callback）
        """

        def _req(name: str) -> str:
            v = os.environ.get(name, "").strip()
            if not v:
                raise RuntimeError(
                    f"缺少环境变量 {name}（微信客服接入需配置 "
                    f"WECHAT_KF_CORP_ID/SECRET/TOKEN/AESKEY）"
                )
            return v

        path = os.environ.get("WECHAT_KF_CALLBACK_PATH", "/wechat/kf/callback").strip()
        return cls(
            corp_id=_req("WECHAT_KF_CORP_ID"),
            secret=_req("WECHAT_KF_SECRET"),
            token=_req("WECHAT_KF_TOKEN"),
            encoding_aes_key=_req("WECHAT_KF_AESKEY"),
            callback_path=path or "/wechat/kf/callback",
        )


# ── API 客户端 ────────────────────────────────────────────────────────
class WeChatKfClient:
    """微信客服 API 客户端：access_token 缓存 + sync_msg 拉取 + send_msg 发送。

    注意（echo 阶段的有意简化，生产需加固）：
    - 游标 `_cursors` 和去重 `_seen_msgids` 都放内存，进程重启即丢；
      上生产应持久化（建议落 user_data_dir() 下文件或 Redis）。
    - 内存版仅适合单进程 echo 自测。
    """

    def __init__(
        self,
        config: WeChatKfConfig,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self._http = http or httpx.AsyncClient(timeout=15)
        self._owns_http = http is None
        self._token = ""
        self._token_expire = 0.0
        self._token_lock = asyncio.Lock()
        # open_kfid -> next_cursor（增量拉取游标）
        self._cursors: dict[str, str] = {}
        # msgid 去重：用 OrderedDict 当有界 LRU，防内存无限增长
        self._seen_msgids: "OrderedDict[str, None]" = OrderedDict()

    # access_token：7200s 有效，带双重检查锁缓存复用
    async def access_token(self) -> str:
        if self._token and time.time() < self._token_expire - 60:
            return self._token
        async with self._token_lock:
            if self._token and time.time() < self._token_expire - 60:
                return self._token
            r = await self._http.get(
                f"{_QYAPI}/gettoken",
                params={"corpid": self.config.corp_id, "corpsecret": self.config.secret},
            )
            data = r.json()
            if data.get("errcode", 0) != 0:
                raise RuntimeError(f"gettoken 失败：{data}")
            self._token = data["access_token"]
            self._token_expire = time.time() + data.get("expires_in", 7200)
            return self._token

    async def sync_msg(self, token: str, open_kfid: str) -> list[dict]:
        """用回调 token 拉取该客服账号的新消息，按游标翻页直到 has_more=0。"""
        out: list[dict] = []
        has_more = 1
        while has_more:
            at = await self.access_token()
            r = await self._http.post(
                f"{_QYAPI}/kf/sync_msg",
                params={"access_token": at},
                json={
                    "token": token,
                    "open_kfid": open_kfid,
                    "cursor": self._cursors.get(open_kfid, ""),
                    "limit": 1000,
                },
            )
            data = r.json()
            if data.get("errcode", 0) != 0:
                raise RuntimeError(f"sync_msg 失败：{data}")
            # 游标推进（即便本批为空也要保存，避免重复拉取）
            self._cursors[open_kfid] = data.get(
                "next_cursor", self._cursors.get(open_kfid, "")
            )
            has_more = data.get("has_more", 0)
            out.extend(data.get("msg_list", []))
        return out

    async def send_text(self, touser: str, open_kfid: str, content: str) -> dict:
        """发送文本消息给指定用户（48 小时会话窗口内有效）。"""
        at = await self.access_token()
        r = await self._http.post(
            f"{_QYAPI}/kf/send_msg",
            params={"access_token": at},
            json={
                "touser": touser,
                "open_kfid": open_kfid,
                "msgtype": "text",
                "text": {"content": content},
            },
        )
        return r.json()

    # ── 去重（有界 LRU）────────────────────────────────────────────
    def seen(self, msgid: str) -> bool:
        """msgid 是否已处理过；首次见到则登记并返回 False。"""
        if not msgid:
            return False
        if msgid in self._seen_msgids:
            return True
        self._seen_msgids[msgid] = None
        # 越界淘汰最旧
        while len(self._seen_msgids) > self.config.dedup_capacity:
            self._seen_msgids.popitem(last=False)
        return False

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()


# ── 事件处理（echo）───────────────────────────────────────────────────
async def handle_event(
    client: WeChatKfClient,
    token: str,
    open_kfid: str,
    on_text: OnText | None = None,
) -> None:
    """处理一次 kf_msg_or_event：拉取消息 → 逐条回显（或经 on_text 生成回复）。

    设计为「整段不抛异常」：任何一步失败只记日志，不影响回调已返回的 success。
    """
    try:
        messages = await client.sync_msg(token, open_kfid)
    except Exception:
        logger.exception("sync_msg 拉取失败 open_kfid=%s", open_kfid)
        return

    for msg in messages:
        msgid = msg.get("msgid", "")
        if client.seen(msgid):
            continue
        # echo 阶段只处理文本；其他类型（图片/语音/事件）先跳过
        if msg.get("msgtype") != "text":
            continue
        external_userid = msg.get("external_userid")
        content = (msg.get("text") or {}).get("content", "")
        if not external_userid or not content:
            continue

        try:
            reply = await on_text(external_userid, content) if on_text else content
        except Exception:
            logger.exception("on_text 生成回复失败 user=%s", external_userid)
            reply = content  # 兜底回显，避免静默丢消息

        if not reply:
            continue
        try:
            resp = await client.send_text(external_userid, open_kfid, reply)
            if resp.get("errcode", 0) != 0:
                logger.warning("send_msg 返回错误：%s", resp)
        except Exception:
            logger.exception("send_msg 回复失败 user=%s", external_userid)


# ── FastAPI 路由 ──────────────────────────────────────────────────────
def build_router(
    config: WeChatKfConfig,
    client: WeChatKfClient | None = None,
    on_text: OnText | None = None,
) -> APIRouter:
    """构建可挂载到任意 FastAPI app 的微信客服回调路由。

    :param on_text: 留给后续接 milu 的接缝；None 时为 echo 原样回显。
    """
    crypto = WeComCrypto(
        token=config.token,
        encoding_aes_key=config.encoding_aes_key,
        receive_id=config.corp_id,
    )
    client = client or WeChatKfClient(config)
    router = APIRouter()

    # 注意：开启 `from __future__ import annotations` 后，FastAPI 需要能从
    # 模块全局解析注解字符串——这里所有注解（str / Request）都在模块顶层可见。
    @router.get(config.callback_path)
    async def verify(
        msg_signature: str, timestamp: str, nonce: str, echostr: str
    ) -> Response:
        """① URL 验证：解密 echostr 原样返回明文。"""
        try:
            plain = crypto.verify_url(msg_signature, timestamp, nonce, echostr)
        except WeComCryptError:
            logger.warning("回调 URL 验证失败")
            return Response("fail", status_code=403)
        return Response(plain)

    @router.post(config.callback_path)
    async def receive(
        request: Request, msg_signature: str, timestamp: str, nonce: str
    ) -> Response:
        """② 接收事件：解密 → 若是 kf_msg_or_event，则后台拉取+回显。

        ⚠️ 企业微信要求 5 秒内返回 success，否则会重试。故立即返回，
        把耗时的 sync_msg + send_msg 放后台任务执行。
        """
        body = (await request.body()).decode("utf-8", "ignore")
        try:
            plain = crypto.decrypt_message(body, msg_signature, timestamp, nonce)
        except WeComCryptError:
            logger.warning("回调消息解密失败")
            return Response("fail", status_code=403)

        try:
            root = ET.fromstring(plain)
        except ET.ParseError:
            logger.warning("回调明文非合法 XML")
            return Response("success")  # 仍回 success，避免企业微信重试风暴

        if root.findtext("Event") == "kf_msg_or_event":
            token = root.findtext("Token") or ""
            open_kfid = root.findtext("OpenKfId") or ""
            if token and open_kfid:
                asyncio.create_task(handle_event(client, token, open_kfid, on_text))
        return Response("success")

    # 把 client 挂在 router 上，供 create_app 在关闭时回收
    router.kf_client = client  # type: ignore[attr-defined]
    return router


def create_app(
    config: WeChatKfConfig | None = None,
    on_text: OnText | None = None,
) -> FastAPI:
    """构建一个独立可跑的微信客服 echo 服务（也可把 build_router 挂到现有 app）。"""
    config = config or WeChatKfConfig.from_env()
    app = FastAPI(title="milu 微信客服 echo")
    router = build_router(config, on_text=on_text)
    app.include_router(router)
    client: WeChatKfClient = router.kf_client  # type: ignore[attr-defined]

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True, "callback_path": config.callback_path}

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await client.aclose()

    return app


def run_server(host: str = "0.0.0.0", port: int = 8800) -> None:
    """启动 echo 服务（前台阻塞）。"""
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = WeChatKfConfig.from_env()
    print(
        f"微信客服 echo 服务启动：监听 http://{host}:{port}\n"
        f"  回调路径：{config.callback_path}\n"
        f"  请把企业微信「微信客服 → API → 回调配置」的 URL 指向："
        f"https://<你的公网域名>{config.callback_path}\n"
        f"  （必须公网 HTTPS；开发期可用 cpolar/frp 等内网穿透）"
    )
    uvicorn.run(app=create_app(config), host=host, port=port)


if __name__ == "__main__":
    run_server(port=int(os.environ.get("PORT", "8800")))
