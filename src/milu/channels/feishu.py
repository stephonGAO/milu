"""飞书（Lark）接入 —— 事件订阅 v2 渠道适配器（webhook 型）。

链路：
    飞书用户发消息
      → 飞书 POST 回调（事件订阅 v2；配了 Encrypt Key 则为密文 {"encrypt":..}）
      → 解密 / 取明文 JSON
      → url_verification 握手回 challenge；im.message.receive_v1 文本事件
      → 规整成 InboundMessage 交 dispatch（AgentRunner）
      → 经 tenant_access_token 调 /im/v1/messages 主动回发文本

文档：
- 事件订阅 / 加密：https://open.feishu.cn/document/server-docs/event-subscription-guide/overview
- 发送消息：https://open.feishu.cn/document/server-docs/im-v1/message/create
- tenant_access_token：https://open.feishu.cn/document/server-docs/authentication-management/access-token/tenant_access_token_internal

配置（环境变量，见 FeishuConfig.from_env）：
    FEISHU_APP_ID / FEISHU_APP_SECRET（必填）
    FEISHU_VERIFY_TOKEN（建议；校验事件来源）/ FEISHU_ENCRYPT_KEY（配了则密文模式）
    FEISHU_CALLBACK_PATH（默认 /feishu/event）/ FEISHU_API_BASE（默认 open.feishu.cn；
    国际版 Lark 用 https://open.larksuite.com）
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass

import httpx
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from .base import Channel, Dispatch, InboundMessage
from .state import InMemoryStateStore, StateStore

logger = logging.getLogger("milu.channels.feishu")


# ── 配置 ──────────────────────────────────────────────────────────────
@dataclass
class FeishuConfig:
    app_id: str
    app_secret: str
    verify_token: str = ""
    encrypt_key: str = ""
    callback_path: str = "/feishu/event"
    api_base: str = "https://open.feishu.cn"
    dedup_capacity: int = 4096

    @classmethod
    def from_env(cls) -> "FeishuConfig":
        def _req(name: str) -> str:
            v = os.environ.get(name, "").strip()
            if not v:
                raise RuntimeError(
                    f"缺少环境变量 {name}（飞书接入需配置 FEISHU_APP_ID / FEISHU_APP_SECRET）"
                )
            return v

        base = os.environ.get("FEISHU_API_BASE", "https://open.feishu.cn").strip()
        path = os.environ.get("FEISHU_CALLBACK_PATH", "/feishu/event").strip()
        return cls(
            app_id=_req("FEISHU_APP_ID"),
            app_secret=_req("FEISHU_APP_SECRET"),
            verify_token=os.environ.get("FEISHU_VERIFY_TOKEN", "").strip(),
            encrypt_key=os.environ.get("FEISHU_ENCRYPT_KEY", "").strip(),
            callback_path=path or "/feishu/event",
            api_base=(base or "https://open.feishu.cn").rstrip("/"),
        )


# ── 事件加密（密文模式）──────────────────────────────────────────────
class FeishuCrypto:
    """飞书事件订阅加密：AES-256-CBC，key=sha256(encrypt_key)，IV=密文前 16 字节。"""

    def __init__(self, encrypt_key: str) -> None:
        self._key = hashlib.sha256(encrypt_key.encode("utf-8")).digest()

    def decrypt(self, encrypt_b64: str) -> str:
        cipher_bytes = base64.b64decode(encrypt_b64)
        iv, ct = cipher_bytes[:16], cipher_bytes[16:]
        decryptor = Cipher(algorithms.AES(self._key), modes.CBC(iv)).decryptor()
        raw = decryptor.update(ct) + decryptor.finalize()
        return _pkcs7_unpad(raw).decode("utf-8")

    def encrypt(self, plaintext: str) -> str:
        """加密明文为 base64(iv + 密文)。生产侧不发密文（仅用于自测/对拍）。"""
        data = plaintext.encode("utf-8")
        pad = 16 - (len(data) % 16)
        data = data + bytes([pad]) * pad
        iv = os.urandom(16)
        encryptor = Cipher(algorithms.AES(self._key), modes.CBC(iv)).encryptor()
        ct = encryptor.update(data) + encryptor.finalize()
        return base64.b64encode(iv + ct).decode()


def _pkcs7_unpad(data: bytes) -> bytes:
    """去除 PKCS7 填充（16 字节块）；填充字节越界时容错为不去填充。"""
    if not data:
        return data
    pad = data[-1]
    if pad < 1 or pad > 16:
        return data
    return data[:-pad]


# ── API 客户端 ────────────────────────────────────────────────────────
class FeishuClient:
    """tenant_access_token 缓存 + 发送文本消息。"""

    def __init__(
        self, config: FeishuConfig, http: httpx.AsyncClient | None = None
    ) -> None:
        self.config = config
        self._http = http
        self._owns_http = http is None
        self._http_lock = asyncio.Lock()
        self._token = ""
        self._token_expire = 0.0
        self._token_lock = asyncio.Lock()

    async def _get_http(self) -> httpx.AsyncClient:
        """按需在线程中创建客户端，避免 TLS 初始化阻塞事件循环。"""
        if self._http is not None:
            return self._http
        async with self._http_lock:
            if self._http is None:
                self._http = await asyncio.to_thread(httpx.AsyncClient, timeout=15)
            return self._http

    async def tenant_token(self) -> str:
        if self._token and time.time() < self._token_expire - 60:
            return self._token
        async with self._token_lock:
            if self._token and time.time() < self._token_expire - 60:
                return self._token
            http = await self._get_http()
            r = await http.post(
                f"{self.config.api_base}/open-apis/auth/v3/tenant_access_token/internal",
                json={
                    "app_id": self.config.app_id,
                    "app_secret": self.config.app_secret,
                },
            )
            data = r.json()
            if data.get("code", 0) != 0:
                raise RuntimeError(f"tenant_access_token 获取失败：{data}")
            self._token = data["tenant_access_token"]
            self._token_expire = time.time() + data.get("expire", 7200)
            return self._token

    async def send_text(self, open_id: str, text: str) -> dict:
        token = await self.tenant_token()
        http = await self._get_http()
        r = await http.post(
            f"{self.config.api_base}/open-apis/im/v1/messages",
            params={"receive_id_type": "open_id"},
            headers={"Authorization": f"Bearer {token}"},
            json={
                "receive_id": open_id,
                "msg_type": "text",
                "content": json.dumps({"text": text}, ensure_ascii=False),
            },
        )
        return r.json()

    async def download_resource(
        self, message_id: str, file_key: str, resource_type: str
    ) -> tuple[bytes, str]:
        """下载消息资源（图片 resource_type=image / 文件 resource_type=file）。

        返回 (二进制内容, content_type)；失败（返回 JSON 错误）时抛异常。
        """
        token = await self.tenant_token()
        http = await self._get_http()
        r = await http.get(
            f"{self.config.api_base}/open-apis/im/v1/messages/{message_id}"
            f"/resources/{file_key}",
            params={"type": resource_type},
            headers={"Authorization": f"Bearer {token}"},
        )
        ct = r.headers.get("Content-Type", "")
        if "application/json" in ct.lower():
            raise RuntimeError(f"飞书资源下载失败：{r.text}")
        return r.content, ct

    async def aclose(self) -> None:
        if self._owns_http and self._http is not None:
            await self._http.aclose()


async def save_image_resource(
    client: "FeishuClient",
    channel: str,
    user_id: str,
    message_id: str,
    image_key: str,
) -> str | None:
    """下载飞书图片资源并落盘，返回绝对路径（失败/超限返回 None，不抛异常）。

    webhook（FeishuChannel）与长连接（FeishuWsChannel）共用，避免重复实现。
    """
    if not (message_id and image_key):
        return None
    try:
        data, ct = await client.download_resource(message_id, image_key, "image")
    except Exception:
        logger.exception("飞书图片下载失败 image_key=%s", image_key)
        return None
    from .media import save_image

    return save_image(channel, user_id, data, content_type=ct, media_id=image_key)


async def save_file_resource(
    client: "FeishuClient",
    channel: str,
    user_id: str,
    message_id: str,
    file_key: str,
    file_name: str = "",
) -> str | None:
    """下载飞书文件资源并落盘，返回绝对路径（失败/超限返回 None，不抛异常）。

    webhook 与长连接共用。文件名取消息 content 里的 file_name（doc_read 靠它选解析器）。
    """
    if not (message_id and file_key):
        return None
    try:
        data, _ = await client.download_resource(message_id, file_key, "file")
    except Exception:
        logger.exception("飞书文件下载失败 file_key=%s", file_key)
        return None
    from .media import save_file

    return save_file(channel, user_id, data, filename=file_name, media_id=file_key)


# ── Channel 适配器 ────────────────────────────────────────────────────
class FeishuChannel(Channel):
    """飞书渠道（webhook 型）。"""

    def __init__(
        self,
        config: FeishuConfig,
        *,
        state: StateStore | None = None,
        client: FeishuClient | None = None,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._crypto = FeishuCrypto(config.encrypt_key) if config.encrypt_key else None
        self._client = client or FeishuClient(config, http=http)
        self._owns_client = client is None
        self._state = state or InMemoryStateStore(dedup_capacity=config.dedup_capacity)

    @property
    def name(self) -> str:
        return "feishu"

    def register(self, app: FastAPI, dispatch: Dispatch) -> None:
        cfg = self._config

        @app.post(cfg.callback_path)
        async def event(request: Request) -> Response:
            raw = (await request.body()).decode("utf-8", "ignore")
            try:
                payload = json.loads(raw)
            except ValueError:
                return Response("bad request", status_code=400)

            # 密文模式：先解密成明文 JSON。解密失败/未配 key 时 **ack 忽略**（回 200）
            # 而非 403——对飞书返回非 2xx 会触发反复重投与「回调失败」告警。
            if "encrypt" in payload:
                if self._crypto is None:
                    logger.warning("收到密文事件但未配置 FEISHU_ENCRYPT_KEY，已忽略")
                    return JSONResponse({"code": 0})
                try:
                    payload = json.loads(self._crypto.decrypt(payload["encrypt"]))
                except Exception:
                    logger.warning("飞书事件解密失败，已忽略")
                    return JSONResponse({"code": 0})

            # token 兼容两种 schema：v2.0 在 header.token，v1.0/URL 验证在顶层 token
            token = (payload.get("header") or {}).get("token") or payload.get("token")

            # ① URL 验证握手：token 不符才拒绝（不能把 challenge 回给别的应用）
            if payload.get("type") == "url_verification":
                if cfg.verify_token and token != cfg.verify_token:
                    logger.warning("飞书 url_verification token 不匹配，拒绝")
                    return Response("forbidden", status_code=403)
                return JSONResponse({"challenge": payload.get("challenge", "")})

            # ② 事件投递：token 不符 / 非目标事件，一律 ack（200）后忽略。
            #    飞书除消息事件外还会发其它回调（重复 URL 校验、非消息事件、v1.0 schema
            #    等），对这些回 403 会被飞书判为端点失败并反复重投——故只 ack 不处理。
            if cfg.verify_token and token != cfg.verify_token:
                logger.debug("飞书事件 token 不匹配，忽略（仍 ack 200）")
                return JSONResponse({"code": 0})

            header = payload.get("header") or {}
            if header.get("event_type") == "im.message.receive_v1":
                event_id = header.get("event_id", "")
                # 后台处理（飞书要求快速回 200），立即 ack
                asyncio.create_task(
                    self._handle(event_id, payload, dispatch)
                )
            return JSONResponse({"code": 0})

    async def _handle(
        self, event_id: str, payload: dict, dispatch: Dispatch
    ) -> None:
        """规整文本消息 → dispatch → 回发（整段不抛异常）。"""
        # 去重：飞书失败重投携带同一 event_id
        if event_id and await self._state.seen(self.name, event_id):
            return
        event = payload.get("event") or {}
        message = event.get("message") or {}
        mtype = message.get("message_type")
        if mtype not in ("text", "image", "file"):  # 暂处理文本/图片/文件
            return
        sender_id = (event.get("sender") or {}).get("sender_id") or {}
        open_id = sender_id.get("open_id")
        if not open_id:
            return
        try:
            content = json.loads(message.get("content") or "{}")
        except ValueError:
            content = {}
        message_id = message.get("message_id", "")
        text, images, files = "", [], []
        if mtype == "text":
            text = (content.get("text") or "").strip()
            if not text:
                return
        elif mtype == "image":  # 下载资源落工作区，走视觉
            path = await save_image_resource(
                self._client, self.name, open_id,
                message_id, content.get("image_key", ""),
            )
            if not path:
                return
            images = [path]
        else:  # file：下载落工作区，模型用 doc_read/file_read 读取
            path = await save_file_resource(
                self._client, self.name, open_id, message_id,
                content.get("file_key", ""), content.get("file_name", ""),
            )
            if not path:
                return
            files = [path]

        inbound = InboundMessage(
            channel=self.name,
            user_id=open_id,
            text=text,
            images=images,
            files=files,
            reply_to={"receive_id": open_id, "chat_id": message.get("chat_id")},
            msg_id=message_id,
            raw=payload,
        )
        try:
            reply = await dispatch(inbound)
        except Exception:
            logger.exception("dispatch 生成回复失败 open_id=%s", open_id)
            reply = None
        if reply is None or not reply.text:
            return
        try:
            resp = await self._client.send_text(open_id, reply.text)
            if resp.get("code", 0) != 0:
                logger.warning("飞书 send 返回错误：%s", resp)
        except Exception:
            logger.exception("飞书回复失败 open_id=%s", open_id)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
        await self._state.close()
