"""微信客服（经企业微信）echo 链路启动示例。

第一阶段：把传输链路打通——回调验证 → 收事件 → sync_msg 拉取 → send_msg 原样回显。
暂不接 milu Agent，先确认「微信用户发消息能原样收到回复」全链路通。

准备：
1. 企业微信管理后台「微信客服 → API」拿到 corpid / 客服 Secret；
   「回调配置」里自定义 Token / EncodingAESKey（43 位）。
2. 配置环境变量（见 .env.example 的 3.5 节）：
       WECHAT_KF_CORP_ID / WECHAT_KF_SECRET / WECHAT_KF_TOKEN / WECHAT_KF_AESKEY
3. 安装可选依赖：pip install "milu[wechat]"

运行：
    python examples/wechat_kf_echo.py
然后用内网穿透（cpolar/frp）把本机端口暴露成公网 HTTPS，把
    https://<公网域名>/wechat/kf/callback
填进企业微信「回调配置」的 URL 保存 → 验证通过即链路打通。

下一步（接 milu）：把 create_app(config, on_text=...) 的 on_text 传成一个
跑 AgentPool 的回调即可，传输层无需改动。
"""
from __future__ import annotations

import os

from milu._env import ensure_dotenv_loaded
from milu.channels.wechat_kf import WeChatKfConfig, run_server


def main() -> None:
    # 先加载 .env：项目级（当前目录的 .env，向上查找）→ 用户级 ~/.milu/.env
    # （进程已有的环境变量始终最优先，不被覆盖）
    ensure_dotenv_loaded()
    # from_env 会校验必填项缺失并给出中文提示
    WeChatKfConfig.from_env()
    run_server(host="0.0.0.0", port=int(os.environ.get("PORT", "8800")))


if __name__ == "__main__":
    main()
