"""milu 多渠道网关示例：一个 AgentRunner + 一个 Gateway 接多个 IM 平台。

把「平台无关的 milu 接入」与「各平台适配器（Channel）」解耦（Ports & Adapters）：
新增平台 = 只写/挂一个 Channel，共用同一个 AgentPool + milu 全配默认 + 隔离设置。

本示例按**已配置的环境变量自动启用**渠道（微信客服 / 飞书 / Telegram），并用
**文件持久化** StateStore（去重/游标重启不丢）。准备好任一渠道的环境变量即可：

    微信客服:  WECHAT_KF_CORP_ID / SECRET / TOKEN / AESKEY     （见 .env.example 3.5）
    飞书:      FEISHU_APP_ID / APP_SECRET（+ VERIFY_TOKEN/ENCRYPT_KEY）（3.6）
    Telegram:  TELEGRAM_BOT_TOKEN                              （3.7）

依赖：pip install milu（微信/飞书 webhook + Telegram 全部开箱即用；cryptography 已在核心）

运行：
    python examples/gateway_multi_channel.py
然后把各 webhook 渠道的回调 URL 指向 https://<公网域名><回调路径>（需公网 HTTPS）。

> 生产部署更建议直接用内置命令：`milu gateway`（自动按凭证启用渠道 + 读分层配置的
> strict 隔离）。本示例用于讲解接线与二次开发。
"""
from __future__ import annotations

import os

from milu._env import ensure_dotenv_loaded
from milu.channels import AgentRunner, FileStateStore, Gateway
from milu.llm.providers import ModelRegistry


def build_channels(store):
    """按环境变量探测并构建启用的渠道。"""
    channels = []
    if os.environ.get("WECHAT_KF_CORP_ID"):
        from milu.channels.wechat_kf import WeChatKfChannel, WeChatKfConfig
        channels.append(WeChatKfChannel(WeChatKfConfig.from_env(), state=store))
    if os.environ.get("FEISHU_APP_ID"):
        from milu.channels.feishu import FeishuChannel, FeishuConfig
        channels.append(FeishuChannel(FeishuConfig.from_env(), state=store))
    if os.environ.get("TELEGRAM_BOT_TOKEN"):
        from milu.channels.telegram import TelegramChannel, TelegramConfig
        channels.append(TelegramChannel(TelegramConfig.from_env(), state=store))
    return channels


def main() -> None:
    ensure_dotenv_loaded()

    provider = os.environ.get("LLM_PROVIDER", "qwen")
    model = os.environ.get("LLM_MODEL", "").strip()
    llm = ModelRegistry.create(provider, **({"model": model} if model else {}))

    # 接 milu：单个共享 LLM 起池。strict 强隔离经 agent_kwargs 显式透传——
    # 公网多用户务必加上（代码进断网 docker 容器 + 文件工具关进工作区）：
    #   agent_kwargs={"sandbox": "docker", "workspace_jail": True}
    # 简单内网/自用可省略（默认 subprocess 子进程隔离）。
    runner = AgentRunner.from_llm(llm)

    store = FileStateStore()                 # 去重/游标落 ~/.milu/gateway/
    channels = build_channels(store)
    if not channels:
        raise SystemExit(
            "没有可用渠道。请配置至少一个渠道的环境变量（见本文件顶部 / .env.example）。"
        )

    print("启用渠道:", ", ".join(c.name for c in channels))
    Gateway.from_runner(runner, channels).run(
        host="0.0.0.0", port=int(os.environ.get("PORT", "8800"))
    )


if __name__ == "__main__":
    main()
