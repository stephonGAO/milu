"""接入渠道层（channels）/ 统一网关（Gateway）。

把各 IM 平台（微信客服、飞书、Telegram…）的消息输入/输出，经统一的
**Ports & Adapters（六边形）** 网关桥接到 milu：

    from milu.channels import AgentRunner, Gateway
    from milu.channels.wechat_kf import WeChatKfChannel, WeChatKfConfig
    from milu.channels.telegram import TelegramChannel, TelegramConfig

    runner = AgentRunner.from_llm(llm, agent_kwargs={...})       # 接 milu（含隔离透传）
    channels = [WeChatKfChannel(WeChatKfConfig.from_env(), state=store),
                TelegramChannel(TelegramConfig.from_env(), state=store)]
    Gateway.from_runner(runner, channels).run(port=8800)         # 一把起多渠道服务

核心件（`base`/`runner`/`gateway`/`state`）只依赖核心库（fastapi/AgentPool），
本 `__init__` 直接导出；**具体渠道适配器**（wechat_kf/feishu 需 cryptography）按需
经子模块惰性加载——`from milu.channels import WeChatKfChannel` 触发时才真正 import，
不把 cryptography 拖进 `import milu.channels`。
"""
from __future__ import annotations

from .base import Channel, Dispatch, InboundMessage, OutboundMessage
from .gateway import Gateway
from .runner import DEFAULT_FALLBACK, AgentRunner
from .state import FileStateStore, InMemoryStateStore, StateStore

__all__ = [
    # 核心契约 / 编排 / 接 milu / 状态
    "Channel",
    "Dispatch",
    "InboundMessage",
    "OutboundMessage",
    "Gateway",
    "AgentRunner",
    "DEFAULT_FALLBACK",
    "StateStore",
    "InMemoryStateStore",
    "FileStateStore",
    # 渠道适配器（惰性加载）
    "WeChatKfChannel",
    "WeChatKfConfig",
    "FeishuChannel",
    "FeishuConfig",
    "TelegramChannel",
    "TelegramConfig",
]

# 适配器名 -> 所在子模块（按需导入，避免拖入 cryptography 等渠道专属依赖）
_LAZY = {
    "WeChatKfChannel": ".wechat_kf",
    "WeChatKfConfig": ".wechat_kf",
    "FeishuChannel": ".feishu",
    "FeishuConfig": ".feishu",
    "TelegramChannel": ".telegram",
    "TelegramConfig": ".telegram",
}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    mod = importlib.import_module(module, __name__)
    return getattr(mod, name)
