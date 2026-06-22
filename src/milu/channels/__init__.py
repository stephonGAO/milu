"""接入渠道层（channels）。

把各 IM 平台（微信客服、飞书、Telegram…）的消息输入/输出桥接到 milu。

当前阶段：先把「微信客服」单条链路在传输层打通（echo 回显），尚未抽象出
统一的 Channel/Gateway 接口——待第一个渠道跑通后再回头抽象（避免过早抽象）。

注意：本包 `__init__` 不主动导入各渠道子模块，以免把 cryptography 等渠道专属
依赖拖进 `import milu`。需要时显式 `from milu.channels import wechat_kf`。
"""
from __future__ import annotations

__all__: list[str] = []
