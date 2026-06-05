"""从 Settings 构建 LLM 与 Agent（CLI 专用，依托 Agent「全配默认」最简创建）。"""
from __future__ import annotations

from milu.agent import Agent
from milu.llm.providers import ModelRegistry
from milu.llm.providers.base import BaseLLM

from milu.cli.config import Settings
from milu.cli.render import confirm_unsafe


def build_llm(s: Settings) -> BaseLLM:
    """按设置创建 LLM 实例（鉴权懒加载，构造时不触网）。

    web_search / enable_thinking 作为构造默认值传入，不支持的厂商由
    BaseLLM._validate_params 静默丢弃。
    """
    kwargs = {
        "model": s.model,
        "web_search": s.web_search,
        "enable_thinking": s.enable_thinking,
    }
    if s.api_key:
        kwargs["api_key"] = s.api_key
    return ModelRegistry.create(s.provider, **kwargs)


def build_agent(s: Settings) -> Agent:
    """按设置构建顶层 Agent（最简创建，其余全部交给 Agent 的「全配默认」）。

    - tools / skills / prompt 不传 → 自动注入全套内置工具、内置技能、内置 main 角色提示词
    - subagents：None → 内置三件套（researcher/reader/coder）；--no-subagents → [] 关闭
    - on_confirm：manual 模式人工审批 + auto 模式 AI 判定器转人工时的交互式确认
    """
    return Agent(
        llm=build_llm(s),
        mode=s.mode,
        session_enabled=s.session_enabled,
        subagents=None if s.use_subagents else [],
        on_confirm=confirm_unsafe,
    )
