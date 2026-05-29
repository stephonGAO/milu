"""技能（Skills）模块 — 专项指令集按需加载"""
from agent_framework.skills.config import SkillConfig
from agent_framework.skills.registry import SkillRegistry

__all__ = [
    "SkillConfig",
    "SkillRegistry",
]
