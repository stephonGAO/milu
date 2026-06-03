"""内置资源（提示词模板 / 技能）的定位辅助。

这些默认资源随 wheel 一起打包（位于本包目录下的 templates/），
pip 安装后即可通过下列函数取到它们的绝对路径，无需依赖仓库结构或当前工作目录。

用法：
    from agent_framework import builtin_prompts_dir, builtin_skills_dir, Agent

    agent = Agent(
        llm=llm,
        prompt_dir=builtin_prompts_dir("main"),
        skills_dir=str(builtin_skills_dir()),
    )
"""
from __future__ import annotations

from pathlib import Path

# 包内模板根目录（随 wheel 一起分发）
_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

# 内置角色提示词
PROMPT_ROLES = ("main", "coder", "researcher", "reviewer")


def templates_dir() -> Path:
    """内置模板根目录（含 prompts/ 与 skills/）。"""
    return _TEMPLATES_DIR


def builtin_prompts_dir(role: str = "main") -> Path:
    """返回内置角色提示词目录。

    :param role: 角色名，可选 main / coder / researcher / reviewer。
    :raises FileNotFoundError: 角色不存在时。
    """
    d = _TEMPLATES_DIR / "prompts" / role
    if not d.is_dir():
        raise FileNotFoundError(
            f"未找到内置提示词角色 '{role}'，可选：{', '.join(PROMPT_ROLES)}"
        )
    return d


def builtin_skills_dir() -> Path:
    """返回内置技能目录（含 code-review / skill-creator / translator 等）。"""
    return _TEMPLATES_DIR / "skills"
