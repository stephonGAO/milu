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

import os
from pathlib import Path

# 包内模板根目录（随 wheel 一起分发）
_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

# 内置角色提示词
PROMPT_ROLES = ("main", "coder", "researcher", "reader", "reviewer")


def templates_dir() -> Path:
    """内置模板根目录（含 prompts/ 与 skills/）。"""
    return _TEMPLATES_DIR


# ── 用户级数据/配置目录（与 CWD 解耦）────────────────────────────
#
# 「内置模板」（prompts/skills）是只读的包内资源，随 wheel 分发，用上面的
# templates_dir() / builtin_*_dir() 定位。
# 「用户级可写数据」（会话日志、MCP 配置等）则不应写进 CWD——作为库被集成或部署到
# 服务器时 CWD 取决于宿主应用，写入位置会漂移。统一锚定到 user_data_dir() 下，
# 默认 ~/.agent_framework，可用环境变量 AGENT_FRAMEWORK_HOME 覆盖。


def user_data_dir() -> Path:
    """用户级数据/配置根目录（与 CWD 解耦）。

    解析优先级：
      1. 环境变量 AGENT_FRAMEWORK_HOME（如显式指定服务器数据盘）
      2. ~/.agent_framework

    :return: 根目录路径（不保证已存在，写入方负责按需 mkdir）。
    """
    env = os.environ.get("AGENT_FRAMEWORK_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".agent_framework"


def default_session_dir() -> Path:
    """会话日志默认根目录：user_data_dir()/sessions。"""
    return user_data_dir() / "sessions"


def default_mcp_config_path() -> Path:
    """MCP 配置文件默认路径：user_data_dir()/mcp_servers.json。"""
    return user_data_dir() / "mcp_servers.json"


def builtin_prompts_dir(role: str = "main") -> Path:
    """返回内置角色提示词目录。

    :param role: 角色名，可选 main / coder / researcher / reader / reviewer。
    :raises FileNotFoundError: 角色不存在时。
    """
    d = _TEMPLATES_DIR / "prompts" / role
    if not d.is_dir():
        raise FileNotFoundError(
            f"未找到内置提示词角色 '{role}'，可选：{', '.join(PROMPT_ROLES)}"
        )
    return d


def builtin_skills_dir() -> Path:
    """返回内置技能目录（含 skill-creator / deep-research / mcp-builder 等）。"""
    return _TEMPLATES_DIR / "skills"
