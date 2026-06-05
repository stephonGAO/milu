"""内置资源（提示词模板 / 技能）的定位辅助。

这些默认资源随 wheel 一起打包（位于本包目录下的 templates/），
pip 安装后即可通过下列函数取到它们的绝对路径，无需依赖仓库结构或当前工作目录。

用法：
    from milu import builtin_prompts_dir, builtin_skills_dir, Agent

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


# ── 目录策略：写数据锚定用户级目录，读配置锚定项目目录（均与裸 CWD 解耦）────────
#
# 「内置模板」（prompts/skills）是只读的包内资源，随 wheel 分发，用上面的
# templates_dir() / builtin_*_dir() 定位。
#
# 「写数据」（会话日志、记忆、CLI 配置等可写状态）不应写进 CWD——作为库被集成或
# 部署到服务器时 CWD 取决于宿主应用，写入位置会漂移。统一锚定到 user_data_dir()
# （默认 ~/.milu，可用环境变量 MILU_HOME 覆盖）。
#
# 「读配置」（mcp_servers.json、.env 等由项目/用户提供的配置）锚定到 project_dir()
# （默认 CWD，可用环境变量 MILU_PROJECT_DIR 覆盖），便于每个项目带自己的配置；
# 项目级找不到时再回退到 user_data_dir() 下的用户级同名文件。


def user_data_dir() -> Path:
    """用户级数据根目录 ——「写数据」的锚点（与 CWD 解耦）。

    会话日志、记忆、CLI 配置等可写状态统一落在这里。

    解析优先级：
      1. 环境变量 MILU_HOME（如显式指定服务器数据盘）
      2. ~/.milu

    :return: 根目录路径（不保证已存在，写入方负责按需 mkdir）。
    """
    env = os.environ.get("MILU_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".milu"


def project_dir() -> Path:
    """项目目录 ——「读配置」的锚点（与 user_data_dir() 写数据锚点相对）。

    mcp_servers.json、.env 等由项目/用户提供的配置从这里加载。集成或部署时由
    宿主应用通过 CWD 或环境变量决定，从而与「写数据」目录区分开、不互相污染。

    解析优先级：
      1. 环境变量 MILU_PROJECT_DIR（显式指定项目根，如多项目部署）
      2. 当前工作目录 CWD

    :return: 项目根目录（不保证存在；读取方自行判断 exists()）。
    """
    env = os.environ.get("MILU_PROJECT_DIR")
    if env:
        return Path(env).expanduser()
    return Path.cwd()


def default_session_dir() -> Path:
    """会话日志默认根目录：user_data_dir()/sessions。"""
    return user_data_dir() / "sessions"


def default_mcp_config_path() -> Path:
    """MCP 配置文件默认路径：user_data_dir()/mcp_servers.json。"""
    return user_data_dir() / "mcp_servers.json"


def user_config_path() -> Path:
    """用户级配置文件路径（「写数据」）：user_data_dir()/config.json。

    由 `milu config set` 写入，覆盖项目级配置（见 milu.config 分层合并）。
    """
    return user_data_dir() / "config.json"


def project_config_path() -> Path:
    """项目级配置文件路径（「读配置」）：project_dir()/config/milu.json。

    提交进仓库、写出全量默认参数；优先级低于用户级配置（见 milu.config）。
    与 mcp_servers.json 同放 config/ 下集中管理。
    """
    return project_dir() / "config" / "milu.json"


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
