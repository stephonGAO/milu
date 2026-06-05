"""自我保护守卫：防止 Agent 修改 milu 自身源码和程序配置文件。

保护范围：
  - milu 包源码目录（import milu 解析得到的 __file__ 所在 src/milu/）
  - 项目构建配置（pyproject.toml、setup.py、setup.cfg）
  - 项目级 milu 配置（config/milu.json、config/mcp_servers.json）
  - 环境变量文件（.env、.env.*）

不在保护范围：
  - 用户级 milu 配置（~/.milu/config.json 等）——用户有权通过 agent 调整
"""
from __future__ import annotations

import os


def _get_milu_pkg_dir() -> str | None:
    """返回 milu 包目录的规范化绝对路径。"""
    try:
        import milu
        return os.path.normcase(os.path.normpath(os.path.dirname(os.path.abspath(milu.__file__))))
    except Exception:
        return None


def _get_project_root() -> str | None:
    """返回项目根目录：milu 包目录若在 src/<pkg>/ 下，则上两级为项目根。"""
    pkg_dir = _get_milu_pkg_dir()
    if not pkg_dir:
        return None
    parent = os.path.dirname(pkg_dir)          # e.g. .../src
    grandparent = os.path.dirname(parent)       # e.g. .../my-agent
    if os.path.basename(parent).lower() == "src":
        return grandparent
    # 非 src 布局时，退回到包目录的上级作为猜测
    return parent


def _build_protected_set() -> tuple[list[str], list[str]]:
    """构建受保护目录列表和受保护文件列表（均已规范化）。

    Returns:
        (protected_dirs, protected_files)
    """
    protected_dirs: list[str] = []
    protected_files: list[str] = []

    # 1. milu 包源码目录
    pkg_dir = _get_milu_pkg_dir()
    if pkg_dir:
        protected_dirs.append(pkg_dir)

    # 2. 项目根级构建/配置文件
    project_root = _get_project_root()
    if project_root:
        for fname in ("pyproject.toml", "setup.py", "setup.cfg", "hatch.ini",
                      ".env", ".env.local", ".env.production"):
            p = os.path.normcase(os.path.normpath(os.path.join(project_root, fname)))
            protected_files.append(p)
        # 项目级 milu 配置
        for fname in ("config/milu.json", "config/mcp_servers.json"):
            p = os.path.normcase(os.path.normpath(os.path.join(project_root, fname)))
            protected_files.append(p)

    # 用户级 milu 配置（~/.milu/config.json 等）不在保护范围内，
    # 用户有权通过 agent 调整自己的配置。

    return protected_dirs, protected_files


# 模块级缓存（路径在进程生命周期内不变）
_PROTECTED_DIRS: list[str] = []
_PROTECTED_FILES: list[str] = []
_INITIALIZED = False

# 功能总开关（默认启用；CLI/builder 启动时按配置覆盖）
_ENABLED: bool = True


def set_enabled(enabled: bool) -> None:
    """设置自我保护功能的启用状态（由 CLI/builder 在启动时调用）。"""
    global _ENABLED
    _ENABLED = enabled


def _ensure_init() -> None:
    global _PROTECTED_DIRS, _PROTECTED_FILES, _INITIALIZED
    if not _INITIALIZED:
        _PROTECTED_DIRS, _PROTECTED_FILES = _build_protected_set()
        _INITIALIZED = True


def is_protected_path(path: str) -> bool:
    """判断路径是否属于受保护区域（读写均禁止）。功能关闭时始终返回 False。"""
    if not _ENABLED:
        return False
    _ensure_init()
    try:
        norm = os.path.normcase(os.path.normpath(os.path.abspath(path)))
        for d in _PROTECTED_DIRS:
            if norm == d or norm.startswith(d + os.sep):
                return True
        for f in _PROTECTED_FILES:
            if norm == f:
                return True
    except Exception:
        pass
    return False


def selfguard_error(path: str, reason: str = "代码") -> dict:
    """返回统一的自我保护错误响应 dict。"""
    return {
        "success": False,
        "error": f"安全限制：禁止访问 milu 自身{reason}",
        "protected_path": path,
        "hint": "milu 程序文件属于受保护区域，Agent 无权读取或修改",
    }
