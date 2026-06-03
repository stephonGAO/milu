"""集中管理 .env 加载，避免库被集成时隐式、重复地扫描宿主进程的当前工作目录（CWD）。

背景：原先 `BaseLLM._get_api_key()` 与 `Agent.connect_mcp()` 会在每次取 API Key /
解析 MCP 配置时静默调用 `load_dotenv()`，扫描调用方 CWD 的 .env。作为被 pip 安装、
嵌入到他人项目里的库，这种隐式 CWD 行为不可预测。

本模块提供：
- `ensure_dotenv_loaded()`：进程内最多加载一次（幂等）；
- 可通过环境变量 `AGENT_FRAMEWORK_NO_DOTENV=1` 完全关闭（嵌入式/生产部署推荐，
  改由宿主应用自行管理环境变量）；
- 任何异常都被吞掉，绝不因 .env 问题让库崩溃；
- `load_env()`：供宿主应用显式加载（公开 API）。
"""
from __future__ import annotations

import os

# 进程内是否已尝试加载过（幂等保护，避免重复扫描 CWD）
_loaded = False

_TRUTHY = {"1", "true", "yes", "on"}


def _disabled() -> bool:
    """是否通过环境变量关闭了自动加载 .env。"""
    return os.environ.get("AGENT_FRAMEWORK_NO_DOTENV", "").strip().lower() in _TRUTHY


def ensure_dotenv_loaded() -> None:
    """库内部使用：进程内首次调用时加载 .env（幂等、可关闭、不抛异常）。

    - 已加载过 → 直接返回；
    - 设置了 AGENT_FRAMEWORK_NO_DOTENV → 跳过，不触碰 CWD；
    - python-dotenv 缺失或 .env 读取失败 → 静默忽略。
    """
    global _loaded
    if _loaded or _disabled():
        _loaded = True
        return
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    finally:
        _loaded = True


def load_env(path: "str | os.PathLike | None" = None, override: bool = False) -> None:
    """显式加载 .env（供宿主应用主动调用，绕过幂等限制）。

    :param path: .env 路径；None 时由 python-dotenv 自动向上查找。
    :param override: 是否覆盖已存在的环境变量（默认 False，不覆盖宿主已设置的值）。
    """
    global _loaded
    try:
        from dotenv import load_dotenv
        load_dotenv(path, override=override)
    except Exception:
        pass
    finally:
        _loaded = True
