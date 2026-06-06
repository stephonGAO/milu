"""milu 内置 Web 服务子包 —— FastAPI 应用 + 全功能演示前端。

一键启动（CLI 子命令）：
    milu serve

程序化：
    from milu.serving.web import create_app, run_server

fastapi / uvicorn / sse-starlette 为 `[serve]` 可选依赖（`pip install "milu[serve]"`）。
本子包延迟导入它们——仅在调用 create_app() / run_server() 时才真正 import，
故未安装这些依赖时 `import milu.serving.web` 不会报错。
"""
from __future__ import annotations

__all__ = ["create_app", "run_server"]


def create_app(*args, **kwargs):
    """构造 FastAPI 应用（延迟导入 fastapi）。"""
    from .app import create_app as _impl
    return _impl(*args, **kwargs)


def run_server(*args, **kwargs):
    """启动 uvicorn 服务（延迟导入 uvicorn）。"""
    from .app import run_server as _impl
    return _impl(*args, **kwargs)
