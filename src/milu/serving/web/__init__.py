"""milu 内置 Web 服务子包 —— FastAPI 应用 + 全功能演示前端。

一键启动（CLI 子命令）：
    milu serve

程序化：
    from milu.serving.web import create_app, run_server

fastapi / uvicorn / sse-starlette 现为核心依赖（随 milu 一并安装）。本子包仍保持
延迟导入——仅在调用 create_app() / run_server() 时才真正 import，故即便这些依赖被
手动卸载，`import milu.serving.web` 也不会报错。
"""
from __future__ import annotations

__all__ = ["create_app", "run_server", "find_available_port"]


def create_app(*args, **kwargs):
    """构造 FastAPI 应用（延迟导入 fastapi）。"""
    from .app import create_app as _impl
    return _impl(*args, **kwargs)


def run_server(*args, **kwargs):
    """启动 uvicorn 服务（延迟导入 uvicorn）。"""
    from .app import run_server as _impl
    return _impl(*args, **kwargs)


def find_available_port(*args, **kwargs):
    """从指定端口起向后探测一个可绑定的端口（延迟导入）。"""
    from .app import find_available_port as _impl
    return _impl(*args, **kwargs)
