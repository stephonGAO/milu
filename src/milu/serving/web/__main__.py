"""支持 `python -m milu.serving.web` 直接启动内置 Web 服务。

等价于 CLI 的 `milu serve`（默认 127.0.0.1:8000）。host/port 经环境变量
HOST / PORT 覆盖；更多选项（厂商/模型/模式/开关）请用 `milu serve --help`，
或在自己的代码里调用 create_app() / run_server()。
"""
from __future__ import annotations

import os

from milu.serving.web import run_server

if __name__ == "__main__":
    run_server(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
    )
