"""跨平台系统通知（零额外依赖）。

- Windows：ctypes MessageBoxW（在独立线程弹出，非阻塞）
- macOS：osascript
- Linux：notify-send
"""
from __future__ import annotations

import logging
import subprocess
import sys
import threading

logger = logging.getLogger(__name__)


def send(title: str, message: str) -> None:
    """弹出系统通知，失败时静默记录 warning。"""
    try:
        if sys.platform == "win32":
            _notify_windows(title, message)
        elif sys.platform == "darwin":
            _notify_macos(title, message)
        else:
            _notify_linux(title, message)
    except Exception as e:
        logger.warning("系统通知发送失败: %s", e)


def _notify_windows(title: str, message: str) -> None:
    # 用 ctypes 弹出信息对话框——零依赖、必然可见
    # 在 daemon 线程里运行，不阻塞调度器主循环
    import ctypes
    MB_ICONINFORMATION = 0x40
    MB_TOPMOST = 0x40000
    MB_SETFOREGROUND = 0x10000
    flags = MB_ICONINFORMATION | MB_TOPMOST | MB_SETFOREGROUND
    threading.Thread(
        target=ctypes.windll.user32.MessageBoxW,
        args=(None, message, title, flags),
        daemon=True,
        name=f"notify-{title[:20]}",
    ).start()


def _notify_macos(title: str, message: str) -> None:
    t = title.replace('"', '\\"')
    m = message.replace('"', '\\"')
    subprocess.Popen(
        ["osascript", "-e", f'display notification "{m}" with title "{t}"'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _notify_linux(title: str, message: str) -> None:
    subprocess.Popen(
        ["notify-send", title, message],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
