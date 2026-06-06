"""调度器单实例锁（PID 文件 + 跨平台进程检活）。

多个调度引擎并存会重复执行同一批任务（引擎 tick 全量扫盘、无任务级防重），
故全局只允许一个引擎运行。三个消费方共用本锁：
  - CLI 守护进程 `milu scheduler start`（拿不到锁则拒绝启动）
  - CLI chat 进程内嵌入（拿不到锁则跳过嵌入，任务由已有进程执行）
  - Web 服务嵌入（同上）

锁文件 {data_dir}/scheduler.lock 内容为持有者 PID；持有者已不存活的
stale 锁可被覆盖。check-then-write 的 TOCTOU 窗口与 store/session 的
跨进程竞态同档取舍（两个进程同毫秒抢锁的概率极低），不引入文件锁新依赖。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def pid_alive(pid: int) -> bool:
    """检查指定 PID 的进程是否存活（跨平台）。

    注意：Windows 下切勿用 os.kill(pid, 0) 探测——它会直接 TerminateProcess。
    """
    if sys.platform == "win32":
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return bool(ok) and code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class SchedulerLock:
    """调度器单实例 PID 锁。

    用法：
        lock = SchedulerLock(user_data_dir())
        if lock.try_acquire():
            ...  # 启动引擎
        ...
        lock.release()  # finally 中释放（非持有者调用为 no-op）
    """

    def __init__(self, data_dir: Path):
        self._path = Path(data_dir) / "scheduler.lock"
        self._acquired = False

    @property
    def path(self) -> Path:
        """锁文件路径（供错误提示打印）。"""
        return self._path

    def holder_pid(self) -> int:
        """返回当前存活持有者的 PID；无持有者/已死/解析失败返回 0。"""
        if not self._path.exists():
            return 0
        try:
            # utf-8-sig：兼容带 BOM 的锁文件（如被外部工具写入）
            pid = int(self._path.read_text(encoding="utf-8-sig").strip())
        except (ValueError, OSError):
            return 0
        return pid if pid and pid_alive(pid) else 0

    def try_acquire(self) -> bool:
        """尝试获取锁：有其他存活持有者返回 False；否则写入本进程 PID（覆盖
        stale 锁）。本进程已持有时重入幂等（探测与正式获取可分离调用）。"""
        holder = self.holder_pid()
        if holder == os.getpid():
            self._acquired = True
            return True
        if holder:
            return False
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(str(os.getpid()), encoding="utf-8")
        self._acquired = True
        return True

    def release(self) -> None:
        """释放锁。仅当本实例成功 acquire 过才删锁文件——嵌入方拿不到锁时
        绝不能误删持有者的锁。"""
        if not self._acquired:
            return
        self._path.unlink(missing_ok=True)
        self._acquired = False
