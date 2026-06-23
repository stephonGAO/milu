"""网关状态存储：游标 + 消息去重的可插拔后端。

每个渠道需要记两类小状态：

- **游标 cursor**：增量拉取/轮询的位置（微信客服 sync_msg 的 next_cursor、
  Telegram getUpdates 的 offset）。重启后从上次位置续拉，**不重放积压**。
- **去重 seen**：已处理过的消息 ID（有界 LRU），防回调重试/重启重放导致**重复回复**。

两个后端：
- `InMemoryStateStore`：进程内，重启即丢（单测与「不要持久化」场景用）。
- `FileStateStore`：落 `user_data_dir()/gateway/{channel}.json`，**原子写**
  （mkstemp+fsync+os.replace，仿 scheduler/knowledge）+ **去抖刷盘**（同一渠道
  interval 内的多次写合并成一次 fsync，不每条消息都落盘）+ per-渠道 asyncio.Lock。
  跨进程并发取舍同 scheduler（单网关进程为常态，不加 flock）。
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path

from milu.resources import user_data_dir


# ── 抽象接口 ──────────────────────────────────────────────────────────
class StateStore(ABC):
    """游标 + 去重的抽象存储（按渠道命名空间隔离）。"""

    @abstractmethod
    async def get_cursor(self, channel: str, key: str) -> str:
        """读某渠道下某 key 的游标（不存在返回空串）。"""

    @abstractmethod
    async def set_cursor(self, channel: str, key: str, cursor: str) -> None:
        """写某渠道下某 key 的游标。"""

    @abstractmethod
    async def seen(self, channel: str, msg_id: str) -> bool:
        """msg_id 是否已处理过；首次见到则登记并返回 False（空 id 恒 False）。"""

    async def flush(self) -> None:
        """把待落盘的状态强制写下去（内存后端为 no-op）。"""

    async def close(self) -> None:
        """收尾：刷盘 + 释放后台任务（内存后端为 no-op）。"""


# ── 内存后端 ──────────────────────────────────────────────────────────
class InMemoryStateStore(StateStore):
    """进程内状态（重启即丢）。"""

    def __init__(self, *, dedup_capacity: int = 4096) -> None:
        self._cap = dedup_capacity
        self._cursors: dict[str, dict[str, str]] = {}
        self._seen: dict[str, "OrderedDict[str, None]"] = {}

    async def get_cursor(self, channel: str, key: str) -> str:
        return self._cursors.get(channel, {}).get(key, "")

    async def set_cursor(self, channel: str, key: str, cursor: str) -> None:
        self._cursors.setdefault(channel, {})[key] = cursor

    async def seen(self, channel: str, msg_id: str) -> bool:
        if not msg_id:
            return False
        bucket = self._seen.setdefault(channel, OrderedDict())
        if msg_id in bucket:
            return True
        bucket[msg_id] = None
        while len(bucket) > self._cap:
            bucket.popitem(last=False)
        return False


# ── 文件后端 ──────────────────────────────────────────────────────────
class FileStateStore(StateStore):
    """文件持久化状态（每渠道一个 JSON，原子写 + 去抖刷盘）。

    :param base_dir: 根目录，默认 user_data_dir()/gateway。
    :param dedup_capacity: 每渠道去重 LRU 上限。
    :param flush_interval: 去抖窗口秒数——同一渠道窗口内的多次写合并成一次落盘；
        进程崩溃最多丢失这段窗口内的状态（close() 会强制刷下最后一次）。
    """

    def __init__(
        self,
        base_dir: Path | str | None = None,
        *,
        dedup_capacity: int = 4096,
        flush_interval: float = 2.0,
    ) -> None:
        self._dir = Path(base_dir) if base_dir else (user_data_dir() / "gateway")
        self._cap = dedup_capacity
        self._interval = flush_interval
        # channel -> {"cursors": {k:v}, "seen": OrderedDict[id, None]}
        self._data: dict[str, dict] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._loaded: set[str] = set()
        self._dirty: set[str] = set()
        self._flush_tasks: dict[str, asyncio.Task] = {}

    # ── 内部：路径 / 锁 / 懒加载 ──────────────────────────────────────
    def _path(self, channel: str) -> Path:
        return self._dir / f"{channel}.json"

    def _lock(self, channel: str) -> asyncio.Lock:
        lk = self._locks.get(channel)
        if lk is None:
            lk = self._locks[channel] = asyncio.Lock()
        return lk

    async def _ensure_loaded(self, channel: str) -> dict:
        """首次访问某渠道时从磁盘载入（坏文件/缺失均退化为空状态）。"""
        if channel in self._loaded:
            return self._data[channel]
        async with self._lock(channel):
            if channel in self._loaded:  # 双重检查
                return self._data[channel]
            cursors: dict[str, str] = {}
            seen: "OrderedDict[str, None]" = OrderedDict()
            try:
                raw = json.loads(self._path(channel).read_text(encoding="utf-8"))
                cursors = dict(raw.get("cursors") or {})
                for mid in (raw.get("seen") or [])[-self._cap:]:
                    seen[str(mid)] = None
            except (FileNotFoundError, ValueError, OSError):
                pass  # 首次运行或文件损坏：从空开始
            self._data[channel] = {"cursors": cursors, "seen": seen}
            self._loaded.add(channel)
            return self._data[channel]

    # ── 读写 API ──────────────────────────────────────────────────────
    async def get_cursor(self, channel: str, key: str) -> str:
        data = await self._ensure_loaded(channel)
        return data["cursors"].get(key, "")

    async def set_cursor(self, channel: str, key: str, cursor: str) -> None:
        data = await self._ensure_loaded(channel)
        if data["cursors"].get(key) == cursor:
            return
        data["cursors"][key] = cursor
        self._schedule_flush(channel)

    async def seen(self, channel: str, msg_id: str) -> bool:
        if not msg_id:
            return False
        data = await self._ensure_loaded(channel)
        bucket: "OrderedDict[str, None]" = data["seen"]
        if msg_id in bucket:
            return True
        bucket[msg_id] = None
        while len(bucket) > self._cap:
            bucket.popitem(last=False)
        self._schedule_flush(channel)
        return False

    # ── 去抖刷盘 ──────────────────────────────────────────────────────
    def _schedule_flush(self, channel: str) -> None:
        """标记脏并安排一次尾随刷盘（窗口内多次写合并为一次落盘）。"""
        self._dirty.add(channel)
        task = self._flush_tasks.get(channel)
        if task is None or task.done():
            self._flush_tasks[channel] = asyncio.create_task(
                self._trailing_flush(channel)
            )

    async def _trailing_flush(self, channel: str) -> None:
        try:
            await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            await self._write(channel)  # 取消（停机）也要把脏数据落下去
            raise
        await self._write(channel)

    async def _write(self, channel: str) -> None:
        async with self._lock(channel):
            if channel not in self._dirty:
                return
            data = self._data.get(channel)
            if data is None:
                self._dirty.discard(channel)
                return
            obj = {
                "cursors": dict(data["cursors"]),
                "seen": list(data["seen"].keys()),
            }
            # 同步原子写（状态文件很小，微秒级；与 scheduler/knowledge 一致）。
            # 不走 to_thread：避免「写未落盘就被取消、dirty 却已清」的竞态——
            # 构造 obj→写→清 dirty 之间无 await，要么整体完成、要么整体未做，
            # 未做则 dirty 仍在、close() 的 flush() 兜底重写。
            _atomic_write_json(self._path(channel), obj)
            self._dirty.discard(channel)

    async def flush(self) -> None:
        for channel in list(self._dirty):
            await self._write(channel)

    async def close(self) -> None:
        # 取消尚未触发的尾随任务（其 CancelledError 分支会补一次落盘），再兜底全量刷
        for task in list(self._flush_tasks.values()):
            if not task.done():
                task.cancel()
        for task in list(self._flush_tasks.values()):
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        await self.flush()


def _atomic_write_json(path: Path, obj: dict) -> None:
    """原子写 JSON：同目录临时文件 + fsync + os.replace（仿 scheduler/store.py）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            json.dump(obj, tmp, ensure_ascii=False)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
