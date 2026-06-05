"""会话持久化 — JSONL 对话日志 + 元数据管理

会话默认落在用户级数据目录 `user_data_dir()/sessions/`（即 `~/.milu/sessions`，
`MILU_HOME` 可覆盖），与 CWD 解耦；base_dir 由调用方显式传入。每个会话存储在
`{base_dir}/{session_id}/` 目录下：
  - conversation.jsonl: 每行一条消息的完整日志（append-only）
  - session.json: 会话元数据（创建时间、消息数、模型等）

日志分段（reset 支持）：reset() 会在同一会话目录下切换到新的空日志段——
首段为 conversation.jsonl，之后依次为 conversation.2.jsonl、conversation.3.jsonl…
旧段只新建不重命名/删除（天然归档）；活动段 = 目录中段号最大的文件（磁盘即真相），
load_messages() / 计数器恢复都只针对活动段，因此 reset 后加载不再包含旧历史。

SYSTEM 消息不记录（每轮由 _build_system_prompt() 重建）。
ASSISTANT 消息触发 round 计数器自增。
"""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any

from milu.llm.base.message import Message, MessageRole

logger = logging.getLogger(__name__)


# ── 日志分段命名 ─────────────────────────────────────────────
#
# 首段 conversation.jsonl（段号 1，向后兼容旧会话），reset 后依次
# conversation.2.jsonl、conversation.3.jsonl…
_SEG_RE = re.compile(r"^conversation(?:\.(\d+))?\.jsonl$")


def _segment_index(filename: str) -> int | None:
    """解析日志段文件名的段号：conversation.jsonl→1，conversation.N.jsonl→N，其它→None。"""
    m = _SEG_RE.match(filename)
    if not m:
        return None
    return int(m.group(1)) if m.group(1) else 1


def _serialize_content(content) -> str | list:
    """序列化消息 content 供 JSONL 记录。

    str 原样；list（多模态轻量块，如 image_path / text，不含 base64）原样存为
    JSON 数组，load 时经 _dict_to_message 透传恢复为 list；其余类型 str() 兜底。
    """
    if isinstance(content, (str, list)):
        return content
    return str(content or "")


# ── 跨进程文件锁（写入健壮性）────────────────────────────────
#
# 多 worker 部署下，若同一会话日志被多个进程写入（未做粘性路由），裸 append
# 可能交错损坏。这里提供独占文件锁：Linux/macOS 用 fcntl.flock，其它平台优雅降级。
# Windows 单进程开发环境下为 no-op（单进程内写入本身是同步、不交错的）。
try:
    import fcntl  # POSIX

    def _lock_file(f) -> None:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)

    def _unlock_file(f) -> None:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
except ImportError:  # 非 POSIX（如 Windows）：降级为 no-op
    def _lock_file(f) -> None:  # noqa: D401
        pass

    def _unlock_file(f) -> None:
        pass


class Session:
    """会话管理器 — 对话日志持久化和元数据跟踪。

    用法（base_dir 通常取 default_session_dir() = ~/.milu/sessions）：
        from milu.resources import default_session_dir
        session = Session(Session.generate_id(), default_session_dir(), model="qwen-max")
        session.log_message(user_msg)     # 追加到 JSONL
        session.log_message(assistant_msg) # round 计数器自增
        session.save_metadata()            # 写 session.json
    """

    def __init__(
        self,
        session_id: str,
        base_dir: Path,
        model: str = "",
    ):
        self._session_id = session_id
        self._base_dir = Path(base_dir)
        self._model = model
        self._dir = self._base_dir / session_id
        self._dir.mkdir(parents=True, exist_ok=True)

        # 活动日志段：取目录中段号最大的文件（reset 后定位到新段）
        self._active_segment = self._scan_max_segment(self._dir)

        self._line_counter = 0
        self._round_counter = 0
        self._message_count = 0
        self._created_at = time.time()

        # 如果活动段 JSONL 已存在（加载已有会话），计算当前行数和 round
        conv_path = self.conversation_path
        if conv_path.exists():
            self._scan_existing()

    @staticmethod
    def _segment_filename(idx: int) -> str:
        """段号 → 文件名：1 → conversation.jsonl，N → conversation.N.jsonl。"""
        return "conversation.jsonl" if idx <= 1 else f"conversation.{idx}.jsonl"

    @staticmethod
    def _scan_max_segment(dir_path: Path) -> int:
        """扫描目录中最大的日志段号；无任何段时返回 1（首段约定名）。"""
        max_idx = 0
        try:
            for entry in dir_path.iterdir():
                idx = _segment_index(entry.name)
                if idx is not None and idx > max_idx:
                    max_idx = idx
        except OSError as e:
            logger.warning("扫描日志段失败: %s", e)
        return max_idx or 1

    def _scan_existing(self) -> None:
        """扫描已有的 JSONL 文件（活动段），恢复计数器。"""
        try:
            with open(self.conversation_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self._line_counter += 1
                    try:
                        obj = json.loads(line)
                        # 跳过压缩事件（不是消息）
                        if obj.get("type") == "compaction":
                            continue
                        self._message_count += 1
                        if obj.get("role") == MessageRole.ASSISTANT.value:
                            round_val = obj.get("round", 0)
                            if round_val > self._round_counter:
                                self._round_counter = round_val
                    except (json.JSONDecodeError, KeyError):
                        pass
        except OSError as e:
            logger.warning("扫描已有 JSONL 失败: %s", e)

    # ── 写入 ──────────────────────────────────────────────────

    def _append_jsonl(self, record: dict) -> bool:
        """以独占文件锁追加一条 JSONL 记录（跨进程写入安全）。

        - 锁保证多 worker 写同一会话文件时整行不交错；
        - 读端（load_messages/_scan_existing）已能跳过损坏/半截行，故无需读锁；
        - flush + 关闭即落盘，保证崩溃后已写记录可恢复。

        :return: 成功返回 True，IO 失败返回 False。
        """
        line = json.dumps(record, ensure_ascii=False) + "\n"
        try:
            with open(self.conversation_path, "a", encoding="utf-8") as f:
                _lock_file(f)
                try:
                    f.write(line)
                    f.flush()
                finally:
                    _unlock_file(f)
        except OSError as e:
            logger.warning("写入 JSONL 失败: %s", e)
            return False
        return True

    def log_message(self, message: Message) -> int:
        """追加一条消息到 JSONL。

        :return: 行号（0-based），SYSTEM 消息返回 -1（不记录）
        """
        if message.role == MessageRole.SYSTEM:
            return -1

        # ASSISTANT 消息自增 round
        if message.role == MessageRole.ASSISTANT:
            self._round_counter += 1

        record: dict[str, Any] = {
            "line": self._line_counter,
            "round": self._round_counter,
            "role": message.role.value,
            "content": _serialize_content(message.content),
            "timestamp": time.time(),
        }

        # 可选字段
        if message.tool_calls:
            record["tool_calls"] = message.tool_calls
        if message.tool_call_id:
            record["tool_call_id"] = message.tool_call_id
        if message.name:
            record["name"] = message.name

        if not self._append_jsonl(record):
            return -1

        line_no = self._line_counter
        self._line_counter += 1
        self._message_count += 1
        return line_no

    def log_compaction(self, messages: list[Message]) -> int:
        """记录压缩事件到 JSONL。

        保存压缩后的消息快照，后续 load_messages() 从此点恢复。

        :param messages: 压缩后的消息列表
        :return: 行号，失败返回 -1
        """
        serialized = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                continue
            entry: dict[str, Any] = {
                "role": msg.role.value,
                "content": _serialize_content(msg.content),
            }
            if msg.tool_calls:
                entry["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id
            if msg.name:
                entry["name"] = msg.name
            serialized.append(entry)

        record: dict[str, Any] = {
            "type": "compaction",
            "line": self._line_counter,
            "timestamp": time.time(),
            "message_count": len(serialized),
            "messages": serialized,
        }

        if not self._append_jsonl(record):
            return -1

        line_no = self._line_counter
        self._line_counter += 1
        return line_no

    # ── 重置（切换日志分段）──────────────────────────────────

    def reset(self) -> int:
        """切换到新的空日志段（同一会话目录，旧段保留），计数器归零。

        - 用独占创建（"x"）探测可用段号：并发/多 worker 同时 reset 时各拿到
          不同的新段，不会互相覆盖；
        - 新空段立即落盘：进程重启 / AgentPool 重建 / load_session 时按
          「最大段号」即可定位到新段，而不会回退到旧段加载旧历史。

        :return: 新段号；创建失败时保持旧段并返回当前段号（降级，不丢数据）。
        """
        next_idx = max(self._active_segment, self._scan_max_segment(self._dir)) + 1
        while True:
            candidate = self._dir / self._segment_filename(next_idx)
            try:
                with open(candidate, "x", encoding="utf-8"):
                    pass
                break
            except FileExistsError:
                next_idx += 1
            except OSError as e:
                logger.warning("创建新日志段失败: %s", e)
                return self._active_segment

        self._active_segment = next_idx
        self._line_counter = 0
        self._round_counter = 0
        self._message_count = 0
        self.save_metadata()
        logger.info("会话 %s 已切换到新日志段: %s", self._session_id, candidate.name)
        return next_idx

    # ── 读取 ──────────────────────────────────────────────────

    def load_messages(self) -> list[Message]:
        """读取 JSONL，重建 Message 列表（不含 system）。

        如果存在压缩事件，从最后一个压缩快照恢复，只加载压缩后的消息。
        跳过损坏的行（崩溃恢复）。
        """
        if not self.conversation_path.exists():
            return []

        # 第一遍：找到最后一个 compaction 事件的位置
        last_compaction_line = -1
        last_compaction_messages: list[Message] | None = None

        try:
            with open(self.conversation_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if obj.get("type") == "compaction":
                            last_compaction_line = i
                            last_compaction_messages = []
                            for m_obj in obj.get("messages", []):
                                msg = self._dict_to_message(m_obj)
                                if msg is not None:
                                    last_compaction_messages.append(msg)
                    except (json.JSONDecodeError, KeyError):
                        continue
        except OSError as e:
            logger.warning("扫描压缩事件失败: %s", e)

        # 第二遍：从压缩事件之后读取后续消息
        messages: list[Message] = []
        if last_compaction_messages is not None:
            messages = list(last_compaction_messages)
            # 读取 compaction 之后的新增消息
            try:
                with open(self.conversation_path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if i <= last_compaction_line:
                            continue
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            # 跳过后续可能出现的 compaction 事件（不应发生，但防御性处理）
                            if obj.get("type") == "compaction":
                                continue
                            msg = self._dict_to_message(obj)
                            if msg is not None:
                                messages.append(msg)
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.debug("跳过损坏的 JSONL 行: %s", e)
                            continue
            except OSError as e:
                logger.warning("读取 JSONL 失败: %s", e)
        else:
            # 无压缩事件，正常全量加载
            try:
                with open(self.conversation_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            if obj.get("type") == "compaction":
                                continue  # 防御性跳过
                            msg = self._dict_to_message(obj)
                            if msg is not None:
                                messages.append(msg)
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.debug("跳过损坏的 JSONL 行: %s", e)
                            continue
            except OSError as e:
                logger.warning("读取 JSONL 失败: %s", e)

        return messages

    def read_line(self, n: int) -> dict | None:
        """读取指定行号的原始记录（调试用）。"""
        if not self.conversation_path.exists():
            return None

        try:
            with open(self.conversation_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i == n:
                        return json.loads(line.strip())
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("读取第 %d 行失败: %s", n, e)

        return None

    @staticmethod
    def _dict_to_message(obj: dict) -> Message | None:
        """将 JSONL 记录转为 Message 对象。"""
        role_str = obj.get("role")
        if not role_str:
            return None

        try:
            role = MessageRole(role_str)
        except ValueError:
            return None

        return Message(
            role=role,
            content=obj.get("content", ""),
            tool_calls=obj.get("tool_calls"),
            tool_call_id=obj.get("tool_call_id"),
            name=obj.get("name"),
        )

    # ── 元数据 ────────────────────────────────────────────────

    def save_metadata(self, **extra) -> None:
        """写 session.json 元数据文件。"""
        metadata = {
            "session_id": self._session_id,
            "model": self._model,
            "created_at": self._created_at,
            "updated_at": time.time(),
            "message_count": self._message_count,
            "round_count": self._round_counter,
            "active_log": self._segment_filename(self._active_segment),
        }
        metadata.update(extra)

        try:
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.warning("写入 session.json 失败: %s", e)

    # ── 类方法 ────────────────────────────────────────────────

    @classmethod
    def generate_id(cls) -> str:
        """生成会话 ID: YYYYMMDD_HHMMSS_{16hex}

        随机部分用 8 字节（64 bit）。原先仅 2 字节（16 bit），同一秒内自动
        创建约 256 个会话即有 ~50% 概率碰撞（生日悖论）→ 不同会话写进同一目录、
        JSONL 混写。64 bit 下即使每秒百万级会话，碰撞概率也可忽略。
        """
        now = time.localtime()
        ts = time.strftime("%Y%m%d_%H%M%S", now)
        rand = secrets.token_hex(8)
        return f"{ts}_{rand}"

    @classmethod
    def list_sessions(cls, base_dir: Path) -> list[dict]:
        """列出所有会话，返回元数据列表。

        按 updated_at 降序排列。缺少 session.json 的会话从 JSONL 推断信息。
        """
        base = Path(base_dir)
        if not base.exists():
            return []

        sessions = []
        for entry in base.iterdir():
            if not entry.is_dir():
                continue

            meta_path = entry / "session.json"
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    meta["session_id"] = entry.name
                    sessions.append(meta)
                    continue
                except (OSError, json.JSONDecodeError):
                    pass

            # 没有 session.json，从目录名和活动段 JSONL 推断
            # （与 metadata 路径语义一致：消息数 = 当前活动段行数，即 load 可恢复的量）
            conv_path = entry / cls._segment_filename(cls._scan_max_segment(entry))
            msg_count = 0
            if conv_path.exists():
                try:
                    with open(conv_path, "r", encoding="utf-8") as f:
                        msg_count = sum(1 for line in f if line.strip())
                except OSError:
                    pass

            sessions.append({
                "session_id": entry.name,
                "model": "",
                "message_count": msg_count,
                "updated_at": entry.stat().st_mtime if entry.exists() else 0,
            })

        # 按 updated_at 降序
        sessions.sort(key=lambda s: s.get("updated_at", 0), reverse=True)
        return sessions

    @classmethod
    def load_session(cls, session_id: str, base_dir: Path) -> "Session":
        """加载已有会话。"""
        base = Path(base_dir)
        meta_path = base / session_id / "session.json"

        model = ""
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                model = meta.get("model", "")
            except (OSError, json.JSONDecodeError):
                pass

        session = cls(session_id, base, model=model)

        # 恢复 created_at
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                session._created_at = meta.get("created_at", session._created_at)
            except (OSError, json.JSONDecodeError):
                pass

        return session

    # ── 属性 ──────────────────────────────────────────────────

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def conversation_path(self) -> Path:
        """当前活动日志段的文件路径（首段 conversation.jsonl，reset 后为 conversation.N.jsonl）"""
        return self._dir / self._segment_filename(self._active_segment)

    @property
    def metadata_path(self) -> Path:
        """session.json 文件路径"""
        return self._dir / "session.json"

    @property
    def dir_path(self) -> Path:
        """会话目录路径"""
        return self._dir

    @property
    def message_count(self) -> int:
        return self._message_count

    @property
    def round_count(self) -> int:
        return self._round_counter

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def __repr__(self) -> str:
        return f"Session(id={self._session_id!r}, messages={self._message_count}, rounds={self._round_counter})"
