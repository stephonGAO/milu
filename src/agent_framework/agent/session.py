"""会话持久化 — JSONL 对话日志 + 元数据管理

每个会话存储在 `.sessions/{session_id}/` 目录下：
  - conversation.jsonl: 每行一条消息的完整日志（append-only）
  - session.json: 会话元数据（创建时间、消息数、模型等）

SYSTEM 消息不记录（每轮由 _build_system_prompt() 重建）。
ASSISTANT 消息触发 round 计数器自增。
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Any

from agent_framework.llm.base.message import Message, MessageRole

logger = logging.getLogger(__name__)


class Session:
    """会话管理器 — 对话日志持久化和元数据跟踪。

    用法：
        session = Session(Session.generate_id(), Path(".sessions"), model="qwen-max")
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

        self._line_counter = 0
        self._round_counter = 0
        self._message_count = 0
        self._created_at = time.time()

        # 如果 conversation.jsonl 已存在（加载已有会话），计算当前行数和 round
        conv_path = self.conversation_path
        if conv_path.exists():
            self._scan_existing()

    def _scan_existing(self) -> None:
        """扫描已有的 JSONL 文件，恢复计数器。"""
        try:
            with open(self.conversation_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self._line_counter += 1
                    try:
                        obj = json.loads(line)
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
            "content": message.content if isinstance(message.content, str) else str(message.content or ""),
            "timestamp": time.time(),
        }

        # 可选字段
        if message.tool_calls:
            record["tool_calls"] = message.tool_calls
        if message.tool_call_id:
            record["tool_call_id"] = message.tool_call_id
        if message.name:
            record["name"] = message.name

        try:
            with open(self.conversation_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning("写入 JSONL 失败: %s", e)
            return -1

        line_no = self._line_counter
        self._line_counter += 1
        self._message_count += 1
        return line_no

    # ── 读取 ──────────────────────────────────────────────────

    def load_messages(self) -> list[Message]:
        """读取全部 JSONL，重建 Message 列表（不含 system）。

        跳过损坏的行（崩溃恢复）。
        """
        messages: list[Message] = []
        if not self.conversation_path.exists():
            return messages

        try:
            with open(self.conversation_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
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
        """生成会话 ID: YYYYMMDD_HHMMSS_{4hex}"""
        now = time.localtime()
        ts = time.strftime("%Y%m%d_%H%M%S", now)
        rand = secrets.token_hex(2)
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

            # 没有 session.json，从目录名和 JSONL 推断
            conv_path = entry / "conversation.jsonl"
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
        """conversation.jsonl 文件路径"""
        return self._dir / "conversation.jsonl"

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
