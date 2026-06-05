"""内置工具：统一文件系统操作

单一 file(action, path, ...) 工具覆盖所有文件读写操作。
支持索引、搜索、分段读取、原子写入、局部编辑、备份还原。

设计原则：
  - 能用局部编辑就不用全文覆盖
  - 能精确定位就不用模糊替换
  - 写入粒度越小，出错影响范围越小
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from typing import Literal, Optional

from milu.tools.decorator import tool

# ── 常量 ─────────────────────────────────────────────────
_MAX_READ_LINES = 200          # 单次读取最大行数
_LANDMARK_INTERVAL = 50         # 索引路标间隔（行）
_MAX_GREP_RESULTS = 20         # grep 最大返回条数
_MAX_GREP_CONTEXT = 10         # grep 上下文最大行数
_STRUCTURAL_PATTERNS = re.compile(
    r"^\s*(def |class |async def )"     # Python 函数/类
    r"|^#{1,6}\s"                        # Markdown 标题
)


# ── 内部工具函数 ──────────────────────────────────────────


def _atomic_write(path: str, content: str, encoding: str = "utf-8") -> None:
    """原子写入：先写临时文件再 rename，不会出现半截文件。"""
    dir_path = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_path, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as tmp:
            tmp.write(content)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        # 写入失败时清理临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _create_backup(path: str) -> str:
    """创建备份文件，返回备份路径。"""
    ts = int(time.time())
    backup_path = f"{path}.bak.{ts}"
    shutil.copy2(path, backup_path)
    return backup_path


def _read_lines(path: str, encoding: str = "utf-8") -> list[str]:
    """读取文件所有行。"""
    with open(path, "r", encoding=encoding) as f:
        return f.readlines()


# ── Action 实现 ───────────────────────────────────────────


def _action_index(path: str, **_) -> dict:
    """建立文件结构索引：总行数、路标、目录。"""
    if not os.path.exists(path):
        return {"success": False, "error": f"文件不存在: {path}"}
    if not os.path.isfile(path):
        return {"success": False, "error": f"不是文件: {path}"}

    lines = _read_lines(path)
    total = len(lines)

    # 每 N 行取一个路标
    landmarks = []
    for i in range(0, total, _LANDMARK_INTERVAL):
        landmarks.append({
            "line": i + 1,
            "preview": lines[i].rstrip()[:80],
        })

    # 识别结构性标记（函数、类、Markdown 标题）
    toc = []
    for i, line in enumerate(lines):
        if _STRUCTURAL_PATTERNS.search(line):
            toc.append({
                "line": i + 1,
                "text": line.rstrip()[:60],
            })

    return {
        "success": True,
        "total_lines": total,
        "landmarks": landmarks,
        "toc": toc,
    }


def _action_read(path: str, start: int | None = None,
                 end: int | None = None, **_) -> dict:
    """读取文件全文或指定行范围。"""
    if not os.path.exists(path):
        return {"success": False, "error": f"文件不存在: {path}"}

    lines = _read_lines(path)
    total = len(lines)

    # 没指定范围 → 读全文
    if start is None and end is None:
        content = "".join(lines)
        truncated = False
        if len(content) > 50000:
            content = content[:50000]
            truncated = True
        result = {
            "success": True,
            "total_lines": total,
            "content": content,
        }
        if truncated:
            result["truncated"] = True
            result["hint"] = "文件较大，请用 action=read 配合 start/end 分段读取"
        return result

    # 指定范围
    s = max(1, start or 1)
    e = min(total, end or total)
    # 硬上限
    if e - s + 1 > _MAX_READ_LINES:
        e = s + _MAX_READ_LINES - 1

    selected = lines[s - 1: e]
    return {
        "success": True,
        "total_lines": total,
        "start": s,
        "end": e,
        "has_more_before": s > 1,
        "has_more_after": e < total,
        "content": "".join(selected),
    }


def _action_grep(path: str, pattern: str | None = None,
                 context_lines: int = 2, **_) -> dict:
    """搜索关键词/正则，返回匹配行号和上下文。"""
    if not pattern:
        return {"success": False, "error": "grep 操作需要 pattern 参数"}
    if not os.path.exists(path):
        return {"success": False, "error": f"文件不存在: {path}"}

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return {"success": False, "error": f"正则表达式无效: {e}"}

    lines = _read_lines(path)
    ctx = max(0, min(context_lines, _MAX_GREP_CONTEXT))
    results = []

    for i, line in enumerate(lines):
        if regex.search(line):
            start = max(0, i - ctx)
            end = min(len(lines), i + ctx + 1)
            results.append({
                "match_line": i + 1,
                "preview": line.rstrip()[:120],
                "context_start": start + 1,
                "context_end": end,
                "context": "".join(lines[start:end]),
            })
            if len(results) >= _MAX_GREP_RESULTS:
                break

    return {
        "success": True,
        "total_matches": len(results),
        "truncated": len(results) >= _MAX_GREP_RESULTS,
        "matches": results,
    }


def _action_write(path: str, content: str | None = None,
                  backup: bool = True, **_) -> dict:
    """原子写入（全量覆盖），可选备份。"""
    if content is None:
        return {"success": False, "error": "write 操作需要 content 参数"}

    backup_path = None
    if backup and os.path.exists(path):
        backup_path = _create_backup(path)

    _atomic_write(path, content)

    result = {
        "success": True,
        "path": path,
        "bytes_written": len(content.encode("utf-8")),
    }
    if backup_path:
        result["backup"] = backup_path
    return result


def _action_append(path: str, content: str | None = None, **_) -> dict:
    """追加内容到文件末尾。"""
    if content is None:
        return {"success": False, "error": "append 操作需要 content 参数"}

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        # 如果文件有内容且不以换行结尾，先加换行
        if os.path.exists(path) and os.path.getsize(path) > 0:
            f.seek(0, 2)  # seek to end
            if f.tell() > 0:
                pass  # 追加模式已经在末尾
        f.write(content)
        if not content.endswith("\n"):
            f.write("\n")

    return {
        "success": True,
        "path": path,
        "appended_bytes": len(content.encode("utf-8")),
    }


def _action_replace(path: str, start: int | None = None,
                    end: int | None = None,
                    old: str | None = None, new: str | None = None,
                    content: str | None = None,
                    backup: bool = True, count: int = 1, **_) -> dict:
    """局部替换：行范围替换或字符串精确替换。"""
    if not os.path.exists(path):
        return {"success": False, "error": f"文件不存在: {path}"}

    # 模式 1: 行范围替换
    if start is not None and end is not None:
        if content is None:
            return {"success": False, "error": "行范围替换需要 content 参数"}
        lines = _read_lines(path)
        total = len(lines)
        if start < 1 or end > total or start > end:
            return {"success": False, "error": f"行号越界，文件共 {total} 行"}

        new_lines = content.splitlines(keepends=True)
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"

        result_lines = lines[:start - 1] + new_lines + lines[end:]

        backup_path = None
        if backup:
            backup_path = _create_backup(path)

        _atomic_write(path, "".join(result_lines))

        result = {
            "success": True,
            "mode": "line_range",
            "replaced_lines": f"{start}-{end}",
            "old_line_count": end - start + 1,
            "new_line_count": len(new_lines),
        }
        if backup_path:
            result["backup"] = backup_path
        return result

    # 模式 2: 字符串替换
    if old is not None and new is not None:
        file_content = "".join(_read_lines(path))
        occurrences = file_content.count(old)

        if occurrences == 0:
            return {"success": False, "error": "未找到目标字符串",
                    "hint": "检查空格/换行是否与文件中的完全一致"}

        if count == 1 and occurrences > 1:
            return {
                "success": False,
                "error": f"找到 {occurrences} 处匹配，请用 count 参数明确指定替换数量",
                "occurrences": occurrences,
            }

        replace_count = occurrences if count == -1 else min(count, occurrences)
        new_content = file_content.replace(old, new, replace_count)

        backup_path = None
        if backup:
            backup_path = _create_backup(path)

        _atomic_write(path, new_content)

        result = {
            "success": True,
            "mode": "string",
            "replaced": replace_count,
        }
        if backup_path:
            result["backup"] = backup_path
        return result

    return {
        "success": False,
        "error": "replace 需要: start+end+content（行范围替换）或 old+new（字符串替换）",
    }


def _action_insert(path: str, after_line: int | None = None,
                   content: str | None = None,
                   backup: bool = True, **_) -> dict:
    """在指定行后插入内容。after_line=0 表示插入到文件开头。"""
    if content is None:
        return {"success": False, "error": "insert 操作需要 content 参数"}
    if after_line is None:
        return {"success": False, "error": "insert 操作需要 after_line 参数"}
    if not os.path.exists(path):
        return {"success": False, "error": f"文件不存在: {path}"}

    lines = _read_lines(path)
    total = len(lines)
    if after_line < 0 or after_line > total:
        return {"success": False, "error": f"after_line 越界，文件共 {total} 行"}

    insert_lines = content.splitlines(keepends=True)
    if insert_lines and not insert_lines[-1].endswith("\n"):
        insert_lines[-1] += "\n"

    result_lines = lines[:after_line] + insert_lines + lines[after_line:]

    backup_path = None
    if backup:
        backup_path = _create_backup(path)

    _atomic_write(path, "".join(result_lines))

    result = {
        "success": True,
        "inserted_after_line": after_line,
        "inserted_lines": len(insert_lines),
    }
    if backup_path:
        result["backup"] = backup_path
    return result


def _action_delete(path: str, start: int | None = None,
                   end: int | None = None,
                   backup: bool = True, **_) -> dict:
    """删除指定行范围 [start, end]。"""
    if start is None or end is None:
        return {"success": False, "error": "delete 操作需要 start 和 end 参数"}
    if not os.path.exists(path):
        return {"success": False, "error": f"文件不存在: {path}"}

    lines = _read_lines(path)
    total = len(lines)
    if start < 1 or end > total or start > end:
        return {"success": False, "error": f"行号越界，文件共 {total} 行"}

    deleted_count = end - start + 1
    deleted_preview = "".join(lines[start - 1: end])[:200]
    result_lines = lines[:start - 1] + lines[end:]

    backup_path = None
    if backup:
        backup_path = _create_backup(path)

    _atomic_write(path, "".join(result_lines))

    result = {
        "success": True,
        "deleted_lines": f"{start}-{end}",
        "deleted_count": deleted_count,
        "deleted_preview": deleted_preview,
    }
    if backup_path:
        result["backup"] = backup_path
    return result


def _action_restore(path: str, **_) -> dict:
    """从备份文件还原。path 为备份文件路径（如 config.yaml.bak.1234567890）。"""
    if not os.path.exists(path):
        return {"success": False, "error": f"备份文件不存在: {path}"}

    # 解析备份路径: original_path.bak.timestamp
    match = re.match(r"^(.+)\.bak\.(\d+)$", path)
    if not match:
        return {"success": False,
                "error": "路径不是有效的备份文件（预期格式: *.bak.时间戳）"}

    original_path = match.group(1)
    shutil.copy2(path, original_path)

    return {
        "success": True,
        "restored_to": original_path,
        "backup_used": path,
    }


# ── Action 分发表 ─────────────────────────────────────────

_ACTIONS = {
    "index":   _action_index,
    "read":    _action_read,
    "grep":    _action_grep,
    "write":   _action_write,
    "append":  _action_append,
    "replace": _action_replace,
    "insert":  _action_insert,
    "delete":  _action_delete,
    "restore": _action_restore,
}


# ── 工具定义 ──────────────────────────────────────────────


@tool(name="file_read", description=(
    "文件读取与结构索引工具。通过 action 参数选择操作类型：\n"
    "index - 建立文件结构索引（总行数、目录、行号地图），不了解文件时先用此操作\n"
    "grep - 关键词/正则搜索，返回匹配行号和上下文\n"
    "read - 读取全文或指定行范围（start-end），单次最多200行"
), is_safe=True)
async def file_read(
    action: Literal["index", "read", "grep"],
    path: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
    pattern: Optional[str] = None,
    context_lines: int = 2,
) -> str:
    """
    文件读取与结构索引工具。

    :param action: 操作类型
    :param path: 文件路径
    :param start: 起始行号（read 时使用，从1开始）
    :param end: 结束行号（read 时使用）
    :param pattern: 搜索关键词或正则表达式（grep 时使用）
    :param context_lines: grep 时匹配行前后显示的上下文行数
    """
    handler = _ACTIONS.get(action)
    if not handler or action not in ("index", "read", "grep"):
        return json.dumps({
            "success": False,
            "error": f"未知 action: {action}",
            "available": ["index", "read", "grep"],
        }, ensure_ascii=False)

    try:
        result = handler(
            path=path, start=start, end=end,
            pattern=pattern, context_lines=context_lines,
        )
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"{action} 操作失败: {e}",
        }, ensure_ascii=False)


@tool(name="file_write", description=(
    "文件修改与写入工具。通过 action 参数选择操作类型：\n"
    "write - 原子写入（全量覆盖），backup=True时自动备份原文件\n"
    "append - 追加内容到文件末尾\n"
    "replace - 局部替换：传start+end+content为行范围替换，传old+new为字符串精确替换\n"
    "insert - 在指定行后插入内容（after_line=0为文件开头）\n"
    "delete - 删除指定行范围\n"
    "restore - 从备份文件还原"
), is_safe=False)
async def file_write(
    action: Literal["write", "append", "replace", "insert", "delete", "restore"],
    path: str,
    content: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    old: Optional[str] = None,
    new: Optional[str] = None,
    after_line: Optional[int] = None,
    backup: bool = True,
    count: int = 1,
) -> str:
    """
    文件修改与写入工具。

    :param action: 操作类型
    :param path: 文件路径（restore 时传备份文件路径）
    :param content: 写入或替换的内容（write/append/replace行范围 时使用）
    :param start: 起始行号（replace/delete 时使用，从1开始）
    :param end: 结束行号（replace/delete 时使用）
    :param old: 替换源字符串（replace 字符串模式时使用）
    :param new: 替换目标字符串（replace 字符串模式时使用）
    :param after_line: 插入位置行号（insert 时使用，0=文件开头）
    :param backup: 写入/替换/删除操作前是否自动备份原文件
    :param count: 字符串替换次数（1=仅第一处，-1=全部）
    """
    handler = _ACTIONS.get(action)
    if not handler or action not in ("write", "append", "replace", "insert", "delete", "restore"):
        return json.dumps({
            "success": False,
            "error": f"未知 action: {action}",
            "available": ["write", "append", "replace", "insert", "delete", "restore"],
        }, ensure_ascii=False)

    try:
        result = handler(
            path=path, content=content, start=start, end=end,
            old=old, new=new, after_line=after_line,
            backup=backup, count=count,
        )
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"{action} 操作失败: {e}",
        }, ensure_ascii=False)
