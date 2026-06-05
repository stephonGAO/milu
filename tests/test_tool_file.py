"""测试内置工具 file - 统一文件系统操作（9 种 action）"""
import json
import os
import re
import time

import pytest
from milu.tools.builtin.file_tool import file_read, file_write


# ── 辅助 ─────────────────────────────────────────────────


def _parse(result_str: str) -> dict:
    """解析工具返回的 JSON 字符串"""
    return json.loads(result_str)


def _make_file(tmp_path, name: str, content: str) -> str:
    """创建测试文件，返回路径"""
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


# ── action: index ─────────────────────────────────────────


class TestFileIndex:
    """file(action='index') 测试"""

    @pytest.mark.asyncio
    async def test_index_python_file(self, tmp_path):
        """索引 Python 文件，识别 def/class"""
        content = "import os\n\nclass Foo:\n    pass\n\ndef bar():\n    return 1\n"
        path = _make_file(tmp_path, "test.py", content)

        result = _parse(await file_read(action="index", path=path))
        assert result["success"] is True
        assert result["total_lines"] == 7
        toc_texts = [item["text"] for item in result["toc"]]
        assert any("class Foo" in t for t in toc_texts)
        assert any("def bar" in t for t in toc_texts)

    @pytest.mark.asyncio
    async def test_index_markdown_file(self, tmp_path):
        """索引 Markdown 文件，识别标题"""
        content = "# 标题\n内容\n## 第二章\n更多内容\n### 2.1 节\n"
        path = _make_file(tmp_path, "doc.md", content)

        result = _parse(await file_read(action="index", path=path))
        assert result["success"] is True
        assert result["total_lines"] == 5
        assert len(result["toc"]) == 3  # 三个标题

    @pytest.mark.asyncio
    async def test_index_landmarks(self, tmp_path):
        """大文件生成路标"""
        lines = [f"line {i}" for i in range(200)]
        path = _make_file(tmp_path, "big.txt", "\n".join(lines))

        result = _parse(await file_read(action="index", path=path))
        assert result["total_lines"] == 200
        # 200行 / 50间隔 = 4 个路标
        assert len(result["landmarks"]) == 4
        assert result["landmarks"][0]["line"] == 1

    @pytest.mark.asyncio
    async def test_index_nonexistent(self):
        """不存在的文件返回错误"""
        result = _parse(await file_read(action="index", path="/no/such/file.txt"))
        assert result["success"] is False
        assert "不存在" in result["error"]


# ── action: read ──────────────────────────────────────────


class TestFileRead:
    """file(action='read') 测试"""

    @pytest.mark.asyncio
    async def test_read_full(self, tmp_path):
        """读取完整文件"""
        path = _make_file(tmp_path, "test.txt", "Hello\nWorld\n")
        result = _parse(await file_read(action="read", path=path))
        assert result["success"] is True
        assert "Hello" in result["content"]
        assert result["total_lines"] == 2  # "Hello\n" + "World\n"

    @pytest.mark.asyncio
    async def test_read_range(self, tmp_path):
        """读取指定行范围"""
        lines = [f"line {i}\n" for i in range(1, 21)]
        path = _make_file(tmp_path, "test.txt", "".join(lines))

        result = _parse(await file_read(action="read", path=path, start=5, end=10))
        assert result["success"] is True
        assert result["start"] == 5
        assert result["end"] == 10
        assert "line 5" in result["content"]
        assert "line 10" in result["content"]
        assert result["has_more_before"] is True
        assert result["has_more_after"] is True

    @pytest.mark.asyncio
    async def test_read_range_max_lines_cap(self, tmp_path):
        """超过 MAX_READ_LINES 被限制"""
        lines = [f"line {i}\n" for i in range(1, 501)]
        path = _make_file(tmp_path, "test.txt", "".join(lines))

        result = _parse(await file_read(action="read", path=path, start=1, end=500))
        assert result["success"] is True
        # 被限制到 200 行
        assert result["end"] - result["start"] + 1 <= 200
        assert result["has_more_after"] is True

    @pytest.mark.asyncio
    async def test_read_first_lines(self, tmp_path):
        """读开头几行，has_more_before=False"""
        lines = [f"line {i}\n" for i in range(1, 21)]
        path = _make_file(tmp_path, "test.txt", "".join(lines))

        result = _parse(await file_read(action="read", path=path, start=1, end=3))
        assert result["has_more_before"] is False
        assert result["has_more_after"] is True

    @pytest.mark.asyncio
    async def test_read_nonexistent(self):
        """不存在的文件"""
        result = _parse(await file_read(action="read", path="/no/file.txt"))
        assert result["success"] is False


# ── action: grep ──────────────────────────────────────────


class TestFileGrep:
    """file(action='grep') 测试"""

    @pytest.mark.asyncio
    async def test_grep_found(self, tmp_path):
        """搜索到匹配行"""
        content = "host: localhost\ntimeout: 30\nport: 5432\nretry: 3\n"
        path = _make_file(tmp_path, "config.yaml", content)

        result = _parse(await file_read(action="grep", path=path, pattern="timeout"))
        assert result["success"] is True
        assert result["total_matches"] == 1
        assert result["matches"][0]["match_line"] == 2
        assert "timeout" in result["matches"][0]["preview"]

    @pytest.mark.asyncio
    async def test_grep_with_context(self, tmp_path):
        """grep 带上下文"""
        lines = [f"line_{i}\n" for i in range(1, 11)]
        path = _make_file(tmp_path, "test.txt", "".join(lines))

        result = _parse(await file_read(
            action="grep", path=path, pattern="line_5", context_lines=2,
        ))
        match = result["matches"][0]
        assert match["context_start"] == 3  # line 5 - 2
        assert match["context_end"] == 7    # line 5 + 2

    @pytest.mark.asyncio
    async def test_grep_case_insensitive(self, tmp_path):
        """grep 忽略大小写"""
        path = _make_file(tmp_path, "test.txt", "Hello World\nhello world\n")

        result = _parse(await file_read(action="grep", path=path, pattern="HELLO"))
        assert result["total_matches"] == 2

    @pytest.mark.asyncio
    async def test_grep_regex(self, tmp_path):
        """grep 支持正则"""
        content = "timeout: 30\nretry: 3\nmax_connections: 100\n"
        path = _make_file(tmp_path, "config.yaml", content)

        result = _parse(await file_read(action="grep", path=path, pattern=r"timeout|retry"))
        assert result["total_matches"] == 2

    @pytest.mark.asyncio
    async def test_grep_no_match(self, tmp_path):
        """无匹配"""
        path = _make_file(tmp_path, "test.txt", "hello world\n")
        result = _parse(await file_read(action="grep", path=path, pattern="xyz"))
        assert result["success"] is True
        assert result["total_matches"] == 0

    @pytest.mark.asyncio
    async def test_grep_missing_pattern(self, tmp_path):
        """缺少 pattern 参数"""
        path = _make_file(tmp_path, "test.txt", "hello\n")
        result = _parse(await file_read(action="grep", path=path))
        assert result["success"] is False
        assert "pattern" in result["error"]

    @pytest.mark.asyncio
    async def test_grep_invalid_regex(self, tmp_path):
        """无效正则表达式"""
        path = _make_file(tmp_path, "test.txt", "hello\n")
        result = _parse(await file_read(action="grep", path=path, pattern="[invalid"))
        assert result["success"] is False


# ── action: write ─────────────────────────────────────────


class TestFileWrite:
    """file(action='write') 测试"""

    @pytest.mark.asyncio
    async def test_write_new_file(self, tmp_path):
        """写入新文件"""
        path = str(tmp_path / "new.txt")
        result = _parse(await file_write(action="write", path=path, content="Hello"))
        assert result["success"] is True
        assert open(path, encoding="utf-8").read() == "Hello"

    @pytest.mark.asyncio
    async def test_write_overwrite(self, tmp_path):
        """覆盖已有文件"""
        path = _make_file(tmp_path, "existing.txt", "原始内容")

        result = _parse(await file_write(
            action="write", path=path, content="新内容",
        ))
        assert result["success"] is True
        assert open(path, encoding="utf-8").read() == "新内容"
        assert "backup" not in result

    @pytest.mark.asyncio
    async def test_write_creates_directories(self, tmp_path):
        """自动创建不存在的父目录"""
        path = str(tmp_path / "sub" / "dir" / "file.txt")
        result = _parse(await file_write(action="write", path=path, content="内容"))
        assert result["success"] is True
        assert os.path.exists(path)

    @pytest.mark.asyncio
    async def test_write_missing_content(self, tmp_path):
        """缺少 content 参数"""
        path = str(tmp_path / "test.txt")
        result = _parse(await file_write(action="write", path=path))
        assert result["success"] is False


class TestFileAppend:
    """file(action='append') 测试"""

    @pytest.mark.asyncio
    async def test_append_to_existing(self, tmp_path):
        """追加到已有文件"""
        path = _make_file(tmp_path, "log.txt", "第一行\n")

        result = _parse(await file_write(action="append", path=path, content="第二行"))
        assert result["success"] is True
        content = open(path, encoding="utf-8").read()
        assert "第一行" in content
        assert "第二行" in content

    @pytest.mark.asyncio
    async def test_append_creates_file(self, tmp_path):
        """追加到不存在的文件（创建新文件）"""
        path = str(tmp_path / "new_log.txt")
        result = _parse(await file_write(action="append", path=path, content="首行"))
        assert result["success"] is True
        assert "首行" in open(path, encoding="utf-8").read()


# ── action: replace ───────────────────────────────────────


class TestFileReplace:
    """file(action='replace') 测试"""

    @pytest.mark.asyncio
    async def test_replace_line_range(self, tmp_path):
        """行范围替换"""
        content = "line1\nline2\nline3\nline4\nline5\n"
        path = _make_file(tmp_path, "test.txt", content)

        result = _parse(await file_write(
            action="replace", path=path,
            start=2, end=4, content="NEW_LINE2\nNEW_LINE3\n",
        ))
        assert result["success"] is True
        assert result["mode"] == "line_range"
        assert result["old_line_count"] == 3
        new_content = open(path, encoding="utf-8").read()
        assert "line1" in new_content
        assert "NEW_LINE2" in new_content
        assert "line5" in new_content
        assert "line2" not in new_content  # 被替换掉了

    @pytest.mark.asyncio
    async def test_replace_string(self, tmp_path):
        """字符串精确替换"""
        path = _make_file(tmp_path, "config.yaml", "timeout: 30\nport: 5432\n")

        result = _parse(await file_write(
            action="replace", path=path,
            old="timeout: 30", new="timeout: 60",
        ))
        assert result["success"] is True
        assert result["mode"] == "string"
        assert "timeout: 60" in open(path, encoding="utf-8").read()

    @pytest.mark.asyncio
    async def test_replace_string_multiple_occurrences(self, tmp_path):
        """字符串多处匹配时要求明确指定 count"""
        path = _make_file(tmp_path, "test.txt", "foo\nfoo\nfoo\n")

        result = _parse(await file_write(
            action="replace", path=path, old="foo", new="bar",
        ))
        assert result["success"] is False
        assert "3 处匹配" in result["error"]
        assert result["occurrences"] == 3

    @pytest.mark.asyncio
    async def test_replace_string_with_count_all(self, tmp_path):
        """count=-1 替换全部"""
        path = _make_file(tmp_path, "test.txt", "foo\nfoo\nfoo\n")

        result = _parse(await file_write(
            action="replace", path=path, old="foo", new="bar", count=-1,
        ))
        assert result["success"] is True
        assert open(path, encoding="utf-8").read() == "bar\nbar\nbar\n"

    @pytest.mark.asyncio
    async def test_replace_string_not_found(self, tmp_path):
        """目标字符串不存在"""
        path = _make_file(tmp_path, "test.txt", "hello world\n")

        result = _parse(await file_write(
            action="replace", path=path, old="xyz", new="abc",
        ))
        assert result["success"] is False
        assert "未找到" in result["error"]

    @pytest.mark.asyncio
    async def test_replace_missing_params(self, tmp_path):
        """参数不完整"""
        path = _make_file(tmp_path, "test.txt", "hello\n")
        result = _parse(await file_write(action="replace", path=path))
        assert result["success"] is False


# ── action: insert ────────────────────────────────────────


class TestFileInsert:
    """file(action='insert') 测试"""

    @pytest.mark.asyncio
    async def test_insert_after_line(self, tmp_path):
        """在指定行后插入"""
        path = _make_file(tmp_path, "test.txt", "line1\nline2\nline3\n")

        result = _parse(await file_write(
            action="insert", path=path,
            after_line=2, content="INSERTED\n",
        ))
        assert result["success"] is True
        lines = open(path, encoding="utf-8").readlines()
        assert lines[0] == "line1\n"
        assert lines[1] == "line2\n"
        assert lines[2] == "INSERTED\n"
        assert lines[3] == "line3\n"

    @pytest.mark.asyncio
    async def test_insert_at_beginning(self, tmp_path):
        """after_line=0 插入到文件开头"""
        path = _make_file(tmp_path, "test.txt", "line1\nline2\n")

        result = _parse(await file_write(
            action="insert", path=path,
            after_line=0, content="HEADER\n",
        ))
        assert result["success"] is True
        lines = open(path, encoding="utf-8").readlines()
        assert lines[0] == "HEADER\n"
        assert lines[1] == "line1\n"

    @pytest.mark.asyncio
    async def test_insert_invalid_line(self, tmp_path):
        """after_line 越界"""
        path = _make_file(tmp_path, "test.txt", "line1\n")
        result = _parse(await file_write(
            action="insert", path=path, after_line=99, content="x",
        ))
        assert result["success"] is False


# ── action: delete ────────────────────────────────────────


class TestFileDelete:
    """file(action='delete') 测试"""

    @pytest.mark.asyncio
    async def test_delete_lines(self, tmp_path):
        """删除指定行范围"""
        content = "line1\nline2\nline3\nline4\nline5\n"
        path = _make_file(tmp_path, "test.txt", content)

        result = _parse(await file_write(
            action="delete", path=path, start=2, end=4,
        ))
        assert result["success"] is True
        assert result["deleted_count"] == 3
        new_content = open(path, encoding="utf-8").read()
        assert "line1" in new_content
        assert "line5" in new_content
        assert "line2" not in new_content
        assert "line3" not in new_content

    @pytest.mark.asyncio
    async def test_delete_invalid_range(self, tmp_path):
        """行号越界"""
        path = _make_file(tmp_path, "test.txt", "line1\nline2\n")
        result = _parse(await file_write(
            action="delete", path=path, start=1, end=99,
        ))
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_delete_missing_params(self, tmp_path):
        """缺少参数"""
        path = _make_file(tmp_path, "test.txt", "line1\n")
        result = _parse(await file_write(action="delete", path=path))
        assert result["success"] is False


# ── 工具元数据和错误处理 ─────────────────────────────────


class TestFileToolMeta:
    """工具元数据和通用错误处理"""

    @pytest.mark.asyncio
    async def test_tool_wrapper_metadata(self):
        """@tool 元数据"""
        r_wrapper = file_read._tool_wrapper
        assert r_wrapper.name == "file_read"
        assert r_wrapper.is_async is True
        assert r_wrapper.is_safe is True

        w_wrapper = file_write._tool_wrapper
        assert w_wrapper.name == "file_write"
        assert w_wrapper.is_async is True
        assert w_wrapper.is_safe is False

    @pytest.mark.asyncio
    async def test_invalid_action(self):
        """无效 action 返回错误"""
        result_read = _parse(await file_read(action="invalid", path="/tmp/x"))
        assert result_read["success"] is False
        assert "未知 action" in result_read["error"]
        assert "available" in result_read
        assert "index" in result_read["available"]

        result_write = _parse(await file_write(action="invalid", path="/tmp/x"))
        assert result_write["success"] is False
        assert "未知 action" in result_write["error"]
        assert "available" in result_write
        assert "write" in result_write["available"]

    @pytest.mark.asyncio
    async def test_full_workflow(self, tmp_path):
        """完整工作流：index → grep → read → replace → read"""
        # 创建一个有结构的文件
        content = (
            "# 配置文件\n\n"
            "database:\n"
            "  host: localhost\n"
            "  port: 5432\n"
            "  timeout: 30\n"
            "  pool_size: 10\n\n"
            "cache:\n"
            "  enabled: true\n"
            "  ttl: 3600\n"
        )
        path = _make_file(tmp_path, "config.yaml", content)

        # 1. index
        idx = _parse(await file_read(action="index", path=path))
        assert idx["success"] is True
        assert idx["total_lines"] > 0

        # 2. grep
        grep_result = _parse(await file_read(
            action="grep", path=path, pattern="timeout",
        ))
        assert grep_result["total_matches"] == 1
        match_line = grep_result["matches"][0]["match_line"]
        assert match_line == 6

        # 3. read 上下文
        read_result = _parse(await file_read(
            action="read", path=path,
            start=match_line - 1, end=match_line + 1,
        ))
        assert "timeout" in read_result["content"]

        # 4. replace
        replace_result = _parse(await file_write(
            action="replace", path=path,
            old="timeout: 30", new="timeout: 60",
        ))
        assert replace_result["success"] is True

        # 5. read 验证
        verify = _parse(await file_read(
            action="read", path=path,
            start=match_line, end=match_line,
        ))
        assert "timeout: 60" in verify["content"]
