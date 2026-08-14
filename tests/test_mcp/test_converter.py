"""测试 MCP Tool → ToolWrapper 转换器"""
import json
from unittest.mock import MagicMock

import pytest
from milu.tools.mcp.converter import convert_mcp_tool, _convert_result


class TestConvertMcpTool:
    """convert_mcp_tool 转换测试"""

    @pytest.mark.asyncio
    async def test_basic_conversion(self, mock_mcp_tool, mock_session):
        wrapper = convert_mcp_tool(
            mcp_tool=mock_mcp_tool,
            session=mock_session,
            server_name="fs",
            prefix=True,
        )
        assert wrapper.name == "fs__read_file"
        assert wrapper.description == "Read file contents"
        assert wrapper.is_async is True
        assert wrapper.is_safe is False
        assert wrapper.parameters_schema == mock_mcp_tool.inputSchema

    @pytest.mark.asyncio
    async def test_no_prefix(self, mock_mcp_tool, mock_session):
        wrapper = convert_mcp_tool(
            mcp_tool=mock_mcp_tool,
            session=mock_session,
            server_name="fs",
            prefix=False,
        )
        assert wrapper.name == "read_file"

    @pytest.mark.asyncio
    async def test_safe_flag(self, mock_mcp_tool, mock_session):
        wrapper = convert_mcp_tool(
            mcp_tool=mock_mcp_tool,
            session=mock_session,
            server_name="fs",
            is_safe=True,
        )
        assert wrapper.is_safe is True

    @pytest.mark.asyncio
    async def test_call_delegates_to_session(self, mock_mcp_tool, mock_session, mock_mcp_result):
        """调用 wrapper.func 应委托给 session.call_tool"""
        wrapper = convert_mcp_tool(
            mcp_tool=mock_mcp_tool,
            session=mock_session,
            server_name="fs",
        )
        result = await wrapper.func(path="/tmp/test.txt")
        mock_session.call_tool.assert_called_once_with("read_file", {"path": "/tmp/test.txt"})
        assert result == "file contents here"

    @pytest.mark.asyncio
    async def test_missing_description(self, mock_session):
        tool = MagicMock()
        tool.name = "no_desc"
        tool.description = None
        tool.inputSchema = {"type": "object", "properties": {}}

        wrapper = convert_mcp_tool(tool, mock_session, "srv")
        assert "MCP tool" in wrapper.description

    @pytest.mark.asyncio
    async def test_missing_input_schema(self, mock_session):
        tool = MagicMock()
        tool.name = "no_schema"
        tool.description = "A tool"
        tool.inputSchema = None

        wrapper = convert_mcp_tool(tool, mock_session, "srv")
        assert wrapper.parameters_schema == {"type": "object", "properties": {}}


class TestConvertResult:
    """_convert_result 结果转换测试"""

    def test_text_content(self):
        text = MagicMock()
        text.__class__.__name__ = "TextContent"
        text.text = "hello world"

        result = MagicMock()
        result.content = [text]
        result.isError = False
        result.structuredContent = None

        output = _convert_result(result)
        assert output == "hello world"

    def test_multiple_text_parts(self):
        t1 = MagicMock()
        t1.__class__.__name__ = "TextContent"
        t1.text = "part 1"

        t2 = MagicMock()
        t2.__class__.__name__ = "TextContent"
        t2.text = "part 2"

        result = MagicMock()
        result.content = [t1, t2]
        result.isError = False
        result.structuredContent = None

        output = _convert_result(result)
        assert output == "part 1\npart 2"

    def test_image_content(self):
        img = MagicMock()
        img.__class__.__name__ = "ImageContent"
        img.mimeType = "image/png"
        img.data = "base64data=="
        # 确保不被识别为 TextContent
        del img.text

        result = MagicMock()
        result.content = [img]
        result.isError = False
        result.structuredContent = None

        output = _convert_result(result)
        assert "[Image: image/png" in output

    def test_structured_content(self):
        result = MagicMock()
        result.content = []
        result.isError = False
        result.structuredContent = {"key": "value", "count": 42}

        output = _convert_result(result)
        parsed = json.loads(output)
        assert parsed == {"key": "value", "count": 42}

    def test_duplicate_json_text_and_structured_content_is_emitted_once(self):
        """FastMCP 同时返回文本 JSON 与 structuredContent 时不得重复拼接。"""
        payload = {"key": "value", "items": [1, 2]}
        text = MagicMock()
        text.__class__.__name__ = "TextContent"
        text.text = json.dumps(payload, ensure_ascii=False, indent=2)

        result = MagicMock()
        result.content = [text]
        result.isError = False
        result.structuredContent = payload

        output = _convert_result(result)
        assert json.loads(output) == payload
        assert output.count('"key"') == 1

    def test_error_result(self):
        text = MagicMock()
        text.__class__.__name__ = "TextContent"
        text.text = "Something went wrong"

        result = MagicMock()
        result.content = [text]
        result.isError = True
        result.structuredContent = None

        output = _convert_result(result)
        assert output.startswith("[MCP Error]")
        assert "Something went wrong" in output

    def test_error_with_empty_content(self):
        result = MagicMock()
        result.content = []
        result.isError = True
        result.structuredContent = None

        output = _convert_result(result)
        assert output == "[MCP Error] 工具执行失败"

    def test_empty_content(self):
        result = MagicMock()
        result.content = []
        result.isError = False
        result.structuredContent = None

        output = _convert_result(result)
        assert output == ""
