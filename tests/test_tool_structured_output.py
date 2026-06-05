"""测试内置工具 structured_output - 结构化输出解析与验证"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from milu.tools.builtin.structured_output import (
    structured_output,
    create_structured_output_tool,
    _extract_json,
)
from milu.llm.base.response import StreamChunk


# ── 测试 Schema ──────────────────────────────────────────

SAMPLE_SCHEMA = {
    "type": "object",
    "required": ["name", "score", "tags"],
    "properties": {
        "name": {"type": "string"},
        "score": {"type": "number"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
}


# ── 辅助：模拟 LLM ──────────────────────────────────────


def _make_mock_llm(fixed_json: str):
    """创建返回指定 JSON 的 mock LLM"""
    llm = MagicMock()

    async def mock_chat(*args, **kwargs):
        yield StreamChunk(content=fixed_json)

    llm.chat = mock_chat
    return llm


# ── 第一层：解析层测试 ────────────────────────────────────


class TestExtractJson:
    """_extract_json 处理各种 LLM 输出噪声"""

    def test_clean_json(self):
        """干净的 JSON"""
        data = _extract_json('{"a": 1}')
        assert data == {"a": 1}

    def test_markdown_code_block(self):
        """```json ... ``` 包裹"""
        raw = '```json\n{"a": 1}\n```'
        assert _extract_json(raw) == {"a": 1}

    def test_markdown_without_lang(self):
        """``` ... ``` 无语言标记"""
        raw = '```\n{"a": 1}\n```'
        assert _extract_json(raw) == {"a": 1}

    def test_trailing_comma_object(self):
        """尾部逗号 ,}"""
        raw = '{"a": 1, "b": 2,}'
        assert _extract_json(raw) == {"a": 1, "b": 2}

    def test_trailing_comma_array(self):
        """尾部逗号 ,]"""
        raw = '[1, 2, 3,]'
        assert _extract_json(raw) == [1, 2, 3]

    def test_line_comments(self):
        """行注释 // ..."""
        raw = '{\n"a": 1, // 这是注释\n"b": 2\n}'
        assert _extract_json(raw) == {"a": 1, "b": 2}

    def test_leading_text(self):
        """JSON 前有解释性文字"""
        raw = '好的，这是结果：\n{"a": 1}'
        assert _extract_json(raw) == {"a": 1}

    def test_leading_text_with_markdown(self):
        """完整噪声：解释文字 + markdown 包裹 + 注释 + 尾部逗号"""
        raw = """好的，这是提取的结果：
```json
{
  "name": "Claude",
  "tags": ["ai", "assistant",]  // 尾部逗号
}
```
"""
        data = _extract_json(raw)
        assert data == {"name": "Claude", "tags": ["ai", "assistant"]}

    def test_json_array(self):
        """JSON 数组"""
        raw = '[1, 2, 3]'
        assert _extract_json(raw) == [1, 2, 3]

    def test_invalid_json_raises(self):
        """无效 JSON 抛异常"""
        with pytest.raises(json.JSONDecodeError):
            _extract_json("not json at all {{{")


# ── 第二层：验证层测试 ────────────────────────────────────


class TestStructuredOutputValidation:
    """structured_output 的解析+验证（不涉及 LLM 修复）"""

    @pytest.mark.asyncio
    async def test_valid_json_passes(self):
        """完全符合 schema 的 JSON"""
        raw = json.dumps({"name": "test", "score": 95, "tags": ["a"]})
        result = await structured_output(raw, SAMPLE_SCHEMA, auto_fix=False)
        assert result["success"] is True
        assert result["data"]["name"] == "test"
        assert result["errors"] == []
        assert result["attempts"] == 1

    @pytest.mark.asyncio
    async def test_type_mismatch(self):
        """类型不匹配返回精确错误路径"""
        raw = json.dumps({"name": "test", "score": "95", "tags": ["a"]})
        result = await structured_output(raw, SAMPLE_SCHEMA, auto_fix=False)
        assert result["success"] is False
        assert len(result["errors"]) == 1
        assert "score" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_missing_required_field(self):
        """缺少必填字段"""
        raw = json.dumps({"name": "test", "score": 95})  # 缺少 tags
        result = await structured_output(raw, SAMPLE_SCHEMA, auto_fix=False)
        assert result["success"] is False
        assert any("tags" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_multiple_errors(self):
        """多个验证错误"""
        raw = json.dumps({"name": 123, "score": "bad"})  # name 应为 string，score 应为 number，缺 tags
        result = await structured_output(raw, SAMPLE_SCHEMA, auto_fix=False)
        assert result["success"] is False
        assert len(result["errors"]) >= 2

    @pytest.mark.asyncio
    async def test_invalid_json_returns_error(self):
        """无效 JSON"""
        result = await structured_output("not json", SAMPLE_SCHEMA, auto_fix=False)
        assert result["success"] is False
        assert "解析失败" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_noisy_json_parsed_correctly(self):
        """带噪声的 JSON 能被正确解析"""
        raw = """好的：
```json
{
  "name": "Claude",
  "score": 95,
  "tags": ["ai",]  // trailing comma
}
```"""
        result = await structured_output(raw, SAMPLE_SCHEMA, auto_fix=False)
        assert result["success"] is True
        assert result["data"]["score"] == 95

    @pytest.mark.asyncio
    async def test_extra_properties_allowed(self):
        """schema 未定义的额外属性默认允许"""
        raw = json.dumps({
            "name": "test", "score": 95, "tags": [], "extra": "field",
        })
        result = await structured_output(raw, SAMPLE_SCHEMA, auto_fix=False)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_nested_schema_validation(self):
        """嵌套 schema 验证"""
        schema = {
            "type": "object",
            "required": ["user"],
            "properties": {
                "user": {
                    "type": "object",
                    "required": ["name", "age"],
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                }
            },
        }
        raw = json.dumps({"user": {"name": "test", "age": "not_a_number"}})
        result = await structured_output(raw, schema, auto_fix=False)
        assert result["success"] is False
        assert any("age" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_auto_fix_false_skips_llm(self):
        """auto_fix=False 时不调用 LLM"""
        raw = json.dumps({"name": "test", "score": "bad", "tags": []})
        result = await structured_output(raw, SAMPLE_SCHEMA, auto_fix=False)
        assert result["success"] is False
        assert result["attempts"] == 1


# ── 第三层：修复层测试 ────────────────────────────────────


class TestStructuredOutputAutoFix:
    """auto_fix 修复层（使用 mock LLM）"""

    @pytest.mark.asyncio
    async def test_auto_fix_success(self):
        """第一次验证失败，LLM 修复后成功"""
        bad_raw = json.dumps({"name": "test", "score": "95", "tags": []})
        fixed_json = json.dumps({"name": "test", "score": 95, "tags": []})

        mock_llm = _make_mock_llm(fixed_json)
        result = await structured_output(bad_raw, SAMPLE_SCHEMA, _fix_llm=mock_llm)
        assert result["success"] is True
        assert result["attempts"] == 2

    @pytest.mark.asyncio
    async def test_auto_fix_still_fails(self):
        """LLM 修复后仍然不符合 schema"""
        bad_raw = json.dumps({"name": "test", "score": "95", "tags": []})
        # LLM 返回的还是错的
        still_bad = json.dumps({"name": "test", "score": "still_bad", "tags": []})

        mock_llm = _make_mock_llm(still_bad)
        result = await structured_output(bad_raw, SAMPLE_SCHEMA, _fix_llm=mock_llm)
        assert result["success"] is False
        assert result["attempts"] == 3  # 1 初始 + 2 次修复

    @pytest.mark.asyncio
    async def test_auto_fix_uses_default_llm(self):
        """不提供 _fix_llm 时使用默认 deepseek-v4-flash"""
        bad_raw = json.dumps({"name": "test", "score": "95", "tags": []})
        fixed_json = json.dumps({"name": "test", "score": 95, "tags": []})
        mock_llm = _make_mock_llm(fixed_json)

        # mock 掉默认 LLM 的创建
        with patch("milu.tools.builtin.structured_output._get_default_fix_llm", return_value=mock_llm):
            result = await structured_output(bad_raw, SAMPLE_SCHEMA, auto_fix=True)
            assert result["success"] is True
            assert result["attempts"] == 2  # 没有 LLM，不尝试修复


# ── Agent 工具集成测试 ────────────────────────────────────


class TestCreateStructuredOutputTool:
    """create_structured_output_tool 工厂函数测试"""

    def test_factory_creates_tool_wrapper(self):
        """工厂函数创建带有 _tool_wrapper 的工具"""
        mock_llm = MagicMock()
        tool_func = create_structured_output_tool(fix_llm=mock_llm)
        assert hasattr(tool_func, "_tool_wrapper")
        wrapper = tool_func._tool_wrapper
        assert wrapper.name == "structured_output"
        assert wrapper.is_async is True

    def test_factory_default_no_args(self):
        """工厂函数可以无参调用（使用默认 deepseek-v4-flash）"""
        tool_func = create_structured_output_tool()
        assert hasattr(tool_func, "_tool_wrapper")
        assert tool_func._tool_wrapper.name == "structured_output"

    def test_schema_excludes_llm_param(self):
        """生成的 schema 不包含 _fix_llm 参数"""
        tool_func = create_structured_output_tool()
        schema = tool_func._tool_wrapper.parameters_schema
        props = schema["properties"]
        assert "_fix_llm" not in props
        assert "_llm" not in props
        assert "raw" in props
        assert "schema" in props
        assert "auto_fix" in props

    @pytest.mark.asyncio
    async def test_tool_returns_json_string(self):
        """工具返回 JSON 字符串格式（无 auto_fix 时直接验证）"""
        tool_func = create_structured_output_tool()

        valid_json = json.dumps({"name": "test", "score": 95, "tags": ["a"]})
        result_str = await tool_func(raw=valid_json, schema=SAMPLE_SCHEMA, auto_fix=False)
        result = json.loads(result_str)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_tool_with_auto_fix(self):
        """工具使用工厂注入的 fix_llm 进行自修复"""
        bad_raw = json.dumps({"name": "test", "score": "95", "tags": []})
        fixed_json = json.dumps({"name": "test", "score": 95, "tags": []})
        mock_llm = _make_mock_llm(fixed_json)

        tool_func = create_structured_output_tool(fix_llm=mock_llm)
        result_str = await tool_func(raw=bad_raw, schema=SAMPLE_SCHEMA)
        result = json.loads(result_str)
        assert result["success"] is True
