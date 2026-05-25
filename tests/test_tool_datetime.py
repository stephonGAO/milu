"""测试内置工具 datetime_tool - 日期时间查询与转换"""
import json
from datetime import datetime

import pytest
from agent_framework.tools.builtin.datetime_tool import datetime_tool, get_current_time


class TestDatetimeTool:
    """datetime_tool 功能测试"""

    @pytest.mark.asyncio
    async def test_now_default(self):
        """获取当前日期时间（默认格式）"""
        result = await datetime_tool(operation="now")
        data = json.loads(result)
        assert "datetime" in data
        assert "date" in data
        assert "time" in data
        assert "timestamp" in data
        assert "weekday" in data

    @pytest.mark.asyncio
    async def test_now_with_format(self):
        """获取当前日期时间（自定义格式）"""
        result = await datetime_tool(operation="now", format="%Y-%m-%d")
        data = json.loads(result)
        # 自定义格式结果应在 datetime 字段
        assert len(data["datetime"]) == 10  # "2026-05-25" 格式

    @pytest.mark.asyncio
    async def test_parse_date(self):
        """解析日期字符串"""
        result = await datetime_tool(
            operation="parse",
            value="2026-01-15 10:30:00",
            format="%Y-%m-%d %H:%M:%S",
        )
        data = json.loads(result)
        assert data["year"] == 2026
        assert data["month"] == 1
        assert data["day"] == 15
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_parse_auto_detect(self):
        """自动识别常见日期格式"""
        result = await datetime_tool(operation="parse", value="2026-05-25")
        data = json.loads(result)
        assert data["year"] == 2026
        assert data["month"] == 5
        assert data["day"] == 25

    @pytest.mark.asyncio
    async def test_timestamp_to_datetime(self):
        """时间戳转日期时间"""
        # 使用一个已知的时间戳: 2024-01-01 00:00:00 UTC = 1704067200
        result = await datetime_tool(operation="timestamp", value="1704067200")
        data = json.loads(result)
        assert data["year"] == 2024
        assert "datetime" in data

    @pytest.mark.asyncio
    async def test_invalid_operation(self):
        """无效操作返回错误"""
        result = await datetime_tool(operation="invalid_op")
        assert "错误" in result or "不支持" in result

    @pytest.mark.asyncio
    async def test_parse_invalid_date(self):
        """解析无效日期返回错误"""
        result = await datetime_tool(operation="parse", value="not-a-date")
        assert "错误" in result or "解析" in result

    @pytest.mark.asyncio
    async def test_tool_wrapper_metadata(self):
        """验证 @tool 元数据正确"""
        wrapper = datetime_tool._tool_wrapper
        assert wrapper.name == "datetime_tool"
        assert wrapper.is_async is True
        assert wrapper.dangerous is False


class TestGetCurrentTime:
    """get_current_time 辅助函数测试"""

    def test_returns_dict(self):
        """返回字典包含所有字段"""
        result = get_current_time()
        assert "datetime" in result
        assert "date" in result
        assert "time" in result
        assert "timestamp" in result
        assert "weekday" in result
        assert isinstance(result["timestamp"], (int, float))
