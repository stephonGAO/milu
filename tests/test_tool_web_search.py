"""测试内置工具 web_search - 网页搜索（ddgs）"""
import json
from unittest.mock import AsyncMock, patch, MagicMock

import httpx
import pytest
from agent_framework.tools.builtin.web_search import web_search


def _make_ddgs_mock(results: list[dict]):
    """构造 DDGS mock，text() 返回指定结果"""
    mock_instance = MagicMock()
    mock_instance.text = MagicMock(return_value=results)
    mock_cls = MagicMock(return_value=mock_instance)
    return mock_cls


class TestWebSearch:
    """web_search 功能测试"""

    @pytest.mark.asyncio
    async def test_search_with_results(self):
        """搜索返回结果"""
        items = [
            {"title": "Python 官网", "body": "Python is a programming language.", "href": "https://python.org"},
            {"title": "Wikipedia", "body": "Python created by Guido.", "href": "https://en.wikipedia.org/wiki/Python"},
        ]

        with patch("agent_framework.tools.builtin.web_search.DDGS", _make_ddgs_mock(items), create=True):
            result = await web_search(query="Python programming")
            assert "Python" in result
            assert "https://python.org" in result

    @pytest.mark.asyncio
    async def test_search_no_results(self):
        """搜索无结果"""
        with patch("agent_framework.tools.builtin.web_search.DDGS", _make_ddgs_mock([]), create=True):
            result = await web_search(query="xyznonexistentquery123")
            assert "未找到" in result

    @pytest.mark.asyncio
    async def test_search_network_error(self):
        """网络错误返回错误信息"""
        mock_instance = MagicMock()
        mock_instance.text = MagicMock(side_effect=Exception("Connection refused"))
        mock_cls = MagicMock(return_value=mock_instance)

        with patch("agent_framework.tools.builtin.web_search.DDGS", mock_cls, create=True):
            result = await web_search(query="test")
            assert "错误" in result

    @pytest.mark.asyncio
    async def test_search_custom_num_results(self):
        """自定义结果数量"""
        items = [
            {"title": f"Topic {i}", "body": f"Body {i}", "href": f"https://example.com/{i}"}
            for i in range(3)
        ]

        with patch("agent_framework.tools.builtin.web_search.DDGS", _make_ddgs_mock(items), create=True):
            result = await web_search(query="test", num_results=3)
            assert result.count("https://example.com/") == 3

    @pytest.mark.asyncio
    async def test_tool_wrapper_metadata(self):
        """@tool 元数据正确"""
        wrapper = web_search._tool_wrapper
        assert wrapper.name == "web_search"
        assert wrapper.is_async is True
        assert wrapper.dangerous is False

    @pytest.mark.asyncio
    async def test_custom_search_api(self):
        """自定义搜索 API 走 httpx 路径"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"title": "Custom Result", "snippet": "A custom snippet", "url": "https://custom.com/1"}
            ]
        }

        with patch.dict("os.environ", {"SEARCH_API_URL": "https://api.custom.com/search", "SEARCH_API_KEY": "mykey"}):
            with patch("agent_framework.tools.builtin.web_search.httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                result = await web_search(query="test")
                assert "Custom Result" in result
                assert "custom.com" in result
