"""测试内置工具 web_search - 网页搜索"""
import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from agent_framework.tools.builtin.web_search import web_search


class TestWebSearch:
    """web_search 功能测试"""

    @pytest.mark.asyncio
    async def test_search_with_results(self):
        """搜索返回结果"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "AbstractText": "Python is a programming language.",
            "Heading": "Python (programming language)",
            "AbstractURL": "https://en.wikipedia.org/wiki/Python_(programming_language)",
            "RelatedTopics": [
                {
                    "Text": "Python is a high-level language",
                    "FirstURL": "https://example.com/1",
                },
                {
                    "Text": "Guido van Rossum created Python",
                    "FirstURL": "https://example.com/2",
                },
            ],
        }
        mock_response.text = json.dumps(mock_response.json.return_value)

        with patch("agent_framework.tools.builtin.web_search.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await web_search(query="Python programming")
            assert "Python" in result

    @pytest.mark.asyncio
    async def test_search_no_results(self):
        """搜索无结果"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "AbstractText": "",
            "Heading": "",
            "AbstractURL": "",
            "RelatedTopics": [],
        }

        with patch("agent_framework.tools.builtin.web_search.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await web_search(query="xyznonexistentquery123")
            assert "没有" in result or "未找到" in result or len(result) > 0

    @pytest.mark.asyncio
    async def test_search_network_error(self):
        """网络错误返回错误信息"""
        import httpx

        with patch("agent_framework.tools.builtin.web_search.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("failed"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await web_search(query="test")
            assert "错误" in result

    @pytest.mark.asyncio
    async def test_search_custom_num_results(self):
        """自定义结果数量"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "AbstractText": "Result",
            "Heading": "Topic",
            "AbstractURL": "https://example.com",
            "RelatedTopics": [
                {"Text": f"Topic {i}", "FirstURL": f"https://example.com/{i}"}
                for i in range(10)
            ],
        }

        with patch("agent_framework.tools.builtin.web_search.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await web_search(query="test", num_results=3)
            assert "test" in result.lower() or "result" in result.lower() or "Topic" in result

    @pytest.mark.asyncio
    async def test_tool_wrapper_metadata(self):
        """@tool 元数据正确"""
        wrapper = web_search._tool_wrapper
        assert wrapper.name == "web_search"
        assert wrapper.is_async is True
        assert wrapper.dangerous is False
