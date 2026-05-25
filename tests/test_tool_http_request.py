"""测试内置工具 http_request - HTTP GET/POST 请求"""
import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from agent_framework.tools.builtin.http_request import http_request


class TestHttpRequest:
    """http_request 功能测试"""

    @pytest.mark.asyncio
    async def test_get_request_success(self):
        """GET 请求成功"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"message": "ok"}'
        mock_response.headers = {"content-type": "application/json"}

        with patch("agent_framework.tools.builtin.http_request.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await http_request(url="https://example.com/api")
            assert "200" in result
            assert "ok" in result

    @pytest.mark.asyncio
    async def test_post_request_with_body(self):
        """POST 请求带 body"""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.text = "Created"
        mock_response.headers = {}

        with patch("agent_framework.tools.builtin.http_request.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await http_request(
                url="https://example.com/api",
                method="POST",
                body='{"name": "test"}',
            )
            assert "201" in result

    @pytest.mark.asyncio
    async def test_custom_headers(self):
        """自定义请求头"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "ok"
        mock_response.headers = {}

        with patch("agent_framework.tools.builtin.http_request.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await http_request(
                url="https://example.com",
                headers='{"Authorization": "Bearer token123"}',
            )
            assert "200" in result

    @pytest.mark.asyncio
    async def test_timeout_error(self):
        """超时返回错误"""
        import httpx

        with patch("agent_framework.tools.builtin.http_request.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await http_request(url="https://example.com", timeout=1)
            assert "错误" in result or "超时" in result

    @pytest.mark.asyncio
    async def test_connection_error(self):
        """连接错误返回错误信息"""
        import httpx

        with patch("agent_framework.tools.builtin.http_request.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await http_request(url="https://nonexistent.example.com")
            assert "错误" in result

    @pytest.mark.asyncio
    async def test_response_truncation(self):
        """超长响应被截断"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "x" * 10000
        mock_response.headers = {}

        with patch("agent_framework.tools.builtin.http_request.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await http_request(url="https://example.com/large")
            assert "截断" in result

    @pytest.mark.asyncio
    async def test_tool_wrapper_metadata(self):
        """@tool 元数据正确"""
        wrapper = http_request._tool_wrapper
        assert wrapper.name == "http_request"
        assert wrapper.is_async is True
        assert wrapper.dangerous is False
