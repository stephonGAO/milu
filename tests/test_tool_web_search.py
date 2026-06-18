"""测试内置工具 web_search - 网页搜索（ddgs）"""
from unittest.mock import AsyncMock, patch, MagicMock

import sys

import pytest
from milu.tools.builtin.web_search import web_search
# 取真正的子模块对象：milu.tools.builtin.web_search 作为 builtin 包属性会被同名的
# web_search 函数遮蔽——patch("…web_search.X") 与 import…as 在 Python 3.10 的 mock
# 上都会解析到「函数」而非「模块」，patch 失效。sys.modules 不受属性遮蔽影响，
# 拿到模块后用 patch.object(_ws_mod, …) 跨版本稳定。
_ws_mod = sys.modules["milu.tools.builtin.web_search"]


def _make_ddgs_mock(results: list[dict]):
    """构造 DDGS mock，text() 返回指定结果"""
    mock_instance = MagicMock()
    mock_instance.text = MagicMock(return_value=results)
    mock_cls = MagicMock(return_value=mock_instance)
    return mock_cls


class TestWebSearch:
    """web_search 功能测试"""

    @pytest.fixture(autouse=True)
    def _isolate_search_backend_env(self):
        """隔离环境中的搜索后端配置。

        开发机的项目 .env 可能设了 WEB_SEARCH_PROVIDER（如 bocha），框架启动时
        会自动加载，导致本类中假设「默认走 DDG」的用例被路由到真实后端、mock 失效。
        这里临时清除相关变量，确保用例在任何机器上都确定性地走默认 DDG 路径。
        """
        import os
        keys = ("WEB_SEARCH_PROVIDER", "SEARCH_API_URL", "SEARCH_API_KEY")
        saved = {k: os.environ.pop(k, None) for k in keys}
        try:
            yield
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    @pytest.mark.asyncio
    async def test_search_with_results(self):
        """搜索返回结果"""
        items = [
            {"title": "Python 官网", "body": "Python is a programming language.", "href": "https://python.org"},
            {"title": "Wikipedia", "body": "Python created by Guido.", "href": "https://en.wikipedia.org/wiki/Python"},
        ]

        with patch.object(_ws_mod, "DDGS", _make_ddgs_mock(items), create=True):
            result = await web_search(query="Python programming")
            assert "Python" in result
            assert "https://python.org" in result

    @pytest.mark.asyncio
    async def test_search_no_results(self):
        """搜索无结果"""
        with patch.object(_ws_mod, "DDGS", _make_ddgs_mock([]), create=True):
            result = await web_search(query="xyznonexistentquery123")
            assert "未找到" in result

    @pytest.mark.asyncio
    async def test_search_network_error(self):
        """网络错误返回错误信息"""
        mock_instance = MagicMock()
        mock_instance.text = MagicMock(side_effect=Exception("Connection refused"))
        mock_cls = MagicMock(return_value=mock_instance)

        with patch.object(_ws_mod, "DDGS", mock_cls, create=True):
            result = await web_search(query="test")
            assert "错误" in result

    @pytest.mark.asyncio
    async def test_search_custom_num_results(self):
        """自定义结果数量"""
        items = [
            {"title": f"Topic {i}", "body": f"Body {i}", "href": f"https://example.com/{i}"}
            for i in range(3)
        ]

        with patch.object(_ws_mod, "DDGS", _make_ddgs_mock(items), create=True):
            result = await web_search(query="test", num_results=3)
            assert result.count("https://example.com/") == 3

    @pytest.mark.asyncio
    async def test_ddgs_not_installed(self):
        """ddgs 为可选依赖（milu[ddg]）：未安装时返回友好提示，不抛 ImportError"""
        import sys
        # DDGS 缓存置 None + 屏蔽 ddgs 模块（sys.modules 值为 None → import 即抛 ImportError）
        with patch.object(_ws_mod, "DDGS", None, create=True):
            with patch.dict(sys.modules, {"ddgs": None}):
                result = await web_search(query="test")
        assert "ddgs" in result
        assert "milu[ddg]" in result
        assert "WEB_SEARCH_PROVIDER" in result

    @pytest.mark.asyncio
    async def test_tool_wrapper_metadata(self):
        """@tool 元数据正确"""
        wrapper = web_search._tool_wrapper
        assert wrapper.name == "web_search"
        assert wrapper.is_async is True
        assert wrapper.is_safe is True

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
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                result = await web_search(query="test")
                assert "Custom Result" in result
                assert "custom.com" in result


# ── 可插拔后端（WEB_SEARCH_PROVIDER）──────────────────────


def _make_post_client(response):
    """构造支持 async with 的 httpx.AsyncClient mock（POST 路径）"""
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=mock_client), mock_client


class TestSearchProviders:
    """WEB_SEARCH_PROVIDER 后端切换"""

    @pytest.mark.asyncio
    async def test_tavily_provider(self):
        """provider=tavily：POST Tavily 端点并格式化 results[].{title,content,url}"""
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"results": [
            {"title": "Tavily 结果", "content": "摘要内容", "url": "https://t.example/1"},
        ]}
        client_cls, client = _make_post_client(resp)

        env = {"WEB_SEARCH_PROVIDER": "tavily", "TAVILY_API_KEY": "tk"}
        with patch.dict("os.environ", env, clear=False):
            with patch("httpx.AsyncClient", client_cls):
                result = await web_search(query="测试")

        assert "Tavily 结果" in result
        assert "https://t.example/1" in result
        # 请求体携带 api_key 与 query
        body = client.post.call_args.kwargs["json"]
        assert body["api_key"] == "tk" and body["query"] == "测试"

    @pytest.mark.asyncio
    async def test_bocha_provider(self):
        """provider=bocha：POST 博查端点并格式化 data.webPages.value[].{name,summary,url}"""
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"data": {"webPages": {"value": [
            {"name": "博查结果", "summary": "中文摘要", "url": "https://b.example/1"},
        ]}}}
        client_cls, client = _make_post_client(resp)

        env = {"WEB_SEARCH_PROVIDER": "bocha", "BOCHA_API_KEY": "bk"}
        with patch.dict("os.environ", env, clear=False):
            with patch("httpx.AsyncClient", client_cls):
                result = await web_search(query="测试")

        assert "博查结果" in result
        assert "https://b.example/1" in result
        # 鉴权走 Bearer header
        headers = client.post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer bk"

    @pytest.mark.asyncio
    async def test_provider_missing_key_errors(self):
        """配置了 provider 却缺 Key → 明确报错，不静默回退"""
        env = {"WEB_SEARCH_PROVIDER": "tavily"}
        with patch.dict("os.environ", env, clear=False):
            # 确保 Key 不存在
            import os
            os.environ.pop("TAVILY_API_KEY", None)
            result = await web_search(query="测试")
        assert "TAVILY_API_KEY" in result
        assert "错误" in result

    @pytest.mark.asyncio
    async def test_default_provider_is_ddg(self):
        """未设置 provider → 默认走 DuckDuckGo"""
        items = [{"title": "DDG 结果", "body": "正文", "href": "https://d.example/1"}]
        env = {"WEB_SEARCH_PROVIDER": ""}  # 空值 → 按缺省处理? 这里直接删除变量验证
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("WEB_SEARCH_PROVIDER", None)
            with patch.object(_ws_mod, "DDGS", _make_ddgs_mock(items), create=True):
                result = await web_search(query="测试")
        assert "DDG 结果" in result
