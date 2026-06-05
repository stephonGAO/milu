"""测试 web_fetch 工具（网页 → Markdown 正文提取）。

httpx 全部 mock，不发真实网络请求。
"""
from __future__ import annotations

import json

import pytest

from milu.tools.builtin import web_fetch
from milu.tools.builtin.web_fetch import _html_to_markdown


# ── 纯函数 _html_to_markdown ─────────────────────────────


class TestHtmlToMarkdown:
    def test_strips_noise_tags(self):
        """script/style/nav 等噪声标签应被剔除"""
        html = """
        <html><head><style>body{color:red}</style></head><body>
        <nav>导航栏链接一堆</nav>
        <script>alert('x')</script>
        <article><h1>正文标题</h1><p>正文内容。</p></article>
        <footer>页脚版权</footer>
        </body></html>
        """
        md = _html_to_markdown(html)
        assert "正文标题" in md
        assert "正文内容" in md
        assert "alert" not in md
        assert "导航栏" not in md
        assert "页脚版权" not in md
        assert "color:red" not in md

    def test_converts_headings_to_atx(self):
        """标题转为 ATX 风格（# 前缀）"""
        md = _html_to_markdown("<body><h2>章节</h2><p>内容</p></body>")
        assert "## 章节" in md

    def test_prefers_article_container(self):
        """优先提取 article 容器，忽略容器外杂讯"""
        html = (
            "<body><div>侧边栏推荐内容</div>"
            "<article><p>这才是正文</p></article></body>"
        )
        md = _html_to_markdown(html)
        assert "这才是正文" in md
        assert "侧边栏推荐内容" not in md

    def test_collapses_blank_lines(self):
        """连续空行压缩"""
        md = _html_to_markdown(
            "<body><p>a</p><br><br><br><br><p>b</p></body>"
        )
        assert "\n\n\n" not in md


# ── 工具函数 web_fetch（mock httpx）──────────────────────


class _FakeResp:
    def __init__(self, text: str, ctype: str = "text/html; charset=utf-8",
                 status: int = 200):
        self.text = text
        self.headers = {"content-type": ctype}
        self.status_code = status

    def json(self):
        return json.loads(self.text)


class _FakeClient:
    """替身 AsyncClient：返回预置响应"""
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, url):
        if isinstance(self._resp, Exception):
            raise self._resp
        return self._resp


def _patch_client(monkeypatch, resp):
    """替换 web_fetch 模块命名空间中的 httpx（保留 HTTPError 供 except 使用）。

    注意：包 __init__ 中 `from ... import web_fetch`（函数）遮蔽了同名子模块属性，
    `import ...web_fetch as mod` 拿到的是函数，须经 sys.modules 取真模块。
    """
    import sys
    from types import SimpleNamespace

    import httpx as real_httpx

    wf_mod = sys.modules["milu.tools.builtin.web_fetch"]
    fake_httpx = SimpleNamespace(
        AsyncClient=lambda **kw: _FakeClient(resp),
        HTTPError=real_httpx.HTTPError,
    )
    monkeypatch.setattr(wf_mod, "httpx", fake_httpx)


class TestWebFetchTool:
    @pytest.mark.asyncio
    async def test_html_page_to_markdown(self, monkeypatch):
        """HTML 页面 → Markdown 正文 + 状态码头"""
        _patch_client(monkeypatch, _FakeResp(
            "<html><body><script>x()</script><article>"
            "<h1>标题</h1><p>正文段落</p></article></body></html>"
        ))
        result = await web_fetch._tool_wrapper.func(url="https://example.com/a")
        assert result.startswith("[200] https://example.com/a")
        assert "# 标题" in result
        assert "正文段落" in result
        assert "x()" not in result

    @pytest.mark.asyncio
    async def test_json_response_pretty_printed(self, monkeypatch):
        """JSON 响应美化输出（API 场景）"""
        _patch_client(monkeypatch, _FakeResp(
            '{"name": "测试", "value": 1}', ctype="application/json",
        ))
        result = await web_fetch._tool_wrapper.func(url="https://api.example.com/x")
        assert '"name": "测试"' in result

    @pytest.mark.asyncio
    async def test_truncation(self, monkeypatch):
        """超长内容按 max_chars 截断并标注总长度"""
        long_text = "字" * 5000
        _patch_client(monkeypatch, _FakeResp(
            f"<body><p>{long_text}</p></body>"
        ))
        result = await web_fetch._tool_wrapper.func(
            url="https://example.com/long", max_chars=1000,
        )
        assert "内容已截断" in result
        # 头部 + 1000 字符 + 截断标注，总长度远小于原文
        assert len(result) < 1500

    @pytest.mark.asyncio
    async def test_invalid_url_rejected(self):
        """非 http/https URL 直接拒绝（不发请求）"""
        result = await web_fetch._tool_wrapper.func(url="file:///etc/passwd")
        assert "仅支持 http/https" in result

    @pytest.mark.asyncio
    async def test_http_error_returns_message(self, monkeypatch):
        """网络错误返回失败信息而非抛异常"""
        import httpx
        _patch_client(monkeypatch, httpx.ConnectError("连接被拒绝"))
        result = await web_fetch._tool_wrapper.func(url="https://unreachable.example")
        assert "抓取失败" in result

    @pytest.mark.asyncio
    async def test_malformed_html_degrades_gracefully(self, monkeypatch):
        """纯文本响应原样返回"""
        _patch_client(monkeypatch, _FakeResp(
            "纯文本内容，没有 HTML 标签", ctype="text/plain",
        ))
        result = await web_fetch._tool_wrapper.func(url="https://example.com/t")
        assert "纯文本内容" in result

    def test_is_safe(self):
        """web_fetch 是只读安全工具（talk 模式可用）"""
        assert web_fetch._tool_wrapper.is_safe is True
