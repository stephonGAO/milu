"""内置工具：网页搜索（可插拔后端）

后端通过环境变量 WEB_SEARCH_PROVIDER 选择（默认 ddg）：
- ddg    : DuckDuckGo（ddgs 库，无需 API Key；中国大陆网络不可直连）
- tavily : Tavily Search API（需 TAVILY_API_KEY；为 LLM 设计，返回已清洗摘要）
- bocha  : 博查搜索 API（需 BOCHA_API_KEY；国内可直连）

向后兼容：设置 SEARCH_API_URL + SEARCH_API_KEY 时优先走自定义通用搜索 API。

中国大陆部署建议（DuckDuckGo 被墙，默认后端不可用）：
1. 配置 WEB_SEARCH_PROVIDER=bocha（或 tavily）及对应 API Key；或
2. 直接使用 LLM 自带联网搜索（如 Qwen 创建时 web_search=True），可不依赖本工具
"""
from __future__ import annotations

import asyncio
import logging
import os

import httpx
from ddgs import DDGS

from agent_framework.tools.decorator import tool

logger = logging.getLogger(__name__)

_TAVILY_ENDPOINT = "https://api.tavily.com/search"
_BOCHA_ENDPOINT = "https://api.bochaai.com/v1/web-search"


def _format_results(items: list[dict], query: str) -> str:
    """统一格式化：每条 {title, snippet, url} → 编号列表。纯函数。"""
    lines: list[str] = []
    for i, item in enumerate(items, 1):
        title = item.get("title") or "无标题"
        snippet = item.get("snippet") or ""
        url = item.get("url") or ""
        lines.append(f"{i}. {title}")
        if snippet:
            lines.append(f"   {snippet}")
        if url:
            lines.append(f"   链接: {url}")
    if not lines:
        return f"未找到与「{query}」相关的结果。请尝试其他关键词。"
    return "\n".join(lines)


@tool(name="web_search", description=(
    "搜索互联网信息，返回相关结果摘要（标题/摘要/链接）。"
    "需要进一步阅读某条结果时，用 web_fetch 抓取该链接的页面全文。"
))
async def web_search(query: str, num_results: int = 5) -> str:
    """
    搜索互联网并返回相关结果。后端由环境变量 WEB_SEARCH_PROVIDER 决定（默认 DuckDuckGo）。

    :param query: 搜索关键词
    :param num_results: 返回结果数量，默认 5
    """
    provider = os.environ.get("WEB_SEARCH_PROVIDER", "ddg").strip().lower()
    custom_api_url = os.environ.get("SEARCH_API_URL")
    custom_api_key = os.environ.get("SEARCH_API_KEY")

    try:
        # 1. 自定义通用搜索 API（向后兼容，优先级最高）
        if custom_api_url and custom_api_key:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                return await _custom_search(client, custom_api_url, custom_api_key, query, num_results)

        # 2. 命名 provider（配置了 provider 却缺 Key 视为配置错误，明确报错不静默回退）
        if provider == "tavily":
            key = os.environ.get("TAVILY_API_KEY")
            if not key:
                return "错误: WEB_SEARCH_PROVIDER=tavily 但未配置 TAVILY_API_KEY"
            async with httpx.AsyncClient(timeout=15) as client:
                return await _search_tavily(client, key, query, num_results)

        if provider == "bocha":
            key = os.environ.get("BOCHA_API_KEY")
            if not key:
                return "错误: WEB_SEARCH_PROVIDER=bocha 但未配置 BOCHA_API_KEY"
            async with httpx.AsyncClient(timeout=15) as client:
                return await _search_bocha(client, key, query, num_results)

        # 3. 默认 DuckDuckGo
        return await _duckduckgo_search(query, num_results)

    except Exception as e:
        logger.error("搜索失败: %s", e)
        return f"错误: 搜索失败 - {e}"


# ── 各后端实现 ────────────────────────────────────────────


async def _duckduckgo_search(query: str, num_results: int) -> str:
    """DuckDuckGo（ddgs 库，同步 API 包装为异步）"""

    def _sync_search():
        d = DDGS()
        return d.text(query, max_results=num_results)

    try:
        results = await asyncio.to_thread(_sync_search)
    except Exception as e:
        return f"错误: DuckDuckGo 搜索失败 - {e}"

    items = [
        {"title": r.get("title"), "snippet": r.get("body"), "url": r.get("href")}
        for r in (results or [])
    ]
    return _format_results(items, query)


async def _search_tavily(
    client: httpx.AsyncClient, api_key: str, query: str, num_results: int,
) -> str:
    """Tavily Search API（https://tavily.com，为 LLM 设计）"""
    resp = await client.post(_TAVILY_ENDPOINT, json={
        "api_key": api_key,
        "query": query,
        "max_results": num_results,
    })
    if resp.status_code != 200:
        return f"错误: Tavily API 返回状态码 {resp.status_code}"

    data = resp.json()
    items = [
        {"title": r.get("title"), "snippet": r.get("content"), "url": r.get("url")}
        for r in data.get("results", [])[:num_results]
    ]
    return _format_results(items, query)


async def _search_bocha(
    client: httpx.AsyncClient, api_key: str, query: str, num_results: int,
) -> str:
    """博查搜索 API（https://open.bochaai.com，国内可直连）"""
    resp = await client.post(
        _BOCHA_ENDPOINT,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"query": query, "count": num_results, "summary": True},
    )
    if resp.status_code != 200:
        return f"错误: 博查 API 返回状态码 {resp.status_code}"

    data = resp.json()
    pages = (data.get("data") or {}).get("webPages") or {}
    items = [
        {
            "title": r.get("name"),
            "snippet": r.get("summary") or r.get("snippet"),
            "url": r.get("url"),
        }
        for r in (pages.get("value") or [])[:num_results]
    ]
    return _format_results(items, query)


async def _custom_search(
    client: httpx.AsyncClient,
    api_url: str,
    api_key: str,
    query: str,
    num_results: int,
) -> str:
    """自定义通用搜索 API（GET + Bearer，宽松解析常见响应格式）"""
    resp = await client.get(
        api_url,
        params={"q": query, "num": num_results},
        headers={"Authorization": f"Bearer {api_key}"},
    )

    if resp.status_code != 200:
        return f"错误: 搜索 API 返回状态码 {resp.status_code}"

    data = resp.json()

    raw_items = data.get("results", data.get("items", data.get("data", [])))
    items = []
    if isinstance(raw_items, list):
        for item in raw_items[:num_results]:
            items.append({
                "title": item.get("title", item.get("name")),
                "snippet": item.get("snippet", item.get("description", item.get("content"))),
                "url": item.get("url", item.get("link")),
            })
    if not items:
        return f"未找到与「{query}」相关的结果。"
    return _format_results(items, query)
