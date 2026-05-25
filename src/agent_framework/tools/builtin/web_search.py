"""内置工具：网页搜索

使用 ddgs 库进行真实搜索（无需 API Key）。
也支持通过环境变量配置自定义搜索 API。
"""
from __future__ import annotations

import asyncio
import logging
import os

import httpx
from ddgs import DDGS

from agent_framework.tools.decorator import tool

logger = logging.getLogger(__name__)


@tool(name="web_search", description="搜索互联网信息，返回相关结果摘要")
async def web_search(query: str, num_results: int = 5) -> str:
    """
    搜索互联网并返回相关结果。默认使用 DuckDuckGo 搜索。

    :param query: 搜索关键词
    :param num_results: 返回结果数量，默认 5
    """
    # 检查是否有自定义搜索 API
    custom_api_url = os.environ.get("SEARCH_API_URL")
    custom_api_key = os.environ.get("SEARCH_API_KEY")

    try:
        if custom_api_url and custom_api_key:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                return await _custom_search(client, custom_api_url, custom_api_key, query, num_results)
        else:
            return await _duckduckgo_search(query, num_results)

    except Exception as e:
        logger.error("搜索失败: %s", e)
        return f"错误: 搜索失败 - {e}"


async def _duckduckgo_search(query: str, num_results: int) -> str:
    """使用 ddgs 库进行真实搜索（同步 API 包装为异步）"""

    def _sync_search():
        d = DDGS()
        return d.text(query, max_results=num_results)

    try:
        results = await asyncio.to_thread(_sync_search)
    except Exception as e:
        return f"错误: DuckDuckGo 搜索失败 - {e}"

    if not results:
        return f"未找到与「{query}」相关的结果。请尝试其他关键词。"

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "无标题")
        body = r.get("body", "")
        href = r.get("href", "")
        lines.append(f"{i}. {title}")
        if body:
            lines.append(f"   {body}")
        if href:
            lines.append(f"   链接: {href}")

    return "\n".join(lines)


async def _custom_search(
    client: httpx.AsyncClient,
    api_url: str,
    api_key: str,
    query: str,
    num_results: int,
) -> str:
    """使用自定义搜索 API"""
    resp = await client.get(
        api_url,
        params={"q": query, "num": num_results},
        headers={"Authorization": f"Bearer {api_key}"},
    )

    if resp.status_code != 200:
        return f"错误: 搜索 API 返回状态码 {resp.status_code}"

    data = resp.json()

    # 尝试通用格式解析
    results = []
    items = data.get("results", data.get("items", data.get("data", [])))
    if isinstance(items, list):
        for i, item in enumerate(items[:num_results], 1):
            title = item.get("title", item.get("name", "无标题"))
            snippet = item.get("snippet", item.get("description", item.get("content", "")))
            url = item.get("url", item.get("link", ""))
            results.append(f"{i}. {title}")
            if snippet:
                results.append(f"   {snippet}")
            if url:
                results.append(f"   链接: {url}")

    if not results:
        return f"未找到与「{query}」相关的结果。"

    return "\n".join(results)
