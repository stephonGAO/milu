"""内置工具：网页搜索

使用 DuckDuckGo Instant Answer API 进行免费搜索（无需 API Key）。
也支持通过环境变量配置自定义搜索 API。
"""
from __future__ import annotations

import os

import httpx

from agent_framework.tools.decorator import tool

# DuckDuckGo Instant Answer API
_DDG_API = "https://api.duckduckgo.com/"


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
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            if custom_api_url and custom_api_key:
                return await _custom_search(client, custom_api_url, custom_api_key, query, num_results)
            else:
                return await _duckduckgo_search(client, query, num_results)

    except httpx.TimeoutException:
        return "错误: 搜索请求超时"
    except httpx.ConnectError:
        return "错误: 无法连接到搜索服务"
    except Exception as e:
        return f"错误: 搜索失败 - {e}"


async def _duckduckgo_search(client: httpx.AsyncClient, query: str, num_results: int) -> str:
    """使用 DuckDuckGo Instant Answer API 搜索"""
    resp = await client.get(_DDG_API, params={"q": query, "format": "json", "no_html": 1})

    if resp.status_code != 200:
        return f"错误: 搜索 API 返回状态码 {resp.status_code}"

    data = resp.json()
    results = []

    # 摘要信息
    if data.get("AbstractText"):
        results.append(f"【摘要】{data['AbstractText']}")
        if data.get("AbstractURL"):
            results.append(f"  来源: {data['AbstractURL']}")

    # 相关主题
    topics = data.get("RelatedTopics", [])
    for i, topic in enumerate(topics[:num_results], 1):
        if isinstance(topic, dict) and "Text" in topic:
            text = topic["Text"]
            url = topic.get("FirstURL", "")
            results.append(f"\n{i}. {text}")
            if url:
                results.append(f"   链接: {url}")

    if not results:
        return f"未找到与「{query}」相关的结果。请尝试其他关键词。"

    return "\n".join(results)


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
