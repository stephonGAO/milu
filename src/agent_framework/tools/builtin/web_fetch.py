"""内置工具：网页正文抓取（HTML → Markdown）

与 http_request 的分工：
- web_fetch：面向「给 LLM 阅读」的网页获取——抽取正文并转 Markdown，
  去除脚本/样式/导航等噪声（实测可节省 20-30% token），适合调研/阅读场景
- http_request：面向 API 调用（JSON、自定义 method/header/body），返回原始响应

对标 MCP 官方 Fetch server 与 Claude Code 的 WebFetch 工具。
"""
from __future__ import annotations

import json
import re

import httpx

from agent_framework.tools.decorator import tool

# 通用 UA（部分站点拒绝无 UA 请求）
_UA = "Mozilla/5.0 (compatible; agent-framework/0.1)"

# 噪声标签：对正文阅读无价值、且显著消耗 token
_NOISE_TAGS = [
    "script", "style", "noscript", "nav", "header", "footer",
    "aside", "form", "iframe", "svg", "button",
]


def _html_to_markdown(html: str) -> str:
    """HTML → Markdown 正文提取（纯函数，便于单测）。

    流程：剔除噪声标签 → 优先定位正文容器（article/main/body）→
    markdownify 转换 → 压缩多余空行。
    """
    from bs4 import BeautifulSoup
    from markdownify import markdownify as md

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_NOISE_TAGS):
        tag.decompose()

    # 优先取语义化正文容器，降低侧边栏/页眉残留
    main = soup.find("article") or soup.find("main") or soup.body or soup
    text = md(str(main), heading_style="ATX")
    # 压缩 3 个以上连续空行为 1 个空行
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


@tool(
    name="web_fetch",
    description=(
        "抓取网页并提取正文转为 Markdown（自动去除脚本/样式/导航等噪声，适合阅读理解）。"
        "需要读取某个 URL 的页面内容时优先用此工具；"
        "调用 JSON API 或需要自定义请求方法/头时才用 http_request。"
    ),
)
async def web_fetch(url: str, max_chars: int = 20000) -> str:
    """
    :param url: 要抓取的网页地址（仅支持 http/https）
    :param max_chars: 返回内容的最大字符数，超出截断并标注总长度
    """
    if not url.lower().startswith(("http://", "https://")):
        return "错误：仅支持 http/https URL"

    try:
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=30.0, headers={"User-Agent": _UA},
        ) as client:
            resp = await client.get(url)
    except httpx.HTTPError as e:
        return f"抓取失败: {e}"

    ctype = resp.headers.get("content-type", "")
    body = resp.text

    if "text/html" in ctype or (not ctype and body.lstrip()[:1] == "<"):
        try:
            content = _html_to_markdown(body)
        except Exception:
            content = body  # 转换失败时降级返回原文，不让工具整体失败
    elif "application/json" in ctype:
        try:
            content = json.dumps(resp.json(), ensure_ascii=False, indent=2)
        except ValueError:
            content = body
    else:
        content = body

    total = len(content)
    if total > max_chars:
        content = content[:max_chars] + (
            f"\n\n…[内容已截断：全文共 {total} 字符，仅显示前 {max_chars}。"
            f"如需后续内容可改用 http_request 或缩小提取目标]"
        )
    return f"[{resp.status_code}] {url}\n\n{content}"
