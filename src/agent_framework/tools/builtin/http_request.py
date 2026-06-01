"""内置工具：HTTP 请求

支持 GET/POST 请求，自定义 headers、body 和超时时间。
"""
from __future__ import annotations

import json
from typing import Literal, Optional

import httpx

from agent_framework.tools.decorator import tool

# 响应体最大字符数
_MAX_RESPONSE_CHARS = 4096


@tool(name="http_request", description="发送 HTTP 请求（GET/POST），返回响应状态码和内容", read_only=True)
async def http_request(
    url: str,
    method: Literal["GET", "POST"] = "GET",
    headers: Optional[str] = None,
    body: Optional[str] = None,
    timeout: int = 30,
) -> str:
    """
    发送 HTTP 请求并返回响应。

    :param url: 请求的完整 URL
    :param method: HTTP 方法，GET 或 POST
    :param headers: 自定义请求头，JSON 格式字符串，如 '{"Authorization": "Bearer xxx"}'
    :param body: 请求体内容（仅 POST 有效）
    :param timeout: 超时时间（秒），默认 30
    """
    try:
        # 解析 headers
        req_headers = {}
        if headers:
            try:
                req_headers = json.loads(headers)
            except json.JSONDecodeError:
                return f"错误: headers 参数不是有效的 JSON: {headers}"

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.request(
                method=method,
                url=url,
                headers=req_headers,
                content=body.encode("utf-8") if body else None,
            )

        # 构建结果
        body_text = response.text
        truncated = False
        if len(body_text) > _MAX_RESPONSE_CHARS:
            body_text = body_text[:_MAX_RESPONSE_CHARS]
            truncated = True

        result = f"状态码: {response.status_code}\n\n{body_text}"
        if truncated:
            result += f"\n\n... [响应已截断，原始长度 {len(response.text)} 字符]"

        return result

    except httpx.TimeoutException:
        return f"错误: 请求超时（{timeout}秒）"
    except httpx.ConnectError as e:
        return f"错误: 连接失败 - {e}"
    except httpx.InvalidURL as e:
        return f"错误: 无效的 URL - {e}"
    except Exception as e:
        return f"错误: 请求失败 - {e}"
