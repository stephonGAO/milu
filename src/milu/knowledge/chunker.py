"""文本分块 —— 段落边界优先 + 超长段落固定窗口滑动。纯函数，零依赖。

策略（对中英文混排均适用，按字符计长）：
1. 按空行（``\\n\\s*\\n``）切段落；
2. 相邻段落聚合进同一块，直到再放一段会超过 chunk_size；
3. 单段超过 chunk_size 时，段内按固定窗口 + overlap 重叠滑动切分
   （重叠保证语义边界附近的内容不会被"切断后两边都检索不到"）。
"""
from __future__ import annotations

import re


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> list[str]:
    """把长文本切成适合 embedding 的块。

    :param text: 原始文本。
    :param chunk_size: 分块目标长度（字符），须为正。
    :param overlap: 超长段落滑动窗口的重叠字符数；自动夹紧到 [0, chunk_size//2]，
        防止 overlap >= chunk_size 导致窗口不前进（死循环）。
    :return: 文本块列表（每块已 strip，无空块）。空文本返回空列表。
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size 须为正数，收到 {chunk_size}")
    overlap = max(0, min(overlap, chunk_size // 2))

    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(para) > chunk_size:
            # 超长段落：先冲掉缓冲区，再段内滑动窗口切分
            if buf:
                chunks.append(buf)
                buf = ""
            start = 0
            step = chunk_size - overlap
            while start < len(para):
                chunks.append(para[start:start + chunk_size])
                if start + chunk_size >= len(para):
                    break
                start += step
        elif not buf:
            buf = para
        elif len(buf) + 2 + len(para) <= chunk_size:
            buf = f"{buf}\n\n{para}"
        else:
            chunks.append(buf)
            buf = para
    if buf:
        chunks.append(buf)
    return chunks
