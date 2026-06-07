"""向量嵌入客户端 —— 走各厂商 OpenAI 兼容 /embeddings 端点（与 LLM 层同栈）。

刻意保持单类 + 静态表（不做 BaseEmbedder 抽象基类 + 注册表）：各厂商的
embeddings 端点形态高度一致，差异只有 base_url 与默认模型名，一张表足够。

embedding 厂商独立于对话模型配置——9 个对话厂商中 claude(anthropic)/
deepseek/kimi 没有 embedding API，对话用它们时 embedding 仍需配到本表
内的厂商（config.json `knowledge.embedding_provider`）。
"""
from __future__ import annotations

import asyncio
import os

from milu.llm.base.exceptions import AuthenticationError

# provider → (base_url, 默认 embedding 模型)。
# base_url 与 llm/providers/ 各实现一致；模型名可经 KnowledgeConfig.embedding_model 覆盖。
_EMBEDDING_PROVIDERS: dict[str, tuple[str, str]] = {
    "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "text-embedding-v4"),
    "glm": ("https://open.bigmodel.cn/api/paas/v4", "embedding-3"),
    "minimax": ("https://api.minimax.chat/v1", "embo-01"),
    "doubao": ("https://ark.cn-beijing.volces.com/api/v3", "doubao-embedding"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-embedding-001"),
    "openai": ("https://api.openai.com/v1", "text-embedding-3-small"),
}

# 无 embedding API 的对话厂商 → 给明确改配指引（而非笼统的"不支持"）
_NO_EMBEDDING = {"anthropic", "claude", "deepseek", "kimi"}


class Embedder:
    """OpenAI 兼容 embeddings 客户端（AsyncOpenAI 协程安全，可跨调用复用连接池）。"""

    def __init__(
        self,
        provider: str = "qwen",
        model: str = "",
        api_key: str | None = None,
        batch_size: int = 10,
        concurrency: int = 4,
    ):
        """
        :param provider: embedding 厂商名（见 _EMBEDDING_PROVIDERS）。
        :param model: embedding 模型名；空 → 厂商默认。
        :param api_key: 显式密钥；None → 环境变量 {PROVIDER}_API_KEY。
        :param batch_size: 单批条数（多数国产厂商单批上限 10）。
        :param concurrency: 批次并发数（大文档入库吞吐受 API 往返延迟主导，
            并发批次可提速数倍；4 路在各厂商默认限流内安全）。
        :raises ValueError: 厂商不在支持表内（含无 embedding API 厂商的改配指引）。
        """
        if provider not in _EMBEDDING_PROVIDERS:
            supported = "/".join(sorted(_EMBEDDING_PROVIDERS))
            hint = (
                f"厂商 '{provider}' 没有 embedding API，"
                if provider in _NO_EMBEDDING
                else f"不支持的 embedding 厂商 '{provider}'，"
            )
            raise ValueError(
                hint + f"知识库 embedding 请配置为：{supported}"
                "（config.json 的 knowledge.embedding_provider，独立于对话模型）"
            )
        self._provider = provider
        base_url, default_model = _EMBEDDING_PROVIDERS[provider]
        self._base_url = base_url
        self._model = model or default_model
        self._api_key = api_key
        self._batch_size = max(1, batch_size)
        self._concurrency = max(1, concurrency)
        self._client = None

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    def _get_api_key(self) -> str:
        """密钥解析：显式传入 > 环境变量 {PROVIDER}_API_KEY（与 BaseLLM 同约定）。"""
        if self._api_key:
            return self._api_key
        env_key = f"{self._provider.upper()}_API_KEY"
        key = os.environ.get(env_key)
        if not key:
            raise AuthenticationError(
                f"缺少 embedding API Key：请设置环境变量 {env_key}"
                "（或在构造 KnowledgeConfig 时显式传入 api_key）"
            )
        return key

    def _get_client(self):
        """懒加载 AsyncOpenAI 客户端（仿 BaseLLM._get_client）。"""
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self._get_api_key(), base_url=self._base_url)
        return self._client

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """批量计算向量（按 batch_size 自动分批、批次间并发，结果顺序与输入对齐）。

        吞吐由 API 往返延迟主导（单批 ~300ms），Semaphore 限幅的并发批次
        可把大文档入库提速约 concurrency 倍；asyncio.gather 保证结果按
        批次顺序拼接，与输入顺序严格对齐。
        """
        if not texts:
            return []
        client = self._get_client()
        batches = [texts[i:i + self._batch_size]
                   for i in range(0, len(texts), self._batch_size)]
        sem = asyncio.Semaphore(self._concurrency)

        async def _embed_batch(batch: list[str]) -> list[list[float]]:
            async with sem:
                resp = await client.embeddings.create(model=self._model, input=batch)
            # 按 index 还原批内顺序（OpenAI 兼容端点通常有序，防御性排序一次）
            data = sorted(resp.data, key=lambda d: d.index)
            return [d.embedding for d in data]

        if len(batches) == 1:
            return await _embed_batch(batches[0])
        results = await asyncio.gather(*(_embed_batch(b) for b in batches))
        return [v for batch_vecs in results for v in batch_vecs]

    async def aclose(self) -> None:
        """关闭 HTTP 客户端，释放连接池。"""
        if self._client is not None:
            await self._client.close()
            self._client = None
