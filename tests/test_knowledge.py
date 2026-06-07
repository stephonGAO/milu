"""向量知识库测试 — chunker / store / embedder / kb_* 工具 / Agent 集成 / pool 派生。

embedding 全部用 FakeEmbedder（确定性向量），不触真实 API；
存储经 conftest 的 MILU_HOME 重定向天然隔离在 tmp_path 下。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from milu.agent import Agent
from milu.agent.events import ToolResult
from milu.knowledge import (
    Embedder,
    KnowledgeConfig,
    KnowledgeStore,
    chunk_text,
    knowledge_dir,
)
from milu.llm.base.exceptions import AuthenticationError
from milu.llm.base.response import StreamChunk, TokenUsage
from milu.llm.providers.base import BaseLLM, ModelCapabilities
from milu.tools.builtin.knowledge_tool import (
    KnowledgeRuntime,
    _current_knowledge,
    kb_ingest,
    kb_manage,
    kb_search,
)


# ── Fake 基础设施 ────────────────────────────────────────


def _fake_vec(text: str, dim: int = 8) -> list[float]:
    """确定性伪向量：同文本同向量（单位化，余弦可比）。"""
    v = [0.0] * dim
    for i, ch in enumerate(text):
        v[i % dim] += ord(ch) / 1000.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class FakeEmbedder:
    """替身 Embedder：不触网，记录调用。"""

    provider = "fake"
    model = "fake-1"

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [_fake_vec(t, self.dim) for t in texts]

    async def aclose(self) -> None:
        pass


def _make_runtime(user_id: str = "t1") -> KnowledgeRuntime:
    """构造注入了 FakeEmbedder 的运行时（存储经 MILU_HOME 隔离）。"""
    rt = KnowledgeRuntime(KnowledgeConfig(user_id=user_id))
    rt._embedder = FakeEmbedder()
    return rt


@dataclass
class _FakeFunction:
    name: str
    arguments: str = ""


@dataclass
class _FakeToolCall:
    function: _FakeFunction
    id: str
    index: int = 0


class _MockLLM(BaseLLM):
    """Echo LLM。"""

    def __init__(self):
        super().__init__(model="mock", provider="mock")

    async def chat(self, messages, **kwargs):
        yield StreamChunk(content="ok", finish_reason="stop")
        yield StreamChunk(usage=TokenUsage(1, 1, 2))

    def _get_available_param_names(self):
        return frozenset()

    @property
    def base_url(self) -> str:
        return "mock://"

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def capabilities(self):
        return ModelCapabilities(supports_streaming=True)


class _KbSearchLLM(_MockLLM):
    """首轮发出 kb_search 调用，次轮结束（验证 run() 注入链路）。"""

    def __init__(self, query: str):
        super().__init__()
        self._query = query
        self._count = 0

    async def chat(self, messages, **kwargs):
        self._count += 1
        if self._count == 1:
            yield StreamChunk(tool_calls=[
                _FakeToolCall(
                    function=_FakeFunction(
                        name="kb_search",
                        arguments=json.dumps({"query": self._query}, ensure_ascii=False),
                    ),
                    id="call_1",
                )
            ])
            yield StreamChunk(finish_reason="tool_calls")
        else:
            yield StreamChunk(content="done", finish_reason="stop")
        yield StreamChunk(usage=TokenUsage(1, 1, 2))

    @property
    def capabilities(self):
        return ModelCapabilities(supports_streaming=True, supports_function_calling=True)


# ── chunker ─────────────────────────────────────────────


class TestChunker:
    def test_empty(self):
        assert chunk_text("") == []
        assert chunk_text("   \n\n  ") == []

    def test_short_text_single_chunk(self):
        assert chunk_text("你好世界") == ["你好世界"]

    def test_paragraphs_merge_within_size(self):
        text = "段落一\n\n段落二\n\n段落三"
        chunks = chunk_text(text, chunk_size=100)
        assert chunks == ["段落一\n\n段落二\n\n段落三"]

    def test_paragraphs_split_when_full(self):
        text = "a" * 60 + "\n\n" + "b" * 60
        chunks = chunk_text(text, chunk_size=100)
        assert chunks == ["a" * 60, "b" * 60]

    def test_long_paragraph_sliding_window(self):
        text = "x" * 1000
        chunks = chunk_text(text, chunk_size=400, overlap=100)
        # 窗口步进 300：[0:400] [300:700] [600:1000]
        assert len(chunks) == 3
        assert all(len(c) <= 400 for c in chunks)
        # 重叠区一致 + 全覆盖
        assert chunks[0][-100:] == chunks[1][:100]
        assert sum(len(c) for c in chunks) >= 1000

    def test_overlap_clamped_no_dead_loop(self):
        # overlap >= chunk_size 时自动夹紧，不死循环
        chunks = chunk_text("y" * 500, chunk_size=100, overlap=100)
        assert chunks
        assert all(len(c) <= 100 for c in chunks)

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError):
            chunk_text("abc", chunk_size=0)


# ── store ───────────────────────────────────────────────


class TestStore:
    def test_knowledge_dir_sanitized(self):
        # 非法字符全部安全化，不含路径分隔符，落在 knowledge/ 下
        d = knowledge_dir("user/../博 客")
        assert d.parent.name == "knowledge"
        assert "/" not in d.name and "\\" not in d.name
        assert knowledge_dir("").name == "default"
        assert knowledge_dir("  ").name == "default"

    def test_add_and_load_roundtrip(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb")
        total = store.add(
            ["文本一", "文本二"], [[1.0, 0.0], [0.0, 1.0]],
            source="a.txt", provider="fake", model="m1",
        )
        assert total == 2
        chunks, vectors = store.load()
        assert [c["text"] for c in chunks] == ["文本一", "文本二"]
        assert vectors.shape == (2, 2)
        meta = store.meta()
        assert (meta["provider"], meta["model"], meta["dim"], meta["count"]) == ("fake", "m1", 2, 2)

    def test_add_appends(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb")
        store.add(["一"], [[1.0, 0.0]], source="a", provider="fake", model="m1")
        total = store.add(["二"], [[0.0, 1.0]], source="b", provider="fake", model="m1")
        assert total == 2
        assert store.count() == 2

    def test_search_ranking(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb")
        store.add(
            ["甲", "乙", "丙"],
            [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]],
            source="s", provider="fake", model="m1",
        )
        results = store.search([1.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0][1]["text"] == "甲"
        assert results[1][1]["text"] == "丙"
        assert results[0][0] > results[1][0]

    def test_delete_source(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb")
        store.add(["一"], [[1.0, 0.0]], source="a", provider="fake", model="m1")
        store.add(["二", "三"], [[0.0, 1.0], [1.0, 1.0]], source="b", provider="fake", model="m1")
        assert store.delete_source("a") == 1
        assert store.count() == 2
        assert store.list_sources() == [("b", 2)]
        assert store.delete_source("不存在") == 0

    def test_delete_last_source_clears(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb")
        store.add(["一"], [[1.0, 0.0]], source="a", provider="fake", model="m1")
        assert store.delete_source("a") == 1
        assert store.is_empty()

    def test_model_mismatch_rejected(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb")
        store.add(["一"], [[1.0, 0.0]], source="a", provider="fake", model="m1")
        with pytest.raises(RuntimeError, match="向量空间不兼容"):
            store.add(["二"], [[0.0, 1.0]], source="b", provider="fake", model="m2")
        with pytest.raises(RuntimeError, match="向量空间不兼容"):
            store.check_model("other", "m1")

    def test_dim_mismatch_on_search(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb")
        store.add(["一"], [[1.0, 0.0]], source="a", provider="fake", model="m1")
        with pytest.raises(RuntimeError, match="维度"):
            store.search([1.0, 0.0, 0.0], top_k=1)

    def test_inconsistent_files_detected(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb")
        store.add(["一"], [[1.0, 0.0]], source="a", provider="fake", model="m1")
        # 人为追加一行 chunk（不动 vectors/meta）→ 数量不一致
        with open(tmp_path / "kb" / "chunks.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"text": "孤儿", "source": "x"}) + "\n")
        with pytest.raises(RuntimeError, match="不一致"):
            store.load()

    def test_stats(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb")
        store.add(["一", "二"], [[1.0, 0.0], [0.0, 1.0]], source="a", provider="fake", model="m1")
        s = store.stats()
        assert s["count"] == 2 and s["sources"] == 1
        assert s["provider"] == "fake" and s["dim"] == 2
        assert s["disk_bytes"] > 0

    def test_load_cache_hit_and_invalidate(self, tmp_path):
        """mtime 缓存：文件未变复用内存对象；写入后签名变化自动失效。"""
        store = KnowledgeStore(tmp_path / "kb")
        store.add(["一"], [[1.0, 0.0]], source="a", provider="fake", model="m1")
        c1, v1 = store.load()
        c2, v2 = store.load()
        assert c1 is c2 and v1 is v2  # 缓存命中：同一对象
        store.add(["二"], [[0.0, 1.0]], source="b", provider="fake", model="m1")
        c3, _ = store.load()
        assert c3 is not c1 and len(c3) == 2  # 写入后失效重载

    def test_cache_invalidated_by_external_write(self, tmp_path):
        """跨进程写入（模拟为第二个 store 实例写盘）后，旧实例缓存自动失效。"""
        store_a = KnowledgeStore(tmp_path / "kb")
        store_a.add(["一"], [[1.0, 0.0]], source="a", provider="fake", model="m1")
        store_a.load()  # 灌入缓存
        store_b = KnowledgeStore(tmp_path / "kb")
        store_b.add(["二"], [[0.0, 1.0]], source="b", provider="fake", model="m1")
        chunks, _ = store_a.load()
        assert len(chunks) == 2  # 磁盘即真相，签名变化触发重载


# ── Embedder ────────────────────────────────────────────


class TestEmbedder:
    def test_unsupported_provider_hint(self):
        with pytest.raises(ValueError, match="没有 embedding API"):
            Embedder(provider="deepseek")
        with pytest.raises(ValueError, match="不支持的 embedding 厂商"):
            Embedder(provider="nope")

    def test_default_model_resolution(self):
        e = Embedder(provider="qwen")
        assert e.model == "text-embedding-v4"
        e2 = Embedder(provider="qwen", model="text-embedding-v3")
        assert e2.model == "text-embedding-v3"

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("QWEN_API_KEY", raising=False)
        e = Embedder(provider="qwen")
        with pytest.raises(AuthenticationError, match="QWEN_API_KEY"):
            e._get_api_key()

    async def test_batching_and_order(self, monkeypatch):
        e = Embedder(provider="qwen", api_key="k", batch_size=10)
        created: list[list[str]] = []

        class _Data:
            def __init__(self, i):
                self.index = i
                self.embedding = [float(i)]

        class _Embeddings:
            async def create(self, model, input):
                created.append(list(input))

                class _Resp:
                    # 倒序返回，验证按 index 还原
                    data = [_Data(i) for i in reversed(range(len(input)))]
                return _Resp()

        class _Client:
            embeddings = _Embeddings()

        monkeypatch.setattr(e, "_get_client", lambda: _Client())
        vectors = await e.embed([f"t{i}" for i in range(25)])
        # 批次并发执行，调用顺序不保证；按多重集比较批大小
        assert sorted(len(b) for b in created) == [5, 10, 10]
        assert len(vectors) == 25
        # gather 保证批次结果顺序 + 批内按 index 升序还原
        assert vectors[0] == [0.0] and vectors[9] == [9.0]

    async def test_concurrent_batches_keep_order(self, monkeypatch):
        """并发批次下结果与输入顺序严格对齐（gather 顺序保证）。"""
        import asyncio as _asyncio

        e = Embedder(provider="qwen", api_key="k", batch_size=2, concurrency=3)

        class _Data:
            def __init__(self, i, v):
                self.index = i
                self.embedding = v

        class _Embeddings:
            async def create(self, model, input):
                # 故意让早批次更慢，验证 gather 仍按批次顺序拼接
                await _asyncio.sleep(0.02 if input[0] == "a0" else 0.001)

                class _Resp:
                    data = [_Data(i, [float(ord(t[0]))]) for i, t in enumerate(input)]
                return _Resp()

        class _Client:
            embeddings = _Embeddings()

        monkeypatch.setattr(e, "_get_client", lambda: _Client())
        texts = ["a0", "a1", "b0", "b1", "c0", "c1"]
        vectors = await e.embed(texts)
        assert vectors == [[float(ord(t[0]))] for t in texts]

    async def test_embed_empty(self):
        e = Embedder(provider="qwen", api_key="k")
        assert await e.embed([]) == []


# ── kb_* 工具（ContextVar 注入）─────────────────────────


class TestKnowledgeTools:
    async def test_not_enabled_message(self):
        assert _current_knowledge.get() is None
        assert "未启用" in await kb_search._tool_wrapper.func(query="任意")
        assert "未启用" in await kb_ingest._tool_wrapper.func(text="任意")
        assert "未启用" in await kb_manage._tool_wrapper.func(action="list")

    async def test_ingest_then_search(self):
        rt = _make_runtime()
        token = _current_knowledge.set(rt)
        try:
            result = await kb_ingest._tool_wrapper.func(
                text="米鹿是一个统一的 Agent 编排框架。\n\n它支持九个大模型厂商。",
                source="intro",
            )
            assert "已入库" in result and "intro" in result

            found = await kb_search._tool_wrapper.func(query="米鹿是一个统一的 Agent 编排框架。")
            assert "intro" in found and "相似度" in found
        finally:
            _current_knowledge.reset(token)

    async def test_ingest_param_validation(self):
        rt = _make_runtime()
        token = _current_knowledge.set(rt)
        try:
            assert "二选一" in await kb_ingest._tool_wrapper.func()
            assert "二选一" in await kb_ingest._tool_wrapper.func(path="a", text="b")
            assert "文件不存在" in await kb_ingest._tool_wrapper.func(path="not/exist.txt")
        finally:
            _current_knowledge.reset(token)

    async def test_ingest_text_file(self, tmp_path):
        f = tmp_path / "notes.md"
        f.write_text("第一段笔记。\n\n第二段笔记。", encoding="utf-8")
        rt = _make_runtime()
        token = _current_knowledge.set(rt)
        try:
            result = await kb_ingest._tool_wrapper.func(path=str(f))
            assert "已入库" in result and "notes.md" in result
            listing = await kb_manage._tool_wrapper.func(action="list")
            assert "notes.md" in listing
        finally:
            _current_knowledge.reset(token)

    async def test_reingest_replaces_source(self):
        rt = _make_runtime()
        token = _current_knowledge.set(rt)
        try:
            await kb_ingest._tool_wrapper.func(text="旧内容", source="doc")
            result = await kb_ingest._tool_wrapper.func(text="新内容", source="doc")
            assert "已替换" in result
            listing = await kb_manage._tool_wrapper.func(action="list")
            assert "doc（1 块）" in listing
        finally:
            _current_knowledge.reset(token)

    async def test_search_empty_kb(self):
        rt = _make_runtime("empty-user")
        token = _current_knowledge.set(rt)
        try:
            assert "知识库为空" in await kb_search._tool_wrapper.func(query="任意")
        finally:
            _current_knowledge.reset(token)

    async def test_search_min_score_filter(self):
        """相似度低于阈值时不返回片段，给出明确的"无相关内容"提示。"""
        rt = KnowledgeRuntime(KnowledgeConfig(user_id="thresh", min_score=0.99))
        rt._embedder = FakeEmbedder()
        token = _current_knowledge.set(rt)
        try:
            await kb_ingest._tool_wrapper.func(text="知识库里的某段内容", source="doc")
            # 完全相同文本 → 相似度 1.0，过阈值
            hit = await kb_search._tool_wrapper.func(query="知识库里的某段内容")
            assert "相似度" in hit and "doc" in hit
            # 不同文本 → 相似度 < 0.99，被阈值拦下
            miss = await kb_search._tool_wrapper.func(query="毫不相干的天文话题")
            assert "足够相关" in miss and "阈值" in miss
        finally:
            _current_knowledge.reset(token)

    async def test_ingest_selfguard_blocked(self):
        """kb_ingest 不得绕过自我保护守卫读取 milu 自身源码。"""
        import milu
        from milu.tools import _selfguard
        protected = str(Path(milu.__file__).parent / "knowledge" / "store.py")
        rt = _make_runtime("guard")
        token = _current_knowledge.set(rt)
        old = _selfguard._ENABLED
        _selfguard.set_enabled(True)
        try:
            result = await kb_ingest._tool_wrapper.func(path=protected)
            assert "安全限制" in result
        finally:
            _selfguard.set_enabled(old)
            _current_knowledge.reset(token)

    async def test_manage_stats_delete_clear(self):
        rt = _make_runtime()
        token = _current_knowledge.set(rt)
        try:
            await kb_ingest._tool_wrapper.func(text="内容甲", source="a")
            await kb_ingest._tool_wrapper.func(text="内容乙", source="b")

            stats = await kb_manage._tool_wrapper.func(action="stats")
            assert "2 块" in stats and "fake/fake-1" in stats

            assert "delete 操作必须指定 source" in await kb_manage._tool_wrapper.func(action="delete")
            assert "不存在" in await kb_manage._tool_wrapper.func(action="delete", source="x")
            assert "已删除" in await kb_manage._tool_wrapper.func(action="delete", source="a")

            assert "已清空" in await kb_manage._tool_wrapper.func(action="clear")
            assert "知识库为空" in await kb_manage._tool_wrapper.func(action="list")
        finally:
            _current_knowledge.reset(token)

    def test_safety_flags(self):
        assert kb_search._tool_wrapper.is_safe is True
        assert kb_ingest._tool_wrapper.is_safe is False
        w = kb_manage._tool_wrapper
        assert w.is_safe is False
        assert w.safe_check({"action": "list"}) is True
        assert w.safe_check({"action": "stats"}) is True
        assert w.safe_check({"action": "delete"}) is False
        assert w.safe_check({"action": "clear"}) is False


# ── Agent 集成 ──────────────────────────────────────────


class TestAgentIntegration:
    def _tool_names(self, agent: Agent) -> list[str]:
        return agent._registry.list_tools()

    def test_disabled_by_default(self):
        agent = Agent(_MockLLM(), tools=[], subagents=[], session_enabled=False)
        assert agent._knowledge is None
        assert not any(n.startswith("kb_") for n in self._tool_names(agent))

    def test_enabled_with_true(self):
        agent = Agent(_MockLLM(), tools=[], subagents=[], session_enabled=False,
                      knowledge=True)
        assert agent._knowledge is not None
        assert agent._knowledge.store.dir_path.name == "default"
        assert {"kb_search", "kb_ingest", "kb_manage"} <= set(self._tool_names(agent))

    def test_enabled_with_user_id(self):
        agent = Agent(_MockLLM(), tools=[], subagents=[], session_enabled=False,
                      knowledge="user-9")
        assert agent._knowledge.store.dir_path.name == "user-9"

    def test_enabled_with_config(self):
        cfg = KnowledgeConfig(user_id="u", chunk_size=500, top_k=3)
        agent = Agent(_MockLLM(), tools=[], subagents=[], session_enabled=False,
                      knowledge=cfg)
        assert agent._knowledge.config is cfg
        assert agent._knowledge.config.top_k == 3

    async def test_run_injects_runtime(self):
        """端到端：Agent.run() 注入 → LLM 调 kb_search → 命中预入库内容。"""
        doc = "米鹿支持向量知识库语义检索。"
        llm = _KbSearchLLM(query=doc)
        agent = Agent(llm, tools=[], subagents=[], session_enabled=False,
                      knowledge="runner")
        agent._knowledge._embedder = fake = FakeEmbedder()
        agent._knowledge.store.add(
            [doc], [_fake_vec(doc, fake.dim)],
            source="intro", provider="fake", model="fake-1",
        )

        results = [e async for e in agent.run("查一下知识库")]
        tool_results = [e for e in results if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert "intro" in tool_results[0].output
        # run 结束后 ContextVar 已清理
        assert _current_knowledge.get() is None

    async def test_close_knowledge_noop_and_idempotent(self):
        agent = Agent(_MockLLM(), tools=[], subagents=[], session_enabled=False)
        await agent.close_knowledge()  # 未启用 no-op
        agent2 = Agent(_MockLLM(), tools=[], subagents=[], session_enabled=False,
                       knowledge=True)
        agent2._knowledge._embedder = FakeEmbedder()
        await agent2.close_knowledge()
        await agent2.close_knowledge()  # 重复调用安全


# ── AgentPool 派生 ──────────────────────────────────────


class TestPoolDerivation:
    def _make_pool(self, **agent_kwargs):
        from milu.serving.pool import AgentPool
        base = {"tools": [], "subagents": [], "session_enabled": False}
        return AgentPool(
            llm_factory=lambda uid, sid: _MockLLM(),
            agent_kwargs={**base, **agent_kwargs},
        )

    def test_knowledge_true_derived_to_user_id(self):
        pool = self._make_pool(knowledge=True)
        agent = pool._default_agent_factory("alice", "s1", _MockLLM())
        assert agent._knowledge.store.dir_path.name == "alice"

    def test_knowledge_string_respected(self):
        pool = self._make_pool(knowledge="team-kb")
        agent = pool._default_agent_factory("alice", "s1", _MockLLM())
        assert agent._knowledge.store.dir_path.name == "team-kb"

    def test_knowledge_default_off(self):
        pool = self._make_pool()
        agent = pool._default_agent_factory("alice", "s1", _MockLLM())
        assert agent._knowledge is None


# ── 配置接入 ────────────────────────────────────────────


class TestConfigIntegration:
    def test_builtin_defaults_has_knowledge_section(self):
        from milu.config import _builtin_defaults
        kn = _builtin_defaults()["knowledge"]
        assert kn["enabled"] is False
        assert kn["embedding_provider"] == "qwen"
        assert "user_id" not in kn and "api_key" not in kn

    def test_from_mapping_ignores_extra_keys(self):
        cfg = KnowledgeConfig.from_mapping(
            {"enabled": True, "chunk_size": 600, "unknown": 1}, user_id="u2"
        )
        assert cfg.user_id == "u2"
        assert cfg.chunk_size == 600

    def test_to_knowledge_config(self):
        from milu.config import load_config
        cfg = load_config().to_knowledge_config("bob")
        assert cfg.user_id == "bob"
        assert cfg.embedding_provider == "qwen"
