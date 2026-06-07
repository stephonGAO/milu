"""内置工具：向量知识库（kb_search / kb_ingest / kb_manage）

对私有语料做语义检索（RAG）。与 memory 互补——memory 是少量条目全量进
system prompt；knowledge 是大语料分块向量化、按需召回。

启用方式（默认关闭，不在 BUILTIN_TOOLS）：`Agent(knowledge=True)`（身份
"default"）/ `Agent(knowledge="user-123")`（按用户隔离）/
`Agent(knowledge=KnowledgeConfig(...))`（程序化定制）。启用后 Agent 负责：
  1. 注册 kb_search / kb_ingest / kb_manage 三个工具；
  2. run() 入口经 ContextVar 注入 KnowledgeRuntime（asyncio 任务级隔离，
     未启用注入 None——子代理不会继承父的知识库而意外写入）。

工具拆分沿用 schedule 的参数正交性原则：kb_search 纯读高频主路径、
kb_ingest 专职入库（写存储 + 产生 embedding API 费用，is_safe=False）、
kb_manage 管理操作（schema 极简，safe_check 判 list/stats 只读安全）。
user_id 经 ContextVar 获取、不暴露为 LLM 参数（防伪造他人身份，同 schedule）。
"""
from __future__ import annotations

import contextvars
import logging
import os
from datetime import datetime
from typing import Literal

from milu.knowledge import (
    Embedder,
    KnowledgeConfig,
    KnowledgeStore,
    chunk_text,
    knowledge_dir,
    store_lock,
)
from milu.llm.base.exceptions import AuthenticationError
from milu.tools.decorator import tool

logger = logging.getLogger(__name__)

# 单次入库块数上限（800 字符/块 × 2000 ≈ 160 万字符，超出建议拆分文档分次入库）
_MAX_INGEST_CHUNKS = 2000
# 纯文本文件读取上限（防误把超大文件整个 embedding）
_MAX_TEXT_BYTES = 5 * 1024 * 1024
# pdf 分页提取的窗口页数（小于 doc_tool 默认 20 页：降低单窗口命中 50k 字符截断的概率）
_PDF_WINDOW_PAGES = 10

# ── 状态注入（asyncio 任务级隔离）──────────────────────
#
# Agent.run() 入口注入当前 Agent 的 KnowledgeRuntime（未启用 knowledge 时注入
# None，显式隔离——子代理不会经 Context 继承父的知识库而意外读写）。

_current_knowledge: contextvars.ContextVar["KnowledgeRuntime | None"] = contextvars.ContextVar(
    "current_knowledge", default=None
)

_NOT_ENABLED = (
    "知识库未启用：构造 Agent 时传 knowledge=True（或用户标识字符串 / "
    "KnowledgeConfig）开启"
)


class KnowledgeRuntime:
    """Agent 持有的知识库运行时：config + store + 懒初始化的 Embedder。

    Embedder 内含 AsyncOpenAI 客户端，生命周期跟随 Agent（复用连接池，
    Agent 关闭时经 aclose() 释放），不在工具调用内反复新建。
    懒初始化使配置错误（如厂商无 embedding API）在工具调用时以友好文案
    暴露，而非让 Agent 构造直接失败。
    """

    def __init__(self, config: KnowledgeConfig):
        self.config = config
        self.store = KnowledgeStore(knowledge_dir(config.user_id))
        self._embedder: Embedder | None = None

    def get_embedder(self) -> Embedder:
        if self._embedder is None:
            c = self.config
            self._embedder = Embedder(
                provider=c.embedding_provider,
                model=c.embedding_model,
                api_key=c.api_key,
                batch_size=c.batch_size,
            )
        return self._embedder

    async def aclose(self) -> None:
        if self._embedder is not None:
            await self._embedder.aclose()
            self._embedder = None


# ── 文件文本提取（复用 doc_tool 解析栈）────────────────────


def _extract_file(abs_path: str) -> "tuple[str, str] | str":
    """从文件提取全文。:return: (文本, 备注) 或错误信息字符串。"""
    from milu.tools.builtin.doc_tool import _LEGACY_HINTS, _READERS

    ext = os.path.splitext(abs_path)[1].lower()
    if ext in _LEGACY_HINTS:
        return f"不支持二进制老格式 {ext}。{_LEGACY_HINTS[ext]}"
    if ext == ".pdf":
        return _extract_pdf(abs_path)
    handler = _READERS.get(ext)
    if handler is not None:
        result = handler(abs_path)
        if not result.get("success"):
            return f"文档解析失败：{result.get('error', '未知错误')}"
        note = ""
        if result.get("truncated"):
            note = f"注意：文档超长已截断，仅入库前半部分（{result.get('hint', '')}）"
        return result.get("content", ""), note
    # 其余扩展名按纯文本读取（md/txt/代码等）
    try:
        if os.path.getsize(abs_path) > _MAX_TEXT_BYTES:
            return (
                f"文件超过 {_MAX_TEXT_BYTES // (1024 * 1024)}MB，"
                "请拆分后分次入库（或确认它确实是文本文件）"
            )
        with open(abs_path, encoding="utf-8", errors="replace") as f:
            return f.read(), ""
    except OSError as e:
        return f"读取文件失败：{e}"


def _extract_pdf(abs_path: str) -> "tuple[str, str] | str":
    """pdf 全文提取：按页窗口循环调 doc_tool._read_pdf，绕过单次 20 页/50k 字符上限。"""
    from milu.tools.builtin.doc_tool import _read_pdf

    parts: list[str] = []
    notes: list[str] = []
    start = 1
    while True:
        result = _read_pdf(abs_path, page_start=start, page_end=start + _PDF_WINDOW_PAGES - 1)
        if not result.get("success"):
            return f"文档解析失败：{result.get('error', '未知错误')}"
        parts.append(result.get("content", ""))
        total = int(result.get("total_pages", 0))
        end = int(result.get("page_end", 0))
        # 单窗口内容命中 50k 字符截断（极密集文本）：窗口内损失少量文本，提示之
        if result.get("truncated") and end < total:
            notes.append(f"第 {start}-{end} 页内容过密，窗口内有少量截断")
        if end >= total:
            break
        start = end + 1
    note = f"注意：{'；'.join(notes)}" if notes else ""
    return "\n\n".join(parts), note


# ── 工具函数 ──────────────────────────────────────────


@tool(
    name="kb_search",
    description=(
        "在本地向量知识库中按语义检索相关内容片段（已入库的文档/文本）。"
        "**主动调用时机**：用户的问题可能涉及知识库中的私有资料"
        "（产品文档、笔记、FAQ 等）时优先检索，再基于检索结果回答。"
        "返回带来源与相似度的片段列表。"
    ),
)
async def kb_search(query: str, top_k: int = 0) -> str:
    """
    :param query: 检索语句（自然语言，描述要找的内容）
    :param top_k: 返回片段数；0 表示用配置默认值
    """
    rt = _current_knowledge.get()
    if rt is None:
        return _NOT_ENABLED
    query = query.strip()
    if not query:
        return "query 不能为空"
    try:
        if rt.store.is_empty():
            return "知识库为空，请先用 kb_ingest 入库文档或文本。"
        embedder = rt.get_embedder()
        rt.store.check_model(embedder.provider, embedder.model)
        query_vec = (await embedder.embed([query]))[0]
        results = rt.store.search(query_vec, top_k or rt.config.top_k)
    except (AuthenticationError, ValueError, RuntimeError) as e:
        return f"检索失败：{e}"
    except Exception as e:
        return f"检索失败（embedding 调用异常）：{e}"

    # 相似度阈值过滤：把低于阈值的无关片段挡在 LLM 上下文之外（防诱导幻觉）
    min_score = rt.config.min_score
    filtered = [(s, c) for s, c in results if s >= min_score]
    if not filtered:
        top1 = results[0][0] if results else 0.0
        return (
            f"知识库中没有与「{query}」足够相关的内容"
            f"（最高相似度 {top1:.3f}，低于阈值 {min_score}）。"
            "该问题可能超出知识库覆盖范围，请改用其他工具或直接回答。"
        )
    results = filtered

    lines = [f"与「{query}」最相关的 {len(results)} 个片段（知识库共 {rt.store.count()} 块）："]
    for i, (score, chunk) in enumerate(results, 1):
        lines.append(
            f"\n[{i}] 来源: {chunk.get('source', '')} | 相似度: {score:.3f}\n"
            f"{chunk.get('text', '')}"
        )
    return "\n".join(lines)


@tool(
    name="kb_ingest",
    description=(
        "把文档或文本入库到本地向量知识库（分块 → embedding → 持久存储），"
        "供 kb_search 语义检索。支持 docx/xlsx/pdf/pptx 及任意文本文件（md/txt/代码）。"
        "同名来源（source）再次入库会**整体替换**旧内容。"
        "注意：入库会产生 embedding API 调用费用，大文档耗时较长。"
    ),
    is_safe=False,
)
async def kb_ingest(path: str = "", text: str = "", source: str = "") -> str:
    """
    :param path: 要入库的本地文件路径（与 text 二选一）
    :param text: 直接入库的文本内容（与 path 二选一）
    :param source: 来源标识，用于 kb_manage 按来源管理/删除；path 入库时默认取文件名
    """
    rt = _current_knowledge.get()
    if rt is None:
        return _NOT_ENABLED
    if bool(path) == bool(text):
        return "参数错误：path 与 text 必须二选一"

    note = ""
    if path:
        abs_path = os.path.abspath(path)
        # 自我保护守卫：与 file_read 同栈，禁止把 milu 自身源码/配置入库
        # （kb_ingest 直接读文件，不走 file_read，须独立检查防旁路）
        from milu.tools._selfguard import is_protected_path
        if is_protected_path(abs_path):
            return f"安全限制：禁止访问 milu 自身代码/配置（受保护路径：{abs_path}）"
        if not os.path.isfile(abs_path):
            return f"文件不存在: {abs_path}"
        extracted = _extract_file(abs_path)
        if isinstance(extracted, str):
            return extracted
        content, note = extracted
        source = source.strip() or os.path.basename(abs_path)
    else:
        content = text
        source = source.strip() or f"text-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    try:
        chunks = chunk_text(content, rt.config.chunk_size, rt.config.chunk_overlap)
        if not chunks:
            return "内容为空，未入库"
        if len(chunks) > _MAX_INGEST_CHUNKS:
            return (
                f"文档过大（分块后 {len(chunks)} 块，上限 {_MAX_INGEST_CHUNKS}），"
                "请拆分后分次入库"
            )
        embedder = rt.get_embedder()
        # embedding 前先校验模型指纹（错配早失败，不白烧 API 费用）
        rt.store.check_model(embedder.provider, embedder.model)
        vectors = await embedder.embed(chunks)
        async with store_lock(rt.store.dir_path):
            replaced = rt.store.delete_source(source)
            total = rt.store.add(
                chunks, vectors, source=source,
                provider=embedder.provider, model=embedder.model,
            )
    except (AuthenticationError, ValueError, RuntimeError) as e:
        return f"入库失败：{e}"
    except Exception as e:
        return f"入库失败（embedding 调用异常）：{e}"

    parts = [f"已入库：来源「{source}」共 {len(chunks)} 块（知识库现有 {total} 块）。"]
    if replaced:
        parts.append(f"已替换同名来源的 {replaced} 个旧块。")
    if note:
        parts.append(note)
    return " ".join(parts)


@tool(
    name="kb_manage",
    description=(
        "管理本地向量知识库。action：list（按来源列出已入库内容）/ "
        "stats（统计信息）/ delete（删除指定来源的全部块，须传 source）/ "
        "clear（清空整个知识库，慎用）。"
    ),
    is_safe=False,
    safe_check=lambda args: args.get("action") in ("list", "stats"),
)
async def kb_manage(
    action: Literal["list", "stats", "delete", "clear"],
    source: str = "",
) -> str:
    """
    :param action: 操作类型：list / stats / delete / clear
    :param source: 来源标识（action=delete 时必填）
    """
    rt = _current_knowledge.get()
    if rt is None:
        return _NOT_ENABLED
    try:
        if action == "list":
            sources = rt.store.list_sources()
            if not sources:
                return "知识库为空。"
            lines = [f"知识库共 {rt.store.count()} 块，{len(sources)} 个来源："]
            lines.extend(f"- {src}（{n} 块）" for src, n in sources)
            return "\n".join(lines)
        if action == "stats":
            s = rt.store.stats()
            return (
                f"知识库统计：{s['count']} 块 / {s['sources']} 个来源 / "
                f"embedding: {s['provider']}/{s['model']}（{s['dim']} 维）/ "
                f"磁盘占用 {s['disk_bytes'] / 1024:.1f} KB / 目录 {s['dir']}"
            )
        if action == "delete":
            if not source.strip():
                return "参数错误：delete 操作必须指定 source（可先用 list 查看来源列表）"
            async with store_lock(rt.store.dir_path):
                removed = rt.store.delete_source(source.strip())
            if removed == 0:
                return f"来源「{source}」不存在（可用 list 查看来源列表）"
            return f"已删除来源「{source}」的 {removed} 块（知识库现有 {rt.store.count()} 块）"
        if action == "clear":
            async with store_lock(rt.store.dir_path):
                rt.store.clear()
            return "知识库已清空。"
        return f"未知操作：{action}（可用：list / stats / delete / clear）"
    except (ValueError, RuntimeError) as e:
        return f"操作失败：{e}"
