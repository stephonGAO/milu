# Knowledge Base（本地向量知识库）设计

> 日期：2026-06-02
> 状态：待审阅
> 阶段：在现有 Agent 框架上增加本地知识库能力

## 一、目标 & 约束

为现有 agent 增加一个**纯本地**的知识库能力：把任意文本入库，自动切块并向量化；agent 可以按 query 检索相关片段作为上下文。

**硬约束**：
- 不引入任何需要独立进程的"向量数据库服务器"（排除 Qdrant server、Milvus server、Weaviate 等）
- 依赖最小化，复用项目现有 LLM 抽象层
- 必须能离线完成持久化，重启 agent 后数据不丢

## 二、技术选型

| 维度 | 选型 | 理由 |
|------|------|------|
| 向量库 | **Chroma 0.5+ `PersistentClient`** | 嵌入式（in-process）运行，零子进程；持久化为单一目录；原生 metadata 过滤 |
| 嵌入生成 | **复用现有 LLM 的 embedding API** | 项目已有 `ModelCapabilities.supports_embedding`；轻量、零新模型依赖 |
| 切块策略 | **固定字符切块 + overlap** | 工业标准，召回质量与实现复杂度平衡 |
| 持久化路径 | `~/.milu/kb/chroma/` | 与项目其他目录约定一致（参考 skills/ 路径） |

**候选方案对比记录**：
- LanceDB：性能更好但生态偏小
- FAISS + 自建元数据层：最轻量但要写 ~200 行胶水代码，性价比低

## 三、架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Agent 运行时                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │kb_ingest │  │ kb_search│  │ kb_manage│                  │
│  │ (3 action)│  │          │  │ (3 action)│                 │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                  │
│       └──────┬──────┴──────┬──────┘                         │
│              │             │                                │
│       ┌──────▼──────┐ ┌───▼────────┐                        │
│       │ KBIngestor  │ │ KBRetriever│                        │
│       └──────┬──────┘ └────┬───────┘                        │
│              │             │                                │
│       ┌──────▼─────────────▼──────┐                         │
│       │   KBStore (Chroma 封装)   │                         │
│       └──────────────┬───────────┘                          │
│                      │                                      │
│       ┌──────────────▼───────────┐                         │
│       │  EmbeddingProvider 接口  │                         │
│       │  - RemoteLLMEmbedder     │  ← 默认实现              │
│       │  - (预留: LocalEmbedder) │  ← 后续扩展              │
│       └──────────────────────────┘                         │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
              ~/.milu/kb/chroma/
```

**6 个内部组件 + 3 个 LLM 可见工具，职责单一**：

| 组件 | 职责 | 文件 |
|------|------|------|
| `EmbeddingProvider` (Protocol) | 文本列表 → 向量列表 | `tools/knowledge_base/embedding.py` |
| `RemoteLLMEmbedder` | 默认实现，调用 provider 的 embedding 接口 | 同上 |
| `KBStore` | 封装 Chroma 的 CRUD、过滤、维度校验 | `tools/knowledge_base/store.py` |
| `Chunker` | 纯函数：固定字符切块，UTF-8 安全 | `tools/knowledge_base/chunker.py` |
| `KBIngestor` | 编排：读文件 → 切块 → 嵌入 → 写入 | `tools/knowledge_base/ingestor.py` |
| `KBRetriever` | 编排：嵌入 query → 检索 → 排序 | `tools/knowledge_base/retriever.py` |
| 3 个 `@tool` 工具 | LLM 可见的接口 | `tools/knowledge_base/tools.py` |

**目录结构**：
```
src/milu/tools/knowledge_base/
├── __init__.py
├── embedding.py        # EmbeddingProvider + RemoteLLMEmbedder
├── chunker.py          # 切块
├── store.py            # KBStore
├── ingestor.py         # 入库编排
├── retriever.py        # 检索编排
├── tools.py            # 3 个 @tool
└── models.py           # Chunk, SearchHit
```

## 四、数据模型

```python
@dataclass
class Chunk:
    """入库的最小单位"""
    text: str
    source: str                    # 来源标识：绝对路径 或 "manual:<tag>"
    chunk_index: int               # 该 source 内的块序号
    start_char: int                # 原文起始字符位置
    end_char: int                  # 原文结束字符位置
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchHit:
    """检索命中"""
    text: str
    source: str
    chunk_index: int
    score: float                   # 相似度 0~1（cosine）
    metadata: dict
```

## 五、EmbeddingProvider 接口

```python
class EmbeddingProvider(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """批量把文本变成向量。失败抛 EmbeddingError。"""

    @property
    def dim(self) -> int: ...          # 向量维度

    @property
    def model_name(self) -> str: ...   # 用于 stats 显示
```

**`RemoteLLMEmbedder` 配置表**（写死在 `embedding.py` 顶部）：

| Provider | Endpoint | Model | Dim |
|----------|----------|-------|-----|
| openai | `/v1/embeddings` | `text-embedding-3-small` | 1536 |
| qwen | `/compatible-mode/v1/embeddings` | `text-embedding-v3` | 1024 |
| doubao | `/api/v3/embeddings` | `doubao-embedding` | 1024 |
| glm | `/api/paas/v4/embeddings` | `embedding-2` | 1024 |
| kimi | `/v1/embeddings` | `embedding-v1` | 1536 |
| gemini | 自定义（非 openai-compat） | `text-embedding-004` | 768 |

不支持 embedding 的 provider（claude / deepseek / minimax）：`RemoteLLMEmbedder(provider_name="claude")` 构造时直接抛 `EmbeddingError`，错误信息列出"支持的 provider 列表"。

**实现要点**：
- 内部用 `httpx.AsyncClient` 直连（不强制复用 openai SDK，因为 gemini 等不在 openai-compat 路径下）
- 批量上限 64 段/次，超出切片循环
- 启动时 KBStore 校验：若 collection 非空且存储 dim 与当前 provider dim 不一致 → 抛 `DimensionMismatchError` 并提示"重新建库或换回原 provider"

## 六、工具表面（3 个工具，参考 file_read/file_write 合并风格）

### `kb_ingest`（写入，`is_safe=False`）

| action | 必填 | 可选 | 行为 |
|--------|------|------|------|
| `text` | `text`, `tag` | `metadata` | 手动加一段文本，`source = "manual:<tag>"` |
| `file` | `path` | `metadata` | 读单文件 → 切块 → 入库 |
| `directory` | `path` | `glob`（默认 `**/*.{md,txt,py,json,csv}`）, `recursive`（默认 `True`） | 批量扫目录 |

**`metadata` 参数语义**（`file` / `directory` action）：每个 source 共享一个 dict，作为公共 metadata 写入该 source 的所有块。例如 `{"project": "milu", "tag": "design"}` 会出现在所有相关 chunk 的 metadata 中。

### `kb_search`（只读，`is_safe=True`）

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `query` | str | 必填 | 查询文本 |
| `top_k` | int | 5 | 返回条数（上限 50） |
| `source_filter` | str \| None | None | 限定来源前缀，如 `docs/` |
| `min_score` | float | 0.0 | 过滤低相似度 |

返回：`{"hits": [{text, source, chunk_index, score, metadata}], "total": int}`

### `kb_manage`（管理，按 action 区分安全级别）

| action | 必填 | 安全级别 | 行为 |
|--------|------|----------|------|
| `list_sources` | — | `is_safe=True` | 列出所有 source（按 source 去重，含块数、首块时间） |
| `stats` | — | `is_safe=True` | 总块数、source 数、embedding 模型、维度、持久化路径 |
| `delete_source` | `source` | `is_safe=False` | 删除某 source 的所有块 |

## 七、关键流程

### 入库（kb_ingest: file）

```
1. 嗅探文件编码（utf-8 → gbk → latin-1 fallback）
2. Chunker.split(text, chunk_size=500, overlap=50)
3. 为每块生成 Chunk（含 start_char / end_char）
4. RemoteLLMEmbedder.embed([chunk.text, ...])  →  按 64 段/批
5. KBStore.add(chunks, vectors)
6. 返回 {"success": True, "added_chunks": N, "total_chars": M, "source": "..."}
```

### 检索（kb_search）

```
1. RemoteLLMEmbedder.embed([query])
2. KBStore.query(q_vec, top_k, source_filter)
   → chroma 返回 cosine distance
3. similarity = 1 - distance
4. 过滤 min_score
5. 构造 SearchHit 列表返回
```

## 八、持久化

- 默认目录：`~/.milu/kb/chroma/`
- `AgentConfig` 新增字段 `kb_persist_dir: str | None = None`（None 即用默认）
- collection 名字固定 `kb_default`（YAGNI：MVP 不做多 collection）
- Chroma 内部用 SQLite + DuckDB + Parquet，单目录可整体备份

## 九、错误处理

### 异常体系（在 `milu/exceptions.py` 新增）

```python
class KnowledgeBaseError(Exception): pass
class EmbeddingError(KnowledgeBaseError): pass
class ChunkingError(KnowledgeBaseError): pass
class IngestionError(KnowledgeBaseError): pass
class RetrievalError(KnowledgeBaseError): pass
class DimensionMismatchError(KnowledgeBaseError): pass
```

工具层统一捕获 → 返回 `{"success": False, "error": "...", "hint": "..."}`，不向 LLM 抛异常。

### 关键场景

| 场景 | 行为 |
|------|------|
| 文件不存在 / 不可读 | 返回 `{"success": False, "error": "文件不存在: ..."}` |
| 文件编码错误 | 嗅探 utf-8 → gbk → latin-1，失败才报错 |
| embedding 401 | "API Key 无效，请检查 .env 中 XXX_API_KEY" |
| embedding 429 | "触发限流，已自动重试 3 次仍失败"（含 retry with backoff） |
| 维度不匹配 | 启动时拒绝；运行中暂不发生 |
| 空库检索 | `{"hits": [], "total": 0, "hint": "知识库为空，先用 kb_ingest 入库"}` |
| directory 无匹配 | `{"success": True, "added_chunks": 0, "scanned": 0, "hint": "..."}` |
| 重复入库同一 source | **按 chunk_index 覆盖**（Chroma ID = `f"{source}::{chunk_index}"`，相同 ID 自动 upsert）；若新块数 < 老块数，残留的老块由 `KBStore.upsert()` 在写入末尾删除 |

## 十、依赖

`pyproject.toml` 新增：
```toml
dependencies = [
    ...
    "chromadb>=0.5.0",
]
```

不新增 optional 依赖组（chromadb 体积可控，无原生编译依赖）。

## 十一、测试策略

### 单元测试（`tests/test_kb_*.py`，mock 全部，不联网）

- `test_chunker.py` — 切块边界、UTF-8 安全（中文/emoji）、overlap、短文本（< chunk_size 不切）、oversize
- `test_embedding_provider.py` — fake provider 测批量、超 64 段切片、错误传递、不支持 provider 报错
- `test_kb_store.py` — 用 `chromadb.EphemeralClient`（纯内存变体）测 add/query/delete/filter/维度校验
- `test_kb_ingestor.py` — mock file reader + fake embedder，验证切→嵌→写全流程
- `test_kb_retriever.py` — fake embedder + 预填 store，验证 score 转换、过滤
- `test_kb_tools.py` — 3 个工具 × 全部 action 的入参校验、JSON 返回结构

### 集成（不入 CI，需 `.env` 中有 `OPENAI_API_KEY`）

- `tests/test_kb_real_e2e.py` — 真用 openai embedding + 真 chroma，验证 1 篇文档入库+检索

### 手动验证

- `examples/7. knowledge_base.py` — 演示：把本项目 `CLAUDE.md` 入库，问"项目支持哪些 LLM"，验证检索质量

## 十二、YAGNI 清单（明确不做）

- 多 collection（统一用 `kb_default`）
- 混合检索（向量 + BM25）—— 纯向量检索
- 文档解析器（PDF / DOCX）—— MVP 只支持纯文本
- 增量更新 / watch 目录变化
- 删除单个 chunk（只能按 source 整源删）
- 多租户 / 权限
- 远程 / 分布式部署
- embedding 模型热切换（启动时检测到维度不一致就拒绝）

## 十三、文件改动清单

```
新增:
  src/milu/tools/knowledge_base/__init__.py
  src/milu/tools/knowledge_base/embedding.py
  src/milu/tools/knowledge_base/chunker.py
  src/milu/tools/knowledge_base/store.py
  src/milu/tools/knowledge_base/ingestor.py
  src/milu/tools/knowledge_base/retriever.py
  src/milu/tools/knowledge_base/tools.py
  src/milu/tools/knowledge_base/models.py
  tests/test_kb_chunker.py
  tests/test_kb_embedding.py
  tests/test_kb_store.py
  tests/test_kb_ingestor.py
  tests/test_kb_retriever.py
  tests/test_kb_tools.py
  tests/test_kb_real_e2e.py
  examples/7. knowledge_base.py

修改:
  pyproject.toml                              # +chromadb
  src/milu/exceptions.py           # +KnowledgeBaseError 等
  src/milu/agent/config.py         # +kb_persist_dir 字段
  src/milu/tools/builtin/__init__.py  # 导出 kb 工具（可选）
```
