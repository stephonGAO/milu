"""轻量向量知识库 —— 对私有语料的语义检索（RAG）。

与 memory 长期记忆互补：memory 是少量条目全量渲染进 system prompt；
knowledge 是大语料分块向量化、按需语义召回（适合 FAQ / 产品手册 / 笔记 /
论文等非结构化文档；代码与结构化文档建议直接用 file_read+grep 式检索）。

启用方式（默认关闭）：`Agent(knowledge=True)` / `Agent(knowledge="user-1")` /
`Agent(knowledge=KnowledgeConfig(...))`。检索依赖可选安装：pip install "milu[kb]"。
"""
from milu.knowledge.chunker import chunk_text
from milu.knowledge.config import KnowledgeConfig
from milu.knowledge.embedder import Embedder
from milu.knowledge.store import KnowledgeStore, knowledge_dir, store_lock

__all__ = [
    "KnowledgeConfig",
    "Embedder",
    "KnowledgeStore",
    "chunk_text",
    "knowledge_dir",
    "store_lock",
]
