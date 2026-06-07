"""知识库存储 —— chunks.jsonl + vectors.npy + meta.json，numpy 暴力余弦检索。

轻量定位：万级 chunks（1 万块 × 1024 维 float32 ≈ 40MB 内存，暴力余弦 < 5ms；
查询总延迟由 embedding API 网络往返主导，ANN 索引在此量级无可感知收益）。
若未来需要十万级以上 / 元数据过滤 / 多进程并发写，在本类接口（add/search/
delete_source/stats）后面换 sqlite-vec 或 usearch 等后端，工具层不动。

文件布局 `{user_data_dir()}/knowledge/{safe_user_id}/`：
    chunks.jsonl — 每行一个块 {"text", "source", "created_at"}
    vectors.npy  — float32 矩阵，行序与 chunks.jsonl 严格对齐
    meta.json    — {"provider", "model", "dim", "count"}（embedding 模型指纹，
                   防换模型后新旧维度/语义空间混存；count 兼做一致性校验锚点）

写策略：**全量重写 + 原子替换**（mkstemp + fsync + os.replace，仿 scheduler/
store.py）。万级规模全量重写成本可忽略（< 1s），换来的是永不出现半截文件。
写入顺序 chunks → vectors → meta，meta 最后落盘作为"提交点"。

并发取舍（与 scheduler 同档）：进程内并发用模块级 per-目录 asyncio.Lock
串行化（store_lock()，工具层 ingest/delete 前获取）；跨进程同毫秒写同一
知识库的丢更新窗口极小，接受之，不引入文件锁新依赖。
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import asyncio

logger = logging.getLogger(__name__)

# 进程内 per-知识库目录写锁（key 为目录的字符串路径）
_store_locks: dict[str, asyncio.Lock] = {}


def store_lock(dir_path: Path) -> asyncio.Lock:
    """返回该知识库目录的进程内写锁（同目录共享同一把锁）。"""
    return _store_locks.setdefault(str(dir_path), asyncio.Lock())


def knowledge_dir(user_id: str) -> Path:
    """返回某用户标识对应的知识库目录。

    `{user_data_dir()}/knowledge/{safe_user_id}/`，user_id 文件系统安全化
    （与 memory_file_path 同一规则）。
    """
    from milu.resources import user_data_dir

    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", (user_id or "default").strip() or "default")[:64]
    return user_data_dir() / "knowledge" / safe


def _require_numpy():
    """导入 numpy（可选依赖 [kb]），未安装时给中文安装指引。"""
    try:
        import numpy
    except ImportError as e:
        raise RuntimeError(
            '向量知识库需要 numpy，请安装：pip install "milu[kb]"（或 pip install numpy）'
        ) from e
    return numpy


def _atomic_write_bytes(path: Path, write_fn) -> None:
    """原子写文件：同目录临时文件 + fsync + os.replace（仿 scheduler/store.py）。

    :param write_fn: 接收已打开的二进制文件对象、负责写入内容的回调。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as tmp:
            write_fn(tmp)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


class KnowledgeStore:
    """单个知识库的读写与检索（磁盘即真相 + mtime 失效的内存缓存）。

    load() 带实例级缓存：三个数据文件的 (mtime_ns, size) 签名未变则直接复用
    内存对象——万级 chunks 时每次检索从"重读 ~40MB 磁盘"降回毫秒级。
    跨进程写入会改变文件签名，缓存自动失效，"磁盘即真相"语义不变。
    注意：load() 返回的是共享缓存对象，调用方**不得原地修改**（本类内部
    一律以"新列表/新矩阵"方式写回）。
    """

    def __init__(self, dir_path: Path):
        self._dir = Path(dir_path)
        # (文件签名, chunks, vectors)；签名不匹配即失效
        self._cache: tuple[tuple, list[dict[str, Any]], Any] | None = None

    def _files_signature(self) -> tuple:
        """三个数据文件的 (mtime_ns, size) 签名（文件缺失记 None）。"""
        sig = []
        for name in ("chunks.jsonl", "vectors.npy", "meta.json"):
            try:
                st = (self._dir / name).stat()
                sig.append((st.st_mtime_ns, st.st_size))
            except OSError:
                sig.append(None)
        return tuple(sig)

    @property
    def dir_path(self) -> Path:
        return self._dir

    # ── 读 ──────────────────────────────────────────────

    def meta(self) -> dict[str, Any]:
        """读取 meta.json（不存在/损坏返回空 dict）。"""
        path = self._dir / "meta.json"
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            logger.warning("读取知识库 meta 失败 %s: %s", path, e)
            return {}
        return data if isinstance(data, dict) else {}

    def count(self) -> int:
        return int(self.meta().get("count", 0))

    def is_empty(self) -> bool:
        return self.count() == 0

    def _load_chunks(self) -> list[dict[str, Any]]:
        path = self._dir / "chunks.jsonl"
        if not path.exists():
            return []
        chunks = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
        return chunks

    def load(self) -> tuple[list[dict[str, Any]], Any]:
        """加载全部块与向量矩阵，并做一致性校验（带 mtime 缓存）。

        :return: (chunks, vectors)；空库返回 ([], None)。返回共享缓存对象，勿原地修改。
        :raises RuntimeError: 三个文件数量不一致（写入中断/手工破坏），提示重建。
        """
        np = _require_numpy()
        sig = self._files_signature()
        if self._cache is not None and self._cache[0] == sig:
            return self._cache[1], self._cache[2]

        chunks = self._load_chunks()
        vec_path = self._dir / "vectors.npy"
        if not chunks and not vec_path.exists():
            self._cache = (sig, [], None)
            return [], None
        vectors = np.load(vec_path) if vec_path.exists() else None
        n_vec = 0 if vectors is None else int(vectors.shape[0])
        if len(chunks) != n_vec or self.count() != len(chunks):
            raise RuntimeError(
                f"知识库文件不一致（chunks={len(chunks)}, vectors={n_vec}, "
                f"meta.count={self.count()}），可能写入中断。"
                f"请删除目录后重新入库：{self._dir}"
            )
        self._cache = (sig, chunks, vectors)
        return chunks, vectors

    # ── 写 ──────────────────────────────────────────────

    def add(
        self,
        texts: list[str],
        vectors: list[list[float]],
        source: str,
        provider: str,
        model: str,
    ) -> int:
        """追加一批块（全量重写落盘）。

        :raises RuntimeError: 与已有库的 embedding 模型指纹不一致。
        :return: 入库后的总块数。
        """
        np = _require_numpy()
        if len(texts) != len(vectors):
            raise ValueError(f"texts({len(texts)}) 与 vectors({len(vectors)}) 数量不一致")
        if not texts:
            return self.count()

        self.check_model(provider, model)

        old_chunks, old_vectors = self.load()
        created = datetime.now().isoformat(timespec="seconds")
        # 拼接为新列表（load() 返回共享缓存对象，不可原地 extend）
        chunks = old_chunks + [
            {"text": t, "source": source, "created_at": created} for t in texts
        ]
        new_vectors = np.asarray(vectors, dtype=np.float32)
        if old_vectors is not None:
            if old_vectors.shape[1] != new_vectors.shape[1]:
                raise RuntimeError(
                    f"向量维度不一致（已有 {old_vectors.shape[1]}，新增 {new_vectors.shape[1]}），"
                    "embedding 模型可能已变更，请重建知识库"
                )
            new_vectors = np.vstack([old_vectors, new_vectors])

        self._write_all(chunks, new_vectors, provider, model)
        return len(chunks)

    def delete_source(self, source: str) -> int:
        """删除某来源的全部块。:return: 删除的块数（0 表示来源不存在）。"""
        chunks, vectors = self.load()
        keep = [i for i, c in enumerate(chunks) if c.get("source") != source]
        removed = len(chunks) - len(keep)
        if removed == 0:
            return 0
        meta = self.meta()
        kept_chunks = [chunks[i] for i in keep]
        kept_vectors = vectors[keep] if keep else None
        if kept_chunks:
            self._write_all(kept_chunks, kept_vectors,
                            meta.get("provider", ""), meta.get("model", ""))
        else:
            self.clear()
        return removed

    def clear(self) -> None:
        """清空整个知识库（删除三个数据文件，目录保留）。"""
        for name in ("chunks.jsonl", "vectors.npy", "meta.json"):
            try:
                (self._dir / name).unlink(missing_ok=True)
            except OSError as e:
                logger.warning("删除知识库文件失败 %s: %s", name, e)

    def _write_all(self, chunks: list[dict], vectors, provider: str, model: str) -> None:
        """全量落盘：chunks → vectors → meta（meta 最后写，作为提交点）。"""
        np = _require_numpy()
        lines = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks)
        _atomic_write_bytes(self._dir / "chunks.jsonl",
                            lambda f: f.write(lines.encode("utf-8")))
        _atomic_write_bytes(self._dir / "vectors.npy",
                            lambda f: np.save(f, vectors))
        meta = {
            "provider": provider,
            "model": model,
            "dim": int(vectors.shape[1]),
            "count": len(chunks),
        }
        _atomic_write_bytes(self._dir / "meta.json",
                            lambda f: f.write(json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8")))

    # ── 校验 / 检索 / 统计 ────────────────────────────────

    def check_model(self, provider: str, model: str) -> None:
        """校验当前 embedding 模型与库内指纹一致（空库跳过）。

        :raises RuntimeError: 不一致时给出处置指引（防新旧向量空间混存）。
        """
        meta = self.meta()
        if not meta or not meta.get("count"):
            return
        if (meta.get("provider"), meta.get("model")) != (provider, model):
            raise RuntimeError(
                f"知识库由 {meta.get('provider')}/{meta.get('model')} 构建，"
                f"当前配置为 {provider}/{model}，向量空间不兼容。"
                "请改回原 embedding 配置，或用 kb_manage(action='clear') 清空后重新入库"
            )

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[float, dict]]:
        """暴力余弦检索。:return: [(相似度, 块)]，按相似度降序，最多 top_k 个。"""
        np = _require_numpy()
        chunks, vectors = self.load()
        if not chunks:
            return []
        q = np.asarray(query_vector, dtype=np.float32)
        if q.shape[0] != vectors.shape[1]:
            raise RuntimeError(
                f"查询向量维度（{q.shape[0]}）与库内维度（{vectors.shape[1]}）不一致，"
                "embedding 模型可能已变更，请重建知识库"
            )
        norms = np.linalg.norm(vectors, axis=1) * (np.linalg.norm(q) or 1.0)
        norms = np.where(norms == 0, 1.0, norms)
        scores = (vectors @ q) / norms
        top_k = max(1, min(top_k, len(chunks)))
        idx = np.argsort(scores)[::-1][:top_k]
        return [(float(scores[i]), chunks[i]) for i in idx]

    def list_sources(self) -> list[tuple[str, int]]:
        """按来源汇总：[(source, 块数)]，按来源名排序。"""
        counts: dict[str, int] = {}
        for c in self._load_chunks():
            src = c.get("source", "")
            counts[src] = counts.get(src, 0) + 1
        return sorted(counts.items())

    def stats(self) -> dict[str, Any]:
        """知识库统计（meta + 来源数 + 磁盘占用）。"""
        meta = self.meta()
        disk = sum(
            (self._dir / name).stat().st_size
            for name in ("chunks.jsonl", "vectors.npy", "meta.json")
            if (self._dir / name).exists()
        )
        return {
            "count": int(meta.get("count", 0)),
            "sources": len(self.list_sources()),
            "provider": meta.get("provider", ""),
            "model": meta.get("model", ""),
            "dim": int(meta.get("dim", 0)),
            "disk_bytes": disk,
            "dir": str(self._dir),
        }
