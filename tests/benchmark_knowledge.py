"""向量知识库全面评测 —— 真实 embedding API，覆盖质量/阈值/成本/性能/端到端五个维度，自动生成 Markdown 报告。

与 tests/test_knowledge.py（FakeEmbedder 单元测试）互补：本脚本衡量"真实模型下的整体效果"。

评测维度：
  A. 检索质量    Hit@1/3/5、MRR@5、答案关键词召回@3、对照 bigram 关键词基线
  B. 双阈值评估  工具路径（min_score=0.35）与自动注入路径（auto_min_score=0.5/auto_top_k=3）
                 的命中保留率、负例拦截率、回捞带分析——验证"自动收精度、工具保召回"设计
  C. 上下文成本  自动注入 vs kb_search top5 vs 钳制上限 8 vs 未钳制 top20 的字符开销
  D. 性能        入库 embedding 并发加速比（1 路 vs 4 路 A/B）、查询延迟、
                 本地检索冷启动（含读盘）vs 热缓存（mtime 缓存命中）
  E. 端到端      kb_ingest/kb_search/kb_manage 真实工具链路 + prompt 渲染 + 自动检索命中/未命中

用法（需 .env 配置对应厂商 API Key，如 QWEN_API_KEY）：
    .venv/Scripts/python tests/benchmark_knowledge.py                  # 默认 qwen，报告写 docs/
    .venv/Scripts/python tests/benchmark_knowledge.py -p glm
    .venv/Scripts/python tests/benchmark_knowledge.py --synthetic 240  # 加大性能测试语料
    .venv/Scripts/python tests/benchmark_knowledge.py --report out.md --keep
"""
from __future__ import annotations

import argparse
import asyncio
import shutil
import statistics
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from milu.knowledge import Embedder, KnowledgeConfig, KnowledgeStore
from milu.knowledge.embedder import _EMBEDDING_PROVIDERS
from milu.llm.base.exceptions import AuthenticationError
from milu.tools.builtin.knowledge_tool import (
    KnowledgeRuntime,
    _current_knowledge,
    kb_ingest,
    kb_manage,
    kb_search,
    render_knowledge_prompt,
)

# ==================== 评测语料（5 篇，内容互不重叠）====================

CORPUS: dict[str, str] = {
    "产品手册-星航S3": (
        "星航 S3 智能手表产品说明\n\n"
        "电池与续航：标准模式下单次充电可连续使用 14 天，开启常亮屏后约 5 天。"
        "使用磁吸充电底座，从零充满约 90 分钟。\n\n"
        "防水与材质：支持 5ATM 防水等级，可佩戴游泳，但不适用于热水淋浴和潜水。"
        "表体采用钛合金中框与蓝宝石玻璃镜面。\n\n"
        "健康监测：内置第四代光学心率传感器，支持全天候血氧监测、睡眠分期分析与心率异常提醒。\n\n"
        "定价与版本：标准版定价 1499 元，尊享版（含钛金属表带）1999 元。"
    ),
    "退换货政策": (
        "退换货政策\n\n"
        "无理由退货：自签收之日起 7 天内支持无理由退货，商品需保持原包装完好、配件齐全。\n\n"
        "运费承担：质量问题退货由商家承担往返运费；无理由退货的退回运费由买家承担。\n\n"
        "例外情形：定制刻字商品、已激活的电子兑换码不支持无理由退货。\n\n"
        "退款时效：仓库验收通过后，退款将在 3 个工作日内原路退回。"
    ),
    "订单系统架构": (
        "订单系统技术架构\n\n"
        "服务拆分：系统按领域拆分为订单、库存、支付、通知四个微服务，服务间通过 gRPC 同步调用。\n\n"
        "异步解耦：订单状态变更事件经 Kafka 消息队列广播，库存与通知服务各自消费，峰值可削峰填谷。\n\n"
        "数据存储：核心交易数据存于 PostgreSQL 主从集群，热点查询走 Redis 缓存，缓存过期时间 300 秒。\n\n"
        "发布策略：采用灰度发布，新版本先放量 5% 流量观察 30 分钟，关键指标正常后全量。"
    ),
    "员工手册": (
        "员工手册（假期与报销）\n\n"
        "年假制度：入职满一年享 10 天带薪年假，每满两年增加 1 天，上限 15 天。\n\n"
        "远程办公：每周可申请 2 天居家办公，需提前一天在 OA 系统报备直属主管。\n\n"
        "差旅报销：市内交通凭发票实报实销；住宿标准一线城市每晚不超过 600 元，"
        "发票需在行程结束后 30 天内提交。\n\n"
        "入职流程：新员工报到当天领取工卡与电脑，第一周完成信息安全培训并签署保密协议。"
    ),
    "API接入指南": (
        "开放平台 API 接入指南\n\n"
        "鉴权方式：调用方先用 AppKey 与 AppSecret 换取 access_token，token 有效期 2 小时，"
        "过期需重新获取。\n\n"
        "调用限制：默认限流为每个应用 100 QPS，超出返回 429 状态码；可在控制台申请提升配额。\n\n"
        "错误处理：业务错误统一返回 JSON 结构 {code, message, request_id}，"
        "code 为 40013 表示签名无效。\n\n"
        "事件推送：支持 Webhook 回调，平台对失败的推送按 1/5/30 分钟间隔重试三次。"
    ),
}

# 正例查询：(改写式问题, 期望来源, 答案关键词列表——任一出现在检索文本中即算召回)
POSITIVE_QUERIES: list[tuple[str, str, list[str]]] = [
    ("这块手表充满电要多久？", "产品手册-星航S3", ["90 分钟"]),
    ("戴着它洗热水澡可以吗？", "产品手册-星航S3", ["热水淋浴"]),
    ("尊享版比标准版贵多少钱？", "产品手册-星航S3", ["1999", "1499"]),
    ("手环能测血氧吗？", "产品手册-星航S3", ["血氧"]),
    ("手表镜面用的是什么材质？", "产品手册-星航S3", ["蓝宝石"]),
    ("收到货后几天内可以不要理由地退掉？", "退换货政策", ["7 天"]),
    ("刻了字的商品还能退吗？", "退换货政策", ["定制刻字"]),
    ("退款多久能到账？", "退换货政策", ["3 个工作日"]),
    ("什么情况下退货运费由商家出？", "退换货政策", ["质量问题"]),
    ("订单服务之间用什么协议通信？", "订单系统架构", ["gRPC"]),
    ("系统用什么消息中间件做异步解耦？", "订单系统架构", ["Kafka"]),
    ("缓存多长时间过期？", "订单系统架构", ["300 秒"]),
    ("新版本上线是怎么放量的？", "订单系统架构", ["5%", "灰度"]),
    ("在家办公需要走什么流程？", "员工手册", ["OA", "报备"]),
    ("出差住酒店每晚的报销标准是多少？", "员工手册", ["600 元"]),
    ("新员工第一周要完成什么培训？", "员工手册", ["信息安全"]),
    ("access_token 多长时间失效？", "API接入指南", ["2 小时"]),
    ("接口被限流了会返回什么状态码？", "API接入指南", ["429"]),
    ("错误码 40013 代表什么意思？", "API接入指南", ["签名无效"]),
    ("回调推送失败平台会重试几次？", "API接入指南", ["三次"]),
]

# 负例查询：语料中完全不存在的话题（理想情况下两级阈值都应拦住）
NEGATIVE_QUERIES: list[str] = [
    "明朝灭亡的主要原因是什么？",
    "如何在家制作提拉米苏？",
    "梅西拿过几次世界杯冠军？",
    "光合作用的化学方程式是什么？",
    "东京到大阪坐新干线要多久？",
]

TOP_K = 5

# 性能段合成语料的主题轮换表
_TOPICS = ["考勤", "采购", "数据安全", "出差", "供应商管理", "知识产权", "设备维护", "应急预案"]


def synth_chunks(n: int) -> list[str]:
    """生成 n 条确定性的合成制度条款（性能/成本测试用，内容可区分但同质）。"""
    return [
        f"制度条款第{i}号：关于{_TOPICS[i % len(_TOPICS)]}的管理规定。"
        f"本条款适用于全体员工与相关外包人员，执行编号 KB-{1000 + i}，"
        f"自发布之日起生效，由第{i % 7 + 1}责任部门负责解释、培训与监督执行，"
        f"违反本条款的处理流程参见员工纪律细则第{i % 12 + 1}章。"
        for i in range(n)
    ]


# ==================== 关键词基线（字符 bigram Dice）====================


def _bigrams(s: str) -> set[str]:
    s = "".join(s.split())
    return {s[i:i + 2] for i in range(len(s) - 1)}


def _dice(a: str, b: str) -> float:
    ba, bb = _bigrams(a), _bigrams(b)
    if not ba or not bb:
        return 0.0
    return 2 * len(ba & bb) / (len(ba) + len(bb))


def baseline_rank(query: str, chunks: list[dict], expected_source: str) -> int | None:
    scored = sorted(((_dice(query, c["text"]), c) for c in chunks),
                    key=lambda x: x[0], reverse=True)
    for rank, (_, chunk) in enumerate(scored[:TOP_K], 1):
        if chunk["source"] == expected_source:
            return rank
    return None


# ==================== 报告 ====================


class Report:
    """边打印边累积 Markdown 行，最后落盘。"""

    def __init__(self):
        self.lines: list[str] = []
        self.verdicts: list[tuple[str, str, bool]] = []  # (指标, 实测, 达标)

    def add(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def verdict(self, name: str, actual: str, ok: bool) -> None:
        self.verdicts.append((name, actual, ok))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def _pct(x: float) -> str:
    return f"{x:.1%}"


# ==================== 评测主流程 ====================


async def run_benchmark(args) -> int:
    kn = KnowledgeConfig()  # 读取代码内默认（auto_top_k / auto_min_score / min_score）
    r = Report()
    embedder = Embedder(provider=args.provider, model=args.model,
                        batch_size=args.batch_size)
    workdir = Path(tempfile.mkdtemp(prefix="milu_kb_bench_"))
    store = KnowledgeStore(workdir / "corpus")

    r.add("# 向量知识库全面评测报告")
    r.add()
    r.add(f"- 评测时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    r.add(f"- embedding：{embedder.provider}/{embedder.model}")
    r.add(f"- 阈值配置：kb_search min_score={kn.min_score} / "
          f"自动注入 auto_min_score={kn.auto_min_score}, auto_top_k={kn.auto_top_k} / "
          f"kb_search top_k 上限 8")
    r.add(f"- 语料：{len(CORPUS)} 篇真实风格文档 + {args.synthetic} 条合成条款（性能段）；"
          f"正例查询 {len(POSITIVE_QUERIES)} 条（全部换说法改写）、负例 {len(NEGATIVE_QUERIES)} 条")

    try:
        # ── A. 入库 ───────────────────────────────────────
        total_chars = sum(len(t) for t in CORPUS.values())
        n_chunks = 0
        t0 = time.perf_counter()
        for source, text in CORPUS.items():
            from milu.knowledge import chunk_text
            chunks = chunk_text(text, args.chunk_size, args.overlap)
            vectors = await embedder.embed(chunks)
            store.add(chunks, vectors, source=source,
                      provider=embedder.provider, model=embedder.model)
            n_chunks += len(chunks)
        ingest_secs = time.perf_counter() - t0
        dim = store.stats()["dim"]
        all_chunks, _ = store.load()
        r.add()
        r.add("## A. 语料入库")
        r.add()
        r.add(f"{len(CORPUS)} 篇 / {total_chars} 字符 → {n_chunks} 块"
              f"（chunk_size={args.chunk_size}），耗时 {ingest_secs:.2f}s，"
              f"向量 {dim} 维，磁盘 {store.stats()['disk_bytes'] / 1024:.1f} KB")

        # ── 查询向量（逐条计时，质量/阈值两段复用）────────
        embed_ms: list[float] = []

        async def embed_query(q: str) -> list[float]:
            t = time.perf_counter()
            v = (await embedder.embed([q]))[0]
            embed_ms.append((time.perf_counter() - t) * 1000)
            return v

        pos_results: list[list[tuple[float, dict]]] = []
        search_ms: list[float] = []
        for query, _, _ in POSITIVE_QUERIES:
            qv = await embed_query(query)
            t = time.perf_counter()
            pos_results.append(store.search(qv, TOP_K))
            search_ms.append((time.perf_counter() - t) * 1000)
        neg_top1: list[float] = []
        for query in NEGATIVE_QUERIES:
            qv = await embed_query(query)
            neg_top1.append(store.search(qv, 1)[0][0])

        # ── B. 检索质量 ───────────────────────────────────
        r.add()
        r.add("## B. 检索质量（向量 vs 关键词基线）")
        r.add()
        r.add("| # | 查询 | 向量rank | top1分 | 基线rank | 关键词@3 |")
        r.add("|---|------|----------|--------|----------|-----------|")
        ranks, kw_hits, base_ranks = [], [], []
        for i, ((query, expected, keywords), results) in enumerate(
                zip(POSITIVE_QUERIES, pos_results), 1):
            rank = next((k for k, (_, c) in enumerate(results, 1)
                         if c["source"] == expected), None)
            kw = any(k in c["text"] for _, c in results[:3] for k in keywords)
            b = baseline_rank(query, all_chunks, expected)
            ranks.append(rank)
            kw_hits.append(kw)
            base_ranks.append(b)
            r.add(f"| {i} | {query} | {rank or '>5'} | {results[0][0]:.3f} "
                  f"| {b or '>5'} | {'✓' if kw else '✗'} |")

        n = len(POSITIVE_QUERIES)
        hit1 = sum(1 for x in ranks if x == 1) / n
        hit3 = sum(1 for x in ranks if x and x <= 3) / n
        hit5 = sum(1 for x in ranks if x and x <= 5) / n
        mrr = sum(1 / x for x in ranks if x) / n
        kw_recall = sum(kw_hits) / n
        b_hit1 = sum(1 for x in base_ranks if x == 1) / n
        b_hit3 = sum(1 for x in base_ranks if x and x <= 3) / n
        r.add()
        r.add(f"**Hit@1 {_pct(hit1)}（基线 {_pct(b_hit1)}）｜ Hit@3 {_pct(hit3)}"
              f"（基线 {_pct(b_hit3)}）｜ Hit@5 {_pct(hit5)} ｜ MRR@5 {mrr:.3f} ｜ "
              f"关键词召回@3 {_pct(kw_recall)}**")
        r.verdict("Hit@3 ≥ 90%", _pct(hit3), hit3 >= 0.90)
        r.verdict("MRR@5 ≥ 0.85", f"{mrr:.3f}", mrr >= 0.85)
        r.verdict("关键词召回@3 ≥ 85%", _pct(kw_recall), kw_recall >= 0.85)

        # ── C. 双阈值评估 ─────────────────────────────────
        r.add()
        r.add("## C. 双阈值评估（自动注入收精度 / 工具检索保召回）")
        r.add()
        auto_injected = 0          # 自动路径注入了至少 1 个片段的正例数
        auto_kw = 0                # 注入片段已含答案关键词（零工具调用即得答案）
        inj_total, inj_correct = 0, 0   # 注入片段总数 / 其中来自期望来源的
        fallback = 0               # 自动未注入但 kb_search（0.35）可捞回的正例数
        lost = 0                   # 两级都拿不到的正例数
        for (query, expected, keywords), results in zip(POSITIVE_QUERIES, pos_results):
            injected = [(s, c) for s, c in results[:kn.auto_top_k]
                        if s >= kn.auto_min_score]
            recoverable = any(c["source"] == expected and s >= kn.min_score
                              for s, c in results)
            if injected:
                auto_injected += 1
                inj_total += len(injected)
                inj_correct += sum(1 for _, c in injected if c["source"] == expected)
                if any(k in c["text"] for _, c in injected for k in keywords):
                    auto_kw += 1
            elif recoverable:
                fallback += 1
            else:
                lost += 1
        neg_tool_leak = sum(1 for s in neg_top1 if s >= kn.min_score)
        neg_auto_leak = sum(1 for s in neg_top1 if s >= kn.auto_min_score)
        inj_precision = inj_correct / inj_total if inj_total else 0.0

        r.add(f"- 自动注入率（top1 过 {kn.auto_min_score}）：**{auto_injected}/{n}"
              f"（{_pct(auto_injected / n)}）**，其中注入片段已含答案关键词 "
              f"{auto_kw}/{auto_injected} 条（零工具调用即可作答）")
        r.add(f"- 注入精度（注入片段来自期望来源的占比）：**{_pct(inj_precision)}**"
              f"（共注入 {inj_total} 片段，平均 {inj_total / max(auto_injected, 1):.1f} 片/查询）")
        r.add(f"- 回捞带（自动未注入、kb_search@{kn.min_score} 可捞回）：{fallback}/{n} 条"
              f"——验证「自动收精度、工具保召回」的兜底通道")
        r.add(f"- 两级全失：{lost}/{n} 条")
        r.add(f"- 负例拦截：工具阈值泄漏 **{neg_tool_leak}/{len(NEGATIVE_QUERIES)}**，"
              f"自动阈值泄漏 **{neg_auto_leak}/{len(NEGATIVE_QUERIES)}**"
              f"（负例 top1 分布 {min(neg_top1):.3f}~{max(neg_top1):.3f}）")
        r.verdict("自动注入率 ≥ 60%", _pct(auto_injected / n), auto_injected / n >= 0.60)
        r.verdict("注入精度 ≥ 90%", _pct(inj_precision), inj_precision >= 0.90)
        r.verdict("负例自动注入 = 0", str(neg_auto_leak), neg_auto_leak == 0)
        r.verdict("负例工具泄漏 = 0", str(neg_tool_leak), neg_tool_leak == 0)
        r.verdict("两级全失 = 0", str(lost), lost == 0)

        # ── 性能段语料（合成大库）─────────────────────────
        texts = synth_chunks(args.synthetic)

        # 并发 A/B：同一批文本，1 路 vs 4 路（默认）embedding
        e1 = Embedder(provider=args.provider, model=args.model,
                      batch_size=args.batch_size, concurrency=1)
        t = time.perf_counter()
        await e1.embed(texts)
        serial_secs = time.perf_counter() - t
        await e1.aclose()

        t = time.perf_counter()
        vectors = await embedder.embed(texts)  # 默认 4 路
        conc_secs = time.perf_counter() - t
        speedup = serial_secs / conc_secs if conc_secs else 0.0

        big_store = KnowledgeStore(workdir / "big")
        big_store.add(texts, vectors, source="合成制度库",
                      provider=embedder.provider, model=embedder.model)

        # 冷/热检索：新实例首查含读盘，后续走 mtime 缓存
        sample_qv = (await embedder.embed(["采购管理的相关规定"]))[0]
        cold_store = KnowledgeStore(workdir / "big")
        t = time.perf_counter()
        cold_results = cold_store.search(sample_qv, 20)
        cold_ms = (time.perf_counter() - t) * 1000
        hot_runs = []
        for _ in range(10):
            t = time.perf_counter()
            cold_store.search(sample_qv, 20)
            hot_runs.append((time.perf_counter() - t) * 1000)
        hot_ms = statistics.mean(hot_runs)

        # ── D. 上下文成本 ─────────────────────────────────
        def chars_top(k: int) -> int:
            return sum(len(c["text"]) for _, c in cold_results[:k])

        auto_chars = sum(len(c["text"]) for s, c in cold_results[:kn.auto_top_k]
                         if s >= kn.auto_min_score)
        r.add()
        r.add(f"## D. 上下文成本（{args.synthetic} 块合成库，单次检索注入的字符量）")
        r.add()
        r.add("| 路径 | 片段数 | 字符量 |")
        r.add("|------|--------|--------|")
        r.add(f"| 自动注入（top{kn.auto_top_k} ≥{kn.auto_min_score}） "
              f"| ≤{kn.auto_top_k} | {auto_chars} |")
        r.add(f"| kb_search 默认（top{kn.top_k}） | {kn.top_k} | {chars_top(kn.top_k)} |")
        r.add(f"| kb_search 钳制上限（top8） | 8 | {chars_top(8)} |")
        r.add(f"| 未钳制（LLM 传 top20，旧行为） | 20 | {chars_top(20)} |")
        saved = 1 - chars_top(8) / chars_top(20) if chars_top(20) else 0
        r.add()
        r.add(f"钳制上限相对旧行为节省 **{_pct(saved)}** 上下文；"
              f"自动注入路径最省（高门槛 + 片段数上限）。")

        # ── E. 性能 ───────────────────────────────────────
        r.add()
        r.add("## E. 性能")
        r.add()
        r.add(f"- 入库 embedding 并发加速：{args.synthetic} 块，1 路 {serial_secs:.2f}s "
              f"→ 4 路 {conc_secs:.2f}s，**加速 {speedup:.1f}×**"
              f"（{args.synthetic / conc_secs:.0f} 块/秒）")
        r.add(f"- 查询 embedding：均值 {statistics.mean(embed_ms):.0f} ms"
              f"（p95 {sorted(embed_ms)[int(len(embed_ms) * 0.95) - 1]:.0f} ms）"
              f"——总延迟主导项")
        r.add(f"- 本地检索（{args.synthetic} 块 × {dim} 维）：冷启动（含读盘）"
              f"{cold_ms:.1f} ms → 热缓存 **{hot_ms:.2f} ms**"
              f"（mtime 缓存收益 {cold_ms / hot_ms:.0f}×）")
        r.verdict("并发加速 ≥ 2×", f"{speedup:.1f}×", speedup >= 2.0)
        r.verdict("热缓存检索 < 5ms", f"{hot_ms:.2f} ms", hot_ms < 5.0)

        # ── F. 工具层端到端 ───────────────────────────────
        r.add()
        r.add("## F. 工具层端到端（真实 API 穿越 kb_* 工具与 prompt 渲染）")
        r.add()
        rt = KnowledgeRuntime(KnowledgeConfig(
            user_id="bench-e2e",
            embedding_provider=args.provider,
            embedding_model=args.model,
            batch_size=args.batch_size,
        ))
        rt.store = KnowledgeStore(workdir / "e2e")  # 重定向到临时目录，不写 ~/.milu
        doc = "公司差旅报销制度：高铁出行一律二等座，机票需提前三天经部门负责人审批。"
        token = _current_knowledge.set(rt)
        try:
            checks = [
                ("kb_ingest 文本入库", "已入库" in await kb_ingest._tool_wrapper.func(
                    text=doc, source="差旅制度")),
                ("kb_search 命中", "差旅制度" in await kb_search._tool_wrapper.func(
                    query="坐高铁出差能买一等座吗")),
                ("kb_manage list", "差旅制度" in await kb_manage._tool_wrapper.func(
                    action="list")),
                ("kb_manage stats", "块" in await kb_manage._tool_wrapper.func(
                    action="stats")),
            ]
            prompt = render_knowledge_prompt(rt.store)
            checks.append(("prompt 含目录与路由规则",
                           "差旅制度" in prompt and "必须先调用 kb_search" in prompt))
            await rt.prepare_auto_context(doc)
            checks.append(("自动检索命中注入", "### 本轮自动检索" in rt.auto_context
                           and "差旅制度" in rt.auto_context))
            await rt.prepare_auto_context("如何制作提拉米苏？")
            checks.append(("自动检索负例拦截", "未命中" in rt.auto_context))
        finally:
            _current_knowledge.reset(token)
            await rt.aclose()
        for name, ok in checks:
            r.add(f"- [{'PASS' if ok else 'FAIL'}] {name}")
        e2e_ok = all(ok for _, ok in checks)
        r.verdict("端到端全部 PASS", f"{sum(ok for _, ok in checks)}/{len(checks)}", e2e_ok)

        # ── G. 总评 ───────────────────────────────────────
        r.add()
        r.add("## G. 总评")
        r.add()
        r.add("| 指标 | 实测 | 达标 |")
        r.add("|------|------|------|")
        for name, actual, ok in r.verdicts:
            r.add(f"| {name} | {actual} | {'✅' if ok else '❌'} |")
        passed = sum(1 for *_, ok in r.verdicts if ok)
        r.add()
        r.add(f"**{passed}/{len(r.verdicts)} 项达标。**")
        if passed == len(r.verdicts):
            r.add("整体结论：检索质量、双阈值精度/召回分层、上下文成本控制、"
                  "性能与端到端链路全部达标，知识库可投入实际使用。")
        else:
            r.add("存在未达标项，建议：自动注入率低 → 适度下调 auto_min_score（如 0.45）；"
                  "负例泄漏 → 上调对应阈值；质量类不达标 → 检查 embedding 模型与分块参数。")

        if args.report:
            report_path = Path(args.report)
            r.save(report_path)
            print(f"\n报告已写入：{report_path}")
        return 0 if passed == len(r.verdicts) else 2
    finally:
        await embedder.aclose()
        if args.keep:
            print(f"\n临时知识库已保留：{workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


def main() -> int:
    default_report = Path(__file__).resolve().parent.parent / "docs" / "知识库评测报告.md"
    parser = argparse.ArgumentParser(description="向量知识库全面评测")
    parser.add_argument("-p", "--provider", default="qwen",
                        choices=sorted(_EMBEDDING_PROVIDERS), help="embedding 厂商")
    parser.add_argument("-m", "--model", default="", help="embedding 模型（默认厂商默认）")
    parser.add_argument("--chunk-size", type=int, default=200,
                        help="真实语料分块长度（默认 200，短文档下指标更有区分度）")
    parser.add_argument("--overlap", type=int, default=50, help="分块重叠")
    parser.add_argument("--batch-size", type=int, default=10, help="embedding 单批条数")
    parser.add_argument("--synthetic", type=int, default=120,
                        help="性能段合成语料块数（默认 120）")
    parser.add_argument("--report", default=str(default_report),
                        help="Markdown 报告输出路径（空字符串=不落盘）")
    parser.add_argument("--keep", action="store_true", help="评测后保留临时知识库目录")
    args = parser.parse_args()

    try:
        return asyncio.run(run_benchmark(args))
    except AuthenticationError as e:
        print(f"\n[中止] {e}")
        print("请在项目根目录 .env 中配置对应厂商的 API Key 后重试。")
        return 1
    except Exception as e:
        print(f"\n[失败] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
