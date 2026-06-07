"""向量知识库真实效果评测 —— 走真实 embedding API，输出可量化指标报告。

与 tests/test_knowledge.py（FakeEmbedder 单元测试）互补：本脚本衡量"真实模型
下的检索效果"。语料为 5 篇内容互不重叠的中文文档；正例查询全部**换说法改写**
（不照抄原文字面），专门考察语义召回（这正是向量检索相对关键词匹配的价值点）；
另设负例查询衡量"无关问题会不会被误检"。

指标：
  检索质量  Hit@1 / Hit@3 / Hit@5（期望来源进前 k）、MRR@5、答案关键词召回@3
  基线对比  字符 bigram Dice 关键词基线的 Hit@1 / Hit@3（量化语义检索的增益）
  分离度    正例 top1 均分 vs 负例 top1 均分（能否用阈值滤掉无关查询）
  性能      入库吞吐（块/秒）、查询 embedding 延迟、本地余弦检索延迟、磁盘占用

用法（需 .env 配置对应厂商 API Key，如 QWEN_API_KEY）：
    .venv/Scripts/python tests/benchmark_knowledge.py                # 默认 qwen
    .venv/Scripts/python tests/benchmark_knowledge.py -p glm
    .venv/Scripts/python tests/benchmark_knowledge.py -m text-embedding-v3
    .venv/Scripts/python tests/benchmark_knowledge.py --keep         # 保留临时库目录
"""
from __future__ import annotations

import argparse
import asyncio
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from milu.knowledge import Embedder, KnowledgeStore, chunk_text
from milu.knowledge.embedder import _EMBEDDING_PROVIDERS
from milu.llm.base.exceptions import AuthenticationError

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

# 正例查询：(改写式问题, 期望来源, 答案关键词列表——任一出现在 top3 文本中即算召回)
POSITIVE_QUERIES: list[tuple[str, str, list[str]]] = [
    ("这块手表充满电要多久？", "产品手册-星航S3", ["90 分钟"]),
    ("戴着它洗热水澡可以吗？", "产品手册-星航S3", ["热水淋浴"]),
    ("尊享版比标准版贵多少钱？", "产品手册-星航S3", ["1999", "1499"]),
    ("手环能测血氧吗？", "产品手册-星航S3", ["血氧"]),
    ("收到货后几天内可以不要理由地退掉？", "退换货政策", ["7 天"]),
    ("刻了字的商品还能退吗？", "退换货政策", ["定制刻字"]),
    ("退款多久能到账？", "退换货政策", ["3 个工作日"]),
    ("订单服务之间用什么协议通信？", "订单系统架构", ["gRPC"]),
    ("系统用什么消息中间件做异步解耦？", "订单系统架构", ["Kafka"]),
    ("缓存多长时间过期？", "订单系统架构", ["300 秒"]),
    ("新版本上线是怎么放量的？", "订单系统架构", ["5%", "灰度"]),
    ("在家办公需要走什么流程？", "员工手册", ["OA", "报备"]),
    ("出差住酒店每晚的报销标准是多少？", "员工手册", ["600 元"]),
    ("access_token 多长时间失效？", "API接入指南", ["2 小时"]),
    ("接口被限流了会返回什么状态码？", "API接入指南", ["429"]),
    ("回调推送失败平台会重试几次？", "API接入指南", ["三次"]),
]

# 负例查询：语料中完全不存在的话题（理想情况下分数显著低于正例）
NEGATIVE_QUERIES: list[str] = [
    "明朝灭亡的主要原因是什么？",
    "如何在家制作提拉米苏？",
    "梅西拿过几次世界杯冠军？",
    "光合作用的化学方程式是什么？",
    "东京到大阪坐新干线要多久？",
]

TOP_K = 5


# ==================== 关键词基线（字符 bigram Dice）====================


def _bigrams(s: str) -> set[str]:
    s = "".join(s.split())
    return {s[i:i + 2] for i in range(len(s) - 1)}


def _dice(a: str, b: str) -> float:
    """字符 bigram Dice 相似度——零依赖的中文关键词匹配基线。"""
    ba, bb = _bigrams(a), _bigrams(b)
    if not ba or not bb:
        return 0.0
    return 2 * len(ba & bb) / (len(ba) + len(bb))


def baseline_rank(query: str, chunks: list[dict], expected_source: str) -> int | None:
    """基线检索：按 Dice 排序，返回期望来源的最高排名（1-based），未进前 TOP_K 返回 None。"""
    scored = sorted(
        ((_dice(query, c["text"]), c) for c in chunks),
        key=lambda x: x[0], reverse=True,
    )
    for rank, (_, chunk) in enumerate(scored[:TOP_K], 1):
        if chunk["source"] == expected_source:
            return rank
    return None


# ==================== 评测主流程 ====================


async def run_benchmark(args) -> int:
    provider = args.provider
    print("=" * 72)
    print(f"向量知识库真实效果评测  |  embedding: {provider}/{args.model or '(厂商默认)'}")
    print("=" * 72)

    embedder = Embedder(provider=provider, model=args.model, batch_size=args.batch_size)
    workdir = Path(tempfile.mkdtemp(prefix="milu_kb_bench_"))
    store = KnowledgeStore(workdir)

    try:
        # ── 1. 入库 ───────────────────────────────────────
        total_chars = sum(len(t) for t in CORPUS.values())
        n_chunks = 0
        ingest_t0 = time.perf_counter()
        for source, text in CORPUS.items():
            chunks = chunk_text(text, args.chunk_size, args.overlap)
            vectors = await embedder.embed(chunks)
            store.add(chunks, vectors, source=source,
                      provider=embedder.provider, model=embedder.model)
            n_chunks += len(chunks)
        ingest_secs = time.perf_counter() - ingest_t0

        s = store.stats()
        all_chunks, _ = store.load()
        print(f"\n[入库] {len(CORPUS)} 篇文档 / {total_chars} 字符 → {n_chunks} 块"
              f"（chunk_size={args.chunk_size}）")
        print(f"       耗时 {ingest_secs:.2f}s（{n_chunks / ingest_secs:.1f} 块/秒，含 embedding API）"
              f" | 向量 {s['dim']} 维 | 磁盘 {s['disk_bytes'] / 1024:.1f} KB")

        # ── 2. 正例查询 ───────────────────────────────────
        print(f"\n[正例查询] {len(POSITIVE_QUERIES)} 条（全部为换说法改写，top{TOP_K} 检索）")
        print("-" * 72)
        ranks: list[int | None] = []
        kw_hits: list[bool] = []
        base_ranks: list[int | None] = []
        pos_top1_scores: list[float] = []
        embed_ms: list[float] = []
        search_ms: list[float] = []

        for i, (query, expected, keywords) in enumerate(POSITIVE_QUERIES, 1):
            t0 = time.perf_counter()
            qv = (await embedder.embed([query]))[0]
            t1 = time.perf_counter()
            results = store.search(qv, TOP_K)
            t2 = time.perf_counter()
            embed_ms.append((t1 - t0) * 1000)
            search_ms.append((t2 - t1) * 1000)

            rank = next((r for r, (_, c) in enumerate(results, 1)
                         if c["source"] == expected), None)
            kw_hit = any(kw in c["text"] for _, c in results[:3] for kw in keywords)
            b_rank = baseline_rank(query, all_chunks, expected)

            ranks.append(rank)
            kw_hits.append(kw_hit)
            base_ranks.append(b_rank)
            pos_top1_scores.append(results[0][0])

            mark = "✓" if rank == 1 else ("~" if rank else "✗")
            print(f"  [{mark}] Q{i:<2} 向量rank={rank or '>5'} top1分={results[0][0]:.3f} "
                  f"| 基线rank={b_rank or '>5'} | 关键词{'✓' if kw_hit else '✗'} | {query}")

        # ── 3. 负例查询 ───────────────────────────────────
        print(f"\n[负例查询] {len(NEGATIVE_QUERIES)} 条（语料中不存在的话题）")
        print("-" * 72)
        neg_top1_scores: list[float] = []
        for i, query in enumerate(NEGATIVE_QUERIES, 1):
            qv = (await embedder.embed([query]))[0]
            results = store.search(qv, 1)
            neg_top1_scores.append(results[0][0])
            print(f"  [N{i}] top1分={results[0][0]:.3f} | {query}")

        # ── 4. 汇总报告 ───────────────────────────────────
        n = len(POSITIVE_QUERIES)
        hit1 = sum(1 for r in ranks if r == 1) / n
        hit3 = sum(1 for r in ranks if r and r <= 3) / n
        hit5 = sum(1 for r in ranks if r and r <= 5) / n
        mrr = sum(1 / r for r in ranks if r) / n
        kw_recall = sum(kw_hits) / n
        b_hit1 = sum(1 for r in base_ranks if r == 1) / n
        b_hit3 = sum(1 for r in base_ranks if r and r <= 3) / n
        pos_avg = statistics.mean(pos_top1_scores)
        neg_avg = statistics.mean(neg_top1_scores)
        separation = pos_avg - neg_avg
        overlap = max(neg_top1_scores) - min(pos_top1_scores)

        print("\n" + "=" * 72)
        print("评测汇总")
        print("=" * 72)
        print(f"  检索质量（向量 vs 关键词基线）")
        print(f"    Hit@1        {hit1:6.1%}   （基线 {b_hit1:.1%}）")
        print(f"    Hit@3        {hit3:6.1%}   （基线 {b_hit3:.1%}）")
        print(f"    Hit@5        {hit5:6.1%}")
        print(f"    MRR@5        {mrr:6.3f}")
        print(f"    关键词召回@3  {kw_recall:6.1%}   （答案原文出现在前 3 个片段中）")
        print(f"  分数分离度（能否用阈值滤掉无关查询）")
        print(f"    正例 top1 均分 {pos_avg:.3f}（最低 {min(pos_top1_scores):.3f}） | "
              f"负例 top1 均分 {neg_avg:.3f}（最高 {max(neg_top1_scores):.3f}）")
        print(f"    分离度 {separation:+.3f}，"
              + (f"正负完全可分，建议阈值 ≈ {(min(pos_top1_scores) + max(neg_top1_scores)) / 2:.2f}"
                 if overlap < 0 else "正负分数区间有重叠，不宜单靠阈值过滤"))
        print(f"  性能")
        print(f"    入库吞吐        {n_chunks / ingest_secs:8.1f} 块/秒（含 embedding API 往返）")
        print(f"    查询 embedding  {statistics.mean(embed_ms):8.1f} ms/次"
              f"（p95 {sorted(embed_ms)[int(len(embed_ms) * 0.95) - 1]:.1f} ms）")
        print(f"    本地余弦检索    {statistics.mean(search_ms):8.2f} ms/次"
              f"（{n_chunks} 块 × {s['dim']} 维）")
        print(f"  参考解读：Hit@3 ≥ 90% 为良好；向量 Hit@1 显著高于基线 = 语义改写召回的增益；")
        print(f"            检索延迟应远小于 embedding 延迟（验证瓶颈在 API 网络而非本地计算）")
        print("=" * 72)
        return 0
    finally:
        if args.keep:
            print(f"\n临时知识库已保留：{workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="向量知识库真实效果评测")
    parser.add_argument("-p", "--provider", default="qwen",
                        choices=sorted(_EMBEDDING_PROVIDERS),
                        help="embedding 厂商（默认 qwen）")
    parser.add_argument("-m", "--model", default="",
                        help="embedding 模型（默认用厂商默认模型）")
    parser.add_argument("--chunk-size", type=int, default=200,
                        help="分块长度（默认 200——语料为短文档，小块使指标更有区分度）")
    parser.add_argument("--overlap", type=int, default=50, help="分块重叠（默认 50）")
    parser.add_argument("--batch-size", type=int, default=10, help="embedding 单批条数")
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
        return 1


if __name__ == "__main__":
    sys.exit(main())
