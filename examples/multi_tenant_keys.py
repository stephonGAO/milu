"""多租户 API Key 隔离示例 — KeyedLLMProvider。

演示：
- 不同租户/用户使用各自的 API Key：resolve_api_key 把 (user_id, session_id) 映射到 Key
- 同 Key 复用同一个 LLM 实例（共享 AsyncOpenAI 连接池，不随用户数增长内存）
- 不同 Key 得到独立实例（配额互不串用），总数受 max_clients 约束（超出 LRU 淘汰并关闭连接池）
- 与 AgentPool 组合：per-user Agent 隔离 + per-tenant Key 隔离 同时成立
- get_stats() 监控与 aclose() 优雅关闭
- 安全实践：Key 从环境变量读取（如 TENANT_KEY_ACME），源码零明文 Key

适用场景：SaaS 多租户（每个企业客户一个 Key）、BYOK（用户自带 Key）、分级配额。

运行：
    .venv/Scripts/python examples/multi_tenant_keys.py

    第 1 部分（机制演示）无需真实 Key；
    第 2 部分（真实调用）需要 .env 中配置 QWEN_API_KEY，未配置时自动跳过。
"""
from __future__ import annotations

import asyncio
import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

from milu import (
    AgentDone,
    AgentPool,
    AgentPoolConfig,
    KeyedLLMProvider,
    TextDelta,
)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _mask(key: str | None) -> str:
    """打印用：遮蔽 Key 中段。"""
    if not key:
        return "<None>"
    return key[:6] + "…" + key[-4:] if len(key) > 12 else key[:3] + "…"


# ── 1. 机制演示（不发起真实请求，构造 LLM 是懒加载的） ─────────────────


async def demo_mechanics() -> None:
    print("=" * 60)
    print("第 1 部分：缓存/隔离/LRU 机制（无需真实 Key）")
    print("=" * 60)

    # ⚠️ 安全实践：API Key 绝不要明文硬编码在源码中（会随代码进 Git、随日志泄漏）。
    # 生产环境应从 环境变量 / 密钥管理服务（Vault、云 KMS）/ 加密配置中心 读取并预载内存。
    # 本演示为了能离线展示缓存机制，用运行时随机生成的假 Key 兜底——不落盘、源码零明文。
    demo_fallback_keys = {
        t: f"sk-demo-{secrets.token_hex(8)}" for t in ("acme", "globex", "initech")
    }

    def resolve_key(user_id: str, session_id: str) -> str | None:
        """user_id 形如 "租户:用户名"。生产取 Key 方式：按租户读环境变量/密钥服务。"""
        tenant = user_id.split(":", 1)[0]
        env_key = os.environ.get(f"TENANT_KEY_{tenant.upper()}")  # ← 生产路径，如 TENANT_KEY_ACME
        if env_key:
            return env_key
        return demo_fallback_keys.get(tenant)  # 演示兜底；查不到返回 None → 默认 Key

    provider = KeyedLLMProvider(
        "qwen", model="qwen-plus",
        resolve_api_key=resolve_key,
        max_clients=2,            # 故意调小，便于演示 LRU 淘汰
    )

    # 同一租户的两个用户 → 复用同一个 LLM 实例（同一连接池）
    llm_alice = provider("acme:alice", "s1")
    llm_bob = provider("acme:bob", "s1")
    print(f"acme:alice 与 acme:bob 复用同一实例: {llm_alice is llm_bob}"
          f"  (key={_mask(llm_alice._api_key)})")

    # 不同租户 → 独立实例，Key 互不串用
    llm_carol = provider("globex:carol", "s1")
    print(f"globex:carol 是独立实例: {llm_carol is not llm_alice}"
          f"  (key={_mask(llm_carol._api_key)})")

    # 第 3 个 Key 超过 max_clients=2 → LRU 淘汰最久未用的 acme，并后台关闭其连接池
    provider("initech:dave", "s1")
    await asyncio.sleep(0.05)     # 让后台关闭任务执行
    stats = provider.get_stats()
    print(f"统计: created={stats['created']} reused={stats['reused']} "
          f"evicted={stats['evicted']} active={stats['active_clients']} "
          f"hit_rate={stats['hit_rate']}")

    await provider.aclose()       # 关闭所有缓存的连接池
    print("已 aclose，active_clients =", provider.get_stats()["active_clients"])


# ── 2. 真实调用：KeyedLLMProvider + AgentPool 组合 ───────────────────


async def demo_with_pool() -> None:
    print()
    print("=" * 60)
    print("第 2 部分：接入 AgentPool 真实对话（需 QWEN_API_KEY）")
    print("=" * 60)
    if not os.environ.get("QWEN_API_KEY"):
        print("未配置 QWEN_API_KEY，跳过真实调用演示。")
        return

    # 演示里所有租户都回退到默认 Key（返回 None → 环境变量）；
    # 实际项目把 resolve 换成上面 demo_mechanics 里那种 租户→Key 查表即可。
    provider = KeyedLLMProvider(
        "qwen", model="qwen-plus",
        resolve_api_key=lambda uid, sid: None,
    )
    pool = AgentPool(
        llm_factory=provider,                       # ← 多租户 Key 隔离用在这里
        config=AgentPoolConfig(max_agents=10, max_concurrent_runs=5),
        agent_kwargs={"session_enabled": False, "tools": []},  # 纯对话演示
    )
    await pool.start()
    try:
        async def ask(user_id: str, text: str) -> str:
            """并发执行，缓冲完整回复后统一打印（避免两路流式输出交错）。"""
            reply = []
            async with pool.acquire(user_id, "s1") as h:
                async for evt in h.agent.run(text):
                    if isinstance(evt, TextDelta) and evt.text:
                        reply.append(evt.text)
                    elif isinstance(evt, AgentDone):
                        break
            answer = "".join(reply).strip()
            print(f"\n[{user_id}] 问: {text}\n[{user_id}] 答: {answer}")
            return answer

        # 两个用户并发提问：Agent 实例彼此隔离，但共享同一个 LLM 客户端
        await asyncio.gather(
            ask("acme:alice", "用一句话介绍你自己"),
            ask("acme:bob", "1+1 等于几？只回答数字"),
        )
        print("\nLLM 客户端统计:", provider.get_stats())
        print("Agent 池统计: created =", pool.get_stats()["created"],
              "(两个用户 → 两个独立 Agent)")
    finally:
        await pool.stop()
        await provider.aclose()


async def main() -> None:
    await demo_mechanics()
    await demo_with_pool()


if __name__ == "__main__":
    asyncio.run(main())
