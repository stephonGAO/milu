"""AgentPool — 多用户场景下的 Agent 生命周期与隔离管理。

设计目标：
- 按 (user_id, session_id) 自动管理 Agent 实例
- 限制最大并发数（信号量）和实例池容量（LRU 淘汰）
- 透明的连接/会话生命周期（async contextmanager）
- 不修改 Agent 本身，纯外层包装

典型用法（FastAPI）：
    pool = AgentPool(llm_factory=..., max_agents=200, max_concurrent=50)
    async with pool.get(user_id, session_id) as agent:
        async for evt in agent.run(user_input):
            yield evt
"""
from .pool import AgentPool, AgentPoolConfig, PooledAgent

__all__ = ["AgentPool", "AgentPoolConfig", "PooledAgent"]
