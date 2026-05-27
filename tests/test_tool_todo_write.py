"""测试内置工具 todo_write - 会话计划管理"""
import pytest
from agent_framework.tools.builtin.todo_write import (
    TodoManager,
    PlanItem,
    PlanningState,
    create_todo_write_tool,
    _MAX_PLAN_ITEMS,
    _PLAN_REMINDER_INTERVAL,
)


# ── TodoManager 单元测试 ────────────────────────────────────


class TestTodoManagerUpdate:
    """TodoManager.update() 验证"""

    def test_basic_update(self):
        """基础计划更新"""
        mgr = TodoManager()
        result = mgr.update([
            {"content": "分析需求", "status": "completed"},
            {"content": "实现功能", "status": "in_progress"},
            {"content": "编写测试", "status": "pending"},
        ])
        assert len(mgr.state.items) == 3
        assert mgr.state.items[0].status == "completed"
        assert mgr.state.items[1].status == "in_progress"
        assert mgr.state.items[2].status == "pending"
        assert "3" in result  # 总数
        assert "1" in result  # 已完成

    def test_empty_plan(self):
        """空计划"""
        mgr = TodoManager()
        result = mgr.update([])
        assert len(mgr.state.items) == 0
        assert "暂无" in result

    def test_with_active_form(self):
        """activeForm 显示"""
        mgr = TodoManager()
        mgr.update([
            {"content": "实现认证", "status": "in_progress", "activeForm": "编写认证模块"},
        ])
        result = mgr.render()
        assert "[>]" in result
        assert "编写认证模块" in result

    def test_overwrite_plan(self):
        """更新应完整重写计划"""
        mgr = TodoManager()
        mgr.update([{"content": "任务 A", "status": "pending"}])
        assert len(mgr.state.items) == 1
        mgr.update([
            {"content": "任务 B", "status": "pending"},
            {"content": "任务 C", "status": "pending"},
        ])
        assert len(mgr.state.items) == 2
        assert mgr.state.items[0].content == "任务 B"

    def test_max_items(self):
        """超过最大条目数应报错"""
        mgr = TodoManager()
        items = [{"content": f"任务 {i}", "status": "pending"} for i in range(_MAX_PLAN_ITEMS + 1)]
        with pytest.raises(ValueError, match="最多"):
            mgr.update(items)

    def test_invalid_status(self):
        """无效状态应报错"""
        mgr = TodoManager()
        with pytest.raises(ValueError, match="无效状态"):
            mgr.update([{"content": "任务", "status": "invalid_status"}])

    def test_multiple_in_progress(self):
        """多个 in_progress 应报错"""
        mgr = TodoManager()
        with pytest.raises(ValueError, match="只能有 1 个"):
            mgr.update([
                {"content": "任务 A", "status": "in_progress"},
                {"content": "任务 B", "status": "in_progress"},
            ])

    def test_empty_content_rejected(self):
        """空 content 应报错"""
        mgr = TodoManager()
        with pytest.raises(ValueError, match="content 不能为空"):
            mgr.update([{"content": "  ", "status": "pending"}])

    def test_default_status(self):
        """省略 status 默认 pending"""
        mgr = TodoManager()
        mgr.update([{"content": "任务"}])
        assert mgr.state.items[0].status == "pending"

    def test_status_case_insensitive(self):
        """状态大小写不敏感"""
        mgr = TodoManager()
        mgr.update([{"content": "任务", "status": "IN_PROGRESS"}])
        assert mgr.state.items[0].status == "in_progress"


class TestTodoManagerRender:
    """TodoManager.render() 格式化"""

    def test_markers(self):
        """状态标记"""
        mgr = TodoManager()
        mgr.update([
            {"content": "待办", "status": "pending"},
            {"content": "进行中", "status": "in_progress"},
            {"content": "已完成", "status": "completed"},
        ])
        result = mgr.render()
        assert "[ ] 待办" in result
        assert "[>] 进行中" in result
        assert "[x] 已完成" in result

    def test_completed_count(self):
        """完成计数"""
        mgr = TodoManager()
        mgr.update([
            {"content": "A", "status": "completed"},
            {"content": "B", "status": "completed"},
            {"content": "C", "status": "pending"},
        ])
        result = mgr.render()
        assert "2/3" in result

    def test_no_plan(self):
        """无计划时的提示"""
        mgr = TodoManager()
        assert "暂无" in mgr.render()


class TestTodoManagerReminder:
    """TodoManager 提醒机制"""

    def test_no_reminder_initially(self):
        """初始状态无提醒"""
        mgr = TodoManager()
        mgr.update([{"content": "任务", "status": "pending"}])
        assert mgr.maybe_reminder() is None

    def test_reminder_after_threshold(self):
        """超过阈值后触发提醒"""
        mgr = TodoManager()
        mgr.update([{"content": "任务", "status": "pending"}])
        for _ in range(_PLAN_REMINDER_INTERVAL):
            mgr.note_round()
        reminder = mgr.maybe_reminder()
        assert reminder is not None
        assert "刷新" in reminder

    def test_no_reminder_before_threshold(self):
        """未达阈值无提醒"""
        mgr = TodoManager()
        mgr.update([{"content": "任务", "status": "pending"}])
        for _ in range(_PLAN_REMINDER_INTERVAL - 1):
            mgr.note_round()
        assert mgr.maybe_reminder() is None

    def test_update_resets_counter(self):
        """update 应重置轮次计数器"""
        mgr = TodoManager()
        mgr.update([{"content": "任务 A", "status": "pending"}])
        for _ in range(_PLAN_REMINDER_INTERVAL):
            mgr.note_round()
        # 更新计划
        mgr.update([{"content": "任务 B", "status": "pending"}])
        assert mgr.state.rounds_since_update == 0
        assert mgr.maybe_reminder() is None

    def test_no_reminder_empty_plan(self):
        """空计划不触发提醒"""
        mgr = TodoManager()
        for _ in range(_PLAN_REMINDER_INTERVAL + 5):
            mgr.note_round()
        assert mgr.maybe_reminder() is None


# ── create_todo_write_tool 集成测试 ─────────────────────────


class TestTodoWriteTool:
    """create_todo_write_tool 工厂与包装函数"""

    @pytest.mark.asyncio
    async def test_tool_metadata(self):
        """工具元数据"""
        tool = create_todo_write_tool()
        wrapper = tool._tool_wrapper
        assert wrapper.name == "todo_write"
        assert wrapper.is_async is True
        assert wrapper.dangerous is False

    @pytest.mark.asyncio
    async def test_tool_invocation(self):
        """工具正常调用"""
        mgr = TodoManager()
        tool = create_todo_write_tool(mgr)
        wrapper = tool._tool_wrapper

        result = await wrapper.func(items=[
            {"content": "分析需求", "status": "completed"},
            {"content": "编写代码", "status": "in_progress"},
        ])
        assert "1/2" in result
        assert len(mgr.state.items) == 2

    @pytest.mark.asyncio
    async def test_tool_error_propagates(self):
        """工具错误通过字符串返回（由 executor 处理）"""
        tool = create_todo_write_tool()
        wrapper = tool._tool_wrapper

        # 无效状态应由 executor 捕获并转为 is_error=True
        # 这里直接测试函数，应抛出 ValueError
        with pytest.raises(ValueError):
            await wrapper.func(items=[{"content": "任务", "status": "bad_status"}])

    @pytest.mark.asyncio
    async def test_wrapped_func_resets_counter(self):
        """包装函数应在调用后重置 rounds_since_update"""
        mgr = TodoManager()
        tool = create_todo_write_tool(mgr)
        wrapper = tool._tool_wrapper

        # 先手动设置轮次
        mgr.state.rounds_since_update = 10
        # 调用工具
        await wrapper.func(items=[{"content": "任务", "status": "pending"}])
        # 应被重置
        assert mgr.state.rounds_since_update == 0

    @pytest.mark.asyncio
    async def test_shared_manager(self):
        """多个工具共享同一个 Manager 状态"""
        mgr = TodoManager()
        tool_a = create_todo_write_tool(mgr)
        tool_b = create_todo_write_tool(mgr)

        await tool_a._tool_wrapper.func(items=[
            {"content": "共享任务", "status": "pending"},
        ])
        # tool_b 应该能看到 tool_a 写入的状态
        assert len(mgr.state.items) == 1
        assert mgr.state.items[0].content == "共享任务"


# ── Agent 集成测试 ──────────────────────────────────────────


class TestTodoWriteAgentIntegration:
    """todo_write 与 Agent 的集成"""

    @pytest.mark.asyncio
    async def test_agent_with_todo_tool(self):
        """Agent 注册并使用 todo_write 工具"""
        import json
        from unittest.mock import AsyncMock
        from agent_framework.agent import Agent, AgentConfig
        from agent_framework.agent.events import ToolResult, AgentDone
        from agent_framework.llm.base.response import StreamChunk

        todo_tool = create_todo_write_tool()
        call_count = 0

        async def mock_chat(*a, **kw):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # 第一轮：调用 todo_write
                yield StreamChunk(tool_calls=[
                    type('obj', (), {
                        'index': 0, 'id': 'call_1',
                        'function': type('obj', (), {
                            'name': 'todo_write',
                            'arguments': json.dumps({
                                "items": [
                                    {"content": "分析需求", "status": "completed"},
                                    {"content": "编码实现", "status": "in_progress"},
                                ]
                            }),
                        })()
                    })()
                ])
                yield StreamChunk(finish_reason="tool_calls")
            else:
                # 第二轮：总结
                yield StreamChunk(content="计划已更新", finish_reason="stop")

        llm = AsyncMock()
        llm.chat = mock_chat

        agent = Agent(
            llm=llm,
            system_prompt="你是助手",
            tools=[todo_tool],
            config=AgentConfig(max_turns=5),
        )

        events = []
        async for event in agent.run("制定计划"):
            events.append(event)

        # 验证工具结果
        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert not tool_results[0].is_error
        assert "1/2" in tool_results[0].output
