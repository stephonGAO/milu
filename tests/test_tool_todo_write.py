"""测试内置工具 todo_write - 会话计划管理（含文件持久化）"""
import json
import tempfile
from pathlib import Path

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
        """超过阈值后触发提醒（包含完整计划）"""
        mgr = TodoManager()
        mgr.update([
            {"content": "任务 A", "status": "completed"},
            {"content": "任务 B", "status": "in_progress"},
        ])
        for _ in range(_PLAN_REMINDER_INTERVAL):
            mgr.note_round()
        reminder = mgr.maybe_reminder()
        assert reminder is not None
        assert "刷新" in reminder
        # 关键：提醒中包含完整计划内容
        assert "任务 A" in reminder
        assert "任务 B" in reminder
        assert "[x]" in reminder
        assert "[>]" in reminder

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


# ── 文件持久化测试 ──────────────────────────────────────────


class TestTodoManagerPersistence:
    """TodoManager 文件持久化"""

    def test_save_to_file(self, tmp_path):
        """update 后自动保存到文件"""
        plan_file = tmp_path / ".plan.json"
        mgr = TodoManager(plan_file=plan_file)
        mgr.update([
            {"content": "分析需求", "status": "completed"},
            {"content": "实现功能", "status": "in_progress", "activeForm": "编码中"},
        ])
        assert plan_file.exists()
        data = json.loads(plan_file.read_text(encoding="utf-8"))
        assert len(data["items"]) == 2
        assert data["items"][0]["content"] == "分析需求"
        assert data["items"][0]["status"] == "completed"
        assert data["items"][1]["activeForm"] == "编码中"

    def test_load_from_file(self, tmp_path):
        """启动时从文件自动加载"""
        plan_file = tmp_path / ".plan.json"
        # 先写入文件
        data = {
            "items": [
                {"content": "任务 A", "status": "completed", "activeForm": ""},
                {"content": "任务 B", "status": "in_progress", "activeForm": "进行中"},
            ]
        }
        plan_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        # 新实例应自动加载
        mgr = TodoManager(plan_file=plan_file)
        assert len(mgr.state.items) == 2
        assert mgr.state.items[0].content == "任务 A"
        assert mgr.state.items[1].status == "in_progress"

    def test_load_nonexistent_file(self, tmp_path):
        """文件不存在时正常启动"""
        plan_file = tmp_path / "nonexistent.json"
        mgr = TodoManager(plan_file=plan_file)
        assert len(mgr.state.items) == 0

    def test_roundtrip(self, tmp_path):
        """保存后加载，内容一致"""
        plan_file = tmp_path / ".plan.json"
        mgr1 = TodoManager(plan_file=plan_file)
        mgr1.update([
            {"content": "设计 API", "status": "completed", "activeForm": ""},
            {"content": "实现核心", "status": "in_progress", "activeForm": "编码中"},
            {"content": "编写测试", "status": "pending", "activeForm": ""},
        ])

        # 新实例从文件加载
        mgr2 = TodoManager(plan_file=plan_file)
        assert len(mgr2.state.items) == 3
        for i in range(3):
            assert mgr2.state.items[i].content == mgr1.state.items[i].content
            assert mgr2.state.items[i].status == mgr1.state.items[i].status
            assert mgr2.state.items[i].active_form == mgr1.state.items[i].active_form

    def test_overwrite_file_on_update(self, tmp_path):
        """每次 update 都覆盖文件"""
        plan_file = tmp_path / ".plan.json"
        mgr = TodoManager(plan_file=plan_file)
        mgr.update([{"content": "初始任务", "status": "pending"}])
        mgr.update([{"content": "新任务", "status": "completed"}])

        data = json.loads(plan_file.read_text(encoding="utf-8"))
        assert len(data["items"]) == 1
        assert data["items"][0]["content"] == "新任务"

    def test_clear_removes_file(self, tmp_path):
        """clear() 删除计划文件"""
        plan_file = tmp_path / ".plan.json"
        mgr = TodoManager(plan_file=plan_file)
        mgr.update([{"content": "任务", "status": "pending"}])
        assert plan_file.exists()
        mgr.clear()
        assert not plan_file.exists()

    def test_no_file_mode(self):
        """plan_file=None 时不保存文件"""
        mgr = TodoManager()  # 不传 plan_file
        mgr.update([{"content": "任务", "status": "pending"}])
        assert mgr.plan_file is None

    def test_to_dict(self):
        """to_dict 序列化"""
        mgr = TodoManager()
        mgr.update([
            {"content": "A", "status": "completed"},
            {"content": "B", "status": "in_progress", "activeForm": "做 B"},
        ])
        d = mgr.to_dict()
        assert len(d["items"]) == 2
        assert d["items"][1]["activeForm"] == "做 B"

    def test_corrupted_file_handled(self, tmp_path):
        """损坏的计划文件不崩溃"""
        plan_file = tmp_path / ".plan.json"
        plan_file.write_text("this is not json!", encoding="utf-8")
        # 不应抛出异常
        mgr = TodoManager(plan_file=plan_file)
        assert len(mgr.state.items) == 0

    def test_set_plan_file_switches_path(self, tmp_path):
        """set_plan_file 切换文件路径并清空旧状态"""
        file_a = tmp_path / "a" / "plan.json"
        file_b = tmp_path / "b" / "plan.json"

        mgr = TodoManager(plan_file=file_a)
        mgr.update([{"content": "任务 A", "status": "completed"}])
        assert file_a.exists()

        # 切换到 file_b（不存在），状态应清空
        mgr.set_plan_file(file_b)
        assert mgr.plan_file == file_b
        assert len(mgr.state.items) == 0

        # 写入新文件
        mgr.update([{"content": "任务 B", "status": "in_progress"}])
        assert file_b.exists()
        data = json.loads(file_b.read_text(encoding="utf-8"))
        assert data["items"][0]["content"] == "任务 B"

    def test_set_plan_file_loads_existing(self, tmp_path):
        """set_plan_file 切换到已有文件时自动加载"""
        file_a = tmp_path / "a" / "plan.json"
        file_b = tmp_path / "b" / "plan.json"

        # 预先写入 file_b
        file_b.parent.mkdir(parents=True, exist_ok=True)
        plan_data = {
            "items": [
                {"content": "已有任务", "status": "in_progress", "activeForm": "进行中"},
            ]
        }
        file_b.write_text(json.dumps(plan_data, ensure_ascii=False), encoding="utf-8")

        mgr = TodoManager(plan_file=file_a)
        mgr.update([{"content": "临时任务", "status": "pending"}])

        # 切换到 file_b，应自动加载
        mgr.set_plan_file(file_b)
        assert len(mgr.state.items) == 1
        assert mgr.state.items[0].content == "已有任务"
        assert mgr.state.items[0].status == "in_progress"

    def test_set_plan_file_none(self, tmp_path):
        """set_plan_file(None) 切换为纯内存模式"""
        plan_file = tmp_path / "plan.json"
        mgr = TodoManager(plan_file=plan_file)
        mgr.update([{"content": "任务", "status": "pending"}])

        mgr.set_plan_file(None)
        assert mgr.plan_file is None
        assert len(mgr.state.items) == 0


# ── create_todo_write_tool 集成测试 ─────────────────────────


class TestTodoWriteTool:
    """create_todo_write_tool 工厂与包装函数"""

    @pytest.mark.asyncio
    async def test_returns_tuple(self):
        """工厂返回 (todo_write, todo_read) 元组"""
        result = create_todo_write_tool()
        assert isinstance(result, tuple)
        assert len(result) == 2
        todo_write, todo_read = result
        assert todo_write._tool_wrapper.name == "todo_write"
        assert todo_read._tool_wrapper.name == "todo_read"

    @pytest.mark.asyncio
    async def test_tool_manager_marker(self):
        """工具函数上有 _todo_manager 标记，供 Agent 自动发现"""
        todo_write, todo_read = create_todo_write_tool()
        # 标记在 wrapped func 上（wrapper.func）
        assert hasattr(todo_write._tool_wrapper.func, '_todo_manager')
        assert hasattr(todo_read._tool_wrapper.func, '_todo_manager')
        # 两个工具共享同一个 TodoManager
        assert todo_write._tool_wrapper.func._todo_manager is todo_read._tool_wrapper.func._todo_manager
        assert isinstance(todo_write._tool_wrapper.func._todo_manager, TodoManager)

    @pytest.mark.asyncio
    async def test_write_tool_metadata(self):
        """todo_write 工具元数据"""
        todo_write, _ = create_todo_write_tool()
        wrapper = todo_write._tool_wrapper
        assert wrapper.name == "todo_write"
        assert wrapper.is_async is True
        assert wrapper.is_safe is False

    @pytest.mark.asyncio
    async def test_read_tool_metadata(self):
        """todo_read 工具元数据"""
        _, todo_read = create_todo_write_tool()
        wrapper = todo_read._tool_wrapper
        assert wrapper.name == "todo_read"
        assert wrapper.is_async is True
        assert wrapper.is_safe is True

    @pytest.mark.asyncio
    async def test_write_invocation(self):
        """todo_write 正常调用"""
        mgr = TodoManager()
        todo_write, _ = create_todo_write_tool(manager=mgr)

        result = await todo_write._tool_wrapper.func(items=[
            {"content": "分析需求", "status": "completed"},
            {"content": "编写代码", "status": "in_progress"},
        ])
        assert "1/2" in result
        assert len(mgr.state.items) == 2

    @pytest.mark.asyncio
    async def test_read_invocation(self):
        """todo_read 查看当前计划"""
        mgr = TodoManager()
        mgr.update([
            {"content": "任务 A", "status": "completed"},
            {"content": "任务 B", "status": "pending"},
        ])
        _, todo_read = create_todo_write_tool(manager=mgr)
        result = await todo_read._tool_wrapper.func()
        assert "[x] 任务 A" in result
        assert "[ ] 任务 B" in result
        assert "1/2" in result

    @pytest.mark.asyncio
    async def test_read_empty_plan(self):
        """todo_read 空计划"""
        mgr = TodoManager()
        _, todo_read = create_todo_write_tool(manager=mgr)
        result = await todo_read._tool_wrapper.func()
        assert "暂无" in result

    @pytest.mark.asyncio
    async def test_write_persists_to_file(self, tmp_path):
        """todo_write 调用后文件被保存"""
        plan_file = tmp_path / ".plan.json"
        todo_write, _ = create_todo_write_tool(plan_file=plan_file)

        await todo_write._tool_wrapper.func(items=[
            {"content": "持久化测试", "status": "pending"},
        ])
        assert plan_file.exists()
        data = json.loads(plan_file.read_text(encoding="utf-8"))
        assert data["items"][0]["content"] == "持久化测试"

    @pytest.mark.asyncio
    async def test_read_after_file_load(self, tmp_path):
        """从文件加载后 todo_read 可见"""
        plan_file = tmp_path / ".plan.json"
        # 预先写入计划文件
        data = {
            "items": [
                {"content": "恢复的任务", "status": "in_progress", "activeForm": ""},
            ]
        }
        plan_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        # 新工厂应自动加载
        _, todo_read = create_todo_write_tool(plan_file=plan_file)
        result = await todo_read._tool_wrapper.func()
        assert "恢复的任务" in result
        assert "[>]" in result

    @pytest.mark.asyncio
    async def test_read_resets_reminder_counter(self):
        """todo_read 调用后重置提醒计数器"""
        mgr = TodoManager()
        mgr.update([{"content": "任务", "status": "pending"}])
        mgr.state.rounds_since_update = 10

        _, todo_read = create_todo_write_tool(manager=mgr)
        await todo_read._tool_wrapper.func()
        assert mgr.state.rounds_since_update == 0

    @pytest.mark.asyncio
    async def test_write_error_propagates(self):
        """todo_write 错误应抛出（由 executor 处理）"""
        todo_write, _ = create_todo_write_tool()
        with pytest.raises(ValueError):
            await todo_write._tool_wrapper.func(items=[
                {"content": "任务", "status": "bad_status"}
            ])

    @pytest.mark.asyncio
    async def test_shared_manager(self):
        """多个工具共享同一个 Manager 状态"""
        mgr = TodoManager()
        write_a, read_a = create_todo_write_tool(manager=mgr)
        write_b, read_b = create_todo_write_tool(manager=mgr)

        await write_a._tool_wrapper.func(items=[
            {"content": "共享任务", "status": "pending"},
        ])
        # read_b 应该能看到 write_a 写入的状态
        result = await read_b._tool_wrapper.func()
        assert "共享任务" in result


# ── Agent 集成测试 ──────────────────────────────────────────


class TestTodoWriteAgentIntegration:
    """todo_write 与 Agent 的集成"""

    @pytest.mark.asyncio
    async def test_agent_with_todo_tools(self):
        """Agent 注册并使用 todo_write + todo_read"""
        import json as _json
        from unittest.mock import AsyncMock
        from agent_framework.agent import Agent, AgentConfig
        from agent_framework.agent.events import ToolResult, AgentDone
        from agent_framework.llm.base.response import StreamChunk

        todo_write, todo_read = create_todo_write_tool()
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
                            'arguments': _json.dumps({
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
            tools=[todo_write, todo_read],
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

    @pytest.mark.asyncio
    async def test_agent_reads_plan_after_truncation(self, tmp_path):
        """模拟上下文截断后，Agent 通过 todo_read 恢复计划"""
        import json as _json
        from unittest.mock import AsyncMock
        from agent_framework.agent import Agent, AgentConfig
        from agent_framework.agent.events import ToolResult
        from agent_framework.llm.base.response import StreamChunk

        plan_file = tmp_path / ".plan.json"
        # 预先保存一个计划（模拟之前的会话）
        plan_data = {
            "items": [
                {"content": "已完成的步骤", "status": "completed", "activeForm": ""},
                {"content": "当前步骤", "status": "in_progress", "activeForm": "处理中"},
                {"content": "待办步骤", "status": "pending", "activeForm": ""},
            ]
        }
        plan_file.write_text(_json.dumps(plan_data, ensure_ascii=False), encoding="utf-8")

        todo_write, todo_read = create_todo_write_tool(plan_file=plan_file)
        call_count = 0

        async def mock_chat(*a, **kw):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # LLM 调用 todo_read 查看当前计划
                yield StreamChunk(tool_calls=[
                    type('obj', (), {
                        'index': 0, 'id': 'call_1',
                        'function': type('obj', (), {
                            'name': 'todo_read',
                            'arguments': "{}",
                        })()
                    })()
                ])
                yield StreamChunk(finish_reason="tool_calls")
            else:
                yield StreamChunk(content="好的，我继续当前步骤", finish_reason="stop")

        llm = AsyncMock()
        llm.chat = mock_chat

        agent = Agent(
            llm=llm,
            system_prompt="你是助手",
            tools=[todo_write, todo_read],
            config=AgentConfig(max_turns=5, session_enabled=False),
        )

        events = []
        async for event in agent.run("我之前的计划是什么"):
            events.append(event)

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert not tool_results[0].is_error
        # 关键：即使上下文被截断，计划内容通过文件恢复
        assert "已完成的步骤" in tool_results[0].output
        assert "当前步骤" in tool_results[0].output
        assert "处理中" in tool_results[0].output
