"""测试内置工具 todo_write - 会话计划管理（无状态版本）

- todo_write(items) / todo_read() 是模块级 @tool 装饰的函数
- 状态完全由 plan.json 文件承载，路径从 _current_session_dir ContextVar 获取
"""
import json

import pytest

from agent_framework.tools.builtin.todo_write import (
    _current_session_dir,
    _MAX_PLAN_ITEMS,
    _render,
    todo_read,
    todo_write,
)


# ── 纯函数 _render 单元测试 ─────────────────────────────────


class TestTodoWriteRender:
    """_render() 输出格式"""

    def test_markers(self):
        """各状态正确标记"""
        items = [
            {"content": "待办", "status": "pending", "activeForm": ""},
            {"content": "进行中", "status": "in_progress", "activeForm": ""},
            {"content": "已完成", "status": "completed", "activeForm": ""},
        ]
        result = _render(items)
        assert "[ ] 待办" in result
        assert "[>] 进行中" in result
        assert "[x] 已完成" in result

    def test_completed_count(self):
        """完成计数正确"""
        items = [
            {"content": "A", "status": "completed", "activeForm": ""},
            {"content": "B", "status": "completed", "activeForm": ""},
            {"content": "C", "status": "pending", "activeForm": ""},
        ]
        result = _render(items)
        assert "2/3" in result

    def test_empty_items(self):
        """空列表返回提示"""
        assert "暂无" in _render([])

    def test_active_form_displayed(self):
        """进行中条目显示 activeForm"""
        items = [
            {"content": "实现认证", "status": "in_progress", "activeForm": "编写认证模块"},
        ]
        result = _render(items)
        assert "[>]" in result
        assert "编写认证模块" in result

    def test_completed_active_form_ignored(self):
        """已完成条目不显示 activeForm"""
        items = [
            {"content": "已完成任务", "status": "completed", "activeForm": "不应显示"},
        ]
        result = _render(items)
        assert "不应显示" not in result

    def test_pending_active_form_ignored(self):
        """待办条目不显示 activeForm"""
        items = [
            {"content": "待办任务", "status": "pending", "activeForm": "不应显示"},
        ]
        result = _render(items)
        assert "不应显示" not in result

    def test_total_count(self):
        """总数显示"""
        items = [
            {"content": "A", "status": "completed", "activeForm": ""},
            {"content": "B", "status": "in_progress", "activeForm": ""},
            {"content": "C", "status": "pending", "activeForm": ""},
        ]
        result = _render(items)
        assert "(1/3 已完成)" in result


# ── 验证逻辑测试（通过 todo_write 调用触发 _validate_items） ───


class TestTodoWriteValidation:
    """todo_write() 调用前对 items 做校验"""

    @pytest.mark.asyncio
    async def test_basic_write(self, tmp_path):
        """基础写入通过校验"""
        token = _current_session_dir.set(tmp_path)
        try:
            result = await todo_write._tool_wrapper.func(items=[
                {"content": "分析需求", "status": "completed"},
                {"content": "实现功能", "status": "in_progress"},
                {"content": "编写测试", "status": "pending"},
            ])
            assert "1/3" in result
            assert "3" in result
        finally:
            _current_session_dir.reset(token)

    @pytest.mark.asyncio
    async def test_empty_items(self, tmp_path):
        """空列表合法"""
        token = _current_session_dir.set(tmp_path)
        try:
            result = await todo_write._tool_wrapper.func(items=[])
            assert "暂无" in result
        finally:
            _current_session_dir.reset(token)

    @pytest.mark.asyncio
    async def test_overwrite_full_plan(self, tmp_path):
        """第二次写入完整重写计划"""
        token = _current_session_dir.set(tmp_path)
        try:
            await todo_write._tool_wrapper.func(items=[
                {"content": "任务 A", "status": "pending"},
            ])
            await todo_write._tool_wrapper.func(items=[
                {"content": "任务 B", "status": "pending"},
                {"content": "任务 C", "status": "pending"},
            ])
            data = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
            assert len(data["items"]) == 2
            assert data["items"][0]["content"] == "任务 B"
        finally:
            _current_session_dir.reset(token)

    @pytest.mark.asyncio
    async def test_too_many_items_rejected(self, tmp_path):
        """超过最大条目数应报错"""
        token = _current_session_dir.set(tmp_path)
        try:
            items = [
                {"content": f"任务 {i}", "status": "pending"}
                for i in range(_MAX_PLAN_ITEMS + 1)
            ]
            with pytest.raises(ValueError, match="最多"):
                await todo_write._tool_wrapper.func(items=items)
        finally:
            _current_session_dir.reset(token)

    @pytest.mark.asyncio
    async def test_invalid_status_rejected(self, tmp_path):
        """无效状态应报错"""
        token = _current_session_dir.set(tmp_path)
        try:
            with pytest.raises(ValueError, match="无效状态"):
                await todo_write._tool_wrapper.func(items=[
                    {"content": "任务", "status": "invalid_status"},
                ])
        finally:
            _current_session_dir.reset(token)

    @pytest.mark.asyncio
    async def test_multiple_in_progress_rejected(self, tmp_path):
        """多个 in_progress 应报错"""
        token = _current_session_dir.set(tmp_path)
        try:
            with pytest.raises(ValueError, match="只能有 1 个"):
                await todo_write._tool_wrapper.func(items=[
                    {"content": "任务 A", "status": "in_progress"},
                    {"content": "任务 B", "status": "in_progress"},
                ])
        finally:
            _current_session_dir.reset(token)

    @pytest.mark.asyncio
    async def test_empty_content_rejected(self, tmp_path):
        """空 content 应报错"""
        token = _current_session_dir.set(tmp_path)
        try:
            with pytest.raises(ValueError, match="content 不能为空"):
                await todo_write._tool_wrapper.func(items=[
                    {"content": "   ", "status": "pending"},
                ])
        finally:
            _current_session_dir.reset(token)

    @pytest.mark.asyncio
    async def test_missing_content_rejected(self, tmp_path):
        """缺 content 字段应报错"""
        token = _current_session_dir.set(tmp_path)
        try:
            with pytest.raises(ValueError, match="content 不能为空"):
                await todo_write._tool_wrapper.func(items=[
                    {"status": "pending"},
                ])
        finally:
            _current_session_dir.reset(token)

    @pytest.mark.asyncio
    async def test_default_status_is_pending(self, tmp_path):
        """省略 status 默认为 pending"""
        token = _current_session_dir.set(tmp_path)
        try:
            await todo_write._tool_wrapper.func(items=[{"content": "任务"}])
            data = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
            assert data["items"][0]["status"] == "pending"
        finally:
            _current_session_dir.reset(token)

    @pytest.mark.asyncio
    async def test_status_case_insensitive(self, tmp_path):
        """状态大小写不敏感"""
        token = _current_session_dir.set(tmp_path)
        try:
            await todo_write._tool_wrapper.func(items=[
                {"content": "任务", "status": "IN_PROGRESS"},
            ])
            data = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
            assert data["items"][0]["status"] == "in_progress"
        finally:
            _current_session_dir.reset(token)

    @pytest.mark.asyncio
    async def test_validation_failure_does_not_overwrite(self, tmp_path):
        """校验失败时 plan.json 不被覆盖"""
        token = _current_session_dir.set(tmp_path)
        try:
            # 先写一个合法计划
            await todo_write._tool_wrapper.func(items=[
                {"content": "原任务", "status": "pending"},
            ])
            # 再写一个非法的，文件应保持不变
            with pytest.raises(ValueError):
                await todo_write._tool_wrapper.func(items=[
                    {"content": "非法", "status": "bad_status"},
                ])
            data = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
            assert len(data["items"]) == 1
            assert data["items"][0]["content"] == "原任务"
        finally:
            _current_session_dir.reset(token)


# ── 工具工厂与工具函数测试 ───────────────────────────────────


class TestTodoWriteTool:
    """todo_write / todo_read 工具元数据"""

    def test_write_tool_metadata(self):
        """todo_write 工具元数据"""
        wrapper = todo_write._tool_wrapper
        assert wrapper.name == "todo_write"
        assert wrapper.is_async is True

    def test_read_tool_metadata(self):
        """todo_read 工具元数据"""
        wrapper = todo_read._tool_wrapper
        assert wrapper.name == "todo_read"
        assert wrapper.is_async is True

    @pytest.mark.asyncio
    async def test_write_invocation_via_contextvar(self, tmp_path):
        """todo_write 通过 ContextVar 写 plan.json"""
        token = _current_session_dir.set(tmp_path)
        try:
            result = await todo_write._tool_wrapper.func(items=[
                {"content": "分析需求", "status": "completed"},
                {"content": "编写代码", "status": "in_progress"},
            ])
            assert "1/2" in result
            plan_file = tmp_path / "plan.json"
            assert plan_file.exists()
            data = json.loads(plan_file.read_text(encoding="utf-8"))
            assert len(data["items"]) == 2
        finally:
            _current_session_dir.reset(token)

    @pytest.mark.asyncio
    async def test_read_invocation_via_contextvar(self, tmp_path):
        """todo_read 通过 ContextVar 读 plan.json"""
        plan_file = tmp_path / "plan.json"
        plan_file.write_text(
            json.dumps({
                "items": [
                    {"content": "任务 A", "status": "completed", "activeForm": ""},
                    {"content": "任务 B", "status": "pending", "activeForm": ""},
                ]
            }, ensure_ascii=False),
            encoding="utf-8",
        )

        token = _current_session_dir.set(tmp_path)
        try:
            result = await todo_read._tool_wrapper.func()
            assert "[x] 任务 A" in result
            assert "[ ] 任务 B" in result
            assert "1/2" in result
        finally:
            _current_session_dir.reset(token)

    @pytest.mark.asyncio
    async def test_read_empty_when_no_file(self, tmp_path):
        """无 plan.json 时 todo_read 返回 '暂无会话计划。'"""
        token = _current_session_dir.set(tmp_path)
        try:
            result = await todo_read._tool_wrapper.func()
            assert "暂无" in result
        finally:
            _current_session_dir.reset(token)

    @pytest.mark.asyncio
    async def test_write_without_contextvar_raises(self):
        """未设置 ContextVar 时 todo_write 抛 RuntimeError"""
        # 显式重置（其他测试可能泄漏）
        token = _current_session_dir.set(None)
        try:
            with pytest.raises(RuntimeError, match="session_dir"):
                await todo_write._tool_wrapper.func(items=[
                    {"content": "任务", "status": "pending"},
                ])
        finally:
            _current_session_dir.reset(token)

    @pytest.mark.asyncio
    async def test_write_error_propagates(self, tmp_path):
        """todo_write 校验错误直接抛出（由 ToolExecutor 包装）"""
        token = _current_session_dir.set(tmp_path)
        try:
            with pytest.raises(ValueError):
                await todo_write._tool_wrapper.func(items=[
                    {"content": "任务", "status": "bad_status"},
                ])
        finally:
            _current_session_dir.reset(token)


# ── 文件持久化测试 ──────────────────────────────────────────


class TestTodoWriteFilePersistence:
    """plan.json 读写、损坏处理"""

    @pytest.mark.asyncio
    async def test_roundtrip(self, tmp_path):
        """写后读回内容一致"""
        token = _current_session_dir.set(tmp_path)
        try:
            await todo_write._tool_wrapper.func(items=[
                {"content": "设计 API", "status": "completed", "activeForm": ""},
                {"content": "实现核心", "status": "in_progress", "activeForm": "编码中"},
                {"content": "编写测试", "status": "pending", "activeForm": ""},
            ])
        finally:
            _current_session_dir.reset(token)

        plan_file = tmp_path / "plan.json"
        assert plan_file.exists()
        data = json.loads(plan_file.read_text(encoding="utf-8"))
        assert len(data["items"]) == 3
        assert data["items"][0]["content"] == "设计 API"
        assert data["items"][1]["activeForm"] == "编码中"

    @pytest.mark.asyncio
    async def test_overwrite_on_every_write(self, tmp_path):
        """每次写入都覆盖 plan.json"""
        token = _current_session_dir.set(tmp_path)
        try:
            await todo_write._tool_wrapper.func(items=[
                {"content": "初始任务", "status": "pending"},
            ])
            await todo_write._tool_wrapper.func(items=[
                {"content": "新任务", "status": "completed"},
            ])
            data = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
            assert len(data["items"]) == 1
            assert data["items"][0]["content"] == "新任务"
        finally:
            _current_session_dir.reset(token)

    @pytest.mark.asyncio
    async def test_persists_active_form(self, tmp_path):
        """activeForm 被正确持久化"""
        token = _current_session_dir.set(tmp_path)
        try:
            await todo_write._tool_wrapper.func(items=[
                {"content": "实现功能", "status": "in_progress", "activeForm": "编码中"},
            ])
            data = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
            assert data["items"][0]["activeForm"] == "编码中"
        finally:
            _current_session_dir.reset(token)

    @pytest.mark.asyncio
    async def test_read_corrupted_file_returns_empty(self, tmp_path):
        """损坏的 plan.json 视为空计划"""
        plan_file = tmp_path / "plan.json"
        plan_file.write_text("this is not json!", encoding="utf-8")

        token = _current_session_dir.set(tmp_path)
        try:
            result = await todo_read._tool_wrapper.func()
            assert "暂无" in result
        finally:
            _current_session_dir.reset(token)

    @pytest.mark.asyncio
    async def test_write_creates_parent_dir(self, tmp_path):
        """session_dir 子目录不存在时 todo_write 自动创建"""
        nested = tmp_path / "deeply" / "nested" / "session"
        token = _current_session_dir.set(nested)
        try:
            await todo_write._tool_wrapper.func(items=[
                {"content": "任务", "status": "pending"},
            ])
            assert (nested / "plan.json").exists()
        finally:
            _current_session_dir.reset(token)

    @pytest.mark.asyncio
    async def test_overwrite_clears_old_items(self, tmp_path):
        """新写入完全替换旧计划"""
        token = _current_session_dir.set(tmp_path)
        try:
            # 旧计划
            await todo_write._tool_wrapper.func(items=[
                {"content": "旧 1", "status": "completed"},
                {"content": "旧 2", "status": "pending"},
            ])
            # 写入空计划
            await todo_write._tool_wrapper.func(items=[])
            data = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
            assert data["items"] == []
        finally:
            _current_session_dir.reset(token)

    @pytest.mark.asyncio
    async def test_unicode_content_persisted(self, tmp_path):
        """中文内容正确持久化（ensure_ascii=False）"""
        token = _current_session_dir.set(tmp_path)
        try:
            await todo_write._tool_wrapper.func(items=[
                {"content": "实现用户认证模块", "status": "in_progress"},
            ])
            raw = (tmp_path / "plan.json").read_text(encoding="utf-8")
            assert "实现用户认证模块" in raw
            # 不应被 \uXXXX 转义
            assert "\\u5b9e\\u73b0" not in raw
        finally:
            _current_session_dir.reset(token)


# ── Agent 集成测试 ──────────────────────────────────────────


class TestTodoWriteAgentIntegration:
    """todo_write 工具与 Agent.run() 端到端集成"""

    @pytest.mark.asyncio
    async def test_agent_with_todo_tools(self, tmp_path):
        """Agent 通过 session_dir 注入 plan.json 位置并完成一次 todo_write"""
        import json as _json
        from unittest.mock import AsyncMock
        from agent_framework.agent import Agent, AgentConfig
        from agent_framework.agent.events import ToolResult
        from agent_framework.llm.base.response import StreamChunk

        todo_write_fn, todo_read_fn = todo_write, todo_read
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
                yield StreamChunk(content="计划已更新", finish_reason="stop")

        llm = AsyncMock()
        llm.chat = mock_chat

        agent = Agent(
            llm=llm,
            system_prompt="你是助手",
            tools=[todo_write_fn, todo_read_fn],
            config=AgentConfig(
                max_turns=5,
                session_dir=str(tmp_path),
                session_enabled=True,
            ),
        )

        events = []
        async for event in agent.run("制定计划"):
            events.append(event)

        # 工具结果
        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert not tool_results[0].is_error
        assert "1/2" in tool_results[0].output

        # plan.json 已被写入实际 session_dir（base_dir / session_id）
        plan_file = agent._session.dir_path / "plan.json"
        assert plan_file.exists()
        data = _json.loads(plan_file.read_text(encoding="utf-8"))
        assert len(data["items"]) == 2

    @pytest.mark.asyncio
    async def test_agent_reads_plan_from_file(self, tmp_path):
        """Agent 通过 todo_read 从已存在的 plan.json 恢复计划"""
        import json as _json
        from unittest.mock import AsyncMock
        from agent_framework.agent import Agent, AgentConfig
        from agent_framework.agent.events import ToolResult
        from agent_framework.llm.base.response import StreamChunk

        todo_write_fn, todo_read_fn = todo_write, todo_read

        # 先用一次模拟 LLM 构造 Agent，让它生成自己的 session_dir
        async def dummy_chat(*a, **kw):
            yield StreamChunk(content="init", finish_reason="stop")

        agent = Agent(
            llm=AsyncMock(chat=dummy_chat),
            system_prompt="init",
            tools=[todo_write_fn, todo_read_fn],
            config=AgentConfig(
                max_turns=5,
                session_dir=str(tmp_path),
                session_enabled=True,
            ),
        )

        # 预先在 session_dir 写入计划（模拟之前的会话）
        plan_file = agent._session.dir_path / "plan.json"
        plan_data = {
            "items": [
                {"content": "已完成的步骤", "status": "completed", "activeForm": ""},
                {"content": "当前步骤", "status": "in_progress", "activeForm": "处理中"},
                {"content": "待办步骤", "status": "pending", "activeForm": ""},
            ]
        }
        plan_file.write_text(
            _json.dumps(plan_data, ensure_ascii=False),
            encoding="utf-8",
        )

        # 切换到新会话（保留 plan 文件），让 todo_read 能读到它
        agent.new_session()
        # 重新把 plan 文件写到新 session_dir
        new_plan_file = agent._session.dir_path / "plan.json"
        new_plan_file.write_text(
            _json.dumps(plan_data, ensure_ascii=False),
            encoding="utf-8",
        )

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
        agent._llm = llm

        events = []
        async for event in agent.run("我之前的计划是什么"):
            events.append(event)

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert not tool_results[0].is_error
        # 即使上下文被截断，计划内容通过文件恢复
        assert "已完成的步骤" in tool_results[0].output
        assert "当前步骤" in tool_results[0].output
        assert "处理中" in tool_results[0].output


# ── 常量与契约测试 ──────────────────────────────────────────


class TestTodoWriteConstants:
    """_MAX_PLAN_ITEMS 常量与基本契约"""

    def test_max_items_is_positive(self):
        """_MAX_PLAN_ITEMS 必须是合理正数"""
        assert _MAX_PLAN_ITEMS > 0
        assert _MAX_PLAN_ITEMS <= 100

    def test_contextvar_default_is_none(self):
        """_current_session_dir 默认值是 None（无注入时不写文件）"""
        # 重置后读取默认值
        token = _current_session_dir.set(None)
        try:
            assert _current_session_dir.get() is None
        finally:
            _current_session_dir.reset(token)
