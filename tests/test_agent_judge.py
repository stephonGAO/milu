"""测试 auto 模式 AI 安全判定器（judge_llm / judge_rules）

分层保证：
- 安全工具走 is_safe/safe_check 快路径，不经过判定器
- 不安全调用被批量判定：allow→执行 / confirm→转人工 / deny→拒绝
- 判定器失败一律 fail-open（回退为直接执行）
- 子代理经 ContextVar 继承判定器，委派不构成判定旁路
"""
import json
import pytest
from unittest.mock import AsyncMock

from agent_framework.agent import Agent, AgentConfig
from agent_framework.agent.config import AgentMode
from agent_framework.agent.events import ToolConfirmRequired, ToolResult
from agent_framework.agent.judge import JudgeVerdict, _parse_verdicts
from agent_framework.llm.base.response import StreamChunk, TokenUsage
from agent_framework.tools import tool


# ── 辅助工厂 ──────────────────────────────────────────────


def _make_mock_llm(tool_name: str, tool_args: str = "{}",
                   call_id: str = "call_1", final_text: str = "完成"):
    """主 LLM mock：第一轮调用指定工具，第二轮返回文本。"""
    call_count = 0

    async def mock_chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield StreamChunk(tool_calls=[
                type('obj', (), {
                    'index': 0, 'id': call_id,
                    'function': type('obj', (), {
                        'name': tool_name,
                        'arguments': tool_args,
                    })()
                })()
            ])
            yield StreamChunk(finish_reason="tool_calls")
        else:
            yield StreamChunk(content=final_text, finish_reason="stop")
            yield StreamChunk(usage=TokenUsage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15,
            ))

    llm = AsyncMock()
    llm.chat = mock_chat
    return llm


def _make_judge_llm(response: str):
    """判定 LLM mock：记录收到的 prompt，返回固定文本。"""
    received: list[str] = []

    async def chat(messages, **kwargs):
        received.append(messages[-1].content)
        yield StreamChunk(content=response)

    llm = AsyncMock()
    llm.chat = chat
    llm.received = received
    return llm


def _make_broken_judge_llm():
    """判定 LLM mock：调用即抛异常（验证 fail-open）。"""
    async def chat(messages, **kwargs):
        raise RuntimeError("judge 服务不可用")
        yield  # noqa: 使其成为 async generator

    llm = AsyncMock()
    llm.chat = chat
    return llm


def _verdict_json(call_id: str, decision: str, reason: str = "测试理由") -> str:
    return json.dumps({"verdicts": [
        {"id": call_id, "decision": decision, "reason": reason},
    ]}, ensure_ascii=False)


def _make_unsafe_tool():
    @tool(name="danger_op", description="危险操作", is_safe=False)
    async def danger_op() -> str:
        return "executed"
    return danger_op


def _make_safe_tool():
    @tool(name="read_data", description="安全读取", is_safe=True)
    async def read_data() -> str:
        return "data"
    return read_data


def _make_agent(llm, tools, judge_llm, mode=AgentMode.AUTO, **kwargs):
    return Agent(
        llm=llm, tools=tools, mode=mode, judge_llm=judge_llm,
        session_enabled=False, register_catalog=False, register_skills=False,
        **kwargs,
    )


async def _run(agent, prompt="执行任务"):
    events = []
    async for event in agent.run(prompt):
        events.append(event)
    return events


def _tool_results(events):
    return [e for e in events if isinstance(e, ToolResult)]


# ── _parse_verdicts 单元测试 ──────────────────────────────


class TestParseVerdicts:
    def test_plain_json(self):
        verdicts = _parse_verdicts(_verdict_json("c1", "deny", "危险"))
        assert verdicts == {"c1": JudgeVerdict(decision="deny", reason="危险")}

    def test_code_fence_stripped(self):
        raw = f"```json\n{_verdict_json('c1', 'allow')}\n```"
        verdicts = _parse_verdicts(raw)
        assert verdicts["c1"].decision == "allow"

    def test_invalid_decision_ignored(self):
        raw = json.dumps({"verdicts": [
            {"id": "c1", "decision": "maybe", "reason": "?"},
            {"id": "c2", "decision": "CONFIRM", "reason": "大小写归一"},
        ]})
        verdicts = _parse_verdicts(raw)
        assert "c1" not in verdicts          # 非法取值忽略（fail-open）
        assert verdicts["c2"].decision == "confirm"

    def test_invalid_json_raises(self):
        with pytest.raises(Exception):
            _parse_verdicts("这不是 JSON")


# ── auto 模式判定行为 ─────────────────────────────────────


class TestAutoJudge:
    @pytest.mark.asyncio
    async def test_allow_executes_without_confirm(self):
        """判定 allow → 直接执行，不触发人工确认"""
        judge = _make_judge_llm(_verdict_json("call_1", "allow"))
        confirm_called = False

        async def on_confirm(name, args):
            nonlocal confirm_called
            confirm_called = True
            return True

        agent = _make_agent(
            _make_mock_llm("danger_op"), [_make_unsafe_tool()], judge,
            on_confirm=on_confirm,
        )
        events = await _run(agent)

        assert len(judge.received) == 1, "判定器应被调用一次"
        assert confirm_called is False
        results = _tool_results(events)
        assert results[0].is_error is False
        assert results[0].output == "executed"

    @pytest.mark.asyncio
    async def test_deny_rejects_with_reason(self):
        """判定 deny → 拒绝执行，理由回传 LLM"""
        judge = _make_judge_llm(_verdict_json("call_1", "deny", "命中危险模式"))
        agent = _make_agent(
            _make_mock_llm("danger_op"), [_make_unsafe_tool()], judge,
        )
        events = await _run(agent)

        results = _tool_results(events)
        assert len(results) == 1
        assert results[0].is_error is True
        assert "安全判定" in results[0].output
        assert "命中危险模式" in results[0].output

    @pytest.mark.asyncio
    async def test_confirm_walks_on_confirm(self):
        """判定 confirm + 有回调 → 并入人工审批流程，批准后执行"""
        judge = _make_judge_llm(_verdict_json("call_1", "confirm"))
        confirm_called = False

        async def on_confirm(name, args):
            nonlocal confirm_called
            confirm_called = True
            return True

        agent = _make_agent(
            _make_mock_llm("danger_op"), [_make_unsafe_tool()], judge,
            on_confirm=on_confirm,
        )
        events = await _run(agent)

        assert confirm_called is True
        assert any(isinstance(e, ToolConfirmRequired) for e in events)
        results = _tool_results(events)
        assert results[0].is_error is False

    @pytest.mark.asyncio
    async def test_confirm_without_callback_rejected(self):
        """判定 confirm + 无回调 → 拒绝执行"""
        judge = _make_judge_llm(_verdict_json("call_1", "confirm"))
        agent = _make_agent(
            _make_mock_llm("danger_op"), [_make_unsafe_tool()], judge,
        )
        events = await _run(agent)

        results = _tool_results(events)
        assert results[0].is_error is True
        assert "确认回调" in results[0].output

    @pytest.mark.asyncio
    async def test_judge_failure_fails_open(self):
        """判定器异常 → fail-open 回退为直接执行"""
        agent = _make_agent(
            _make_mock_llm("danger_op"), [_make_unsafe_tool()],
            _make_broken_judge_llm(),
        )
        events = await _run(agent)

        results = _tool_results(events)
        assert results[0].is_error is False
        assert results[0].output == "executed"

    @pytest.mark.asyncio
    async def test_judge_garbage_output_fails_open(self):
        """判定器输出无法解析 → fail-open 回退为直接执行"""
        judge = _make_judge_llm("抱歉，我无法判断。")
        agent = _make_agent(
            _make_mock_llm("danger_op"), [_make_unsafe_tool()], judge,
        )
        events = await _run(agent)

        results = _tool_results(events)
        assert results[0].is_error is False

    @pytest.mark.asyncio
    async def test_missing_verdict_fails_open(self):
        """判定结果缺失该调用条目 → 按 allow 处理"""
        judge = _make_judge_llm(_verdict_json("other_id", "deny"))
        agent = _make_agent(
            _make_mock_llm("danger_op"), [_make_unsafe_tool()], judge,
        )
        events = await _run(agent)

        results = _tool_results(events)
        assert results[0].is_error is False

    @pytest.mark.asyncio
    async def test_safe_tool_not_judged(self):
        """安全工具走快路径，不调用判定器"""
        judge = _make_judge_llm(_verdict_json("call_1", "deny"))
        agent = _make_agent(
            _make_mock_llm("read_data"), [_make_safe_tool()], judge,
        )
        events = await _run(agent)

        assert judge.received == [], "安全工具不应触发判定"
        results = _tool_results(events)
        assert results[0].is_error is False


# ── 其他模式不启用判定 ────────────────────────────────────


class TestJudgeOnlyInAutoMode:
    @pytest.mark.asyncio
    async def test_manual_mode_no_judge(self):
        """manual 模式：人工审批所有不安全工具，判定器不参与"""
        judge = _make_judge_llm(_verdict_json("call_1", "deny"))
        confirm_called = False

        async def on_confirm(name, args):
            nonlocal confirm_called
            confirm_called = True
            return True

        agent = _make_agent(
            _make_mock_llm("danger_op"), [_make_unsafe_tool()], judge,
            mode=AgentMode.MANUAL, on_confirm=on_confirm,
        )
        events = await _run(agent)

        assert judge.received == []
        assert confirm_called is True
        assert _tool_results(events)[0].is_error is False

    @pytest.mark.asyncio
    async def test_superwork_mode_no_judge(self):
        """superwork 模式：跳过所有安全检查，判定器不参与"""
        judge = _make_judge_llm(_verdict_json("call_1", "deny"))
        agent = _make_agent(
            _make_mock_llm("danger_op"), [_make_unsafe_tool()], judge,
            mode=AgentMode.SUPERWORK,
        )
        events = await _run(agent)

        assert judge.received == []
        assert _tool_results(events)[0].is_error is False


# ── 判定 prompt 内容 ──────────────────────────────────────


class TestJudgePrompt:
    @pytest.mark.asyncio
    async def test_prompt_contains_user_input_and_rules(self):
        """判定 prompt 应包含用户请求、工具参数和自定义规则"""
        judge = _make_judge_llm(_verdict_json("call_1", "allow"))
        agent = _make_agent(
            _make_mock_llm("danger_op", tool_args='{"path": "/tmp/x"}'),
            [_make_unsafe_tool()], judge,
            judge_rules="禁止访问生产数据库",
        )
        await _run(agent, prompt="帮我清理临时文件")

        assert len(judge.received) == 1
        prompt = judge.received[0]
        assert "帮我清理临时文件" in prompt, "应包含用户请求"
        assert "danger_op" in prompt, "应包含工具名"
        assert "/tmp/x" in prompt, "应包含工具参数"
        assert "禁止访问生产数据库" in prompt, "应包含自定义规则"


# ── judge_llm 默认解析（None→主 llm / False→关闭 / 实例→指定）──


class TestJudgeDefaults:
    def _bare_agent(self, llm, **kwargs):
        return Agent(
            llm=llm, tools=[], session_enabled=False,
            register_catalog=False, register_skills=False, **kwargs,
        )

    def test_default_uses_main_llm(self):
        """judge_llm 不传（None）→ 默认复用主 llm"""
        llm = _make_mock_llm("danger_op")
        agent = self._bare_agent(llm)
        assert agent._judge_llm is llm

    def test_true_uses_main_llm(self):
        """judge_llm=True → 同默认，复用主 llm"""
        llm = _make_mock_llm("danger_op")
        agent = self._bare_agent(llm, judge_llm=True)
        assert agent._judge_llm is llm

    def test_false_disables(self):
        """judge_llm=False → 显式关闭判定器"""
        llm = _make_mock_llm("danger_op")
        agent = self._bare_agent(llm, judge_llm=False)
        assert agent._judge_llm is None

    def test_explicit_instance_used(self):
        """judge_llm=实例 → 使用指定判定模型"""
        llm = _make_mock_llm("danger_op")
        judge = _make_judge_llm(_verdict_json("call_1", "allow"))
        agent = self._bare_agent(llm, judge_llm=judge)
        assert agent._judge_llm is judge

    @pytest.mark.asyncio
    async def test_default_judge_fail_open_with_plain_llm(self):
        """默认判定器复用主 llm：输出非 JSON 时 fail-open 直接执行"""
        llm = _make_mock_llm("danger_op")  # 判定调用会拿到"完成"文本 → 解析失败
        agent = Agent(
            llm=llm, tools=[_make_unsafe_tool()],
            session_enabled=False, register_catalog=False, register_skills=False,
        )
        events = await _run(agent)

        results = _tool_results(events)
        assert results[0].is_error is False
        assert results[0].output == "executed"


# ── 子代理继承判定器 ──────────────────────────────────────


class TestSubAgentJudgeInheritance:
    @pytest.mark.asyncio
    async def test_subagent_inherits_judge(self):
        """auto 模式下，子代理内的不安全工具同样经过父的判定器（deny 生效）"""
        from agent_framework.agent.subagent import (
            SubAgentConfig, create_subagent_tools,
        )

        executed = []

        @tool(name="sub_danger", description="子代理危险操作", is_safe=False)
        async def sub_danger() -> str:
            executed.append(True)
            return "done"

        sub_llm = _make_mock_llm("sub_danger", call_id="call_sub")
        sub_tools = create_subagent_tools(
            llm=sub_llm,
            subagents=[SubAgentConfig(
                name="worker", description="测试子代理", tools=[sub_danger],
                config=AgentConfig(),
            )],
        )

        # 判定器：拒绝子代理内的调用（id 对应子代理的 tool_call）
        judge = _make_judge_llm(_verdict_json("call_sub", "deny", "子代理危险"))

        parent = _make_agent(
            _make_mock_llm("worker", tool_args=json.dumps({"task": "执行"}),
                           call_id="call_parent"),
            sub_tools, judge,
        )
        await _run(parent, prompt="委派任务")

        assert len(judge.received) == 1, "子代理应继承并调用判定器"
        assert not executed, "deny 后子代理内工具不应执行"

    @pytest.mark.asyncio
    async def test_subagent_disabled_when_parent_disabled(self):
        """父 judge_llm=False → 子代理判定器也关闭（不因默认规则用自身 llm 重新启用）"""
        from agent_framework.agent.subagent import (
            SubAgentConfig, create_subagent_tools,
        )

        @tool(name="sub_danger2", description="子代理危险操作", is_safe=False)
        async def sub_danger() -> str:
            return "done"

        # 计数的子代理 LLM：判定器若被启用会多消耗一次 chat 调用
        chat_calls = []

        async def sub_chat(*args, **kwargs):
            chat_calls.append(1)
            if len(chat_calls) == 1:
                yield StreamChunk(tool_calls=[
                    type('obj', (), {
                        'index': 0, 'id': 'call_sub2',
                        'function': type('obj', (), {
                            'name': 'sub_danger2', 'arguments': '{}',
                        })()
                    })()
                ])
                yield StreamChunk(finish_reason="tool_calls")
            else:
                yield StreamChunk(content="子代理完成", finish_reason="stop")

        sub_llm = AsyncMock()
        sub_llm.chat = sub_chat

        sub_tools = create_subagent_tools(
            llm=sub_llm,
            subagents=[SubAgentConfig(
                name="worker2", description="测试子代理", tools=[sub_danger],
                config=AgentConfig(),
            )],
        )

        parent = _make_agent(
            _make_mock_llm("worker2", tool_args=json.dumps({"task": "执行"}),
                           call_id="call_parent2"),
            sub_tools, judge_llm=False,
        )
        await _run(parent, prompt="委派任务")

        # 判定关闭：子代理只有 工具调用轮 + 收尾轮 两次 chat（启用判定则为 3 次）
        assert len(chat_calls) == 2, f"判定应关闭，实际 chat 调用 {len(chat_calls)} 次"
