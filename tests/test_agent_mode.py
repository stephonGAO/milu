"""测试 Agent 操作模式（talk / manual / auto / superwork）"""
import json
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from milu.agent import Agent, AgentConfig
from milu.agent.config import AgentMode
from milu.agent.events import (
    ToolResult, AgentDone,
)
from milu.llm.base.response import StreamChunk, TokenUsage
from milu.tools import tool


# ── 辅助工厂 ──────────────────────────────────────────────


def _make_mock_llm(tool_name: str, tool_args: str = "{}",
                     final_text: str = "完成"):
    """创建模拟 LLM：第一轮调用指定工具，第二轮返回文本。"""
    call_count = 0

    async def mock_chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield StreamChunk(tool_calls=[
                type('obj', (), {
                    'index': 0, 'id': f'call_{call_count}',
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


def _make_safe_tool():
    @tool(name="read_data", description="安全数据", is_safe=True)
    async def read_data() -> str:
        return "data"
    return read_data


def _make_write_tool():
    @tool(name="write_data", description="写入数据", is_safe=False)
    async def write_data() -> str:
        return "written"
    return write_data


def _make_unsafe_tool():
    @tool(name="danger_op", description="危险操作", is_safe=False)
    async def danger_op() -> str:
        return "executed"
    return danger_op


def _make_mixed_tool():
    # 动态安全工具：is_safe=False 才会让 _is_safe_call 去咨询 safe_check
    # （与内置 shell_command 的 is_safe=False + safe_check 模式一致）
    @tool(
        name="file_op",
        description="混合工具",
        is_safe=False,
        safe_check=lambda args: args.get("action", "") in {"read", "index"},
    )
    async def file_op(action: str = "read") -> str:
        return f"file:{action}"
    return file_op


# ── AgentMode 枚举 ────────────────────────────────────────


class TestAgentMode:
    def test_default_mode_is_auto(self):
        agent = Agent(
            llm=_make_mock_llm("read_data"),
            session_enabled=False,
            register_catalog=False, register_skills=False,
        )
        assert agent.mode == AgentMode.AUTO

    def test_mode_from_string(self):
        assert AgentMode("talk") == AgentMode.TALK
        assert AgentMode("manual") == AgentMode.MANUAL
        assert AgentMode("auto") == AgentMode.AUTO
        assert AgentMode("superwork") == AgentMode.SUPERWORK

    def test_mode_invalid_string(self):
        with pytest.raises(ValueError):
            AgentMode("invalid")


# ── set_mode ──────────────────────────────────────────────


class TestSetMode:
    def test_set_mode_string(self):
        llm = _make_mock_llm("read_data")
        agent = Agent(
            llm=llm, tools=[_make_safe_tool()],
            session_enabled=False,
        )
        assert agent.mode == AgentMode.AUTO

        agent.set_mode("talk")
        assert agent.mode == AgentMode.TALK

        agent.set_mode("superwork")
        assert agent.mode == AgentMode.SUPERWORK

    def test_set_mode_enum(self):
        llm = _make_mock_llm("read_data")
        agent = Agent(
            llm=llm, tools=[_make_safe_tool()],
            session_enabled=False,
        )

        agent.set_mode(AgentMode.TALK)
        assert agent.mode == AgentMode.TALK

    def test_set_mode_invalid(self):
        llm = _make_mock_llm("read_data")
        agent = Agent(
            llm=llm, tools=[_make_safe_tool()],
            session_enabled=False,
        )

        with pytest.raises(ValueError):
            agent.set_mode("bogus")


# ── Talk 模式 ─────────────────────────────────────────────


class TestTalkMode:
    @pytest.mark.asyncio
    async def test_talk_blocks_write_tool(self):
        """talk 模式应阻止非安全工具"""
        llm = _make_mock_llm("write_data")
        agent = Agent(
            llm=llm, tools=[_make_write_tool()],
            mode=AgentMode.TALK, session_enabled=False,
        )

        events = []
        async for event in agent.run("写入数据"):
            events.append(event)

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert tool_results[0].is_error is True
        assert "talk 模式" in tool_results[0].output

    @pytest.mark.asyncio
    async def test_talk_allows_safe_tool(self):
        """talk 模式应允许安全工具"""
        llm = _make_mock_llm("read_data")
        agent = Agent(
            llm=llm, tools=[_make_safe_tool()],
            mode=AgentMode.TALK, session_enabled=False,
        )

        events = []
        async for event in agent.run("读取数据"):
            events.append(event)

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert tool_results[0].is_error is False
        assert tool_results[0].output == "data"

        # 应正常完成
        assert any(isinstance(e, AgentDone) for e in events)

    @pytest.mark.asyncio
    async def test_talk_allows_safe_action_on_mixed_tool(self):
        """talk 模式应允许混合工具的 safe_check 通过 action"""
        args = json.dumps({"action": "read"})
        llm = _make_mock_llm("file_op", tool_args=args)
        agent = Agent(
            llm=llm, tools=[_make_mixed_tool()],
            mode=AgentMode.TALK, session_enabled=False,
        )

        events = []
        async for event in agent.run("读文件"):
            events.append(event)

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert tool_results[0].is_error is False
        assert "file:read" in tool_results[0].output

    @pytest.mark.asyncio
    async def test_talk_blocks_unsafe_action_on_mixed_tool(self):
        """talk 模式应阻止混合工具的写 action"""
        args = json.dumps({"action": "write"})
        llm = _make_mock_llm("file_op", tool_args=args)
        agent = Agent(
            llm=llm, tools=[_make_mixed_tool()],
            mode=AgentMode.TALK, session_enabled=False,
        )

        events = []
        async for event in agent.run("写文件"):
            events.append(event)

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert tool_results[0].is_error is True
        assert "talk 模式" in tool_results[0].output

    @pytest.mark.asyncio
    async def test_talk_blocks_unsafe_tool(self):
        """talk 模式应阻止非安全工具（即使有 on_confirm）"""
        llm = _make_mock_llm("danger_op")
        confirm_called = False

        async def mock_confirm(name, args):
            nonlocal confirm_called
            confirm_called = True
            return True

        agent = Agent(
            llm=llm, tools=[_make_unsafe_tool()],
            mode=AgentMode.TALK, session_enabled=False,
            on_confirm=mock_confirm,
        )

        events = []
        async for event in agent.run("执行危险操作"):
            events.append(event)

        # talk 模式拦截在确认之前，不应调用 confirm
        assert confirm_called is False

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert tool_results[0].is_error is True
        assert "talk 模式" in tool_results[0].output

    @pytest.mark.asyncio
    async def test_talk_partial_block(self):
        """talk 模式下混合批次：安全工具应执行，写入工具应被阻止"""
        call_count = 0

        async def mock_chat(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # 同时调用两个工具
                yield StreamChunk(tool_calls=[
                    type('obj', (), {
                        'index': 0, 'id': 'call_r',
                        'function': type('obj', (), {
                            'name': 'read_data', 'arguments': '{}',
                        })()
                    })(),
                    type('obj', (), {
                        'index': 1, 'id': 'call_w',
                        'function': type('obj', (), {
                            'name': 'write_data', 'arguments': '{}',
                        })()
                    })(),
                ])
                yield StreamChunk(finish_reason="tool_calls")
            else:
                yield StreamChunk(content="完成", finish_reason="stop")

        llm = AsyncMock()
        llm.chat = mock_chat

        agent = Agent(
            llm=llm, tools=[_make_safe_tool(), _make_write_tool()],
            mode=AgentMode.TALK, session_enabled=False,
        )

        events = []
        async for event in agent.run("混合调用"):
            events.append(event)

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        # read_data 应成功，write_data 应被阻止
        read_result = next(r for r in tool_results if r.tool_name == "read_data")
        write_result = next(r for r in tool_results if r.tool_name == "write_data")

        assert read_result.is_error is False
        assert write_result.is_error is True
        assert "talk 模式" in write_result.output


# ── Superwork 模式 ────────────────────────────────────────


class TestSuperworkMode:
    @pytest.mark.asyncio
    async def test_superwork_skips_confirm(self):
        """superwork 模式应跳过非安全工具确认"""
        llm = _make_mock_llm("danger_op")
        confirm_called = False

        async def mock_confirm(name, args):
            nonlocal confirm_called
            confirm_called = True
            return True

        agent = Agent(
            llm=llm, tools=[_make_unsafe_tool()],
            mode=AgentMode.SUPERWORK,
            session_enabled=False,
            on_confirm=mock_confirm,
        )

        events = []
        async for event in agent.run("执行"):
            events.append(event)

        # 不应调用 confirm
        assert confirm_called is False

        # 工具应正常执行
        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert tool_results[0].is_error is False
        assert tool_results[0].output == "executed"


# ── Manual 模式（人工审批，原 auto 行为）──────────────────


class TestManualMode:
    @pytest.mark.asyncio
    async def test_manual_unsafe_triggers_confirm(self):
        """manual 模式下非安全工具应触发确认"""
        llm = _make_mock_llm("danger_op")
        confirm_called = False

        async def mock_confirm(name, args):
            nonlocal confirm_called
            confirm_called = True
            return True

        agent = Agent(
            llm=llm, tools=[_make_unsafe_tool()],
            mode=AgentMode.MANUAL,
            session_enabled=False,
            on_confirm=mock_confirm,
        )

        events = []
        async for event in agent.run("执行"):
            events.append(event)

        assert confirm_called is True

        # 同意后工具应正常执行
        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert tool_results[0].is_error is False

    @pytest.mark.asyncio
    async def test_manual_no_confirm_callback_executes(self):
        """manual 模式无确认回调时维持原行为：普通不安全工具直接执行"""
        llm = _make_mock_llm("danger_op")
        agent = Agent(
            llm=llm, tools=[_make_unsafe_tool()],
            mode=AgentMode.MANUAL,
            session_enabled=False,
        )

        events = []
        async for event in agent.run("执行"):
            events.append(event)

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert tool_results[0].is_error is False
        assert tool_results[0].output == "executed"


# ── Auto 模式（自主决策，默认）────────────────────────────


class TestAutoMode:
    @pytest.mark.asyncio
    async def test_auto_unsafe_executes_without_confirm(self):
        """auto 模式下非安全工具应自主执行，不触发确认"""
        llm = _make_mock_llm("danger_op")
        confirm_called = False

        async def mock_confirm(name, args):
            nonlocal confirm_called
            confirm_called = True
            return True

        agent = Agent(
            llm=llm, tools=[_make_unsafe_tool()],
            mode=AgentMode.AUTO,
            session_enabled=False,
            on_confirm=mock_confirm,
        )

        events = []
        async for event in agent.run("执行"):
            events.append(event)

        # auto 模式自主决策：不应调用 confirm
        assert confirm_called is False

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert tool_results[0].is_error is False
        assert tool_results[0].output == "executed"

    @pytest.mark.asyncio
    async def test_auto_allows_all_tools(self):
        """auto 模式应允许所有工具"""
        call_count = 0

        async def mock_chat(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield StreamChunk(tool_calls=[
                    type('obj', (), {
                        'index': 0, 'id': 'call_1',
                        'function': type('obj', (), {
                            'name': 'write_data', 'arguments': '{}',
                        })()
                    })()
                ])
                yield StreamChunk(finish_reason="tool_calls")
            else:
                yield StreamChunk(content="完成", finish_reason="stop")

        llm = AsyncMock()
        llm.chat = mock_chat

        agent = Agent(
            llm=llm, tools=[_make_safe_tool(), _make_write_tool()],
            mode=AgentMode.AUTO, session_enabled=False,
        )

        events = []
        async for event in agent.run("写入"):
            events.append(event)

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert tool_results[0].is_error is False
        assert tool_results[0].output == "written"


# ── SubAgent 模式继承 ─────────────────────────────────────


class TestSubAgentModeInheritance:
    @pytest.mark.asyncio
    async def test_subagent_inherits_parent_talk_mode(self):
        """子代理应继承父 Agent 的 talk 模式"""
        from milu.agent.subagent import create_subagent_tools, SubAgentConfig

        write_tool_called = False

        @tool(name="sub_write", description="写入", is_safe=False)
        async def sub_write() -> str:
            nonlocal write_tool_called
            write_tool_called = True
            return "written"

        # 子代理 LLM：调用 sub_write 然后返回文本
        sub_call_count = 0

        async def sub_mock_chat(*args, **kwargs):
            nonlocal sub_call_count
            sub_call_count += 1
            if sub_call_count == 1:
                yield StreamChunk(tool_calls=[
                    type('obj', (), {
                        'index': 0, 'id': 'call_sub',
                        'function': type('obj', (), {
                            'name': 'sub_write', 'arguments': '{}',
                        })()
                    })()
                ])
                yield StreamChunk(finish_reason="tool_calls")
            else:
                yield StreamChunk(content="子代理完成", finish_reason="stop")

        sub_llm = AsyncMock()
        sub_llm.chat = sub_mock_chat

        subagent_tools = create_subagent_tools(
            llm=sub_llm,
            subagents=[SubAgentConfig(
                name="test_sub",
                description="测试子代理",
                tools=[sub_write],
                config=AgentConfig(),
            )],
            get_parent_mode=lambda: AgentMode.TALK,
        )

        # 父 LLM：调用子代理
        parent_call_count = 0

        async def parent_mock_chat(*args, **kwargs):
            nonlocal parent_call_count
            parent_call_count += 1
            if parent_call_count == 1:
                yield StreamChunk(tool_calls=[
                    type('obj', (), {
                        'index': 0, 'id': 'call_parent',
                        'function': type('obj', (), {
                            'name': 'test_sub',
                            'arguments': json.dumps({"task": "测试"}),
                        })()
                    })()
                ])
                yield StreamChunk(finish_reason="tool_calls")
            else:
                yield StreamChunk(content="父代理完成", finish_reason="stop")

        parent_llm = AsyncMock()
        parent_llm.chat = parent_mock_chat

        agent = Agent(
            llm=parent_llm,
            tools=subagent_tools,
            mode=AgentMode.TALK,
            session_enabled=False,
        )

        events = []
        async for event in agent.run("委派任务"):
            events.append(event)

        # 子代理继承了 talk 模式，sub_write 被阻止
        # 所以 write_tool_called 应为 False
        assert write_tool_called is False

    @pytest.mark.asyncio
    async def test_subagent_inherits_mode_via_contextvar(self):
        """不传 get_parent_mode 时，子代理通过 ContextVar 继承父 Agent 的 talk 模式。

        验证 Phase A.2：Agent.run() 注入 _current_parent_mode，子代理工具读取它，
        无需「先建 Agent 再回填 get_parent_mode 闭包」。
        """
        from milu.agent.subagent import create_subagent_tools, SubAgentConfig

        write_tool_called = False

        @tool(name="sub_write2", description="写入", is_safe=False)
        async def sub_write() -> str:
            nonlocal write_tool_called
            write_tool_called = True
            return "written"

        sub_call_count = 0

        async def sub_mock_chat(*args, **kwargs):
            nonlocal sub_call_count
            sub_call_count += 1
            if sub_call_count == 1:
                yield StreamChunk(tool_calls=[
                    type('obj', (), {
                        'index': 0, 'id': 'call_sub',
                        'function': type('obj', (), {
                            'name': 'sub_write2', 'arguments': '{}',
                        })()
                    })()
                ])
                yield StreamChunk(finish_reason="tool_calls")
            else:
                yield StreamChunk(content="子代理完成", finish_reason="stop")

        sub_llm = AsyncMock()
        sub_llm.chat = sub_mock_chat

        # 关键：不传 get_parent_mode，子代理工具可在 Agent 之前创建
        subagent_tools = create_subagent_tools(
            llm=sub_llm,
            subagents=[SubAgentConfig(
                name="test_sub2",
                description="测试子代理",
                tools=[sub_write],
                config=AgentConfig(),
            )],
        )

        parent_call_count = 0

        async def parent_mock_chat(*args, **kwargs):
            nonlocal parent_call_count
            parent_call_count += 1
            if parent_call_count == 1:
                yield StreamChunk(tool_calls=[
                    type('obj', (), {
                        'index': 0, 'id': 'call_parent',
                        'function': type('obj', (), {
                            'name': 'test_sub2',
                            'arguments': json.dumps({"task": "测试"}),
                        })()
                    })()
                ])
                yield StreamChunk(finish_reason="tool_calls")
            else:
                yield StreamChunk(content="父代理完成", finish_reason="stop")

        parent_llm = AsyncMock()
        parent_llm.chat = parent_mock_chat

        agent = Agent(
            llm=parent_llm,
            tools=subagent_tools,
            mode=AgentMode.TALK,
            session_enabled=False,
        )

        async for _ in agent.run("委派任务"):
            pass

        # 子代理经 ContextVar 继承 talk 模式，sub_write 被阻止
        assert write_tool_called is False


# ── _is_safe_call 辅助方法 ────────────────────────────────


class TestIsSafeCall:
    def test_is_safe_true(self):
        """is_safe=True 的工具应判定为安全"""
        wrapper = SimpleNamespace(is_safe=True, safe_check=None)
        assert Agent._is_safe_call(wrapper, "{}") is True

    def test_not_safe(self):
        """is_safe=False 且 safe_check=None 应判定为非安全"""
        wrapper = SimpleNamespace(is_safe=False, safe_check=None)
        assert Agent._is_safe_call(wrapper, "{}") is False

    def test_safe_check_pass(self):
        """safe_check 返回 True 应判定为安全"""
        wrapper = SimpleNamespace(
            is_safe=False,
            safe_check=lambda args: args.get("cmd") == "ls",
        )
        args = json.dumps({"cmd": "ls"})
        assert Agent._is_safe_call(wrapper, args) is True

    def test_safe_check_fail(self):
        """safe_check 返回 False 应判定为非安全"""
        wrapper = SimpleNamespace(
            is_safe=False,
            safe_check=lambda args: args.get("cmd") == "ls",
        )
        args = json.dumps({"cmd": "rm"})
        assert Agent._is_safe_call(wrapper, args) is False

    def test_safe_check_invalid_json(self):
        """无效 JSON 参数应判定为非安全"""
        wrapper = SimpleNamespace(
            is_safe=False,
            safe_check=lambda args: True,
        )
        assert Agent._is_safe_call(wrapper, "invalid{") is False

    def test_safe_check_exception(self):
        """safe_check 抛出异常应判定为非安全"""
        def bad_check(args):
            raise RuntimeError("检查失败")

        wrapper = SimpleNamespace(is_safe=False, safe_check=bad_check)
        args = json.dumps({"cmd": "ls"})
        assert Agent._is_safe_call(wrapper, args) is False


# ── Shell 白名单 ─────────────────────────────────────────


class TestShellSafe:
    """测试 shell_command 的安全白名单检查"""

    def test_safe_commands_allowed(self):
        """白名单中的命令应判定为安全"""
        from milu.tools.builtin.shell_command import _is_safe_shell

        allowed = ["ls", "cat file.txt", "grep pattern file", "curl http://example.com",
                   "git status", "git log", "ps aux", "df -h", "du -sh", "pwd",
                   "head -n 10 file", "tail -f log.txt", "wc -l file",
                   "find . -name '*.py'", "sort file", "uniq file"]
        for cmd in allowed:
            assert _is_safe_shell({"command": cmd}) is True, f"should allow: {cmd}"

    def test_non_safe_commands_blocked(self):
        """不在白名单中的命令应判定为非安全"""
        from milu.tools.builtin.shell_command import _is_safe_shell

        blocked = ["rm file", "cp src dst", "mv old new", "mkdir dir",
                   "chmod 755 file", "chown user file", "pip install pkg",
                   "npm install", "apt install pkg", "python script.py"]
        for cmd in blocked:
            assert _is_safe_shell({"command": cmd}) is False, f"should block: {cmd}"

    def test_redirect_blocked(self):
        """包含重定向的命令应判定为非安全"""
        from milu.tools.builtin.shell_command import _is_safe_shell

        blocked = [
            "echo hello > file.txt",
            "cat input >> log.txt",
            "ls | tee output.txt",
        ]
        for cmd in blocked:
            assert _is_safe_shell({"command": cmd}) is False, f"should block: {cmd}"

    def test_fd_redirect_not_treated_as_write(self):
        """fd 复制（2>&1 / 1>&2）与丢弃到 /dev/null 是只读，不应被当成写文件。

        回归：旧正则 `>\\s*\\S` 把 `2>&1` 的 `>&` 误判为写重定向，导致只读的
        `ls ... 2>&1` 被判非安全（进而误触发已删除的 todo 守卫、talk 模式误拦）。
        """
        from milu.tools.builtin.shell_command import _is_safe_shell

        allowed = [
            "ls -la /tmp 2>&1 | head -50",
            "ls -la 2>/dev/null",
            "cat a.txt 1>&2",
            "grep foo bar.txt 2>&1",
        ]
        for cmd in allowed:
            assert _is_safe_shell({"command": cmd}) is True, f"should allow: {cmd}"

    def test_path_prefix_stripped(self):
        """带路径前缀的命令应正确提取基础命令名"""
        from milu.tools.builtin.shell_command import _is_safe_shell

        assert _is_safe_shell({"command": "/usr/bin/ls -la"}) is True
        assert _is_safe_shell({"command": "/bin/cat file"}) is True
        assert _is_safe_shell({"command": "/usr/local/bin/git status"}) is True

    def test_empty_command(self):
        """空命令应判定为非安全"""
        from milu.tools.builtin.shell_command import _is_safe_shell

        assert _is_safe_shell({"command": ""}) is False
        assert _is_safe_shell({}) is False


# ── Python REPL ───────────────────────────────────────────


class TestPythonReplSafe:
    """测试 python_repl 在 talk 模式下的行为"""

    @pytest.mark.asyncio
    async def test_talk_allows_python_repl(self):
        """talk 模式应允许 python_repl（沙箱执行）"""
        from milu.tools.builtin.python_repl import python_repl

        args = json.dumps({"code": "print(1+1)"})
        llm = _make_mock_llm("python_repl", tool_args=args)
        agent = Agent(
            llm=llm, tools=[python_repl],
            mode=AgentMode.TALK, session_enabled=False,
        )

        events = []
        async for event in agent.run("计算 1+1"):
            events.append(event)

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert tool_results[0].is_error is False
        assert "2" in tool_results[0].output


class TestShellCommandTalkMode:
    """测试 shell_command 在 talk 模式下的行为"""

    @pytest.mark.asyncio
    async def test_talk_allows_safe_shell(self):
        """talk 模式应允许白名单中的 shell 命令"""
        from milu.tools.builtin.shell_command import shell_command

        args = json.dumps({"command": "echo hello"})
        llm = _make_mock_llm("shell_command", tool_args=args)
        agent = Agent(
            llm=llm, tools=[shell_command],
            mode=AgentMode.TALK, session_enabled=False,
        )

        events = []
        async for event in agent.run("运行 echo"):
            events.append(event)

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert tool_results[0].is_error is False

    @pytest.mark.asyncio
    async def test_talk_skips_confirm_for_safe(self):
        """talk 模式下安全工具不应触发确认"""
        from milu.tools.builtin.shell_command import shell_command

        args = json.dumps({"command": "ls -la workspace/"})
        llm = _make_mock_llm("shell_command", tool_args=args)
        confirm_called = False

        async def mock_confirm(name, args_str):
            nonlocal confirm_called
            confirm_called = True
            return True

        agent = Agent(
            llm=llm, tools=[shell_command],
            mode=AgentMode.TALK,
            session_enabled=False,
            on_confirm=mock_confirm,
        )

        events = []
        async for event in agent.run("查看目录"):
            events.append(event)

        # talk 模式已验证安全，不应再弹确认
        assert confirm_called is False

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert tool_results[0].is_error is False

    @pytest.mark.asyncio
    async def test_talk_blocks_non_safe_shell(self):
        """talk 模式应阻止非白名单的 shell 命令"""
        from milu.tools.builtin.shell_command import shell_command

        args = json.dumps({"command": "rm -rf /tmp/test"})
        llm = _make_mock_llm("shell_command", tool_args=args)
        agent = Agent(
            llm=llm, tools=[shell_command],
            mode=AgentMode.TALK, session_enabled=False,
        )

        events = []
        async for event in agent.run("删除文件"):
            events.append(event)

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert tool_results[0].is_error is True
        assert "talk 模式" in tool_results[0].output
