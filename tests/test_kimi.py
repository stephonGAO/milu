"""测试 KimiLLM - 月之暗面Moonshot Kimi模型实现"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import MockChunk, MockChoice, MockDelta, MockUsage


class TestKimiCapabilities:
    """验证 KimiLLM 的能力标志是否与规格一致"""

    def test_capabilities(self):
        from milu.llm.providers.kimi import KimiLLM

        llm = KimiLLM(api_key="test", model="kimi-k2.6")
        caps = llm.capabilities

        # 基础能力
        assert caps.supports_streaming is True
        assert caps.supports_function_calling is True
        assert caps.supports_json_mode is True
        assert caps.supports_web_search is True
        assert caps.supports_thinking is True
        assert caps.supports_embedding is False

        # 多模态理解能力
        assert caps.supports_vision is True
        assert caps.supports_audio_understand is False
        assert caps.supports_video is False
        assert caps.supports_document is True

        # 多模态生成能力
        assert caps.supports_image_generation is False
        assert caps.supports_audio_generation is False

        # 模型规格：按模型解析——kimi-k2.x 当前系列均为 256K
        assert caps.max_context_window == 262144
        assert caps.supported_output_formats == ("text", "json")


class TestKimiAvailableParams:
    """验证 KimiLLM 支持的参数名集合"""

    def test_available_params_include_basic(self):
        from milu.llm.providers.kimi import KimiLLM

        llm = KimiLLM(api_key="test", model="moonshot-v1-8k")
        params = llm._get_available_param_names()

        # 基础参数
        assert "temperature" in params
        assert "top_p" in params
        assert "max_tokens" in params
        assert "stop" in params

    def test_available_params_include_web_search(self):
        from milu.llm.providers.kimi import KimiLLM

        llm = KimiLLM(api_key="test", model="moonshot-v1-8k")
        params = llm._get_available_param_names()

        assert "web_search" in params
        assert "web_search_strategy" in params

    def test_available_params_include_thinking(self):
        from milu.llm.providers.kimi import KimiLLM

        llm = KimiLLM(api_key="test", model="moonshot-v1-8k")
        params = llm._get_available_param_names()

        assert "enable_thinking" in params
        assert "thinking_level" in params

    def test_available_params_include_tools(self):
        from milu.llm.providers.kimi import KimiLLM

        llm = KimiLLM(api_key="test", model="moonshot-v1-8k")
        params = llm._get_available_param_names()

        assert "tools" in params
        assert "tool_choice" in params

    def test_no_image_generation_param(self):
        """Kimi不支持图片生成，相关参数不应存在"""
        from milu.llm.providers.kimi import KimiLLM

        llm = KimiLLM(api_key="test", model="moonshot-v1-8k")
        available = llm.get_available_params()

        assert "image_size" not in available
        assert "image_quality" not in available
        assert "num_images" not in available


class TestKimiChat:
    """测试 KimiLLM 流式聊天功能"""

    @pytest.mark.asyncio
    async def test_basic_chat(self):
        """基础聊天：3个chunk（内容"你"+"好"+结束带usage），验证StreamChunk字段"""
        from milu.llm.base.message import Message, MessageRole
        from milu.llm.providers.kimi import KimiLLM

        # 构造模拟的流式响应：3个chunk
        chunks = [
            # chunk 1: 内容 "你"
            MockChunk(
                choices=[MockChoice(delta=MockDelta(content="你", role="assistant"))]
            ),
            # chunk 2: 内容 "好"
            MockChunk(
                choices=[MockChoice(delta=MockDelta(content="好"))]
            ),
            # chunk 3: 结束，带usage
            MockChunk(
                choices=[MockChoice(delta=MockDelta(), finish_reason="stop")],
                usage=MockUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            ),
        ]

        # 将chunks包装成异步迭代器
        async def mock_stream():
            for chunk in chunks:
                yield chunk

        llm = KimiLLM(api_key="test-key", model="moonshot-v1-8k")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        with patch.object(llm, "_get_client", return_value=mock_client):
            messages = [Message(role=MessageRole.USER, content="你好")]
            results = []
            async for chunk in llm.chat(messages):
                results.append(chunk)

        # 验证返回了3个 StreamChunk
        assert len(results) == 3

        # chunk 1: 内容 "你"
        assert results[0].content == "你"
        assert results[0].finish_reason is None

        # chunk 2: 内容 "好"
        assert results[1].content == "好"
        assert results[1].finish_reason is None

        # chunk 3: 结束，带usage
        assert results[2].content is None
        assert results[2].finish_reason == "stop"
        assert results[2].usage is not None
        assert results[2].usage.prompt_tokens == 10
        assert results[2].usage.completion_tokens == 5
        assert results[2].usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_thinking_toggle_uses_thinking_type(self):
        """思考开关：走 Kimi 官方 thinking.type（enabled/disabled），
        不再用 reasoning_effort（"none" 会被 k2.6 等思考模型 400 拒绝）。
        thinking_level 无深度档位，应被静默忽略。
        """
        from milu.llm.base.message import Message, MessageRole
        from milu.llm.providers.kimi import KimiLLM

        def _mock_stream():
            async def gen():
                yield MockChunk(choices=[MockChoice(delta=MockDelta(content="ok"))])
                yield MockChunk(
                    choices=[MockChoice(delta=MockDelta(), finish_reason="stop")],
                    usage=MockUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
                )
            return gen()

        # ① 开启思考
        llm = KimiLLM(api_key="test-key", model="kimi-k2.6")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_mock_stream())
        with patch.object(llm, "_get_client", return_value=mock_client):
            messages = [Message(role=MessageRole.USER, content="请思考一下")]
            async for _ in llm.chat(messages, enable_thinking=True, thinking_level="high"):
                pass
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["extra_body"]["thinking"] == {"type": "enabled"}
        # thinking_level 不应映射成 reasoning_effort
        assert "reasoning_effort" not in call_kwargs["extra_body"]

        # ② 关闭思考 → disabled（绝不能是 reasoning_effort="none"）
        llm2 = KimiLLM(api_key="test-key", model="kimi-k2.6")
        mock_client2 = AsyncMock()
        mock_client2.chat.completions.create = AsyncMock(return_value=_mock_stream())
        with patch.object(llm2, "_get_client", return_value=mock_client2):
            messages = [Message(role=MessageRole.USER, content="别想太多直接答")]
            async for _ in llm2.chat(messages, enable_thinking=False):
                pass
        call_kwargs2 = mock_client2.chat.completions.create.call_args.kwargs
        assert call_kwargs2["extra_body"]["thinking"] == {"type": "disabled"}
        assert "reasoning_effort" not in call_kwargs2["extra_body"]

    @pytest.mark.asyncio
    async def test_reasoning_content_echoed_on_tool_call_message(self):
        """thinking 模型回传约束：带 tool_calls 的 assistant 历史消息必须携带
        reasoning_content，否则 Kimi 报 "thinking is enabled but reasoning_content
        is missing in assistant tool call message"。验证 _messages_to_dicts 会把
        Message.reasoning_content 注入到带 tool_calls 的 assistant 消息上，
        而普通（无 tool_calls）assistant 消息不注入、tool 消息也不受影响。
        """
        from milu.llm.base.message import Message, MessageRole
        from milu.llm.providers.kimi import KimiLLM

        llm = KimiLLM(api_key="test-key", model="kimi-k2-thinking")
        messages = [
            Message(role=MessageRole.USER, content="帮我建个计划"),
            # 带 tool_calls + reasoning_content 的 assistant 消息（需回传真实 reasoning）
            Message(
                role=MessageRole.ASSISTANT,
                content=None,
                tool_calls=[{
                    "id": "call_1", "type": "function",
                    "function": {"name": "todo_write", "arguments": "{}"},
                }],
                reasoning_content="我先列个待办计划……",
            ),
            Message(
                role=MessageRole.TOOL, content="ok",
                tool_call_id="call_1", name="todo_write",
            ),
            # 带 tool_calls 但【无】reasoning_content（模型直接发工具调用）→ 占位兜底
            Message(
                role=MessageRole.ASSISTANT,
                content=None,
                tool_calls=[{
                    "id": "call_2", "type": "function",
                    "function": {"name": "web_search", "arguments": "{}"},
                }],
                reasoning_content=None,
            ),
            Message(
                role=MessageRole.TOOL, content="结果",
                tool_call_id="call_2", name="web_search",
            ),
            # 普通终结回答：有 reasoning 但无 tool_calls，不应注入 reasoning_content
            Message(
                role=MessageRole.ASSISTANT,
                content="计划已建好。",
                reasoning_content="思考过程……",
            ),
        ]

        dicts = llm._messages_to_dicts(messages)

        # 带 tool_calls 的 assistant 消息被注入【真实】reasoning_content
        assert dicts[1]["role"] == "assistant" and dicts[1]["tool_calls"]
        assert dicts[1]["reasoning_content"] == "我先列个待办计划……"
        # tool 消息不含 reasoning_content
        assert "reasoning_content" not in dicts[2]
        # 带 tool_calls 但无 reasoning 的消息：注入【非空】占位，避免 400 missing
        assert dicts[3]["tool_calls"]
        assert dicts[3].get("reasoning_content")  # 存在且非空
        # tool 消息不含 reasoning_content
        assert "reasoning_content" not in dicts[4]
        # 普通 assistant 终结回答不注入 reasoning_content（避免无谓膨胀）
        assert "reasoning_content" not in dicts[5]
