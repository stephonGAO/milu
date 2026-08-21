"""测试 DeepSeekLLM - 深度求索模型实现"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import MockChunk, MockChoice, MockDelta, MockUsage


class TestDeepSeekCapabilities:
    """验证 DeepSeekLLM 的能力标志是否与规格一致"""

    def test_capabilities(self):
        from milu.llm.providers.deepseek import DeepSeekLLM

        llm = DeepSeekLLM(api_key="test", model="deepseek-chat")
        caps = llm.capabilities

        # 基础能力
        assert caps.supports_streaming is True
        assert caps.supports_function_calling is True
        assert caps.supports_json_mode is True
        assert caps.supports_web_search is False
        assert caps.supports_thinking is True
        assert caps.supports_embedding is False

        # 多模态理解能力 - 全部不支持
        assert caps.supports_vision is False
        assert caps.supports_audio_understand is False
        assert caps.supports_video is False
        assert caps.supports_document is False

        # 多模态生成能力 - 全部不支持
        assert caps.supports_image_generation is False
        assert caps.supports_audio_generation is False

        # 模型规格
        assert caps.max_context_window == 131072
        assert caps.supported_output_formats == ("text", "json")


class TestDeepSeekAvailableParams:
    """验证 DeepSeekLLM 支持的参数名集合"""

    def test_no_web_search(self):
        from milu.llm.providers.deepseek import DeepSeekLLM

        llm = DeepSeekLLM(api_key="test", model="deepseek-chat")
        params = llm._get_available_param_names()

        assert "web_search" not in params
        assert "web_search_strategy" not in params

    def test_has_thinking(self):
        from milu.llm.providers.deepseek import DeepSeekLLM

        llm = DeepSeekLLM(api_key="test", model="deepseek-chat")
        params = llm._get_available_param_names()

        assert "enable_thinking" in params
        assert "thinking_level" in params

    def test_has_basic_params(self):
        from milu.llm.providers.deepseek import DeepSeekLLM

        llm = DeepSeekLLM(api_key="test", model="deepseek-chat")
        params = llm._get_available_param_names()

        assert "temperature" in params
        assert "top_p" in params
        assert "max_tokens" in params
        assert "stop" in params
        assert "frequency_penalty" in params
        assert "presence_penalty" in params

    def test_has_tools(self):
        from milu.llm.providers.deepseek import DeepSeekLLM

        llm = DeepSeekLLM(api_key="test", model="deepseek-chat")
        params = llm._get_available_param_names()

        assert "tools" in params
        assert "tool_choice" in params


class TestDeepSeekChat:
    """测试 DeepSeekLLM 流式聊天功能"""

    @pytest.mark.asyncio
    async def test_basic_chat(self):
        """基础聊天：3个chunk（内容"你"+"好"+结束带usage），验证StreamChunk字段"""
        from milu.llm.base.message import Message, MessageRole
        from milu.llm.providers.deepseek import DeepSeekLLM

        chunks = [
            MockChunk(
                choices=[MockChoice(delta=MockDelta(content="你", role="assistant"))]
            ),
            MockChunk(
                choices=[MockChoice(delta=MockDelta(content="好"))]
            ),
            MockChunk(
                choices=[MockChoice(delta=MockDelta(), finish_reason="stop")],
                usage=MockUsage(
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                    prompt_cache_hit_tokens=8,
                ),
            ),
        ]

        async def mock_stream():
            for chunk in chunks:
                yield chunk

        llm = DeepSeekLLM(api_key="test-key", model="deepseek-chat")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        with patch.object(llm, "_get_client", return_value=mock_client):
            messages = [Message(role=MessageRole.USER, content="你好")]
            results = []
            async for chunk in llm.chat(messages):
                results.append(chunk)

        assert len(results) == 3
        assert results[0].content == "你"
        assert results[0].finish_reason is None
        assert results[1].content == "好"
        assert results[1].finish_reason is None
        assert results[2].content is None
        assert results[2].finish_reason == "stop"
        assert results[2].usage is not None
        assert results[2].usage.prompt_tokens == 10
        assert results[2].usage.completion_tokens == 5
        assert results[2].usage.total_tokens == 15
        assert results[2].usage.cached_tokens == 8

    @pytest.mark.asyncio
    async def test_thinking_enabled_uses_thinking_type_and_reasoning_effort(self):
        """开启思考：extra_body.thinking={"type":"enabled"}（不含 budget_tokens——
        DeepSeek 官方 thinking 对象无此字段，那是 Claude 的字段），思考力度走顶层
        reasoning_effort。"""
        from milu.llm.base.message import Message, MessageRole
        from milu.llm.providers.deepseek import DeepSeekLLM

        def _mk_stream():
            async def gen():
                yield MockChunk(choices=[MockChoice(delta=MockDelta(content="思考中"))])
                yield MockChunk(
                    choices=[MockChoice(delta=MockDelta(), finish_reason="stop")],
                    usage=MockUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
                )
            return gen()

        llm = DeepSeekLLM(api_key="test-key", model="deepseek-v4-flash")
        mock_client = AsyncMock()

        for level in ("low", "medium", "high"):
            mock_client.chat.completions.create = AsyncMock(return_value=_mk_stream())
            with patch.object(llm, "_get_client", return_value=mock_client):
                messages = [Message(role=MessageRole.USER, content="请思考一下")]
                async for _ in llm.chat(messages, enable_thinking=True, thinking_level=level):
                    pass

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["extra_body"]["thinking"] == {"type": "enabled"}
            # 不得再出现 Claude 风格的 budget_tokens
            assert "budget_tokens" not in call_kwargs["extra_body"]["thinking"]
            # 思考力度走顶层 reasoning_effort
            assert call_kwargs["reasoning_effort"] == level

    @pytest.mark.asyncio
    async def test_thinking_disabled_uses_thinking_type(self):
        """关闭思考：extra_body.thinking={"type":"disabled"}。"""
        from milu.llm.base.message import Message, MessageRole
        from milu.llm.providers.deepseek import DeepSeekLLM

        async def mock_stream():
            yield MockChunk(
                choices=[MockChoice(delta=MockDelta(), finish_reason="stop")],
                usage=MockUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
            )

        llm = DeepSeekLLM(api_key="test-key", model="deepseek-v4-flash")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        with patch.object(llm, "_get_client", return_value=mock_client):
            messages = [Message(role=MessageRole.USER, content="快速回答")]
            async for _ in llm.chat(messages, enable_thinking=False):
                pass

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["extra_body"]["thinking"] == {"type": "disabled"}
        assert "reasoning_effort" not in call_kwargs
