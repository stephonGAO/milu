"""测试 GeminiLLM - Google Gemini 模型实现（OpenAI 兼容模式）"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import MockChunk, MockChoice, MockDelta, MockUsage


class TestGeminiCapabilities:

    def test_capabilities(self):
        from agent_framework.llm.providers.gemini import GeminiLLM

        llm = GeminiLLM(api_key="test", model="gemini-2.5-pro")
        caps = llm.capabilities

        assert caps.supports_streaming is True
        assert caps.supports_function_calling is True
        assert caps.supports_json_mode is True
        assert caps.supports_thinking is True
        assert caps.supports_vision is True
        assert caps.supports_audio_understand is True
        assert caps.supports_video is True
        assert caps.supports_document is True
        assert caps.supports_embedding is True
        assert caps.supports_web_search is False
        assert caps.supports_image_generation is False
        assert caps.max_context_window == 1048576
        assert caps.supported_output_formats == ("text", "json")

    def test_provider_name(self):
        from agent_framework.llm.providers.gemini import GeminiLLM

        llm = GeminiLLM(api_key="test", model="gemini-2.5-pro")
        assert llm.provider_name == "gemini"

    def test_base_url(self):
        from agent_framework.llm.providers.gemini import GeminiLLM

        llm = GeminiLLM(api_key="test", model="gemini-2.5-pro")
        assert "generativelanguage.googleapis.com" in llm.base_url

    def test_registered(self):
        from agent_framework.llm.providers import ModelRegistry
        assert "gemini" in ModelRegistry.list_providers()


class TestGeminiParams:

    def test_has_basic_params(self):
        from agent_framework.llm.providers.gemini import GeminiLLM

        llm = GeminiLLM(api_key="test", model="gemini-2.5-pro")
        params = llm._get_available_param_names()
        assert "temperature" in params
        assert "top_p" in params
        assert "max_tokens" in params

    def test_has_thinking(self):
        from agent_framework.llm.providers.gemini import GeminiLLM

        llm = GeminiLLM(api_key="test", model="gemini-2.5-pro")
        params = llm._get_available_param_names()
        assert "enable_thinking" in params

    def test_no_web_search(self):
        from agent_framework.llm.providers.gemini import GeminiLLM

        llm = GeminiLLM(api_key="test", model="gemini-2.5-pro")
        params = llm._get_available_param_names()
        assert "web_search" not in params

    def test_has_tools(self):
        from agent_framework.llm.providers.gemini import GeminiLLM

        llm = GeminiLLM(api_key="test", model="gemini-2.5-pro")
        params = llm._get_available_param_names()
        assert "tools" in params
        assert "tool_choice" in params


class TestGeminiChat:

    @pytest.mark.asyncio
    async def test_basic_chat(self):
        from agent_framework.llm.base.message import Message, MessageRole
        from agent_framework.llm.providers.gemini import GeminiLLM

        chunks = [
            MockChunk(choices=[MockChoice(delta=MockDelta(content="你", role="assistant"))]),
            MockChunk(choices=[MockChoice(delta=MockDelta(content="好"))]),
            MockChunk(
                choices=[MockChoice(delta=MockDelta(), finish_reason="stop")],
                usage=MockUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            ),
        ]

        async def mock_stream():
            for c in chunks:
                yield c

        llm = GeminiLLM(api_key="test-key", model="gemini-2.5-pro")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        with patch.object(llm, "_get_client", return_value=mock_client):
            messages = [Message(role=MessageRole.USER, content="你好")]
            results = []
            async for chunk in llm.chat(messages):
                results.append(chunk)

        assert len(results) == 3
        assert results[0].content == "你"
        assert results[1].content == "好"
        assert results[2].finish_reason == "stop"
        assert results[2].usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_thinking_enabled(self):
        from agent_framework.llm.base.message import Message, MessageRole
        from agent_framework.llm.providers.gemini import GeminiLLM

        async def mock_stream():
            yield MockChunk(
                choices=[MockChoice(delta=MockDelta(content="思考中"))]
            )
            yield MockChunk(
                choices=[MockChoice(delta=MockDelta(), finish_reason="stop")],
                usage=MockUsage(5, 3, 8),
            )

        llm = GeminiLLM(api_key="test-key", model="gemini-2.5-pro")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        with patch.object(llm, "_get_client", return_value=mock_client):
            messages = [Message(role=MessageRole.USER, content="请思考")]
            async for _ in llm.chat(messages, enable_thinking=True):
                pass

        kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "extra_body" in kwargs
        assert kwargs["extra_body"]["thinking"]["type"] == "enabled"
        assert kwargs["extra_body"]["thinking"]["budget_tokens"] == -1

    @pytest.mark.asyncio
    async def test_thinking_disabled(self):
        from agent_framework.llm.base.message import Message, MessageRole
        from agent_framework.llm.providers.gemini import GeminiLLM

        async def mock_stream():
            yield MockChunk(
                choices=[MockChoice(delta=MockDelta(), finish_reason="stop")],
                usage=MockUsage(5, 3, 8),
            )

        llm = GeminiLLM(api_key="test-key", model="gemini-2.5-pro")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        with patch.object(llm, "_get_client", return_value=mock_client):
            messages = [Message(role=MessageRole.USER, content="快速回答")]
            async for _ in llm.chat(messages, enable_thinking=False):
                pass

        kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"]["thinking"]["type"] == "disabled"

    @pytest.mark.asyncio
    async def test_reasoning_content(self):
        from agent_framework.llm.base.message import Message, MessageRole
        from agent_framework.llm.providers.gemini import GeminiLLM

        async def mock_stream():
            yield MockChunk(
                choices=[MockChoice(delta=MockDelta(
                    content="答案",
                    reasoning_content="推理过程",
                ))]
            )
            yield MockChunk(
                choices=[MockChoice(delta=MockDelta(), finish_reason="stop")],
                usage=MockUsage(5, 3, 8),
            )

        llm = GeminiLLM(api_key="test-key", model="gemini-2.5-pro")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        with patch.object(llm, "_get_client", return_value=mock_client):
            messages = [Message(role=MessageRole.USER, content="问题")]
            results = []
            async for chunk in llm.chat(messages):
                results.append(chunk)

        assert results[0].content == "答案"
        assert results[0].reasoning_content == "推理过程"
