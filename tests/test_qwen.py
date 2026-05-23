"""测试 QwenLLM - 通义千问模型实现"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import MockChunk, MockChoice, MockDelta, MockUsage


class TestQwenCapabilities:
    """验证 QwenLLM 的能力标志是否与规格一致"""

    def test_capabilities(self):
        from agent_framework.providers.qwen import QwenLLM

        llm = QwenLLM(api_key="test", model="qwen-max")
        caps = llm.capabilities

        # 基础能力 - 全部支持
        assert caps.supports_streaming is True
        assert caps.supports_function_calling is True
        assert caps.supports_json_mode is True
        assert caps.supports_web_search is True
        assert caps.supports_thinking is True
        assert caps.supports_embedding is True

        # 多模态理解能力 - 全部支持
        assert caps.supports_vision is True
        assert caps.supports_audio_understand is True
        assert caps.supports_video is True
        assert caps.supports_document is True

        # 多模态生成能力 - 仅支持图片生成
        assert caps.supports_image_generation is True
        assert caps.supports_audio_generation is False

        # 模型规格
        assert caps.max_context_window == 131072
        assert caps.supported_output_formats == ("text", "json")


class TestQwenAvailableParams:
    """验证 QwenLLM 支持的参数名集合"""

    def test_available_params_include_basic(self):
        from agent_framework.providers.qwen import QwenLLM

        llm = QwenLLM(api_key="test", model="qwen-max")
        params = llm._get_available_param_names()

        # 基础参数
        assert "temperature" in params
        assert "top_p" in params
        assert "max_tokens" in params
        assert "stop" in params

    def test_available_params_include_web_search(self):
        from agent_framework.providers.qwen import QwenLLM

        llm = QwenLLM(api_key="test", model="qwen-max")
        params = llm._get_available_param_names()

        assert "web_search" in params
        assert "web_search_strategy" in params

    def test_available_params_include_thinking(self):
        from agent_framework.providers.qwen import QwenLLM

        llm = QwenLLM(api_key="test", model="qwen-max")
        params = llm._get_available_param_names()

        assert "enable_thinking" in params
        assert "thinking_level" in params

    def test_available_params_include_tools(self):
        from agent_framework.providers.qwen import QwenLLM

        llm = QwenLLM(api_key="test", model="qwen-max")
        params = llm._get_available_param_names()

        assert "tools" in params
        assert "tool_choice" in params


class TestQwenChat:
    """测试 QwenLLM 流式聊天功能"""

    @pytest.mark.asyncio
    async def test_basic_chat(self):
        """基础聊天：3个chunk（内容"你"+"好"+结束带usage），验证StreamChunk字段"""
        from agent_framework.models.message import Message, MessageRole
        from agent_framework.providers.qwen import QwenLLM

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

        llm = QwenLLM(api_key="test-key", model="qwen-max")
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
    async def test_thinking_mode_mapping(self):
        """思考模式映射：extra_body 中应包含 enable_thinking=True，thinking_level 被忽略"""
        from agent_framework.models.message import Message, MessageRole
        from agent_framework.providers.qwen import QwenLLM

        async def mock_stream():
            yield MockChunk(
                choices=[MockChoice(delta=MockDelta(content="思考中"))]
            )
            yield MockChunk(
                choices=[MockChoice(delta=MockDelta(), finish_reason="stop")],
                usage=MockUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
            )

        llm = QwenLLM(api_key="test-key", model="qwen-max")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        with patch.object(llm, "_get_client", return_value=mock_client):
            messages = [Message(role=MessageRole.USER, content="请思考一下")]
            results = []
            async for chunk in llm.chat(
                messages, enable_thinking=True, thinking_level="high"
            ):
                results.append(chunk)

        # 验证调用参数
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "extra_body" in call_kwargs
        assert call_kwargs["extra_body"]["enable_thinking"] is True
        # thinking_level 不在 extra_body 中（Qwen不支持，静默忽略）
        assert "thinking_level" not in call_kwargs.get("extra_body", {})

    @pytest.mark.asyncio
    async def test_web_search_param(self):
        """联网搜索参数：extra_body 中应包含 enable_search=True"""
        from agent_framework.models.message import Message, MessageRole
        from agent_framework.providers.qwen import QwenLLM

        async def mock_stream():
            yield MockChunk(
                choices=[MockChoice(delta=MockDelta(content="搜索结果"))]
            )
            yield MockChunk(
                choices=[MockChoice(delta=MockDelta(), finish_reason="stop")],
                usage=MockUsage(prompt_tokens=8, completion_tokens=4, total_tokens=12),
            )

        llm = QwenLLM(api_key="test-key", model="qwen-max")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        with patch.object(llm, "_get_client", return_value=mock_client):
            messages = [Message(role=MessageRole.USER, content="今天天气")]
            results = []
            async for chunk in llm.chat(messages, web_search=True):
                results.append(chunk)

        # 验证调用参数
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "extra_body" in call_kwargs
        assert call_kwargs["extra_body"]["enable_search"] is True
