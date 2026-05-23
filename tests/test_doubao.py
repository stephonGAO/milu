"""测试 DoubaoLLM - 豆包/火山引擎模型实现"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import MockChunk, MockChoice, MockDelta, MockUsage


class TestDoubaoCapabilities:
    """验证 DoubaoLLM 的能力标志是否与规格一致"""

    def test_capabilities(self):
        from agent_framework.llm.providers.doubao import DoubaoLLM

        llm = DoubaoLLM(api_key="test", model="doubao-pro")
        caps = llm.capabilities

        # 基础能力
        assert caps.supports_streaming is True
        assert caps.supports_function_calling is True
        assert caps.supports_json_mode is True
        assert caps.supports_web_search is True
        assert caps.supports_thinking is False
        assert caps.supports_embedding is True

        # 多模态理解能力
        assert caps.supports_vision is True
        assert caps.supports_audio_understand is False
        assert caps.supports_video is False
        assert caps.supports_document is False

        # 多模态生成能力
        assert caps.supports_image_generation is True
        assert caps.supports_audio_generation is False

        # 模型规格
        assert caps.max_context_window == 131072


class TestDoubaoAvailableParams:
    """验证 DoubaoLLM 支持的参数名集合"""

    def test_has_web_search(self):
        from agent_framework.llm.providers.doubao import DoubaoLLM

        llm = DoubaoLLM(api_key="test", model="doubao-pro")
        params = llm._get_available_param_names()

        assert "web_search" in params
        assert "web_search_strategy" in params

    def test_has_image_params(self):
        from agent_framework.llm.providers.doubao import DoubaoLLM

        llm = DoubaoLLM(api_key="test", model="doubao-pro")
        params = llm._get_available_param_names()

        assert "image_size" in params
        assert "image_quality" in params
        assert "num_images" in params

    def test_no_thinking_params(self):
        """豆包不支持思考模式，参数集合中不应包含 thinking 相关参数"""
        from agent_framework.llm.providers.doubao import DoubaoLLM

        llm = DoubaoLLM(api_key="test", model="doubao-pro")
        params = llm._get_available_param_names()

        assert "enable_thinking" not in params
        assert "thinking_level" not in params

    def test_has_basic_params(self):
        from agent_framework.llm.providers.doubao import DoubaoLLM

        llm = DoubaoLLM(api_key="test", model="doubao-pro")
        params = llm._get_available_param_names()

        assert "temperature" in params
        assert "top_p" in params
        assert "max_tokens" in params
        assert "stop" in params
        assert "frequency_penalty" in params
        assert "presence_penalty" in params

    def test_has_tools(self):
        from agent_framework.llm.providers.doubao import DoubaoLLM

        llm = DoubaoLLM(api_key="test", model="doubao-pro")
        params = llm._get_available_param_names()

        assert "tools" in params
        assert "tool_choice" in params


class TestDoubaoChat:
    """测试 DoubaoLLM 流式聊天功能"""

    @pytest.mark.asyncio
    async def test_basic_chat(self):
        """基础聊天：验证流式输出和 streaming 参数"""
        from agent_framework.llm.base.message import Message, MessageRole
        from agent_framework.llm.providers.doubao import DoubaoLLM

        chunks = [
            MockChunk(choices=[MockChoice(delta=MockDelta(content="你", role="assistant"))]),
            MockChunk(choices=[MockChoice(delta=MockDelta(content="好"))]),
            MockChunk(
                choices=[MockChoice(delta=MockDelta(), finish_reason="stop")],
                usage=MockUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            ),
        ]

        async def mock_stream():
            for chunk in chunks:
                yield chunk

        llm = DoubaoLLM(api_key="test-key", model="doubao-pro")
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
        assert results[2].usage is not None
        assert results[2].usage.prompt_tokens == 10

        # 验证 stream=True
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["stream"] is True

    @pytest.mark.asyncio
    async def test_web_search_injects_tool(self):
        """联网搜索：web_search=True 应在 tools 列表中追加 {"type": "web_search"}"""
        from agent_framework.llm.base.message import Message, MessageRole
        from agent_framework.llm.providers.doubao import DoubaoLLM

        async def mock_stream():
            yield MockChunk(choices=[MockChoice(delta=MockDelta(content="搜索结果"))])
            yield MockChunk(
                choices=[MockChoice(delta=MockDelta(), finish_reason="stop")],
                usage=MockUsage(prompt_tokens=8, completion_tokens=4, total_tokens=12),
            )

        llm = DoubaoLLM(api_key="test-key", model="doubao-pro")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        with patch.object(llm, "_get_client", return_value=mock_client):
            messages = [Message(role=MessageRole.USER, content="今天天气如何")]
            results = []
            async for chunk in llm.chat(messages, web_search=True):
                results.append(chunk)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "tools" in call_kwargs
        tools_list = call_kwargs["tools"]
        assert {"type": "web_search"} in tools_list
