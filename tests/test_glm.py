"""测试 GLMLLM - 智谱AI GLM模型实现"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import MockChunk, MockChoice, MockDelta, MockUsage


class TestGLMCapabilities:
    """验证 GLMLLM 的能力标志是否与规格一致"""

    def test_capabilities(self):
        from milu.llm.providers.glm import GLMLLM

        llm = GLMLLM(api_key="test", model="glm-4")
        caps = llm.capabilities

        # 基础能力
        assert caps.supports_streaming is True
        assert caps.supports_function_calling is True
        assert caps.supports_json_mode is True
        assert caps.supports_web_search is True
        assert caps.supports_thinking is True
        assert caps.supports_embedding is True

        # 多模态理解能力
        assert caps.supports_vision is True
        assert caps.supports_audio_understand is False
        assert caps.supports_video is True
        assert caps.supports_document is False

        # 多模态生成能力 - 都不支持
        assert caps.supports_image_generation is False
        assert caps.supports_audio_generation is False

        # 模型规格
        assert caps.max_context_window == 131072


class TestGLMAvailableParams:
    """验证 GLMLLM 支持的参数名集合"""

    def test_has_web_search(self):
        from milu.llm.providers.glm import GLMLLM

        llm = GLMLLM(api_key="test", model="glm-4")
        params = llm._get_available_param_names()

        assert "web_search" in params
        assert "web_search_strategy" in params

    def test_has_thinking(self):
        from milu.llm.providers.glm import GLMLLM

        llm = GLMLLM(api_key="test", model="glm-4")
        params = llm._get_available_param_names()

        assert "enable_thinking" in params
        assert "thinking_level" in params

    def test_has_basic_params(self):
        from milu.llm.providers.glm import GLMLLM

        llm = GLMLLM(api_key="test", model="glm-4")
        params = llm._get_available_param_names()

        assert "temperature" in params
        assert "top_p" in params
        assert "max_tokens" in params
        assert "stop" in params
        assert "frequency_penalty" in params
        assert "presence_penalty" in params

    def test_has_tools(self):
        from milu.llm.providers.glm import GLMLLM

        llm = GLMLLM(api_key="test", model="glm-4")
        params = llm._get_available_param_names()

        assert "tools" in params
        assert "tool_choice" in params


class TestGLMChat:
    """测试 GLMLLM 流式聊天功能"""

    @pytest.mark.asyncio
    async def test_basic_chat(self):
        """基础聊天：验证流式输出和streaming参数"""
        from milu.llm.base.message import Message, MessageRole
        from milu.llm.providers.glm import GLMLLM

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

        llm = GLMLLM(api_key="test-key", model="glm-4")
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

        # 验证stream=True
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["stream"] is True

    @pytest.mark.asyncio
    async def test_thinking_ignores_level(self):
        """思考模式：extra_body有enable_thinking=True，但NO thinking_level映射"""
        from milu.llm.base.message import Message, MessageRole
        from milu.llm.providers.glm import GLMLLM

        async def mock_stream():
            yield MockChunk(choices=[MockChoice(delta=MockDelta(content="思考中"))])
            yield MockChunk(
                choices=[MockChoice(delta=MockDelta(), finish_reason="stop")],
                usage=MockUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
            )

        llm = GLMLLM(api_key="test-key", model="glm-4")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        with patch.object(llm, "_get_client", return_value=mock_client):
            messages = [Message(role=MessageRole.USER, content="请思考一下")]
            results = []
            async for chunk in llm.chat(
                messages, enable_thinking=True, thinking_level="high"
            ):
                results.append(chunk)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "extra_body" in call_kwargs
        assert call_kwargs["extra_body"]["enable_thinking"] is True
        # GLM不支持thinking_level，不应出现在extra_body中
        assert "thinking_level" not in call_kwargs.get("extra_body", {})
        assert "effort" not in call_kwargs.get("extra_body", {})

    @pytest.mark.asyncio
    async def test_web_search_injects_nonempty_subobject(self):
        """联网搜索：GLM 要求 web_search 工具项带【非空】的 web_search 子对象，
        否则 400 "tools[N].web_search 不能为空"。验证注入的项含合法子对象，
        且与既有函数工具共存时不破坏函数调用。
        """
        from milu.llm.base.message import Message, MessageRole
        from milu.llm.providers.glm import GLMLLM

        async def mock_stream():
            yield MockChunk(choices=[MockChoice(delta=MockDelta(content="搜索结果"))])
            yield MockChunk(
                choices=[MockChoice(delta=MockDelta(), finish_reason="stop")],
                usage=MockUsage(prompt_tokens=8, completion_tokens=4, total_tokens=12),
            )

        function_tool = {
            "type": "function",
            "function": {"name": "get_weather", "description": "查天气", "parameters": {}},
        }
        llm = GLMLLM(api_key="test-key", model="glm-4")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        with patch.object(llm, "_get_client", return_value=mock_client):
            messages = [Message(role=MessageRole.USER, content="今天天气如何")]
            results = []
            async for chunk in llm.chat(messages, web_search=True, tools=[function_tool]):
                results.append(chunk)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        tools_list = call_kwargs["tools"]
        # 不得出现裸的、子对象为空的非法 web_search 项
        assert {"type": "web_search"} not in tools_list
        ws_items = [t for t in tools_list if t.get("type") == "web_search"]
        assert len(ws_items) == 1
        sub = ws_items[0].get("web_search")
        assert isinstance(sub, dict) and sub  # 子对象存在且非空
        assert sub.get("enable") == "True"
        # 原有函数工具原样保留
        assert function_tool in tools_list
