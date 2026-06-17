"""测试内置工具 image_read 与多模态图片管道

覆盖:
- llm/base/vision.py: MIME 映射、data URL 物化、占位降级、build_user_content
- tools/builtin/image_tool.py: 校验（存在/格式/大小/视觉能力/注入通路）、去重
- providers/base.py _messages_to_dicts: image_path 块物化为 base64 data URL
- providers/chatgpt.py _convert_user_content: Responses 格式转换
- session.py: list content 的 JSONL 往返
- Agent 集成: image_read 工具调用后注入多模态 user 消息；run(images=...) 参数
"""
import base64
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from milu.agent import Agent, AgentConfig
from milu.agent.events import AgentDone, ToolResult
from milu.agent.session import Session
from milu.llm.base.message import Message, MessageRole
from milu.llm.base.response import StreamChunk, TokenUsage
from milu.llm.base.vision import (
    build_user_content,
    encode_image_data_url,
    image_mime_type,
    materialize_content,
)
from milu.tools.builtin.image_tool import (
    _current_pending_images,
    _current_vision_support,
    image_read,
)

# 1×1 透明 PNG（最小合法图片）
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def png_file(tmp_dir):
    p = tmp_dir / "test.png"
    p.write_bytes(_PNG_BYTES)
    return p


def _call_image_read(**kwargs):
    """直接调用工具函数（绕过 Agent）。"""
    return image_read._tool_wrapper.func(**kwargs)


# ── vision.py 纯函数 ──────────────────────────────────────


class TestVisionHelpers:

    def test_mime_mapping(self):
        assert image_mime_type("a.png") == "image/png"
        assert image_mime_type("a.JPG") == "image/jpeg"
        assert image_mime_type("a.jpeg") == "image/jpeg"
        assert image_mime_type("a.webp") == "image/webp"
        assert image_mime_type("a.txt") is None
        assert image_mime_type("noext") is None

    def test_encode_data_url(self, png_file):
        url = encode_image_data_url(str(png_file))
        assert url.startswith("data:image/png;base64,")
        # base64 解码后应还原为原始字节
        b64 = url.split(",", 1)[1]
        assert base64.b64decode(b64) == _PNG_BYTES

    def test_encode_rejects_bad_ext(self, tmp_dir):
        bad = tmp_dir / "a.txt"
        bad.write_text("x")
        with pytest.raises(ValueError):
            encode_image_data_url(str(bad))

    def test_build_user_content_order(self, png_file):
        content = build_user_content("描述图片", [str(png_file)])
        assert content[-1] == {"type": "text", "text": "描述图片"}
        assert content[0]["type"] == "image_path"
        # 路径转为绝对路径
        assert Path(content[0]["path"]).is_absolute()

    def test_materialize_str_passthrough(self):
        assert materialize_content("纯文本") == "纯文本"
        assert materialize_content(None) is None

    def test_materialize_image_path(self, png_file):
        out = materialize_content([
            {"type": "image_path", "path": str(png_file)},
            {"type": "text", "text": "hi"},
        ])
        assert out[0]["type"] == "image_url"
        assert out[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert out[1] == {"type": "text", "text": "hi"}

    def test_materialize_missing_file_degrades(self):
        out = materialize_content([
            {"type": "image_path", "path": "Z:/nonexistent/x.png"},
        ])
        assert out[0]["type"] == "text"
        assert "图片不可用" in out[0]["text"]

    def test_materialize_other_blocks_passthrough(self):
        block = {"type": "image_url", "image_url": {"url": "https://e.com/a.png"}}
        assert materialize_content([block]) == [block]


# ── image_read 工具 ───────────────────────────────────────


class TestImageReadTool:

    async def test_file_not_exists(self):
        token = _current_pending_images.set([])
        try:
            result = json.loads(await _call_image_read(path="Z:/no/such.png"))
        finally:
            _current_pending_images.reset(token)
        assert result["success"] is False
        assert "不存在" in result["error"]

    async def test_bad_extension(self, tmp_dir):
        bad = tmp_dir / "a.txt"
        bad.write_text("x")
        token = _current_pending_images.set([])
        try:
            result = json.loads(await _call_image_read(path=str(bad)))
        finally:
            _current_pending_images.reset(token)
        assert result["success"] is False
        assert "不支持的图片格式" in result["error"]

    async def test_too_large(self, png_file, monkeypatch):
        monkeypatch.setattr("milu.tools.builtin.image_tool.MAX_IMAGE_BYTES", 10)
        token = _current_pending_images.set([])
        try:
            result = json.loads(await _call_image_read(path=str(png_file)))
        finally:
            _current_pending_images.reset(token)
        assert result["success"] is False
        assert "过大" in result["error"]

    async def test_vision_unsupported(self, png_file):
        v_token = _current_vision_support.set(False)
        p_token = _current_pending_images.set([])
        try:
            result = json.loads(await _call_image_read(path=str(png_file)))
        finally:
            _current_vision_support.reset(v_token)
            _current_pending_images.reset(p_token)
        assert result["success"] is False
        assert "不支持视觉" in result["error"]

    async def test_no_injection_context(self, png_file):
        # 默认 pending=None（非 Agent.run 上下文）
        result = json.loads(await _call_image_read(path=str(png_file)))
        assert result["success"] is False
        assert "Agent" in result["error"]

    async def test_success_appends_and_dedupes(self, png_file):
        pending: list = []
        token = _current_pending_images.set(pending)
        try:
            r1 = json.loads(await _call_image_read(path=str(png_file)))
            r2 = json.loads(await _call_image_read(path=str(png_file)))
        finally:
            _current_pending_images.reset(token)
        assert r1["success"] is True
        assert r1["format"] == "image/png"
        assert r2["success"] is True
        assert len(pending) == 1  # 同批去重
        # resolve() 规避 Windows 8.3 短路径名差异（ADMINI~1 vs Administrator）
        assert Path(pending[0]).resolve() == png_file.resolve()


# ── provider 物化 ─────────────────────────────────────────


class TestProviderMaterialization:

    def test_messages_to_dicts_materializes(self, png_file):
        from milu.llm.providers.qwen import QwenLLM
        llm = QwenLLM(api_key="test", model="qwen-vl-plus")
        messages = [
            Message(role=MessageRole.SYSTEM, content="sys"),
            Message(role=MessageRole.USER, content=[
                {"type": "image_path", "path": str(png_file)},
                {"type": "text", "text": "看图"},
            ]),
        ]
        dicts = llm._messages_to_dicts(messages)
        assert dicts[0]["content"] == "sys"
        blocks = dicts[1]["content"]
        assert blocks[0]["type"] == "image_url"
        assert blocks[0]["image_url"]["url"].startswith("data:image/png;base64,")
        # 历史中的原始消息不被改动（轻量块仍是 image_path）
        assert messages[1].content[0]["type"] == "image_path"

    def test_chatgpt_responses_conversion(self, png_file):
        from milu.llm.providers.chatgpt import ChatGPTLLM
        items = ChatGPTLLM._convert_user_content([
            {"type": "image_path", "path": str(png_file)},
            {"type": "text", "text": "看图"},
        ])
        assert items[0]["type"] == "input_image"
        assert items[0]["image_url"].startswith("data:image/png;base64,")
        assert items[1] == {"type": "input_text", "text": "看图"}

    def test_chatgpt_str_passthrough(self):
        from milu.llm.providers.chatgpt import ChatGPTLLM
        assert ChatGPTLLM._convert_user_content("纯文本") == "纯文本"


# ── session 往返 ──────────────────────────────────────────


class TestSessionMultimodalRoundtrip:

    def test_list_content_roundtrip(self, tmp_dir):
        session = Session("img_rt", tmp_dir)
        blocks = [
            {"type": "image_path", "path": "D:/x/test.png"},
            {"type": "text", "text": "[image_read] 已加载"},
        ]
        session.log_message(Message(role=MessageRole.USER, content=blocks))
        loaded = session.load_messages()
        assert len(loaded) == 1
        assert loaded[0].content == blocks  # list 结构保持，不被 str() 破坏


# ── Agent 集成 ────────────────────────────────────────────


def _make_image_agent(tmp_dir, png_path):
    """构造 mock LLM：第 1 轮调 image_read，第 2 轮输出文本。"""
    call_count = 0

    async def mock_chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield StreamChunk(tool_calls=[
                type('obj', (), {
                    'index': 0, 'id': 'call_img',
                    'function': type('obj', (), {
                        'name': 'image_read',
                        'arguments': json.dumps({"path": str(png_path)}),
                    })()
                })()
            ])
            yield StreamChunk(finish_reason="tool_calls")
        else:
            yield StreamChunk(content="图中是一个像素", finish_reason="stop")
            yield StreamChunk(usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30))

    llm = AsyncMock()
    llm.chat = mock_chat
    llm._model_config = None
    llm.capabilities = AsyncMock()
    llm.capabilities.max_context_window = 8192
    return Agent(llm=llm, system_prompt="test", tools=[image_read],
                 config=AgentConfig(), session_dir=str(tmp_dir),
                 skills_dir="/tmp/_nonexistent_")


class TestAgentImageInjection:

    async def test_tool_call_injects_multimodal_message(self, tmp_dir, png_file):
        agent = _make_image_agent(tmp_dir, png_file)

        events = []
        async for event in agent.run("分析这张图"):
            events.append(event)

        # 工具执行成功
        tool_result = next(e for e in events if isinstance(e, ToolResult))
        assert json.loads(tool_result.output)["success"] is True
        assert any(isinstance(e, AgentDone) for e in events)

        # 历史中应注入多模态 user 消息，且位于 tool 消息之后
        msgs = agent.history.all_messages
        tool_idx = next(i for i, m in enumerate(msgs) if m.role == MessageRole.TOOL)
        injected = [
            (i, m) for i, m in enumerate(msgs)
            if m.role == MessageRole.USER and isinstance(m.content, list)
        ]
        assert len(injected) == 1
        idx, msg = injected[0]
        assert idx > tool_idx
        assert msg.content[0]["type"] == "image_path"
        assert Path(msg.content[0]["path"]).resolve() == png_file.resolve()
        assert msg.content[-1]["type"] == "text"

    async def test_run_images_param(self, tmp_dir, png_file):
        async def mock_chat(*args, **kwargs):
            yield StreamChunk(content="收到", finish_reason="stop")

        llm = AsyncMock()
        llm.chat = mock_chat
        llm._model_config = None
        llm.capabilities = AsyncMock()
        llm.capabilities.max_context_window = 8192
        agent = Agent(llm=llm, system_prompt="test", tools=[],
                      config=AgentConfig(), session_dir=str(tmp_dir),
                      skills_dir="/tmp/_nonexistent_")

        async for _ in agent.run("看看这张图", images=[str(png_file)]):
            pass

        user_msg = next(m for m in agent.history.all_messages
                        if m.role == MessageRole.USER)
        assert isinstance(user_msg.content, list)
        assert user_msg.content[0]["type"] == "image_path"
        assert user_msg.content[-1] == {"type": "text", "text": "看看这张图"}

    async def test_run_images_degrades_without_vision(self, tmp_dir, png_file):
        async def mock_chat(*args, **kwargs):
            yield StreamChunk(content="收到", finish_reason="stop")

        llm = AsyncMock()
        llm.chat = mock_chat
        llm._model_config = None
        llm.capabilities = AsyncMock()
        llm.capabilities.max_context_window = 8192
        llm.capabilities.supports_vision = False  # 不支持视觉
        agent = Agent(llm=llm, system_prompt="test", tools=[],
                      config=AgentConfig(), session_dir=str(tmp_dir),
                      skills_dir="/tmp/_nonexistent_")

        async for _ in agent.run("看看这张图", images=[str(png_file)]):
            pass

        user_msg = next(m for m in agent.history.all_messages
                        if m.role == MessageRole.USER)
        assert user_msg.content == "看看这张图"  # 降级纯文本
