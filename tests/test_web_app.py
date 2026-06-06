"""内置 Web 服务（milu serve）端点测试 —— FastAPI TestClient + mock LLM。

覆盖：页面/厂商/设置读写与驱逐重建/模式/工具技能/定时任务 CRUD/对话 SSE
首包/确认边界。LLM 经 monkeypatch 替换为 FakeLLM（不触网），MILU_HOME 指向
tmp_path（任务/会话/记忆写盘隔离）。
"""
from __future__ import annotations

import pytest

from milu.llm.base.response import StreamChunk, TokenUsage
from milu.llm.providers import ModelRegistry
from milu.llm.providers.base import BaseLLM, ModelCapabilities

# fastapi 为 [serve] 可选依赖；未安装则跳过本模块
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


class FakeLLM(BaseLLM):
    """立即返回固定文本的 mock LLM（无工具调用、不触网）。"""

    def __init__(self, text: str = "你好，我是 milu。"):
        super().__init__(model="mock", provider="mock")
        self._text = text

    async def chat(self, messages, **kwargs):
        yield StreamChunk(content=self._text, finish_reason="stop")
        yield StreamChunk(usage=TokenUsage(1, 1, 2))

    def _get_available_param_names(self):
        return frozenset()

    @property
    def base_url(self) -> str:
        return "mock://"

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def capabilities(self):
        return ModelCapabilities(
            supports_streaming=True, supports_function_calling=True)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MILU_HOME", str(tmp_path))
    monkeypatch.setenv("MILU_NO_DOTENV", "1")
    fake = FakeLLM()
    # 替换 ModelRegistry.create：服务内所有 _get_llm 都拿到 FakeLLM
    monkeypatch.setattr(ModelRegistry, "create", lambda *a, **k: fake)

    from milu.serving.web.app import create_app
    app = create_app(use_mcp=False, use_scheduler=False, provider="qwen")
    with TestClient(app) as c:
        yield c


def _h(uid="alice", sid="s1"):
    return {"X-User-Id": uid, "X-Session-Id": sid}


# ── 页面与厂商 ────────────────────────────────────────────

def test_index_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "milu" in r.text


def test_providers(client):
    items = client.get("/api/providers").json()["providers"]
    names = {p["name"] for p in items}
    assert "qwen" in names
    assert len(items) >= 9
    for p in items:
        assert "key_configured" in p and isinstance(p["key_configured"], bool)
        assert "default_model" in p


# ── 设置读写 + 驱逐重建 ───────────────────────────────────

def test_settings_roundtrip(client):
    h = _h()
    s = client.get("/api/settings", headers=h).json()
    assert s["provider"] == "qwen"
    assert s["mode"] == "auto"

    r = client.post("/api/settings", headers=h,
                    json={"provider": "deepseek", "model": "custom-x",
                          "web_search": False})
    assert r.status_code == 200
    s2 = client.get("/api/settings", headers=h).json()
    assert s2["provider"] == "deepseek"
    assert s2["model"] == "custom-x"
    assert s2["web_search"] is False


def test_settings_triggers_rebuild(client):
    """切换设置后实例被驱逐：先建实例，POST settings 后池中该 key 被移除。"""
    h = _h(uid="bob")
    # 触发一次创建（通过 /api/tools → get_or_create_agent）
    client.get("/api/tools", headers=h)
    pool = client.app.state.pool
    assert ("bob", "s1") in pool._entries
    client.post("/api/settings", headers=h, json={"model": "another"})
    assert ("bob", "s1") not in pool._entries


def test_show_subagent_toggle(client):
    """子代理显示开关：默认隐藏、可经 /api/settings 切换，且不驱逐实例。"""
    h = _h(uid="erin")
    # 默认隐藏（服务启动默认 show_subagent_events=False）
    assert client.get("/api/settings", headers=h).json()["show_subagent_events"] is False
    # 先建实例
    client.get("/api/tools", headers=h)
    pool = client.app.state.pool
    assert ("erin", "s1") in pool._entries
    # 仅切换展示开关 → 偏好生效，但不应驱逐实例（纯展示，无需重建）
    r = client.post("/api/settings", headers=h, json={"show_subagent_events": True})
    assert r.status_code == 200
    assert client.get("/api/settings", headers=h).json()["show_subagent_events"] is True
    assert ("erin", "s1") in pool._entries


def test_mode_switch(client):
    h = _h(uid="carol")
    assert client.post("/api/mode", headers=h, json={"mode": "talk"}).status_code == 200
    assert client.get("/api/settings", headers=h).json()["mode"] == "talk"
    assert client.post("/api/mode", headers=h, json={"mode": "bogus"}).status_code == 400


# ── 工具 / 技能 ───────────────────────────────────────────

def test_tools_and_skills(client):
    h = _h(uid="dan")
    tools = client.get("/api/tools", headers=h).json()
    assert len(tools["active"]) > 0
    assert all("is_safe" in t for t in tools["active"])
    skills = client.get("/api/skills", headers=h).json()
    assert isinstance(skills["skills"], list)


# ── 定时任务 CRUD ─────────────────────────────────────────

def test_schedule_crud(client):
    h = {"X-User-Id": "erin"}
    r = client.post("/api/schedule/create", headers=h, json={
        "name": "t1", "prompt": "提醒喝水", "trigger_type": "once",
        "run_at": "2099-01-01T09:00:00"})
    assert r.status_code == 200

    tasks = client.get("/api/schedule/tasks", headers=h).json()["tasks"]
    assert any(t["name"] == "t1" and t["user_id"] == "erin" for t in tasks)

    # 过去时间被拒
    r2 = client.post("/api/schedule/create", headers=h, json={
        "name": "t2", "prompt": "p", "trigger_type": "once",
        "run_at": "2000-01-01T00:00:00"})
    assert r2.status_code == 400

    # 禁用 / 删除
    assert client.post("/api/schedule/action", headers=h,
                       json={"action": "disable", "name": "t1"}).status_code == 200
    assert client.post("/api/schedule/action", headers=h,
                       json={"action": "delete", "name": "t1"}).status_code == 200
    tasks2 = client.get("/api/schedule/tasks", headers=h).json()["tasks"]
    assert not any(t["name"] == "t1" for t in tasks2)


def test_schedule_create_rejects_invalid_trigger_params(client):
    """触发参数缺失/无效的任务被 400 拒绝——配置无效的任务永远算不出
    next_run，会「创建了却永不执行」（回归：曾放行空 run_at 的 once 任务）。"""
    h = {"X-User-Id": "erin"}
    # once 缺 run_at
    r = client.post("/api/schedule/create", headers=h, json={
        "name": "x1", "prompt": "一分钟后提醒我喝水", "trigger_type": "once"})
    assert r.status_code == 400 and "run_at" in r.json()["detail"]
    # interval 无间隔
    r = client.post("/api/schedule/create", headers=h, json={
        "name": "x2", "prompt": "p", "trigger_type": "interval", "interval_minutes": 0})
    assert r.status_code == 400
    # cron 缺表达式
    r = client.post("/api/schedule/create", headers=h, json={
        "name": "x3", "prompt": "p", "trigger_type": "cron"})
    assert r.status_code == 400
    # 全部未落库
    assert client.get("/api/schedule/tasks", headers=h).json()["tasks"] == []


def test_schedule_isolated_by_user(client):
    """不同用户的任务互不可见/不可删。"""
    client.post("/api/schedule/create", headers={"X-User-Id": "ua"}, json={
        "name": "task", "prompt": "p", "trigger_type": "once",
        "run_at": "2099-01-01T09:00:00"})
    # ub 看不到 ua 的任务
    assert client.get("/api/schedule/tasks", headers={"X-User-Id": "ub"}).json()["tasks"] == []
    # ub 删不到
    assert client.post("/api/schedule/action", headers={"X-User-Id": "ub"},
                       json={"action": "delete", "name": "task"}).status_code == 404


# ── 对话 SSE ──────────────────────────────────────────────

def test_chat_sse_stream(client):
    h = _h(uid="frank")
    events = []
    with client.stream("POST", "/api/chat", headers=h,
                       json={"input": "你好"}) as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if line.startswith("event:"):
                name = line[6:].strip()
                events.append(name)
                if name == "AgentDone":
                    break
    assert "TextDelta" in events
    assert "AgentDone" in events


def test_command_stream(client):
    """/ 命令走命令事件流（不进 LLM）。"""
    h = _h(uid="grace")
    got = []
    with client.stream("POST", "/api/chat", headers=h,
                       json={"input": "/help"}) as r:
        for line in r.iter_lines():
            if line.startswith("event:"):
                got.append(line[6:].strip())
                if len(got) >= 1:
                    break
    assert "CommandResult" in got


# ── 确认边界 ──────────────────────────────────────────────

def test_confirm_no_pending(client):
    r = client.post("/api/confirm", headers=_h(uid="henry"),
                    json={"approved": True})
    assert r.status_code == 404
