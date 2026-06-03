"""pytest 公共 fixtures - 用于所有provider测试的模拟工具"""

from __future__ import annotations

from dataclasses import dataclass

import pytest


@pytest.fixture(autouse=True)
def _isolate_cwd(monkeypatch, tmp_path):
    """每个测试在独立临时目录下运行，避免污染仓库工作区。

    根因：``AgentConfig.session_enabled`` 默认 ``True`` 且 ``session_dir=None`` →
    Agent 会在「当前工作目录」下创建 ``.sessions/{id}/``。单元测试大量构造 Agent
    时，会在仓库根目录残留几十上百个空/半截 session 目录（虽被 .gitignore 忽略，
    但污染工作区、误导排查）。

    将每个测试的 CWD 切到 pytest 自动清理的 ``tmp_path``，让这些 ``.sessions``
    落到临时目录并随测试结束被清理。已自行 ``chdir`` 的测试会覆盖本设置，无副作用；
    技能自动发现（扫描 CWD 的 ``skills/``）在临时目录下自然落空，与测试预期一致。
    """
    monkeypatch.chdir(tmp_path)


@dataclass
class MockChoice:
    delta: "MockDelta"
    finish_reason: str | None = None
    index: int = 0


@dataclass
class MockDelta:
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list | None = None
    role: str | None = None


@dataclass
class MockUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class MockChunk:
    choices: list[MockChoice]
    usage: MockUsage | None = None


@pytest.fixture
def mock_openai_client():
    """创建模拟的 AsyncOpenAI 客户端"""
    from unittest.mock import AsyncMock
    client = AsyncMock()
    client.chat.completions.create = AsyncMock()
    return client
