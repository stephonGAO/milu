"""pytest 公共 fixtures - 用于所有provider测试的模拟工具"""

from __future__ import annotations

from dataclasses import dataclass

import pytest


@pytest.fixture(autouse=True)
def _isolate_paths(monkeypatch, tmp_path):
    """每个测试在隔离的临时目录/数据目录下运行，避免污染仓库工作区与用户主目录。

    - CWD 切到 pytest 自动清理的 ``tmp_path``：兼容仍依赖相对路径
      （如显式 ``session_dir="./.sessions"``）的测试。
    - ``AGENT_FRAMEWORK_HOME`` 指向 ``tmp_path`` 下的数据目录：Agent 默认
      ``session_enabled=True`` 且 ``session_dir=None`` → 解析为
      ``user_data_dir()/sessions``（即 ``~/.agent_framework/sessions``）。若不重定向，
      大量构造 Agent 的单元测试会在真实用户主目录残留 session/plan 文件。重定向后
      这些会话/计划落到临时目录并随测试结束被清理。
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_FRAMEWORK_HOME", str(tmp_path / ".agent_framework"))


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
