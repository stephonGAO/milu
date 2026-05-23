"""pytest 公共 fixtures - 用于所有provider测试的模拟工具"""

from __future__ import annotations

from dataclasses import dataclass

import pytest


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
