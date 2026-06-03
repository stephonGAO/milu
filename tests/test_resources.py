"""测试内置资源（提示词模板 / 技能）随包分发与定位 API。"""
import pytest

from agent_framework import builtin_prompts_dir, builtin_skills_dir, templates_dir
from agent_framework.prompts import PromptBuilder


def test_templates_dir_exists():
    assert templates_dir().is_dir()


@pytest.mark.parametrize("role", ["main", "coder", "researcher", "reviewer"])
def test_builtin_prompts_dir(role):
    d = builtin_prompts_dir(role)
    assert d.is_dir()
    assert list(d.glob("*.md")), f"{role} 角色应含 .md 提示词片段"


def test_builtin_prompts_dir_invalid_role():
    with pytest.raises(FileNotFoundError):
        builtin_prompts_dir("nonexistent")


def test_builtin_skills_dir():
    d = builtin_skills_dir()
    assert d.is_dir()
    assert (d / "code-review.md").exists()


def test_promptbuilder_builds_from_builtin():
    prompt = PromptBuilder(builtin_prompts_dir("main")).build()
    assert isinstance(prompt, str)
    assert prompt.strip(), "内置 main 提示词不应为空"
