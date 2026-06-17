"""测试 Skill 系统（按需加载模式）"""
import os
import tempfile

import pytest
from unittest.mock import AsyncMock

from milu.skills.config import SkillConfig
from milu.skills.registry import SkillRegistry
from milu.tools.registry import ToolRegistry
from milu.agent import Agent
from milu.agent.events import ToolResult
from milu.llm.base.response import StreamChunk, TokenUsage


# ---------------------------------------------------------------------------
# SkillConfig 测试
# ---------------------------------------------------------------------------

class TestSkillConfig:

    def test_from_markdown_basic(self):
        """正常解析 Markdown frontmatter"""
        text = """---
name: code-review
description: 代码审查专家
triggers:
  - 审查
  - review
---

你是一位代码审查专家。
"""
        cfg = SkillConfig.from_markdown(text)
        assert cfg.name == "code-review"
        assert cfg.description == "代码审查专家"
        assert cfg.triggers == ["审查", "review"]
        assert "代码审查专家" in cfg.content

    def test_from_markdown_no_frontmatter(self):
        """缺少 frontmatter 应抛 ValueError"""
        with pytest.raises(ValueError, match="frontmatter"):
            SkillConfig.from_markdown("没有 frontmatter 的文本")

    def test_from_markdown_missing_name(self):
        """缺少 name 字段应抛 ValueError"""
        text = """---
description: 只有描述
---

正文
"""
        with pytest.raises(ValueError, match="name"):
            SkillConfig.from_markdown(text)

    def test_from_markdown_missing_description(self):
        """缺少 description 字段应抛 ValueError"""
        text = """---
name: test-skill
---

正文
"""
        with pytest.raises(ValueError, match="description"):
            SkillConfig.from_markdown(text)

    def test_from_markdown_minimal(self):
        """只有必填字段"""
        text = """---
name: minimal
description: 最小配置
---

内容
"""
        cfg = SkillConfig.from_markdown(text)
        assert cfg.name == "minimal"
        assert cfg.triggers == []
        assert cfg.content == "内容"

    def test_from_file(self):
        """从文件读取并解析"""
        text = """---
name: file-test
description: 文件测试
---

文件内容
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(text)
            f.flush()
            path = f.name

        try:
            cfg = SkillConfig.from_file(path)
            assert cfg.name == "file-test"
            assert cfg.source == os.path.abspath(path)
            assert "文件内容" in cfg.content
        finally:
            os.unlink(path)

    def test_from_dict(self):
        """从字典构造"""
        data = {
            "description": "字典配置",
            "triggers": ["测试"],
            "content": "字典内容",
        }
        cfg = SkillConfig.from_dict("dict-skill", data)
        assert cfg.name == "dict-skill"
        assert cfg.description == "字典配置"
        assert cfg.content == "字典内容"


# ---------------------------------------------------------------------------
# SkillRegistry 测试
# ---------------------------------------------------------------------------

class TestSkillRegistry:

    def test_add_and_list(self):
        """add 后可在 list_names 中看到"""
        reg = SkillRegistry()
        reg.add(SkillConfig(name="test", description="测试技能"))
        assert "test" in reg.list_names()
        assert len(reg) == 1

    def test_describe_available(self):
        """describe_available 应返回所有技能的元数据目录"""
        reg = SkillRegistry()
        reg.add(SkillConfig(name="code-review", description="代码审查", triggers=["审查", "CR"]))
        reg.add(SkillConfig(name="translator", description="翻译专家"))

        catalog = reg.describe_available()
        assert "code-review" in catalog
        assert "代码审查" in catalog
        assert "审查, CR" in catalog
        assert "translator" in catalog
        assert "翻译专家" in catalog
        assert "load_skill" in catalog  # 应提示使用 load_skill

    def test_describe_available_empty(self):
        """无技能时返回空字符串"""
        reg = SkillRegistry()
        assert reg.describe_available() == ""

    def test_load_skill_body(self):
        """load_skill 应返回技能的完整正文"""
        reg = SkillRegistry()
        reg.add(SkillConfig(name="expert", description="专家", content="专家指令内容"))

        result = reg.load_skill("expert")
        assert '<skill name="expert">' in result
        assert "专家指令内容" in result
        assert "</skill>" in result

    def test_load_skill_not_found(self):
        """加载不存在的技能应返回错误信息"""
        reg = SkillRegistry()
        reg.add(SkillConfig(name="alpha", description="A"))

        result = reg.load_skill("nonexistent")
        assert "错误" in result
        assert "alpha" in result  # 应列出可用技能

    def test_get(self):
        """get 应返回 SkillConfig 或 None"""
        reg = SkillRegistry()
        cfg = SkillConfig(name="test", description="测试")
        reg.add(cfg)

        assert reg.get("test") is cfg
        assert reg.get("nonexistent") is None

    def test_load_from_directory(self):
        """从目录加载 .md 技能文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 平铺布局
            with open(os.path.join(tmpdir, "test.md"), "w", encoding="utf-8") as f:
                f.write("---\nname: test\ndescription: 测试\n---\n\n内容\n")

            # 子目录布局
            subdir = os.path.join(tmpdir, "sub")
            os.makedirs(subdir)
            with open(os.path.join(subdir, "SKILL.md"), "w", encoding="utf-8") as f:
                f.write("---\nname: sub-test\ndescription: 子目录测试\n---\n\n子内容\n")

            reg = SkillRegistry()
            count = reg.load_from_directory(tmpdir)

            assert count == 2
            assert "test" in reg.list_names()
            assert "sub-test" in reg.list_names()


# ---------------------------------------------------------------------------
# Skill 元工具测试
# ---------------------------------------------------------------------------

class TestAsTool:
    """测试 SkillRegistry.as_tool()"""

    def _make_registry(self):
        reg = SkillRegistry()
        reg.add(SkillConfig(name="alpha", description="Alpha 技能", content="Alpha 正文"))
        return reg

    def test_as_tool_returns_wrapper(self):
        """as_tool 应返回 ToolWrapper"""
        reg = self._make_registry()
        wrapper = reg.as_tool()
        assert wrapper.name == "load_skill"
        assert wrapper.meta is True
        assert wrapper.is_async is False

    def test_as_tool_func_returns_body(self):
        """as_tool 的 func 调用应返回技能正文"""
        reg = self._make_registry()
        wrapper = reg.as_tool()
        result = wrapper.func("alpha")
        assert '<skill name="alpha">' in result
        assert "Alpha 正文" in result

    def test_as_tool_func_not_found(self):
        """as_tool 的 func 调用不存在的技能应返回错误"""
        reg = self._make_registry()
        wrapper = reg.as_tool()
        result = wrapper.func("nonexistent")
        assert "错误" in result

    def test_as_tool_registerable(self):
        """as_tool 返回的 wrapper 可注册到 ToolRegistry"""
        reg = self._make_registry()
        tool_registry = ToolRegistry()
        tool_registry.register_wrapper(reg.as_tool())
        assert "load_skill" in tool_registry.list_tools()


# ---------------------------------------------------------------------------
# Agent 集成测试
# ---------------------------------------------------------------------------

class TestAgentSkillIntegration:

    @pytest.mark.asyncio
    async def test_system_prompt_contains_catalog(self):
        """构造时 system prompt 应包含技能元数据目录"""
        skills = [
            SkillConfig(name="expert", description="专家技能", content="专家正文"),
            SkillConfig(name="translator", description="翻译技能", triggers=["翻译"], content="翻译正文"),
        ]

        llm = AsyncMock()
        agent = Agent(
            llm=llm,
            system_prompt="你是助手",
            skills=skills,
            skills_dir=tempfile.mkdtemp(),  # 空目录，阻止自动搜索
            register_catalog=False,
            register_skills=False,
        )

        agent._build_system_prompt()  # 手动触发动态拼装
        system_msg = agent.history.all_messages[0]
        # 应包含元数据
        assert "expert" in system_msg.content
        assert "专家技能" in system_msg.content
        assert "translator" in system_msg.content
        assert "翻译技能" in system_msg.content
        # 不应包含正文
        assert "专家正文" not in system_msg.content
        assert "翻译正文" not in system_msg.content

    @pytest.mark.asyncio
    async def test_system_prompt_no_skills(self):
        """无技能时 system prompt 不变"""
        llm = AsyncMock()
        agent = Agent(
            llm=llm,
            system_prompt="你是助手",
            skills_dir=tempfile.mkdtemp(),  # 空目录，阻止自动搜索
            register_catalog=False,
            register_skills=False,
        )

        agent._build_system_prompt()  # 手动触发动态拼装
        system_msg = agent.history.all_messages[0]
        assert system_msg.content == "你是助手"

    @pytest.mark.asyncio
    async def test_skill_tools_registered(self):
        """register_skills=True 时应注册 load_skill 工具"""
        llm = AsyncMock()
        agent = Agent(
            llm=llm,
            system_prompt="你是助手",
            register_catalog=False,
            register_skills=True,
        )

        tool_names = agent.tools.list_tools()
        assert "load_skill" in tool_names

    @pytest.mark.asyncio
    async def test_skill_tools_not_registered_for_subagent(self):
        """register_skills=False 时不注册元工具"""
        llm = AsyncMock()
        agent = Agent(
            llm=llm,
            system_prompt="你是助手",
            register_catalog=False,
            register_skills=False,
        )

        tool_names = agent.tools.list_tools()
        assert "load_skill" not in tool_names

    @pytest.mark.asyncio
    async def test_skill_registry_property(self):
        """skill_registry 属性应可访问"""
        llm = AsyncMock()
        skills = [SkillConfig(name="test", description="测试")]
        agent = Agent(
            llm=llm,
            system_prompt="你是助手",
            skills=skills,
            register_catalog=False,
            register_skills=False,
        )

        assert "test" in agent.skill_registry.list_names()

    @pytest.mark.asyncio
    async def test_llm_loads_skill_via_tool(self):
        """LLM 通过 load_skill 工具按需获取技能正文"""
        skills = [
            SkillConfig(name="expert", description="专家技能", content="专家指令详细内容"),
        ]

        call_count = 0

        async def mock_chat(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # 第一轮：调用 load_skill
                yield StreamChunk(tool_calls=[
                    type('obj', (), {
                        'index': 0, 'id': 'call_ls',
                        'function': type('obj', (), {
                            'name': 'load_skill',
                            'arguments': '{"name": "expert"}'
                        })()
                    })()
                ])
                yield StreamChunk(finish_reason="tool_calls")
            else:
                # 第二轮：正常文本结束
                yield StreamChunk(content="已按专家指令执行", finish_reason="stop")
                yield StreamChunk(usage=TokenUsage(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15
                ))

        llm = AsyncMock()
        llm.chat = mock_chat

        agent = Agent(
            llm=llm,
            system_prompt="你是助手",
            skills=skills,
            register_catalog=False,
            register_skills=True,
        )

        events = []
        async for event in agent.run("请帮我处理任务"):
            events.append(event)

        # 应有 ToolResult 表示 load_skill 成功返回正文
        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert not tool_results[0].is_error
        assert "专家指令详细内容" in tool_results[0].output

    @pytest.mark.asyncio
    async def test_skills_dir_loading(self):
        """skills_dir 参数应加载目录中的技能"""
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_content = """---
name: dir-test
description: 目录测试技能
---

目录测试指令
"""
            with open(os.path.join(tmpdir, "dir-test.md"), "w", encoding="utf-8") as f:
                f.write(skill_content)

            llm = AsyncMock()
            agent = Agent(
                llm=llm,
                system_prompt="你是助手",
                skills_dir=tmpdir,
                register_catalog=False,
                register_skills=False,
            )

            assert "dir-test" in agent.skill_registry.list_names()

            # system prompt 应包含目录技能的元数据
            agent._build_system_prompt()  # 手动触发动态拼装
            system_msg = agent.history.all_messages[0]
            assert "dir-test" in system_msg.content
            assert "目录测试技能" in system_msg.content
            # 不应包含正文
            assert "目录测试指令" not in system_msg.content


# ---------------------------------------------------------------------------
# 第三方移植技能（anthropics/skills Apache-2.0 + obra/superpowers MIT）
# ---------------------------------------------------------------------------

_VENDORED = (
    "frontend-design", "internal-comms", "mcp-builder",
    "systematic-debugging", "test-driven-development",
)


class TestVendoredBuiltinSkills:
    """移植技能加载 + 多文件技能目录注入"""

    def _registry(self) -> SkillRegistry:
        from milu.resources import builtin_skills_dir
        reg = SkillRegistry()
        reg.load_from_directory(str(builtin_skills_dir()))
        return reg

    def test_vendored_skills_loaded(self):
        """5 个移植技能均能从内置目录加载"""
        names = set(self._registry().list_names())
        for n in _VENDORED:
            assert n in names, f"移植技能 {n} 未加载"

    def test_multifile_skill_injects_dir(self):
        """多文件技能：load_skill 注明技能目录绝对路径，且目录真实存在"""
        import re
        from pathlib import Path

        out = self._registry().load_skill("internal-comms")
        assert "[技能目录]" in out
        m = re.search(r"\[技能目录\] (.+)", out)
        skill_dir = Path(m.group(1).strip())
        assert skill_dir.is_dir()
        assert (skill_dir / "examples").is_dir(), "附属 examples/ 应一并移植"

    def test_flat_skill_no_dir_note(self):
        """平铺单文件技能不注入目录说明"""
        out = self._registry().load_skill("skill-creator")
        assert not out.startswith("错误")  # 守卫：确认技能真实加载（未知技能返回错误文本而非抛异常）
        assert "[技能目录]" not in out

    def test_licenses_present(self):
        """每个移植技能目录均保留上游 LICENSE.txt（合规要求）"""
        from milu.resources import builtin_skills_dir
        for n in _VENDORED:
            assert (builtin_skills_dir() / n / "LICENSE.txt").exists(), f"{n} 缺 LICENSE.txt"
