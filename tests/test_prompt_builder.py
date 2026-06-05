"""测试 PromptBuilder — 文件化提示词拼装"""
import os
import tempfile

import pytest

from milu.prompts.builder import PromptBuilder, PromptSection


class TestPromptSection:

    def test_fields(self):
        """所有字段应正确设置"""
        s = PromptSection(section="soul", order=0, enabled=True, content="内容")
        assert s.section == "soul"
        assert s.order == 0
        assert s.enabled is True
        assert s.content == "内容"
        assert s.source == ""


class TestPromptBuilderBasic:

    def test_build_simple(self):
        """加载简单目录并拼装"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "soul.md"), "w", encoding="utf-8") as f:
                f.write("---\nsection: soul\norder: 0\n---\n\n你是助手。\n")
            with open(os.path.join(tmpdir, "agent.md"), "w", encoding="utf-8") as f:
                f.write("---\nsection: agent\norder: 0\n---\n\n使用工具完成任务。\n")

            builder = PromptBuilder(tmpdir)
            result = builder.build()
            assert "你是助手。" in result
            assert "使用工具完成任务。" in result
            # soul 应在 agent 前面
            assert result.index("你是助手") < result.index("使用工具")

    def test_build_section_ordering(self):
        """不同 section 按全局排序：safeguard < soul < agent < memory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 故意按反序创建文件
            with open(os.path.join(tmpdir, "memory.md"), "w", encoding="utf-8") as f:
                f.write("---\nsection: memory\n---\n\n记忆内容\n")
            with open(os.path.join(tmpdir, "agent.md"), "w", encoding="utf-8") as f:
                f.write("---\nsection: agent\n---\n\n工作指令\n")
            with open(os.path.join(tmpdir, "soul.md"), "w", encoding="utf-8") as f:
                f.write("---\nsection: soul\n---\n\n人格定义\n")
            with open(os.path.join(tmpdir, "safeguard.md"), "w", encoding="utf-8") as f:
                f.write("---\nsection: safeguard\n---\n\n安全约束\n")

            builder = PromptBuilder(tmpdir)
            result = builder.build()
            assert result.index("安全约束") < result.index("人格定义")
            assert result.index("人格定义") < result.index("工作指令")
            assert result.index("工作指令") < result.index("记忆内容")

    def test_build_same_section_ordering(self):
        """同 section 内按 order 排序"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "a.md"), "w", encoding="utf-8") as f:
                f.write("---\nsection: agent\norder: 10\n---\n\n第二段\n")
            with open(os.path.join(tmpdir, "b.md"), "w", encoding="utf-8") as f:
                f.write("---\nsection: agent\norder: 1\n---\n\n第一段\n")

            builder = PromptBuilder(tmpdir)
            result = builder.build()
            assert result.index("第一段") < result.index("第二段")

    def test_build_disabled(self):
        """enabled: false 的文件应被跳过"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "soul.md"), "w", encoding="utf-8") as f:
                f.write("---\nsection: soul\nenabled: true\n---\n\n启用内容\n")
            with open(os.path.join(tmpdir, "disabled.md"), "w", encoding="utf-8") as f:
                f.write("---\nsection: agent\nenabled: false\n---\n\n禁用内容\n")

            builder = PromptBuilder(tmpdir)
            result = builder.build()
            assert "启用内容" in result
            assert "禁用内容" not in result

    def test_variable_interpolation(self):
        """{{key}} 应被替换"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "soul.md"), "w", encoding="utf-8") as f:
                f.write("---\nsection: soul\n---\n\n你好 {{user_name}}，欢迎 {{project}}。\n")

            builder = PromptBuilder(tmpdir)
            result = builder.build(user_name="张三", project="milu")
            assert "你好 张三" in result
            assert "欢迎 milu" in result

    def test_variable_interpolation_missing(self):
        """未提供的 {{key}} 应保留原样"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "soul.md"), "w", encoding="utf-8") as f:
                f.write("---\nsection: soul\n---\n\n你好 {{user_name}}，{{missing}}。\n")

            builder = PromptBuilder(tmpdir)
            result = builder.build(user_name="李四")
            assert "你好 李四" in result
            assert "{{missing}}" in result

    def test_no_frontmatter(self):
        """无 frontmatter 时应使用文件名作为 section"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "soul.md"), "w", encoding="utf-8") as f:
                f.write("你是助手，没有 frontmatter。\n")

            builder = PromptBuilder(tmpdir)
            result = builder.build()
            assert "你是助手" in result

    def test_directory_not_found(self):
        """目录不存在时应抛 FileNotFoundError"""
        builder = PromptBuilder("/nonexistent/path")
        with pytest.raises(FileNotFoundError):
            builder.build()

    def test_empty_directory(self):
        """空目录应返回空字符串"""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = PromptBuilder(tmpdir)
            assert builder.build() == ""

    def test_reload_same_as_build(self):
        """reload() 应与 build() 等价"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "soul.md"), "w", encoding="utf-8") as f:
                f.write("---\nsection: soul\n---\n\n内容\n")
            builder = PromptBuilder(tmpdir)
            assert builder.build() == builder.reload()

    def test_prompt_dir_property(self):
        """prompt_dir 属性应返回 Path"""
        from pathlib import Path
        builder = PromptBuilder("/some/path")
        assert isinstance(builder.prompt_dir, Path)
        assert builder.prompt_dir == Path("/some/path")


class TestPromptBuilderHotReload:

    def test_hot_reload_detects_changes(self):
        """修改文件后 build() 应反映最新内容"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "soul.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("---\nsection: soul\n---\n\n原始内容\n")

            builder = PromptBuilder(tmpdir)
            assert "原始内容" in builder.build()

            with open(path, "w", encoding="utf-8") as f:
                f.write("---\nsection: soul\n---\n\n修改后内容\n")

            result2 = builder.build()
            assert "修改后内容" in result2
            assert "原始内容" not in result2

    def test_hot_reload_detects_new_files(self):
        """新增文件后 build() 应包含新内容"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "soul.md"), "w", encoding="utf-8") as f:
                f.write("---\nsection: soul\n---\n\n人格\n")

            builder = PromptBuilder(tmpdir)
            result1 = builder.build()
            assert "新增" not in result1

            with open(os.path.join(tmpdir, "agent.md"), "w", encoding="utf-8") as f:
                f.write("---\nsection: agent\n---\n\n新增指令\n")

            result2 = builder.build()
            assert "新增指令" in result2


class TestPromptBuilderListFiles:

    def test_list_files(self):
        """应列出所有文件及其元数据"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "soul.md"), "w", encoding="utf-8") as f:
                f.write("---\nsection: soul\norder: 0\n---\n\n内容\n")
            with open(os.path.join(tmpdir, "agent.md"), "w", encoding="utf-8") as f:
                f.write("---\nsection: agent\norder: 5\nenabled: true\n---\n\n内容\n")

            builder = PromptBuilder(tmpdir)
            files = builder.list_files()
            assert len(files) == 2
            names = {f["file"] for f in files}
            assert names == {"soul.md", "agent.md"}

    def test_list_files_empty(self):
        """空目录应返回空列表"""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = PromptBuilder(tmpdir)
            assert builder.list_files() == []

    def test_list_files_excludes_disabled(self):
        """list_files 不包含 disabled 文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "active.md"), "w", encoding="utf-8") as f:
                f.write("---\nsection: soul\nenabled: true\n---\n\n活跃\n")
            with open(os.path.join(tmpdir, "inactive.md"), "w", encoding="utf-8") as f:
                f.write("---\nsection: agent\nenabled: false\n---\n\n禁用\n")

            builder = PromptBuilder(tmpdir)
            files = builder.list_files()
            assert len(files) == 1
            assert files[0]["file"] == "active.md"
