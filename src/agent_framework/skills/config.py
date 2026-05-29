"""Skill 配置 — 从 Markdown（YAML frontmatter）解析技能定义"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import yaml


@dataclass
class SkillConfig:
    """单个技能的配置。

    :param name: 技能唯一标识（如 "code-review"）
    :param description: 技能描述（展示在 system prompt 目录中）
    :param triggers: 触发关键词列表（辅助 LLM 判断何时加载）
    :param content: 技能指令正文（通过 load_skill 工具按需返回给 LLM）
    :param source: 来源文件路径（用于调试）
    """
    name: str
    description: str
    triggers: list[str] = field(default_factory=list)
    content: str = ""
    source: str = ""

    @classmethod
    def from_markdown(cls, text: str, source: str = "") -> SkillConfig:
        """从 Markdown 文本（YAML frontmatter + 正文）解析。

        格式：
            ---
            name: code-review
            description: 代码审查专家
            triggers: [审查, review]
            ---

            正文内容（通过 load_skill 按需加载）

        Args:
            text: 完整的 Markdown 文本（含 frontmatter）
            source: 来源路径（记录到 SkillConfig.source）

        Returns:
            解析后的 SkillConfig

        Raises:
            ValueError: frontmatter 缺失必填字段
        """
        fm, body = _split_frontmatter(text)
        if fm is None:
            raise ValueError("Markdown 文本缺少 YAML frontmatter（以 --- 开头和结尾）")

        data = yaml.safe_load(fm)
        if not isinstance(data, dict):
            raise ValueError("YAML frontmatter 必须是字典格式")

        return cls._build(data, body, source)

    @classmethod
    def from_file(cls, path: str) -> SkillConfig:
        """读取 .md 文件并解析。

        Args:
            path: Markdown 文件路径

        Returns:
            解析后的 SkillConfig
        """
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls.from_markdown(text, source=os.path.abspath(path))

    @classmethod
    def from_dict(cls, name: str, data: dict) -> SkillConfig:
        """从字典构造。"""
        data = dict(data)
        data.setdefault("name", name)
        return cls._build(data, data.pop("content", ""), source="")

    @classmethod
    def _build(cls, data: dict, body: str, source: str) -> SkillConfig:
        """从解析后的数据构造 SkillConfig。"""
        name = data.get("name")
        if not name:
            raise ValueError("Skill 缺少必填字段: name")
        description = data.get("description", "")
        if not description:
            raise ValueError(f"Skill '{name}' 缺少必填字段: description")

        return cls(
            name=name,
            description=description,
            triggers=data.get("triggers") or [],
            content=body.strip(),
            source=source,
        )


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

_FM_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)", re.DOTALL)


def _split_frontmatter(text: str) -> tuple[str | None, str]:
    """将文本拆分为 YAML frontmatter 和正文。

    Returns:
        (frontmatter_str, body_str) — 无 frontmatter 时返回 (None, text)
    """
    m = _FM_PATTERN.match(text.strip())
    if m:
        return m.group(1), m.group(2)
    return None, text
