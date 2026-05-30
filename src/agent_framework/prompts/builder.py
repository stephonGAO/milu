"""PromptBuilder — 从 Markdown 文件目录分层拼装系统提示词

设计原则：
  - 每个 .md 文件代表一个提示词片段（safeguard / soul / agent / memory）
  - YAML frontmatter 控制拼装行为（section、order、enabled）
  - 每次 build() 重新读取文件，支持热重载（编辑即生效，无需重启）
  - {{key}} 变量插值，运行时注入动态值

用法：
    from agent_framework.prompts import PromptBuilder

    builder = PromptBuilder("config/prompts/main")
    prompt = builder.build(user_name="张三")
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# frontmatter 正则 — 复用 SkillConfig 的模式
_FM_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)", re.DOTALL)

# 变量插值模式：{{key}}
_VAR_PATTERN = re.compile(r"\{\{(\w+)\}\}")


@dataclass
class PromptSection:
    """单个提示词片段（对应一个 .md 文件）。

    :param section: 分类标识（safeguard / soul / agent / memory / custom）
    :param order: 排序权重（数字越小越靠前，同 section 内排序）
    :param enabled: 是否启用（False 时跳过此文件）
    :param content: Markdown 正文（frontmatter 之后的部分）
    :param source: 来源文件路径（用于调试日志）
    """
    section: str
    order: int
    enabled: bool
    content: str
    source: str = ""


# section 间的全局排序（不同 section 之间的拼装顺序）
_SECTION_ORDER: dict[str, int] = {
    "safeguard": 0,   # 安全约束 — 最高优先级，最先注入
    "soul": 1,        # 人格定义 — 确立角色身份
    "agent": 2,       # 工作指令 — 核心行为规范和 SOP
    "memory": 3,      # 上下文记忆 — 用户维护的参考信息
    "custom": 4,      # 自定义扩展 — 兜底
}


class PromptBuilder:
    """从文件目录拼装系统提示词。

    用法：
        builder = PromptBuilder("config/prompts/main")
        prompt = builder.build()                    # 无变量
        prompt = builder.build(user_name="张三")     # 带变量插值

    每次调用 build() 都会重新读取文件，实现热重载。
    """

    def __init__(self, prompt_dir: str | Path) -> None:
        self._prompt_dir = Path(prompt_dir)

    @property
    def prompt_dir(self) -> Path:
        return self._prompt_dir

    def build(self, **variables: str) -> str:
        """加载 .md 文件 → 解析 frontmatter → 排序拼装 → 变量插值。

        :param variables: 变量键值对，替换文件中的 {{key}}
        :return: 拼装后的完整提示词字符串
        """
        sections = self._load_sections()
        assembled = self._assemble(sections)
        if variables:
            assembled = self._interpolate(assembled, variables)
        return assembled

    def reload(self, **variables: str) -> str:
        """热重载别名 — 等价于 build()。"""
        return self.build(**variables)

    def list_files(self) -> list[dict]:
        """列出所有提示词文件及其元数据（调试用）。

        :return: 字典列表，每项包含 file, section, order, enabled
        """
        sections = self._load_sections()
        return [
            {
                "file": os.path.basename(s.source),
                "section": s.section,
                "order": s.order,
                "enabled": s.enabled,
            }
            for s in sections
        ]

    # ── 内部方法 ──

    def _load_sections(self) -> list[PromptSection]:
        """扫描目录，加载所有 .md 文件并解析 frontmatter。"""
        if not self._prompt_dir.is_dir():
            raise FileNotFoundError(f"提示词目录不存在: {self._prompt_dir}")
        sections: list[PromptSection] = []
        for md_file in sorted(self._prompt_dir.glob("*.md")):
            try:
                section = self._parse_file(md_file)
                if section and section.enabled:
                    sections.append(section)
            except Exception as e:
                logger.warning("加载提示词文件失败 %s: %s", md_file, e)
        return sections

    def _parse_file(self, path: Path) -> PromptSection | None:
        """解析单个 .md 文件，返回 PromptSection。"""
        text = path.read_text(encoding="utf-8")
        fm, body = self._split_frontmatter(text)
        if fm is None:
            # 无 frontmatter 时，用文件名推断 section
            return PromptSection(
                section=path.stem, order=50, enabled=True,
                content=text.strip(), source=str(path),
            )
        data = yaml.safe_load(fm)
        if not isinstance(data, dict):
            logger.warning("frontmatter 格式错误 %s，跳过", path)
            return None
        section_name = data.get("section", path.stem)
        order = data.get("order", _SECTION_ORDER.get(section_name, 50))
        enabled = data.get("enabled", True)
        return PromptSection(
            section=section_name, order=order, enabled=enabled,
            content=body.strip(), source=str(path),
        )

    def _assemble(self, sections: list[PromptSection]) -> str:
        """按 section 全局排序 + 同 section 内 order 排序，拼装内容。"""
        sorted_sections = sorted(
            sections,
            key=lambda s: (_SECTION_ORDER.get(s.section, 99), s.order),
        )
        parts = [s.content for s in sorted_sections if s.content]
        return "\n\n".join(parts)

    @staticmethod
    def _interpolate(text: str, variables: dict[str, str]) -> str:
        """替换 {{key}} 为变量值，未提供的 key 保留原样。"""
        def _replace(match: re.Match) -> str:
            key = match.group(1)
            return variables.get(key, match.group(0))
        return _VAR_PATTERN.sub(_replace, text)

    @staticmethod
    def _split_frontmatter(text: str) -> tuple[str | None, str]:
        """将文本拆分为 YAML frontmatter 和正文。"""
        m = _FM_PATTERN.match(text.strip())
        if m:
            return m.group(1), m.group(2)
        return None, text
