"""SkillRegistry — 轻量技能注册表

设计原则（参考 Claude Code skill 模式）：
  - 所有技能的元数据（name/description/triggers）始终注入 system prompt
  - 技能正文（body）不放入 system prompt，由 LLM 按需调用 load_skill 获取
  - 无激活/卸载生命周期，所有已发现的技能始终可用
"""
from __future__ import annotations

import logging
from pathlib import Path

from agent_framework.skills.config import SkillConfig
from agent_framework.tools.decorator import ToolWrapper

logger = logging.getLogger(__name__)

_LOAD_SKILL_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "要加载的技能名称",
        },
    },
    "required": ["name"],
}


class SkillRegistry:
    """技能注册表 — 扫描目录、维护目录、按需返回正文。"""

    def __init__(self) -> None:
        self._skills: dict[str, SkillConfig] = {}

    # -- 注册 --------------------------------------------------------------

    def add(self, config: SkillConfig) -> None:
        """注册一个技能（同名覆盖）。"""
        self._skills[config.name] = config
        logger.debug("技能 %s 已注册", config.name)

    # -- 目录扫描 ----------------------------------------------------------

    def load_from_directory(self, path: str) -> int:
        """扫描目录加载所有 .md 技能文件。

        支持两种布局：
        - 平铺：skills/code-review.md
        - 子目录：skills/code-review/SKILL.md

        Returns:
            成功加载的技能数量
        """
        root = Path(path)
        if not root.is_dir():
            logger.warning("技能目录不存在: %s", path)
            return 0

        count = 0
        for md_file in sorted(root.glob("*.md")):
            try:
                self.add(SkillConfig.from_file(str(md_file)))
                count += 1
            except Exception as e:
                logger.warning("加载技能文件失败 %s: %s", md_file, e)

        for md_file in sorted(root.glob("*/SKILL.md")):
            try:
                self.add(SkillConfig.from_file(str(md_file)))
                count += 1
            except Exception as e:
                logger.warning("加载技能文件失败 %s: %s", md_file, e)

        logger.info("从 %s 加载了 %d 个技能", path, count)
        return count

    # -- 查询 --------------------------------------------------------------

    def describe_available(self) -> str:
        """返回所有技能的元数据目录（注入 system prompt 用）。

        格式示例：
            可用技能（使用 load_skill 加载完整指令）：
            - code-review: 代码审查专家，关注安全性、性能和可维护性 [审查, review, CR]
            - translator: 多语言翻译专家 [翻译, translate]
        """
        if not self._skills:
            return ""

        lines = ["可用技能（使用 load_skill 加载完整指令）："]
        for cfg in sorted(self._skills.values(), key=lambda c: c.name):
            trigger_part = f" [{', '.join(cfg.triggers)}]" if cfg.triggers else ""
            lines.append(f"- {cfg.name}: {cfg.description}{trigger_part}")
        return "\n".join(lines)

    def load_skill(self, name: str) -> str:
        """返回技能的完整正文（作为 load_skill 工具的返回值）。

        格式：
            <skill name="code-review">
            正文内容...
            </skill>
        """
        cfg = self._skills.get(name)
        if not cfg:
            known = ", ".join(sorted(self._skills)) or "(无)"
            return f"错误：未知技能 '{name}'。可用技能：{known}"

        return f'<skill name="{cfg.name}">\n{cfg.content}\n</skill>'

    def as_tool(self) -> ToolWrapper:
        """将 load_skill 包装为 ToolWrapper，可直接注册到 ToolRegistry。"""
        return ToolWrapper(
            name="load_skill",
            description=(
                "加载指定技能的完整指令。"
                "系统提示中已列出所有可用技能的名称和描述，"
                "当你需要使用某个技能的专项指令时，调用此工具获取完整内容。"
            ),
            parameters_schema=_LOAD_SKILL_SCHEMA,
            func=self.load_skill,
            is_async=False,
            meta=True,
        )

    def get(self, name: str) -> SkillConfig | None:
        """获取技能配置。"""
        return self._skills.get(name)

    def list_names(self) -> list[str]:
        """列出所有已注册技能名。"""
        return sorted(self._skills.keys())

    def __len__(self) -> int:
        return len(self._skills)
