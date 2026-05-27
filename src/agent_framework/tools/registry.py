"""工具注册表 - 管理 @tool 装饰过的函数"""
from __future__ import annotations

import logging

from agent_framework.tools.decorator import ToolWrapper

logger = logging.getLogger(__name__)

# 注释：
# 这里相当于把所有装饰后的方法集中保存在这里。
# 并且增加get_schemas和其他统一管理功能。
class ToolRegistry:
    """工具注册表 - 管理 @tool 装饰过的函数。

    维护两个工具池：
    - 活跃池 (_tools)：工具 schema 注入 LLM，可被直接调用
    - 休眠池 (_dormant)：工具已注册但不注入 LLM，需先激活
    """

    def __init__(self):
        self._tools: dict[str, ToolWrapper] = {}       # 活跃池
        self._dormant: dict[str, ToolWrapper] = {}     # 休眠池

    def register(self, func) -> None:
        """注册一个 @tool 装饰的函数（进入活跃池）"""
        if not hasattr(func, "_tool_wrapper"):
            raise ValueError(f"函数 {func.__name__} 未被 @tool 装饰")

        wrapper: ToolWrapper = func._tool_wrapper
        self._tools[wrapper.name] = wrapper

    def register_many(self, funcs: list) -> None:
        """批量注册"""
        for func in funcs:
            self.register(func)

    def register_wrapper(self, wrapper: ToolWrapper) -> None:
        """直接注册 ToolWrapper 实例到活跃池（无需 @tool 装饰器，用于 MCP 等外部工具）"""
        if wrapper.name in self._tools:
            logger.warning("工具 %s 已存在，将被覆盖", wrapper.name)
        self._tools[wrapper.name] = wrapper

    # -- 休眠池管理 ----------------------------------------------------------

    def register_dormant(self, wrapper: ToolWrapper, category: str = "") -> None:
        """注册工具到休眠池。

        Args:
            wrapper: 工具包装器
            category: 分类标签，用于 list_catalog 分组显示。
                      为空时自动从工具名前缀提取（如 fetch__html → fetch）。
        """
        if category:
            wrapper.category = category
        elif not wrapper.category:
            # 自动从工具名前缀提取分类
            wrapper.category = wrapper.name.split("__")[0] if "__" in wrapper.name else ""
        self._dormant[wrapper.name] = wrapper

    def activate(self, name: str) -> ToolWrapper | None:
        """将工具从休眠池移到活跃池。

        Returns:
            被激活的 ToolWrapper，如果工具不存在则返回 None
        """
        wrapper = self._dormant.pop(name, None)
        if wrapper:
            self._tools[name] = wrapper
            logger.debug("工具 %s 已激活", name)
        return wrapper

    def deactivate(self, name: str) -> ToolWrapper | None:
        """将工具从活跃池移回休眠池。

        元工具（meta=True）不可被移回休眠池。

        Returns:
            被停用的 ToolWrapper，如果工具不存在或不可停用则返回 None
        """
        wrapper = self._tools.get(name)
        if wrapper and wrapper.meta:
            return None
        wrapper = self._tools.pop(name, None)
        if wrapper:
            self._dormant[name] = wrapper
            logger.debug("工具 %s 已停用", name)
        return wrapper

    def list_dormant_tools(self) -> list[dict]:
        """返回休眠工具摘要列表。

        Returns:
            [{name, description, category}] 列表
        """
        return [
            {
                "name": w.name,
                "description": w.description[:100],
                "category": w.category,
            }
            for w in self._dormant.values()
        ]

    def search_dormant_tools(self, query: str, limit: int = 20) -> list[dict]:
        """关键词搜索休眠工具（支持多关键词、评分排序）。

        搜索逻辑：
        1. 将 query 按空格和标点拆分为多个关键词
        2. 对每个休眠工具计算命中分（name 和 description 中命中关键词的数量）
        3. 按命中分降序返回

        Args:
            query: 搜索关键词（支持空格分隔多词，如 "fetch url html"）
            limit: 最大返回数量

        Returns:
            匹配的 [{name, description, category}] 列表，按相关性排序
        """
        # 拆分关键词（小写，去重，过滤短词）
        import re
        keywords = list({kw for kw in re.split(r'[\s,\-_/]+', query.lower()) if len(kw) >= 2})
        if not keywords:
            # 降级：整个 query 作为单一关键词
            keywords = [query.lower()] if query.strip() else []

        scored: list[tuple[float, dict]] = []
        for w in self._dormant.values():
            name_lower = w.name.lower()
            desc_lower = w.description.lower()

            score = 0.0
            for kw in keywords:
                # name 命中权重更高（工具名通常最精准）
                if kw in name_lower:
                    score += 2.0
                if kw in desc_lower:
                    score += 1.0

            if score > 0:
                scored.append((score, {
                    "name": w.name,
                    "description": w.description[:200],
                    "category": w.category,
                }))

        # 按得分降序，取 top N
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def is_dormant(self, name: str) -> bool:
        """检查工具是否在休眠池中（O(1) 查找）。"""
        return name in self._dormant

    # -- 查询方法 ------------------------------------------------------------

    def get_schemas(self) -> list[dict]:
        """
        返回活跃池工具的 OpenAI function calling schema 列表。
        仅包含活跃工具，休眠工具不会出现在 LLM 的工具列表中。
        可直接传给 llm.chat(tools=...) 使用。
        """
        schemas = []
        for wrapper in self._tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": wrapper.name,
                    "description": wrapper.description,
                    "parameters": wrapper.parameters_schema,
                },
            })
        return schemas

    def get_tool(self, name: str) -> ToolWrapper | None:
        """根据名称获取工具（活跃池优先，再查休眠池）。

        搜索双池使 ToolExecutor 能找到已激活的工具；
        休眠工具的调用拦截由 ToolExecutor 层负责。
        """
        return self._tools.get(name) or self._dormant.get(name)

    def list_tools(self) -> list[str]:
        """列出活跃池中的工具名"""
        return list(self._tools.keys())

    def list_dormant_names(self) -> list[str]:
        """列出休眠池中的工具名"""
        return list(self._dormant.keys())
