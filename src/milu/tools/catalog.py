"""工具目录元工具 - 让 LLM 自主发现和激活休眠工具。

提供 3 个元工具：
- list_catalog：列出所有休眠工具，按分类分组
- search_tools：关键词搜索休眠工具
- activate_tools：激活指定工具，使其出现在 LLM 工具列表中

使用方式：
    from milu.tools.catalog import create_catalog_tools
    from milu.tools.registry import ToolRegistry

    registry = ToolRegistry()
    catalog_tools = create_catalog_tools(registry)
    registry.register_many(catalog_tools)
"""

from milu.tools.decorator import tool
from milu.tools.registry import ToolRegistry


def create_catalog_tools(registry: ToolRegistry) -> list:
    """创建 3 个元工具，绑定到给定 registry。

    Args:
        registry: 工具注册表实例，元工具通过闭包持有引用

    Returns:
        @tool 装饰的函数列表，可直接传给 registry.register_many()
    """

    @tool(
        name="list_catalog",
        description=(
            "列出所有可用但未激活的工具目录。返回工具名、描述和分类。"
            "用于了解有哪些外部工具可供使用。"
        ),
    )
    async def list_catalog() -> str:
        """列出所有休眠工具，按 category 分组显示。"""
        tools = registry.list_dormant_tools()
        if not tools:
            return "当前没有待激活的工具。所有工具已处于活跃状态。"

        # 按 category 分组
        grouped: dict[str, list] = {}
        for t in tools:
            cat = t["category"] or "未分类"
            grouped.setdefault(cat, []).append(t)

        lines = ["可用工具目录："]
        for cat, items in grouped.items():
            lines.append(f"\n【{cat}】({len(items)} 个工具)")
            for item in items:
                lines.append(f"  - {item['name']}: {item['description']}")
        return "\n".join(lines)

    @tool(
        name="search_tools",
        description=(
            "按关键词搜索可用工具。输入关键词，返回匹配的工具名和描述。"
            "用于查找特定功能的工具。"
        ),
        is_safe=True,
    )
    async def search_tools(query: str) -> str:
        """搜索休眠工具，返回匹配结果。"""
        results = registry.search_dormant_tools(query)
        if not results:
            return f"未找到匹配 '{query}' 的工具。"

        lines = [f"搜索 '{query}' 的结果（{len(results)} 个）："]
        for r in results:
            lines.append(f"  - {r['name']}: {r['description']}")
        lines.append("\n使用 activate_tools 激活需要的工具。")
        return "\n".join(lines)

    @tool(
        name="activate_tools",
        description=(
            "激活一个或多个工具，使其可以被调用。"
            "传入工具名列表。激活后的工具会出现在可用工具中。"
        ),
        is_safe=True,
    )
    async def activate_tools(tool_names: list[str]) -> str:
        """激活指定的休眠工具。"""
        if not tool_names:
            return "没有指定要激活的工具。"

        activated = []
        not_found = []
        already_active = []

        for name in tool_names:
            if name in registry.list_tools():
                already_active.append(name)
            elif registry.activate(name):
                activated.append(name)
            else:
                not_found.append(name)

        parts = []
        if activated:
            parts.append(f"已激活: {', '.join(activated)}")
        if already_active:
            parts.append(f"已处于活跃状态: {', '.join(already_active)}")
        if not_found:
            parts.append(f"未找到: {', '.join(not_found)}")
        return "\n".join(parts)

    tools = [list_catalog, search_tools, activate_tools]

    # 标记为元工具，使其不可被 deactivate
    for t in tools:
        t._tool_wrapper.meta = True

    return tools


def render_catalog_prompt(registry: ToolRegistry, per_group_limit: int = 12) -> str:
    """渲染休眠（未激活）工具的能力清单，供注入 system prompt。

    休眠池里的工具（主要是 MCP 工具）默认不进 LLM 工具列表，模型若不知道
    它们的存在就不会去激活——本函数把"有哪些可激活的能力"按服务/分类分组、
    紧凑列出工具**全名**，让模型知道有什么、需要时主动用 activate_tools 激活。
    与技能"元数据常驻 prompt、正文按需 load_skill"同思路：这里只放工具全名
    （省 token，全名保证可直接传给 activate_tools 命中），完整描述留给
    list_catalog/search_tools 按需查询。

    Args:
        registry: 工具注册表
        per_group_limit: 每个分组最多列出的工具名数量，超出显示 "…(+N)"
                         （详情可经 list_catalog 查看），防止 prompt 膨胀

    Returns:
        Markdown 文本；无休眠工具时返回空串（如 mcp_tools_active_by_default
        =True、或未配置 MCP），调用方据此决定是否注入。
    """
    tools = registry.list_dormant_tools()
    if not tools:
        return ""

    # 按 category 分组（MCP 工具的 category 即服务器名，如 playwright/fetch）
    grouped: dict[str, list[str]] = {}
    for t in tools:
        cat = t["category"] or "未分类"
        grouped.setdefault(cat, []).append(t["name"])

    lines = [
        "\n\n## 可激活的外部工具（已连接但未激活）",
        "以下能力已就绪但默认未加载，**需要时先用 `activate_tools` 激活对应工具"
        "再调用**（也可用 `list_catalog`/`search_tools` 查看完整描述）：",
    ]
    for cat in sorted(grouped):
        names = grouped[cat]
        shown = names[:per_group_limit]
        suffix = (
            f" …(+{len(names) - per_group_limit})"
            if len(names) > per_group_limit else ""
        )
        lines.append(f"- **{cat}**（{len(names)} 个）：{', '.join(shown)}{suffix}")
    return "\n".join(lines)
