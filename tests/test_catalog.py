"""测试工具目录元工具 (catalog.py)"""
import pytest
from milu.tools import ToolRegistry, ToolWrapper
from milu.tools.catalog import create_catalog_tools, render_catalog_prompt


def _make_wrapper(name: str, description: str = "") -> ToolWrapper:
    """创建测试用 ToolWrapper"""
    async def _noop(**kwargs):
        return "ok"
    return ToolWrapper(
        name=name,
        description=description or f"Tool {name}",
        parameters_schema={"type": "object", "properties": {}},
        func=_noop,
        is_async=True,
    )


@pytest.fixture
def registry():
    """创建带元工具和休眠工具的注册表"""
    reg = ToolRegistry()
    # 注册元工具
    catalog = create_catalog_tools(reg)
    reg.register_many(catalog)
    # 注册一些休眠工具
    reg.register_dormant(_make_wrapper("fetch__html", "获取URL的HTML内容"), category="fetch")
    reg.register_dormant(_make_wrapper("fetch__markdown", "获取URL的Markdown内容"), category="fetch")
    reg.register_dormant(_make_wrapper("fetch__json", "获取URL的JSON数据"), category="fetch")
    reg.register_dormant(_make_wrapper("figma__create_file", "创建Figma文件"), category="figma")
    reg.register_dormant(_make_wrapper("figma__get_design", "获取Figma设计"), category="figma")
    return reg


@pytest.fixture
def catalog_funcs(registry):
    """获取元工具函数"""
    return [
        registry.get_tool("list_catalog"),
        registry.get_tool("search_tools"),
        registry.get_tool("activate_tools"),
    ]


class TestCreateCatalogTools:
    """工厂函数测试"""

    def test_returns_three_tools(self):
        """返回 3 个元工具"""
        reg = ToolRegistry()
        tools = create_catalog_tools(reg)
        assert len(tools) == 3

    def test_tool_names(self):
        """元工具名称正确"""
        reg = ToolRegistry()
        tools = create_catalog_tools(reg)
        names = {t._tool_wrapper.name for t in tools}
        assert names == {"list_catalog", "search_tools", "activate_tools"}

    def test_tools_have_schemas(self):
        """元工具有完整的 schema"""
        reg = ToolRegistry()
        tools = create_catalog_tools(reg)
        for t in tools:
            w = t._tool_wrapper
            assert w.name
            assert w.description
            assert w.parameters_schema is not None


class TestListCatalog:
    """list_catalog 元工具测试"""

    @pytest.mark.asyncio
    async def test_list_catalog_shows_grouped(self, registry):
        """按分类分组显示休眠工具"""
        wrapper = registry.get_tool("list_catalog")
        result = await wrapper.func()

        assert "fetch" in result
        assert "figma" in result
        assert "fetch__html" in result
        assert "figma__create_file" in result
        assert "3 个工具" in result  # fetch 有 3 个
        assert "2 个工具" in result  # figma 有 2 个

    @pytest.mark.asyncio
    async def test_list_catalog_empty(self):
        """无休眠工具时返回提示"""
        reg = ToolRegistry()
        catalog = create_catalog_tools(reg)
        reg.register_many(catalog)

        wrapper = reg.get_tool("list_catalog")
        result = await wrapper.func()
        assert "没有待激活的工具" in result

    @pytest.mark.asyncio
    async def test_list_catalog_after_activate(self, registry):
        """激活部分工具后只显示剩余休眠工具"""
        registry.activate("fetch__html")

        wrapper = registry.get_tool("list_catalog")
        result = await wrapper.func()

        # fetch__html 不应出现在列表中
        assert "fetch__html" not in result
        # 其他 fetch 工具仍在
        assert "fetch__markdown" in result
        assert "fetch__json" in result
        assert "2 个工具" in result  # fetch 只剩 2 个


class TestSearchTools:
    """search_tools 元工具测试"""

    @pytest.mark.asyncio
    async def test_search_by_keyword(self, registry):
        """关键词搜索"""
        wrapper = registry.get_tool("search_tools")
        result = await wrapper.func(query="html")

        assert "fetch__html" in result
        assert "fetch__markdown" not in result

    @pytest.mark.asyncio
    async def test_search_by_server(self, registry):
        """按服务器名搜索"""
        wrapper = registry.get_tool("search_tools")
        result = await wrapper.func(query="figma")

        assert "figma__create_file" in result
        assert "figma__get_design" in result
        assert "fetch__html" not in result

    @pytest.mark.asyncio
    async def test_search_no_results(self, registry):
        """无匹配结果"""
        wrapper = registry.get_tool("search_tools")
        result = await wrapper.func(query="nonexistent")

        assert "未找到" in result

    @pytest.mark.asyncio
    async def test_search_shows_activate_hint(self, registry):
        """搜索结果包含激活提示"""
        wrapper = registry.get_tool("search_tools")
        result = await wrapper.func(query="fetch")

        assert "activate_tools" in result


class TestActivateTools:
    """activate_tools 元工具测试"""

    @pytest.mark.asyncio
    async def test_activate_single(self, registry):
        """激活单个工具"""
        wrapper = registry.get_tool("activate_tools")
        result = await wrapper.func(tool_names=["fetch__html"])

        assert "已激活" in result
        assert "fetch__html" in result
        assert "fetch__html" in registry.list_tools()
        assert "fetch__html" not in registry.list_dormant_names()

    @pytest.mark.asyncio
    async def test_activate_multiple(self, registry):
        """批量激活多个工具"""
        wrapper = registry.get_tool("activate_tools")
        result = await wrapper.func(tool_names=["fetch__html", "fetch__markdown"])

        assert "fetch__html" in result
        assert "fetch__markdown" in result
        assert "fetch__html" in registry.list_tools()
        assert "fetch__markdown" in registry.list_tools()

    @pytest.mark.asyncio
    async def test_activate_already_active(self, registry):
        """激活已活跃的工具"""
        registry.activate("fetch__html")

        wrapper = registry.get_tool("activate_tools")
        result = await wrapper.func(tool_names=["fetch__html"])

        assert "已处于活跃状态" in result
        assert "fetch__html" in result

    @pytest.mark.asyncio
    async def test_activate_not_found(self, registry):
        """激活不存在的工具"""
        wrapper = registry.get_tool("activate_tools")
        result = await wrapper.func(tool_names=["nonexistent_tool"])

        assert "未找到" in result
        assert "nonexistent_tool" in result

    @pytest.mark.asyncio
    async def test_activate_mixed(self, registry):
        """混合场景：部分成功、部分已活跃、部分不存在"""
        registry.activate("fetch__html")

        wrapper = registry.get_tool("activate_tools")
        result = await wrapper.func(tool_names=[
            "fetch__html",      # 已活跃
            "fetch__markdown",  # 正常激活
            "ghost_tool",       # 不存在
        ])

        assert "已处于活跃状态" in result
        assert "已激活" in result
        assert "未找到" in result

    @pytest.mark.asyncio
    async def test_activate_empty_list(self, registry):
        """空列表返回提示"""
        wrapper = registry.get_tool("activate_tools")
        result = await wrapper.func(tool_names=[])
        assert "没有指定" in result


class TestMetaToolsSchema:
    """元工具出现在活跃 schema 中"""

    def test_meta_tools_in_schemas(self, registry):
        """元工具始终在 get_schemas 结果中"""
        schemas = registry.get_schemas()
        names = [s["function"]["name"] for s in schemas]

        assert "list_catalog" in names
        assert "search_tools" in names
        assert "activate_tools" in names

    def test_dormant_tools_not_in_schemas(self, registry):
        """休眠工具不在 get_schemas 结果中"""
        schemas = registry.get_schemas()
        names = [s["function"]["name"] for s in schemas]

        assert "fetch__html" not in names
        assert "figma__create_file" not in names

    def test_activated_tool_in_schemas(self, registry):
        """激活后工具出现在 schemas 中"""
        registry.activate("fetch__html")

        schemas = registry.get_schemas()
        names = [s["function"]["name"] for s in schemas]

        assert "fetch__html" in names
        assert "figma__create_file" not in names  # 未激活的仍不在


class TestRenderCatalogPrompt:
    """render_catalog_prompt：把休眠工具清单渲染进 system prompt"""

    def test_renders_grouped_with_full_names(self, registry):
        """按分类分组、列出工具全名（保证可直接 activate_tools 激活）"""
        text = render_catalog_prompt(registry)
        # 分组标题
        assert "fetch" in text
        assert "figma" in text
        # 工具全名（带 server__ 前缀，模型据此激活才能命中休眠池）
        assert "fetch__html" in text
        assert "figma__create_file" in text
        # 数量与激活引导
        assert "（3 个）" in text  # fetch 3 个
        assert "（2 个）" in text  # figma 2 个
        assert "activate_tools" in text

    def test_empty_when_no_dormant(self):
        """无休眠工具时返回空串（如 mcp_tools_active_by_default=True / 无 MCP）"""
        reg = ToolRegistry()
        reg.register_many(create_catalog_tools(reg))  # 元工具在活跃池，不算休眠
        assert render_catalog_prompt(reg) == ""

    def test_empty_after_all_activated(self, registry):
        """全部激活后不再渲染（清单只反映尚未激活的）"""
        for name in list(registry.list_dormant_names()):
            registry.activate(name)
        assert render_catalog_prompt(registry) == ""

    def test_per_group_limit_truncates(self):
        """单组工具过多时截断显示 …(+N)，防 prompt 膨胀"""
        reg = ToolRegistry()
        for i in range(20):
            reg.register_dormant(
                _make_wrapper(f"big__tool_{i}", f"工具 {i}"), category="big"
            )
        text = render_catalog_prompt(reg, per_group_limit=12)
        assert "（20 个）" in text
        assert "…(+8)" in text          # 20 - 12 = 8 个未列出
        assert "big__tool_0" in text     # 前 12 个里有
        assert "big__tool_19" not in text  # 被截断的不出现


class TestEndToEndFlow:
    """端到端流程：搜索 → 激活 → 可调用"""

    @pytest.mark.asyncio
    async def test_search_activate_verify(self, registry):
        """完整流程：搜索工具 → 激活 → 验证可用"""
        # 1. 搜索
        search = registry.get_tool("search_tools")
        result = await search.func(query="markdown")
        assert "fetch__markdown" in result

        # 2. 激活
        activate = registry.get_tool("activate_tools")
        result = await activate.func(tool_names=["fetch__markdown"])
        assert "已激活" in result

        # 3. 验证活跃
        assert "fetch__markdown" in registry.list_tools()
        schemas = registry.get_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert "fetch__markdown" in names

        # 4. 验证可执行
        wrapper = registry.get_tool("fetch__markdown")
        assert wrapper is not None
        output = await wrapper.func()
        assert output == "ok"
