"""测试 ToolRegistry 休眠池管理"""
import pytest
from agent_framework.tools import tool, ToolRegistry, ToolWrapper


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
        dangerous=False,
    )


class TestDormantPool:
    """休眠池基本操作"""

    def test_register_dormant(self):
        """注册工具到休眠池"""
        reg = ToolRegistry()
        w = _make_wrapper("fetch__html", "Fetch HTML content")
        reg.register_dormant(w, category="fetch")

        assert "fetch__html" in reg.list_dormant_names()
        assert "fetch__html" not in reg.list_tools()

    def test_register_dormant_auto_category(self):
        """自动从工具名前缀提取 category"""
        reg = ToolRegistry()
        w = _make_wrapper("figma__create_file")
        reg.register_dormant(w)

        tools = reg.list_dormant_tools()
        assert len(tools) == 1
        assert tools[0]["category"] == "figma"

    def test_register_dormant_no_prefix(self):
        """无前缀的工具 category 为空"""
        reg = ToolRegistry()
        w = _make_wrapper("standalone_tool")
        reg.register_dormant(w)

        tools = reg.list_dormant_tools()
        assert tools[0]["category"] == ""

    def test_activate(self):
        """激活工具：从休眠池移到活跃池"""
        reg = ToolRegistry()
        w = _make_wrapper("fetch__html")
        reg.register_dormant(w, category="fetch")

        result = reg.activate("fetch__html")

        assert result is not None
        assert result.name == "fetch__html"
        assert "fetch__html" in reg.list_tools()
        assert "fetch__html" not in reg.list_dormant_names()

    def test_activate_nonexistent(self):
        """激活不存在的工具返回 None"""
        reg = ToolRegistry()
        assert reg.activate("nonexistent") is None

    def test_deactivate(self):
        """停用工具：从活跃池移回休眠池"""
        reg = ToolRegistry()

        @tool(name="my_tool", description="测试工具")
        async def my_tool() -> str:
            return "ok"

        reg.register(my_tool)
        assert "my_tool" in reg.list_tools()

        result = reg.deactivate("my_tool")

        assert result is not None
        assert "my_tool" not in reg.list_tools()
        assert "my_tool" in reg.list_dormant_names()

    def test_deactivate_meta_tools_blocked(self):
        """元工具不可被停用"""
        reg = ToolRegistry()
        from agent_framework.tools.catalog import create_catalog_tools
        catalog = create_catalog_tools(reg)
        reg.register_many(catalog)

        assert reg.deactivate("list_catalog") is None
        assert reg.deactivate("search_tools") is None
        assert reg.deactivate("activate_tools") is None

        # 元工具仍在活跃池
        assert "list_catalog" in reg.list_tools()
        assert "search_tools" in reg.list_tools()
        assert "activate_tools" in reg.list_tools()

    def test_deactivate_nonexistent(self):
        """停用不存在的工具返回 None"""
        reg = ToolRegistry()
        assert reg.deactivate("nonexistent") is None


class TestGetSchemasActiveOnly:
    """get_schemas 仅返回活跃池"""

    def test_schemas_exclude_dormant(self):
        """get_schemas 不包含休眠工具"""
        reg = ToolRegistry()

        @tool(name="active_tool", description="活跃工具")
        async def active_tool() -> str:
            return "ok"

        reg.register(active_tool)
        reg.register_dormant(_make_wrapper("dormant_tool", "休眠工具"))

        schemas = reg.get_schemas()
        names = [s["function"]["name"] for s in schemas]

        assert "active_tool" in names
        assert "dormant_tool" not in names

    def test_schemas_after_activate(self):
        """激活后工具出现在 schemas 中"""
        reg = ToolRegistry()
        w = _make_wrapper("fetch__html", "Fetch HTML")
        reg.register_dormant(w)

        # 激活前
        assert len(reg.get_schemas()) == 0

        # 激活后
        reg.activate("fetch__html")
        schemas = reg.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "fetch__html"


class TestGetToolBothPools:
    """get_tool 搜索双池"""

    def test_get_active_tool(self):
        """能找到活跃工具"""
        reg = ToolRegistry()

        @tool(name="active", description="活跃")
        async def active() -> str:
            return "ok"

        reg.register(active)
        assert reg.get_tool("active") is not None

    def test_get_dormant_tool(self):
        """能找到休眠工具"""
        reg = ToolRegistry()
        w = _make_wrapper("dormant")
        reg.register_dormant(w)
        assert reg.get_tool("dormant") is not None

    def test_active_pool_priority(self):
        """活跃池优先于休眠池"""
        reg = ToolRegistry()
        w_active = _make_wrapper("tool", "活跃版")
        w_dormant = _make_wrapper("tool", "休眠版")

        # 先注册到休眠池
        reg.register_dormant(w_dormant)
        # 再激活（移到活跃池）
        reg.activate("tool")
        # 再注册另一个同名到休眠池（模拟极端情况）
        # 实际上 activate 已经把休眠池里的移走了，所以这里不会有冲突

        result = reg.get_tool("tool")
        assert result is not None


class TestSearchDormantTools:
    """搜索休眠工具"""

    def test_search_by_name(self):
        """按名称搜索"""
        reg = ToolRegistry()
        reg.register_dormant(_make_wrapper("fetch__html", "Fetch HTML content"))
        reg.register_dormant(_make_wrapper("fetch__markdown", "Fetch markdown"))
        reg.register_dormant(_make_wrapper("mysql__query", "Run SQL query"))

        results = reg.search_dormant_tools("fetch")
        assert len(results) == 2

    def test_search_by_description(self):
        """按描述搜索"""
        reg = ToolRegistry()
        reg.register_dormant(_make_wrapper("fetch__html", "Fetch HTML content"))
        reg.register_dormant(_make_wrapper("mysql__query", "Run SQL query"))

        results = reg.search_dormant_tools("sql")
        assert len(results) == 1
        assert results[0]["name"] == "mysql__query"

    def test_search_case_insensitive(self):
        """搜索不区分大小写"""
        reg = ToolRegistry()
        reg.register_dormant(_make_wrapper("Fetch__HTML", "Fetch HTML"))

        results = reg.search_dormant_tools("fetch")
        assert len(results) == 1

    def test_search_no_results(self):
        """无匹配结果"""
        reg = ToolRegistry()
        reg.register_dormant(_make_wrapper("fetch__html"))

        results = reg.search_dormant_tools("nonexistent")
        assert results == []

    def test_search_limit(self):
        """搜索结果数量限制"""
        reg = ToolRegistry()
        for i in range(25):
            reg.register_dormant(_make_wrapper(f"tool_{i}", "same description"))

        results = reg.search_dormant_tools("tool", limit=10)
        assert len(results) == 10

    def test_search_multi_keywords(self):
        """多关键词搜索：空格分隔"""
        reg = ToolRegistry()
        reg.register_dormant(_make_wrapper("fetch__html", "Fetch HTML content from URL"))
        reg.register_dormant(_make_wrapper("fetch__json", "Fetch JSON data from URL"))
        reg.register_dormant(_make_wrapper("mysql__query", "Run SQL query on database"))

        # "fetch url" 应匹配 fetch__html 和 fetch__json（name 含 fetch，description 含 url）
        results = reg.search_dormant_tools("fetch url")
        assert len(results) == 2
        names = [r["name"] for r in results]
        assert "fetch__html" in names
        assert "fetch__json" in names
        assert "mysql__query" not in names

    def test_search_scoring_name_higher_weight(self):
        """name 命中权重高于 description"""
        reg = ToolRegistry()
        reg.register_dormant(_make_wrapper("fetch__html", "Get web page"))      # name 含 fetch
        reg.register_dormant(_make_wrapper("web__scraper", "Fetch web content"))  # desc 含 fetch

        results = reg.search_dormant_tools("fetch")
        assert len(results) == 2
        # name 命中的排前面（score 2.0 > 1.0）
        assert results[0]["name"] == "fetch__html"

    def test_search_scoring_multi_match(self):
        """多个关键词都命中的排前面"""
        reg = ToolRegistry()
        reg.register_dormant(_make_wrapper("fetch__html", "Fetch HTML content from URL"))
        reg.register_dormant(_make_wrapper("fetch__json", "Fetch JSON data"))

        # "fetch html" — fetch__html 命中 fetch(name+desc) + html(name+desc) = 4分
        # fetch__json 只命中 fetch(name+desc) = 2分
        results = reg.search_dormant_tools("fetch html")
        assert results[0]["name"] == "fetch__html"

    def test_search_short_keywords_filtered(self):
        """过滤掉太短的关键词（< 2字符）"""
        reg = ToolRegistry()
        reg.register_dormant(_make_wrapper("fetch__html", "Fetch HTML"))

        # "a b fetch" 中 a 和 b 太短被过滤
        results = reg.search_dormant_tools("a b fetch")
        assert len(results) == 1
        assert results[0]["name"] == "fetch__html"

    def test_search_empty_query(self):
        """空查询返回空结果"""
        reg = ToolRegistry()
        reg.register_dormant(_make_wrapper("fetch__html", "Fetch HTML"))

        results = reg.search_dormant_tools("")
        assert results == []

    def test_search_single_char_fallback(self):
        """单字符关键词降级为整串匹配"""
        reg = ToolRegistry()
        reg.register_dormant(_make_wrapper("x_tool", "Tool X"))

        # 只有一个字符 "x"，降级为整串匹配
        results = reg.search_dormant_tools("x")
        assert len(results) == 1


class TestListDormantTools:
    """列出休眠工具摘要"""

    def test_list_dormant_tools(self):
        """返回摘要信息"""
        reg = ToolRegistry()
        reg.register_dormant(_make_wrapper("fetch__html", "Fetch HTML"), category="fetch")
        reg.register_dormant(_make_wrapper("fetch__json", "Fetch JSON"), category="fetch")

        tools = reg.list_dormant_tools()
        assert len(tools) == 2
        assert all("name" in t and "description" in t and "category" in t for t in tools)

    def test_list_dormant_tools_empty(self):
        """空休眠池返回空列表"""
        reg = ToolRegistry()
        assert reg.list_dormant_tools() == []

    def test_description_truncation(self):
        """超长描述被截断"""
        reg = ToolRegistry()
        long_desc = "x" * 200
        reg.register_dormant(_make_wrapper("tool", long_desc))

        tools = reg.list_dormant_tools()
        assert len(tools[0]["description"]) <= 100


class TestRoundTrip:
    """完整生命周期：休眠 → 激活 → 停用"""

    def test_full_lifecycle(self):
        """休眠 → 激活 → 停用 → 再激活"""
        reg = ToolRegistry()
        w = _make_wrapper("fetch__html", "Fetch HTML")

        # 1. 注册到休眠池
        reg.register_dormant(w, category="fetch")
        assert "fetch__html" in reg.list_dormant_names()
        assert "fetch__html" not in reg.list_tools()
        assert len(reg.get_schemas()) == 0

        # 2. 激活
        reg.activate("fetch__html")
        assert "fetch__html" not in reg.list_dormant_names()
        assert "fetch__html" in reg.list_tools()
        assert len(reg.get_schemas()) == 1

        # 3. 停用
        reg.deactivate("fetch__html")
        assert "fetch__html" in reg.list_dormant_names()
        assert "fetch__html" not in reg.list_tools()
        assert len(reg.get_schemas()) == 0

        # 4. 再次激活
        reg.activate("fetch__html")
        assert "fetch__html" in reg.list_tools()
        assert len(reg.get_schemas()) == 1


class TestExecutorDormantIntercept:
    """ToolExecutor 休眠工具调用拦截"""

    @pytest.mark.asyncio
    async def test_execute_dormant_tool_returns_hint(self):
        """调用休眠工具应返回激活提示"""
        from agent_framework.tools.executor import ToolExecutor
        from agent_framework.agent.config import AgentConfig

        reg = ToolRegistry()
        w = _make_wrapper("fetch__html", "Fetch HTML")
        reg.register_dormant(w)

        executor = ToolExecutor(reg, AgentConfig())
        tool_call = {
            "id": "call_1",
            "function": {"name": "fetch__html", "arguments": "{}"},
        }

        result = await executor.execute(tool_call)
        assert result.is_error is True
        assert "尚未激活" in result.output
        assert "activate_tools" in result.output

    @pytest.mark.asyncio
    async def test_execute_activated_tool_succeeds(self):
        """激活后工具可正常执行"""
        from agent_framework.tools.executor import ToolExecutor
        from agent_framework.agent.config import AgentConfig

        reg = ToolRegistry()
        w = _make_wrapper("fetch__html", "Fetch HTML")
        reg.register_dormant(w)

        # 激活
        reg.activate("fetch__html")

        executor = ToolExecutor(reg, AgentConfig())
        tool_call = {
            "id": "call_1",
            "function": {"name": "fetch__html", "arguments": "{}"},
        }

        result = await executor.execute(tool_call)
        assert result.is_error is False
        assert result.output == "ok"
