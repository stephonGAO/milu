"""测试 MCPServerConfig 配置数据类和文件加载"""
import json
import tempfile
from pathlib import Path

import pytest
from milu.tools.mcp.config import MCPServerConfig


class TestMCPServerConfigConstruction:
    """MCPServerConfig 构建测试"""

    def test_stdio_factory(self):
        config = MCPServerConfig.stdio(name="test", command="python", args=["-m", "server"])
        assert config.name == "test"
        assert config.transport == "stdio"
        assert config.command == "python"
        assert config.args == ["-m", "server"]
        assert config.prefix_tools is True

    def test_streamable_http_factory(self):
        config = MCPServerConfig.streamable_http(
            name="remote", url="http://localhost:8080/mcp",
            headers={"Authorization": "Bearer token"},
        )
        assert config.name == "remote"
        assert config.transport == "streamable_http"
        assert config.url == "http://localhost:8080/mcp"
        assert config.headers == {"Authorization": "Bearer token"}

    def test_sse_factory(self):
        config = MCPServerConfig.sse(name="legacy", url="http://localhost:3000/sse")
        assert config.name == "legacy"
        assert config.transport == "sse"

    def test_defaults(self):
        config = MCPServerConfig(name="x", transport="stdio", command="echo")
        assert config.args == []
        assert config.env is None
        assert config.prefix_tools is True
        assert config.safe_tools == []
        assert config.tool_filter is None
        assert config.connect_timeout == 30.0

    def test_custom_options(self):
        config = MCPServerConfig.stdio(
            name="db", command="python",
            prefix_tools=False,
            safe_tools=["query", "read"],
            tool_filter=["query", "read"],
            connect_timeout=60.0,
        )
        assert config.prefix_tools is False
        assert config.safe_tools == ["query", "read"]
        assert config.tool_filter == ["query", "read"]
        assert config.connect_timeout == 60.0


class TestFromDict:
    """from_dict 构建测试"""

    def test_basic_stdio(self):
        data = {
            "type": "stdio",
            "command": "python",
            "args": ["-m", "server"],
            "timeout": 60,
        }
        config = MCPServerConfig.from_dict("myserver", data)
        assert config.name == "myserver"
        assert config.transport == "stdio"
        assert config.command == "python"
        assert config.connect_timeout == 60.0

    def test_http_with_headers(self):
        data = {
            "type": "streamable_http",
            "url": "http://localhost:8080/mcp",
            "headers": {"X-Key": "abc"},
        }
        config = MCPServerConfig.from_dict("remote", data)
        assert config.transport == "streamable_http"
        assert config.url == "http://localhost:8080/mcp"
        assert config.headers == {"X-Key": "abc"}

    def test_defaults_for_missing_fields(self):
        data = {"type": "stdio", "command": "echo"}
        config = MCPServerConfig.from_dict("minimal", data)
        assert config.args == []
        assert config.env is None
        assert config.prefix_tools is True
        assert config.connect_timeout == 30.0

    def test_default_transport_type(self):
        """无 type 字段时默认 stdio"""
        data = {"command": "echo"}
        config = MCPServerConfig.from_dict("default", data)
        assert config.transport == "stdio"

    def test_with_env(self):
        data = {
            "type": "stdio",
            "command": "python",
            "env": {"API_KEY": "secret"},
        }
        config = MCPServerConfig.from_dict("env_server", data)
        assert config.env == {"API_KEY": "secret"}


class TestLoadFile:
    """load_file 配置文件加载测试"""

    def test_load_valid_file(self, tmp_path):
        config_file = tmp_path / "mcp_servers.json"
        config_file.write_text(json.dumps({
            "fetch": {
                "type": "stdio",
                "command": "fetch-server",
                "args": [],
            },
            "db": {
                "type": "streamable_http",
                "url": "http://localhost:8080/mcp",
            },
        }, ensure_ascii=False), encoding="utf-8")

        configs = MCPServerConfig.load_file(str(config_file))
        assert len(configs) == 2
        names = {c.name for c in configs}
        assert names == {"fetch", "db"}

    def test_load_nonexistent_file(self):
        configs = MCPServerConfig.load_file("/nonexistent/path.json")
        assert configs == []

    def test_load_invalid_json(self, tmp_path):
        config_file = tmp_path / "bad.json"
        config_file.write_text("not json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            MCPServerConfig.load_file(str(config_file))

    def test_load_non_dict_root(self, tmp_path):
        config_file = tmp_path / "array.json"
        config_file.write_text("[1, 2, 3]", encoding="utf-8")
        configs = MCPServerConfig.load_file(str(config_file))
        assert configs == []

    def test_load_skip_invalid_entry(self, tmp_path):
        config_file = tmp_path / "mixed.json"
        config_file.write_text(json.dumps({
            "good": {"type": "stdio", "command": "echo"},
            "bad": "not a dict",
        }), encoding="utf-8")
        configs = MCPServerConfig.load_file(str(config_file))
        assert len(configs) == 1
        assert configs[0].name == "good"

    def test_auto_search_no_file(self, monkeypatch):
        """无参数时搜索默认路径，找不到返回空"""
        monkeypatch.chdir(tmp_path := Path(tempfile.mkdtemp()))
        configs = MCPServerConfig.load_file()
        assert configs == []

    def test_auto_search_finds_project_config(self, monkeypatch, tmp_path):
        """自动搜索找到项目级配置"""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "mcp_servers.json"
        config_file.write_text(json.dumps({
            "test_server": {"type": "stdio", "command": "echo"},
        }), encoding="utf-8")

        monkeypatch.chdir(tmp_path)
        configs = MCPServerConfig.load_file()
        assert len(configs) == 1
        assert configs[0].name == "test_server"
