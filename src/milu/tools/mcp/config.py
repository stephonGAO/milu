"""MCP 服务器配置

MCPServerConfig 数据类 + 从 JSON 配置文件加载。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# 配置文件搜索顺序（仅在未显式传入路径时生效），见 _find_config_file()：
#   1. ./config/mcp_servers.json —— 项目级（相对 CWD，向后兼容仓库示例）
#   2. user_data_dir()/mcp_servers.json —— 用户级（默认 ~/.milu，
#      可被环境变量 MILU_HOME 覆盖；与 CWD 解耦，适合部署）
# ⚠️ 作为库集成时 CWD 取决于宿主应用，建议显式传入 mcp_config_path
#    或用环境变量 MCP_CONFIG_PATH 指定绝对路径。


@dataclass
class MCPServerConfig:
    """单个 MCP 服务器的连接配置。"""

    name: str
    transport: Literal["stdio", "streamable_http", "sse"]

    # stdio 参数
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None

    # streamable_http / sse 参数
    url: str | None = None
    headers: dict[str, str] | None = None

    # 选项
    prefix_tools: bool = True
    safe_tools: list[str] = field(default_factory=list)
    tool_filter: list[str] | None = None
    connect_timeout: float = 30.0

    # -- 工厂方法 ----------------------------------------------------------

    @classmethod
    def stdio(
        cls,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        **kwargs,
    ) -> MCPServerConfig:
        """创建 stdio 类型的配置。"""
        return cls(
            name=name,
            transport="stdio",
            command=command,
            args=args or [],
            env=env,
            **kwargs,
        )

    @classmethod
    def streamable_http(
        cls,
        name: str,
        url: str,
        headers: dict[str, str] | None = None,
        **kwargs,
    ) -> MCPServerConfig:
        """创建 streamable_http 类型的配置。"""
        return cls(
            name=name,
            transport="streamable_http",
            url=url,
            headers=headers,
            **kwargs,
        )

    @classmethod
    def sse(
        cls,
        name: str,
        url: str,
        headers: dict[str, str] | None = None,
        **kwargs,
    ) -> MCPServerConfig:
        """创建 sse 类型的配置。"""
        return cls(
            name=name,
            transport="sse",
            url=url,
            headers=headers,
            **kwargs,
        )

    # -- 配置文件加载 ------------------------------------------------------

    @classmethod
    def from_dict(cls, name: str, data: dict) -> MCPServerConfig:
        """从单个服务器的字典构建 MCPServerConfig。

        字段映射：type → transport, timeout → connect_timeout
        """
        return cls(
            name=name,
            transport=data.get("type", "stdio"),
            command=data.get("command"),
            args=data.get("args", []),
            env=data.get("env"),
            url=data.get("url"),
            headers=data.get("headers"),
            prefix_tools=data.get("prefix_tools", True),
            safe_tools=data.get("safe_tools", []),
            tool_filter=data.get("tool_filter"),
            connect_timeout=float(data.get("timeout", 30.0)),
        )

    @classmethod
    def load_file(cls, path: str | None = None) -> list[MCPServerConfig]:
        """从 JSON 文件加载 MCP 服务器配置列表。

        Args:
            path: 配置文件路径。为 None 时按优先级搜索：
                  1. ./config/mcp_servers.json（项目级）
                  2. ~/.milu/mcp_servers.json（用户级）

        Returns:
            MCPServerConfig 列表。找不到文件时返回空列表。
        """
        if path:
            config_path = Path(path)
            if not config_path.exists():
                logger.warning("配置文件不存在: %s", config_path)
                return []
        else:
            config_path = cls._find_config_file()
            if config_path is None:
                logger.info("未找到 MCP 配置文件，跳过 MCP 工具加载")
                return []

        logger.info("加载 MCP 配置: %s", config_path)
        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            logger.warning("配置文件格式错误（应为 JSON 对象）: %s", config_path)
            return []

        configs = []
        for name, server_data in data.items():
            if not isinstance(server_data, dict):
                logger.warning("跳过无效配置项 %s", name)
                continue
            try:
                configs.append(cls.from_dict(name, server_data))
            except Exception as e:
                logger.warning("解析配置项 %s 失败: %s", name, e)

        logger.info("已加载 %d 个 MCP 服务器配置", len(configs))
        return configs

    @classmethod
    def _find_config_file(cls) -> Path | None:
        """按优先级搜索配置文件：项目级 CWD → 用户级（user_data_dir，可被环境变量覆盖）。

        用户级路径在调用时解析（而非 import 时），以尊重运行期设置的
        MILU_HOME。
        """
        from milu.resources import default_mcp_config_path

        search_paths = [
            Path("./config/mcp_servers.json"),   # 项目级（相对 CWD）
            default_mcp_config_path(),           # 用户级（~/.milu，env 可覆盖）
        ]
        for p in search_paths:
            if p.exists():
                return p
        return None
