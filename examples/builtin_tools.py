"""内置工具完整测试 —— 7 个工具逐项验证"""
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from agent_framework import (
    Agent, AgentConfig,
    TextDelta, ReasoningDelta,
    ToolCallStart, ToolResult,
    AgentDone, AgentError,
)
from agent_framework.llm.providers import ModelRegistry
from agent_framework.tools.builtin import BUILTIN_TOOLS, create_structured_output_tool

# 加载 .env 中的 API Key
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── 测试任务 ────────────────────────────────────────────────

TEST_TASKS = [
    {
        "name": "datetime_tool",
        "title": "日期时间查询",
        "question": "现在几点了？距离 2026 年元旦还有多少天？",
    },
    {
        "name": "python_repl",
        "title": "Python 代码执行",
        "question": "计算斐波那契数列的前 20 项并求和。",
    },
    {
        "name": "file",
        "title": "文件操作",
        "setup": lambda: Path("test_builtin.txt").write_text(
            "Hello Agent\nbuiltin tools test\nPython is awesome\n", encoding="utf-8"
        ),
        "question": (
            "请读取 test_builtin.txt 的内容，然后把第二行替换为 'modified by agent'，"
            "最后再读取一次确认修改成功。"
        ),
        "cleanup": lambda: [f.unlink() for f in Path(".").glob("test_builtin.txt*")],
    },
    {
        "name": "http_request",
        "title": "HTTP 请求",
        "question": "请求 https://httpbin.org/get 看看返回什么。",
    },
    {
        "name": "web_search",
        "title": "网页搜索",
        "question": "搜索 Python 3.13 有什么新特性，给我 3 条结果。",
    },
    {
        "name": "shell_command",
        "title": "Shell 命令",
        "question": "执行命令 echo hello_agent && dir 查看当前目录文件列表。",
    },
    {
        "name": "structured_output",
        "title": "结构化输出验证与自修复",
        "question": (
            "请验证并修复以下 JSON 输出：\n\n"
            "Schema: {\"type\":\"object\",\"required\":[\"product\",\"price\",\"in_stock\"],"
            "\"properties\":{\"product\":{\"type\":\"string\"},\"price\":{\"type\":\"number\"},"
            "\"in_stock\":{\"type\":\"boolean\"}}}\n\n"
            "原始输出:\n"
            "```json\n{\"product\": \"机械键盘\", \"price\": \"299\", \"in_stock\": \"yes\"}\n```\n\n"
            "使用 structured_output 工具，auto_fix 设为 true。"
        ),
    },
]


async def run_single_task(agent, task: dict) -> dict:
    """执行单个测试任务，返回结果摘要"""
    result = {"name": task["name"], "title": task["title"], "tools_used": [], "ok": False}

    # 准备
    if "setup" in task:
        task["setup"]()

    try:
        thinking_visible = False

        async for event in agent.run(task["question"]):
            if isinstance(event, ReasoningDelta):
                if not thinking_visible:
                    print("  [thinking...] ", end="", flush=True)
                    thinking_visible = True
                print(event.text, end="", flush=True)

            elif isinstance(event, TextDelta):
                if thinking_visible:
                    print(flush=True)
                    thinking_visible = False
                print(f"  {event.text}", end="", flush=True)

            elif isinstance(event, ToolCallStart):
                result["tools_used"].append(event.tool_name)
                args = str(event.arguments)
                short = args[:120] + "..." if len(args) > 120 else args
                print(f"\n  -> {event.tool_name}({short})", flush=True)

            elif isinstance(event, ToolResult):
                tag = "ERR" if event.is_error else "OK"
                output = event.output[:300] + "..." if len(event.output) > 300 else event.output
                print(f"  <- [{tag}] {output}", flush=True)

            elif isinstance(event, AgentDone):
                u = event.total_usage
                result["ok"] = True
                result["turns"] = event.turn_count
                result["tokens"] = u.total_tokens
                print(f"\n  [done] turns={event.turn_count}, tokens={u.total_tokens}")

            elif isinstance(event, AgentError):
                result["error"] = event.message
                print(f"\n  [error] {event.message}")

    except Exception as e:
        result["error"] = str(e)
        print(f"\n  [exception] {e}")

    finally:
        if "cleanup" in task:
            task["cleanup"]()

    print()
    return result


async def main():
    # ── 创建 Agent ──
    llm = ModelRegistry.create("qwen", model="qwen3.7-max")
    so_tool = create_structured_output_tool()
    all_tools = [*BUILTIN_TOOLS, so_tool]

    agent = Agent(
        llm=llm,
        system_prompt="你是一个全能助手，可以使用各种工具完成任务。回答简洁。",
        tools=all_tools,
        config=AgentConfig(max_turns=8, timeout=120),
    )

    # ── 打印工具列表 ──
    print("=" * 60)
    print("  内置工具完整测试 (7 个工具)")
    print("=" * 60)
    for f in all_tools:
        w = f._tool_wrapper
        flag = " [D]" if w.dangerous else ""
        print(f"  {w.name:<20} {w.description[:42]}{flag}")
    print("=" * 60)

    # ── 逐项测试 ──
    results = []
    for i, task in enumerate(TEST_TASKS, 1):
        print(f"\n[{i}/{len(TEST_TASKS)}] {task['title']} ({task['name']})")
        print("-" * 50)
        r = await run_single_task(agent, task)
        results.append(r)

    # ── 汇总报告 ──
    print("\n" + "=" * 60)
    print("  测试汇总")
    print("=" * 60)
    for r in results:
        status = "PASS" if r["ok"] else "FAIL"
        tools = ", ".join(set(r["tools_used"])) or "none"
        tokens = r.get("tokens", "-")
        err = f" ({r.get('error', '')})" if not r["ok"] else ""
        print(f"  [{status}] {r['title']:<16} tools={tools}, tokens={tokens}{err}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
