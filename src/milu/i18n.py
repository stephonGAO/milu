"""CLI 中英双语支持（默认中文）。

设计：以**中文原文为键**维护英文译表 `_EN`；`t(zh, **kw)` 在英文模式下查表回退
中文原文，故任何未包裹/漏译的字符串仍能正常显示（中文），不会因缺键而报错。
带 `{name}` 占位的模板经 `str.format(**kw)` 填充（zh/en 用同名占位）。

语言来源（高→低）：`--lang` > 环境变量 `MILU_LANG` > config.json 的 `lang` > 默认 zh。
由 CLI 入口在解析子命令前调用 `set_lang(...)` 设定，故 argparse 帮助文本也随之切换。
"""
from __future__ import annotations

_LANG = "zh"


def set_lang(lang) -> None:
    """设定当前语言：以 'en' 开头视作英文，其余一律中文。"""
    global _LANG
    _LANG = "en" if str(lang or "").strip().lower().startswith("en") else "zh"


def get_lang() -> str:
    return _LANG


def t(zh: str, **kw) -> str:
    """取词：英文模式查 _EN（缺则回退中文原文），再按 {name} 占位填充。"""
    s = _EN.get(zh, zh) if _LANG == "en" else zh
    if kw:
        try:
            s = s.format(**kw)
        except Exception:
            pass
    return s


# ── 英文译表（键 = 中文原文）────────────────────────────────
_EN: dict[str, str] = {
    # —— app.py: 顶层/帮助 ——
    "milu 命令行：启动 Agent、一次性执行或多轮对话。":
        "milu CLI: launch the agent, run a one-off, or chat.",
    "厂商名（默认 {p}，或配置文件）": "Provider name (default {p}, or config file)",
    "模型名（默认按厂商内置，或配置文件）": "Model name (default per provider, or config file)",
    "临时指定 API Key（优先级最高）": "Temporary API Key (highest priority)",
    "操作模式": "Operation mode",
    "禁用会话持久化": "Disable session persistence",
    "启动时不连接 MCP 服务器": "Don't connect MCP servers at startup",
    "不挂载内置子代理": "Don't mount built-in subagents",
    "对话期间不嵌入定时任务调度引擎（仅 chat 生效）":
        "Don't embed the scheduler during chat (chat only)",
    "选择语言（zh/en，默认 zh）": "Language (zh/en, default zh)",
    "交互式多轮对话": "Interactive multi-turn chat",
    "续接指定会话 ID": "Resume a given session ID",
    "一次性执行单条指令": "Run a single instruction once",
    "指令文本（省略则从 stdin 读取）": "Instruction text (omit to read from stdin)",
    "只输出最终回答（适合管道）": "Output only the final answer (pipe-friendly)",
    "初始化引导：选厂商/模型、配置 API Key 与搜索工具":
        "Setup wizard: pick provider/model, configure API Key and search tools",
    "查看/修改配置": "View/modify config",
    "打印合并后的生效配置及各文件路径": "Print the merged effective config and file paths",
    "打印项目级 / 用户级配置文件路径": "Print project / user config file paths",
    "在项目 config/milu.json 生成全量默认配置模板":
        "Generate a full default config template at project config/milu.json",
    "读取配置项（点号路径，如 agent.max_turns）":
        "Read a config item (dotted path, e.g. agent.max_turns)",
    "设置配置项到用户级配置（点号路径）": "Set a config item in the user config (dotted path)",
    "查看历史会话": "View past sessions",
    "列出全部会话（默认）": "List all sessions (default)",
    "打印某会话的消息": "Print a session's messages",
    "列出支持的厂商及 Key 配置状态": "List supported providers and key status",
    "显示版本": "Show version",
    "定时任务管理（list/run/delete/enable/disable）":
        "Scheduled-task management (list/run/delete/enable/disable)",
    "列出定时任务（默认）": "List scheduled tasks (default)",
    "用户标识（默认 default）": "User id (default: default)",
    "列出全部用户的任务": "List tasks of all users",
    "立即同步执行指定任务": "Run a task immediately (sync)",
    "任务名称": "Task name",
    "删除指定任务": "Delete a task",
    "启用指定任务": "Enable a task",
    "禁用指定任务": "Disable a task",
    "调度守护进程管理（start）": "Scheduler daemon management (start)",
    "启动调度守护进程（前台运行，Ctrl+C 停止）":
        "Start the scheduler daemon (foreground, Ctrl+C to stop)",
    "启动内置 Web 服务（多用户对话 + 演示前端）":
        "Start the built-in web service (multi-user chat + demo frontend)",
    "监听地址（默认 127.0.0.1）": "Listen address (default 127.0.0.1)",
    "监听端口（默认 8000）": "Listen port (default 8000)",
    "默认厂商（默认按配置）": "Default provider (per config)",
    "默认模型（默认按厂商内置）": "Default model (per provider)",
    "临时 API Key（一般用 .env 配置）": "Temporary API Key (usually via .env)",
    "默认操作模式": "Default operation mode",
    "不连接 MCP 服务器": "Don't connect MCP servers",
    "不嵌入定时任务调度引擎": "Don't embed the scheduler",
    "开发模式：代码变更自动重载": "Dev mode: auto-reload on code change",
    # —— app.py: 运行期 ——
    "错误：未提供指令。用法：milu run \"你的问题\"":
        "Error: no instruction. Usage: milu run \"your question\"",
    "项目级: {p}": "Project: {p}",
    "用户级: {p}": "User: {p}",
    "已生成项目配置模板 → {p}": "Generated project config template → {p}",
    "未知配置项：{k}": "Unknown config item: {k}",
    "错误：{e}": "Error: {e}",
    "已设置 {k} = {v}": "Set {k} = {v}",
    "（生效 = 内置默认 ← 项目级 ← 用户级；CLI 参数运行时再叠加）":
        "(effective = builtin ← project ← user; CLI args layered at runtime)",
    "会话不存在: {id}": "Session not found: {id}",
    "会话 {id}（{n} 条消息）": "Session {id} ({n} messages)",
    "暂无历史会话（{base}）": "No past sessions ({base})",
    "历史会话 ({n} 个)": "Past sessions ({n})",
    "{n} 条消息": "{n} messages",
    "支持的厂商 ({n} 个)": "Supported providers ({n})",
    "已配置 (env {env})": "configured (env {env})",
    "未配置（在 .env 设置 {env}）": "not set (set {env} in .env)",
    "默认模型": "default model",
    " *默认": " *default",
    "\n  提示：运行 `milu setup` 可交互式配置厂商、模型与 API Key。":
        "\n  Tip: run `milu setup` to configure provider, model and API Key interactively.",
    "milu (版本未知)": "milu (version unknown)",
    # —— app.py / repl.py: 定时任务 ——
    "暂无定时任务。可在对话中让 Agent 调用 schedule_create 工具创建。":
        "No scheduled tasks. Ask the agent to call schedule_create in chat.",
    "定时任务 ({n} 个)": "Scheduled tasks ({n})",
    "启用": "enabled",
    "禁用": "disabled",
    "从未": "never",
    "待计算": "pending",
    "运行 {n} 次": "ran {n}x",
    "触发: ": "Trigger: ",
    "说明: ": "Desc: ",
    "上次: ": "Last: ",
    "下次: ": "Next: ",
    "错误：任务 '{name}' 不存在": "Error: task '{name}' not found",
    "  正在执行任务 '{name}'...": "  Running task '{name}'...",
    "  任务 '{name}' 执行完成": "  Task '{name}' completed",
    "  执行失败: {e}": "  Failed: {e}",
    "  任务 '{name}' 已删除": "  Task '{name}' deleted",
    "已启用": "enabled",
    "已禁用": "disabled",
    "  任务 '{name}' {verb}": "  Task '{name}' {verb}",
    "未知操作: {action}": "Unknown action: {action}",
    # —— app.py: scheduler daemon ——
    "调度器已在运行（PID {pid}），同一时间只能有一个引擎。":
        "Scheduler already running (PID {pid}); only one engine at a time.",
    "如确认它已不存在，请删除锁文件后重试: {path}":
        "If it's gone, delete the lock file and retry: {path}",
    "  milu 调度守护进程": "  milu scheduler daemon",
    "  任务目录: {p}": "  Tasks dir: {p}",
    "  日志目录: {p}": "  Logs dir: {p}",
    "  已启用任务: {n} / {tot} 个（{u} 个用户）":
        "  Enabled tasks: {n} / {tot} ({u} users)",
    "\n  调度器已停止。": "\n  Scheduler stopped.",
    "未知操作: {action}（可用: start）": "Unknown action: {action} (available: start)",
    # —— app.py: serve ——
    "加载 Web 服务失败：{e}": "Failed to load web service: {e}",
    "  milu Web 服务": "  milu Web Service",
    "  访问地址: {url}": "  URL: {url}",
    "  默认厂商: {p}  模型: {m}": "  Provider: {p}  Model: {m}",
    "  操作模式: {mode}  调度: {sch}  MCP: {mcp}":
        "  Mode: {mode}  Scheduler: {sch}  MCP: {mcp}",
    "开": "on",
    "关": "off",
    "  不同浏览器标签用不同「用户ID」即可演示多用户隔离。Ctrl+C 停止。":
        "  Use a different 'User ID' per browser tab to demo multi-user isolation. Ctrl+C to stop.",
    "缺少 Web 服务依赖（通常已随 milu 安装）。请修复：":
        "Missing web deps (usually installed with milu). Fix:",
    "\n  Web 服务已停止。": "\n  Web service stopped.",
    # —— app.py: main ——
    "未知子命令: {c}": "Unknown subcommand: {c}",
    "\n已中断。": "\nInterrupted.",
    "\n鉴权失败：{e}": "\nAuthentication failed: {e}",
    "可运行 `milu setup` 进行初始化引导，或在 .env / 环境变量中设置 {env}。":
        "Run `milu setup`, or set {env} in .env / environment.",
    "\n错误：{e}": "\nError: {e}",
    "\n运行失败：{e}": "\nFailed: {e}",
    # —— render.py ——
    "输入 y 同意 / n 拒绝 / 或直接输入指示发给 Agent":
        "Type y to approve / n to reject / or type instructions for the agent",
    "用户选择拒绝执行": "User rejected execution",
    "用户同意执行": "User approved",
    "用户指示: ": "User instruction: ",
    "用户拒绝执行": "User rejected",
    "开始执行": "started",
    "完成": "done",
    "对话历史已压缩: {a} → {b} 条消息": "History compacted: {a} → {b} messages",
    # —— repl.py: header ——
    "  milu — 交互式对话": "  milu — Interactive Chat",
    "（已禁用 --no-subagents）": "(disabled via --no-subagents)",
    "厂商/模型": "Provider/Model",
    "模式": "Mode",
    "内置工具": "Built-in tools",
    "子代理": "Subagents",
    "元工具": "Meta tools",
    "（发现/激活 MCP 工具）": " (discover/activate MCP tools)",
    "技能": "Skills",
    "内置技能元数据已注入，LLM 按需 load_skill 加载":
        "Built-in skill metadata injected; the LLM loads bodies via load_skill on demand",
    "命令: /help 查看全部命令  ·  /quit 退出":
        "Commands: /help for all  ·  /quit to exit",
    # —— repl.py: 命令 ——
    "\n再见!": "\nGoodbye!",
    "\n  对话已重置，上下文和计划已清空。":
        "\n  Conversation reset; context and plan cleared.",
    "\n  新日志段: {name}": "\n  New log segment: {name}",
    "  对话历史": "  Conversation history",
    " ({n} 条消息)": " ({n} messages)",
    "  会话 ID: {id}": "  Session ID: {id}",
    "  日志文件: {p}": "  Log file: {p}",
    "[图片×{n}] ": "[images x{n}] ",
    "  活跃工具": "  Active tools",
    " ({n} 个)": " ({n})",
    "  已激活 MCP 工具": "  Activated MCP tools",
    "未分类": "Uncategorized",
    "  休眠工具（search_tools / activate_tools 激活）":
        "  Dormant tools (activate via search_tools / activate_tools)",
    "暂无可用技能。": "No skills available.",
    "  可用技能": "  Available skills",
    "LLM 会自动调用 load_skill 按需加载技能正文":
        "The LLM calls load_skill to load skill bodies on demand",
    "暂无会话计划。": "No session plan.",
    "  当前会话计划": "  Current session plan",
    " ({n} 个条目)": " ({n} items)",
    "({c}/{n} 已完成)": "({c}/{n} done)",
    "只读模式（仅允许安全操作）": "Read-only (safe ops only)",
    "人工审批模式（不安全操作逐一确认）": "Manual approval (confirm each unsafe op)",
    "自主模式（不安全操作自动执行，AI 判定兜底）":
        "Autonomous (unsafe ops auto-run, AI safety net)",
    "全权限模式（跳过所有安全检查）": "Full access (skip all safety checks)",
    "  操作模式": "  Operation mode",
    " → 当前": " → current",
    "  用法: /mode <talk|manual|auto|superwork>\n":
        "  Usage: /mode <talk|manual|auto|superwork>\n",
    "模式已切换为:": "Mode switched to:",
    "无效模式: {m}": "Invalid mode: {m}",
    "（可选: talk, manual, auto, superwork）": " (options: talk, manual, auto, superwork)",
    "当前系统提示词为空。": "Current system prompt is empty.",
    "  当前系统提示词": "  Current system prompt",
    "  ({n} 行, {c} 字符)": "  ({n} lines, {c} chars)",
    "上下文压缩未启用。": "Context compaction is disabled.",
    "  手动压缩完成": "  Manual compaction done",
    "  ({a} → {b} 条消息)": "  ({a} → {b} messages)",
    "会话已保存": "Session saved",
    "  路径: {p}": "  Path: {p}",
    "会话功能未启用。": "Session feature is disabled.",
    "暂无历史会话。": "No past sessions.",
    "  历史会话": "  Past sessions",
    " ← 当前": " ← current",
    "新会话已创建": "New session created",
    "用法: /load <session_id>": "Usage: /load <session_id>",
    "会话已加载": "Session loaded",
    "  ID: {id}  消息数: {n}": "  ID: {id}  Messages: {n}",
    "加载失败: {e}": "Load failed: {e}",
    "  定时任务": "  Scheduled tasks",
    "  milu scheduler start  — 启动调度守护进程\n":
        "  milu scheduler start  — start the scheduler daemon\n",
    ("\n"
     "  /history    — 查看对话历史\n"
     "  /reset      — 重置对话（清空上下文）\n"
     "  /tools      — 查看可用工具（含休眠工具）\n"
     "  /skills     — 查看可用技能\n"
     "  /plan       — 查看当前会话计划\n"
     "  /schedule   — 查看定时任务列表\n"
     "  /mode       — 查看/切换操作模式（talk/manual/auto/superwork）\n"
     "  /prompt     — 查看当前系统提示词\n"
     "  /compact    — 手动压缩对话历史\n"
     "  /save       — 保存当前会话\n"
     "  /sessions   — 查看所有会话\n"
     "  /new        — 新建会话（自动保存当前）\n"
     "  /load <id>  — 加载历史会话\n"
     "  /help       — 显示帮助\n"
     "  /quit       — 退出\n"):
        "\n"
        "  /history    — view conversation history\n"
        "  /reset      — reset conversation (clear context)\n"
        "  /tools      — list available tools (incl. dormant)\n"
        "  /skills     — list available skills\n"
        "  /plan       — show the current session plan\n"
        "  /schedule   — list scheduled tasks\n"
        "  /mode       — view/switch mode (talk/manual/auto/superwork)\n"
        "  /prompt     — show the current system prompt\n"
        "  /compact    — compact conversation history manually\n"
        "  /save       — save the current session\n"
        "  /sessions   — list all sessions\n"
        "  /new        — new session (auto-saves current)\n"
        "  /load <id>  — load a past session\n"
        "  /help       — show this help\n"
        "  /quit       — exit\n",
    "\n  未知命令: {cmd}  (输入 /help 查看帮助)\n":
        "\n  Unknown command: {cmd}  (type /help for help)\n",
    # —— repl.py: run_chat ——
    "  检测到调度器已在运行（PID {pid}），任务由它执行；其退出后本会话自动接管\n":
        "  Scheduler already running (PID {pid}); it runs the tasks; this session takes over after it exits\n",
    "  定时任务调度已启动：{n} 个启用任务，对话期间自动执行":
        "  Scheduler started: {n} enabled task(s), auto-run during chat",
    "（--no-scheduler 可关）\n": " (disable with --no-scheduler)\n",
    "  MCP 工具已加载（休眠态）: {summary}": "  MCP tools loaded (dormant): {summary}",
    "  输入 /tools 查看，或 search_tools / activate_tools 按需激活\n":
        "  Type /tools to view, or activate via search_tools / activate_tools\n",
    "{cat}({n}个)": "{cat}({n})",
    "  已加载会话 {id}（{n} 条消息）\n": "  Loaded session {id} ({n} messages)\n",
    "  会话 {id} 不存在，使用新会话\n": "  Session {id} not found; using a new session\n",
    "  日志路径: {p}\n": "  Log path: {p}\n",
    "  计划已恢复: {c}/{n} 已完成": "  Plan restored: {c}/{n} done",
    " | 当前: {x}": " | current: {x}",
    "  输入 /plan 查看详情\n": "  Type /plan for details\n",
    "  ESC 中断": "  ESC to interrupt",
    "\n  [已中断]": "\n  [interrupted]",
    # —— setup_wizard.py: 厂商标签 / URL / 搜索后端 ——
    "通义千问（阿里）": "Qwen (Alibaba)",
    "Kimi（月之暗面）": "Kimi (Moonshot)",
    "智谱 GLM": "Zhipu GLM",
    "豆包（火山引擎）": "Doubao (Volcengine)",
    "ChatGPT（OpenAI）": "ChatGPT (OpenAI)",
    "Gemini（Google）": "Gemini (Google)",
    "Claude（Anthropic）": "Claude (Anthropic)",
    "https://bailian.console.aliyun.com/（百炼控制台 → API-KEY）":
        "https://bailian.console.aliyun.com/ (Bailian console → API-KEY)",
    "https://platform.minimaxi.com/（账户管理 → 接口密钥）":
        "https://platform.minimaxi.com/ (Account → API keys)",
    "https://console.volcengine.com/ark（火山方舟 → API Key 管理）":
        "https://console.volcengine.com/ark (Volcano Ark → API Key)",
    "博查搜索 —— 国内直连，推荐": "Bocha — China direct, recommended",
    "Tavily —— 为 LLM 设计，需国际网络": "Tavily — built for LLMs, needs intl network",
    "DuckDuckGo —— 免 Key，国内网络不可用": "DuckDuckGo — no key, unavailable in China",
    # —— setup_wizard.py: 流程 ——
    "\n[1/4] 选择默认厂商": "\n[1/4] Choose default provider",
    "已配置 Key": "Key set",
    "未配置 Key": "No key",
    "  ← 当前默认": "  ← current",
    "请输入编号或厂商名（回车 = {cur}）: ": "Enter number or provider name (Enter = {cur}): ",
    "  无效输入：{raw}，请输入 1-{n} 或厂商名。":
        "  Invalid input: {raw}. Enter 1-{n} or a provider name.",
    "\n[2/4] 选择模型（厂商: {p}）": "\n[2/4] Choose model (provider: {p})",
    "回车使用默认 {d}，或输入其他模型名: ": "Enter to use default {d}, or type another model: ",
    "该厂商无内置默认模型，请输入模型名: ":
        "This provider has no built-in default; enter a model name: ",
    "  模型名不能为空。": "  Model name cannot be empty.",
    "\n[3/4] 配置 API Key（环境变量 {env}）": "\n[3/4] Configure API Key (env {env})",
    "  申请地址: {url}": "  Get key: {url}",
    "  当前已配置: {masked}": "  Currently set: {masked}",
    "  回车保留现有，或粘贴新 Key: ": "  Enter to keep current, or paste a new key: ",
    "  请粘贴 API Key（回车跳过，稍后可再运行 milu setup）: ":
        "  Paste API Key (Enter to skip; rerun milu setup later): ",
    "  未配置 Key，调用 {p} 时会鉴权失败。": "  No key set; calls to {p} will fail auth.",
    "\n[4/4] 配置联网搜索工具（web_search 后端）":
        "\n[4/4] Configure web search (web_search backend)",
    "  ← 当前": "  ← current",
    "  {n}. 跳过（保持现状: {cur}）": "  {n}. Skip (keep current: {cur})",
    "请输入编号（回车 = 跳过）: ": "Enter a number (Enter = skip): ",
    "  无效输入：{raw}。": "  Invalid input: {raw}.",
    "  {env} 已配置: {masked}": "  {env} is set: {masked}",
    "  请粘贴 {env}（回车跳过）: ": "  Paste {env} (Enter to skip): ",
    "  未配置 {env}，搜索工具运行时会报错提示。":
        "  {env} not set; the search tool will warn at runtime.",
    "  milu 初始化引导": "  milu Setup Wizard",
    "  共 4 步：选厂商 → 选模型 → API Key → 搜索工具（Ctrl+C 随时退出）":
        "  4 steps: provider → model → API Key → search (Ctrl+C to quit)",
    "  密钥写入 {p}，厂商/模型写入用户配置。":
        "  Keys written to {p}; provider/model to the user config.",
    "\n已取消初始化引导，未写入任何配置。": "\nSetup cancelled; nothing written.",
    "\n是否立即验证 {p} 的 API Key（发送一次最小请求）？":
        "\nVerify {p}'s API Key now (send one minimal request)?",
    "  验证中...": "  Verifying...",
    "  验证通过，Key 可用。": "  Verified; the key works.",
    "  验证失败：{err}": "  Verification failed: {err}",
    "  请检查 Key 是否正确，可重新运行 milu setup 修改。":
        "  Check the key; rerun milu setup to change it.",
    "\n  已跳过验证。": "\n  Verification skipped.",
    "  配置完成！": "  Setup complete!",
    "  厂商/模型: {p} / {m}  ": "  Provider/Model: {p} / {m}  ",
    "  搜索后端:  {b}{keystr}  ": "  Search:  {b}{keystr}  ",
    "\n  现在运行 {milu} 即可开始对话；重新配置请再运行 {setup}。":
        "\n  Run {milu} to start chatting; rerun {setup} to reconfigure.",
    "\n未检测到 {p} 的 API Key（环境变量 {env}）。":
        "\nNo API Key found for {p} (env {env}).",
    "是否现在进行初始化引导？": "Run the setup wizard now?",
    "已跳过。稍后可运行 `milu setup`，或在 .env 中设置 {env}。":
        "Skipped. Run `milu setup` later, or set {env} in .env.",
    "语言 / Language [zh/en]（回车 = {cur}）: ": "Language / 语言 [zh/en] (Enter = {cur}): ",
    # —— config.py（核心）: config set 报错 ——
    "需要布尔值（true/false），收到：{raw}": "expected a boolean (true/false), got {raw}",
    "未知配置项：{dotted}": "Unknown config item: {dotted}",
    "'{dotted}' 是配置分节，请设置其下具体项（如 {dotted}.xxx）":
        "'{dotted}' is a config section; set a specific item (e.g. {dotted}.xxx)",
    # —— cli/config.py: 模型解析 ——
    "厂商 '{p}' 没有内置默认模型，请用 --model 指定，或在 config.json 的 default_models 中补充。":
        "Provider '{p}' has no built-in default model; specify --model or add it to default_models in config.json.",
}
