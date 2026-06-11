"""首次使用初始化引导（milu setup）：交互式选择厂商/模型、配置 API Key 与搜索工具。

pip 安装后用户无需手工创建 .env——引导程序把密钥写入**用户级**
`~/.milu/.env`（`MILU_HOME` 可覆盖），CLI 启动时自动加载（见 `milu._env`
的用户级兜底），任意目录运行 milu 均生效；厂商/模型选择写入用户级
`~/.milu/config.json`（与 `milu config set` 同一路径，稀疏覆盖）。

入口：
- `run_setup_wizard()`：`milu setup` 子命令的完整引导流程；
- `offer_first_run_setup(provider)`：`milu chat` 检测不到 Key 时的首次使用询问。
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from milu.i18n import get_lang, set_lang, t
from milu.cli.render import BANNER, DIVIDER, c

# ── 厂商展示信息（中文名 + API Key 申请地址，仅引导展示用）──────────────

PROVIDER_LABELS: dict[str, str] = {
    "qwen": "通义千问（阿里）",
    "deepseek": "DeepSeek",
    "kimi": "Kimi（月之暗面）",
    "glm": "智谱 GLM",
    "minimax": "MiniMax",
    "doubao": "豆包（火山引擎）",
    "openai": "ChatGPT（OpenAI）",
    "gemini": "Gemini（Google）",
    "anthropic": "Claude（Anthropic）",
}

PROVIDER_KEY_URLS: dict[str, str] = {
    "qwen": "https://bailian.console.aliyun.com/（百炼控制台 → API-KEY）",
    "deepseek": "https://platform.deepseek.com/api_keys",
    "kimi": "https://platform.moonshot.cn/console/api-keys",
    "glm": "https://open.bigmodel.cn/usercenter/apikeys",
    "minimax": "https://platform.minimaxi.com/（账户管理 → 接口密钥）",
    "doubao": "https://console.volcengine.com/ark（火山方舟 → API Key 管理）",
    "openai": "https://platform.openai.com/api-keys",
    "gemini": "https://aistudio.google.com/apikey",
    "anthropic": "https://console.anthropic.com/settings/keys",
}

# 搜索后端：(标识, 说明, Key 环境变量名或 None, 申请地址或 None)
SEARCH_BACKENDS: list[tuple[str, str, str | None, str | None]] = [
    ("bocha", "博查搜索 —— 国内直连，推荐", "BOCHA_API_KEY", "https://open.bochaai.com"),
    ("tavily", "Tavily —— 为 LLM 设计，需国际网络", "TAVILY_API_KEY", "https://app.tavily.com"),
    ("ddg", 'DuckDuckGo —— 免 Key，国内网络不可用；需 pip install "milu[ddg]"', None, None),
]


# ── 工具函数 ───────────────────────────────────────────────

def mask_key(key: str) -> str:
    """掩码显示密钥：保留前 4 后 4 位，中间打码。"""
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}{'*' * 6}{key[-4:]}"


def _format_env_line(key: str, value: str) -> str:
    """生成一行 .env 赋值；值含空白或 # 时加双引号。"""
    if any(ch.isspace() for ch in value) or "#" in value:
        return f'{key}="{value}"'
    return f"{key}={value}"


def update_env_file(path: Path, updates: dict[str, str]) -> Path:
    """合并写 .env：已有键原位更新，新键追加末尾；注释与无关行原样保留。

    :param path: .env 文件路径（不存在则创建，父目录自动建立）。
    :param updates: 要写入/更新的 {环境变量名: 值}。
    :return: 写入的路径。
    """
    lines: list[str] = []
    if path.exists():
        lines = path.read_text(encoding="utf-8-sig").splitlines()

    remaining = dict(updates)
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in remaining:
                out.append(_format_env_line(key, remaining.pop(key)))
                continue
        out.append(line)
    for key, value in remaining.items():
        out.append(_format_env_line(key, value))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


def _ask(prompt: str, default: str = "") -> str:
    """读取一行输入（空输入返回 default）；EOF 视同 Ctrl+C 由上层统一处理。

    剥离 BOM/零宽字符——Windows 下管道喂入（如 echo ... | milu setup）首行常带 BOM。
    """
    try:
        raw = input(prompt).replace("\ufeff", "").replace("\u200b", "").strip()
    except EOFError:
        raise KeyboardInterrupt
    return raw or default


def _ask_yes_no(prompt: str, default: bool) -> bool:
    """y/n 询问，回车取默认。"""
    suffix = "[Y/n]" if default else "[y/N]"
    raw = _ask(f"{prompt} {suffix} ").lower()
    if not raw:
        return default
    return raw in ("y", "yes")


# ── 各步骤 ─────────────────────────────────────────────────

def _step_provider(providers: list[str], default_models: dict, current: str) -> str:
    """步骤 1：选择默认厂商（编号或名称，回车取当前配置）。"""
    from milu.cli.config import env_key_name

    print(c("bold", t("\n[1/4] 选择默认厂商")))
    for i, name in enumerate(providers, 1):
        label = t(PROVIDER_LABELS.get(name, ""))
        has_key = bool(os.environ.get(env_key_name(name)))
        status = c("green", t("已配置 Key")) if has_key else c("dim", t("未配置 Key"))
        model = default_models.get(name, "—")
        mark = c("yellow", t("  ← 当前默认")) if name == current else ""
        print(f"  {i:>2}. {c('cyan', f'{name:<10}')} {label:<18} "
              f"{t('默认模型')} {c('dim', f'{model:<22}')} {status}{mark}")

    while True:
        raw = _ask(t("请输入编号或厂商名（回车 = {cur}）: ", cur=current), current).lower()
        if raw.isdigit() and 1 <= int(raw) <= len(providers):
            return providers[int(raw) - 1]
        if raw in providers:
            return raw
        print(c("red", t("  无效输入：{raw}，请输入 1-{n} 或厂商名。", raw=raw, n=len(providers))))


def _step_model(provider: str, default_models: dict) -> str:
    """步骤 2：选择模型（回车用厂商内置默认）。"""
    default = default_models.get(provider) or ""
    print(c("bold", t("\n[2/4] 选择模型（厂商: {p}）", p=provider)))
    while True:
        if default:
            model = _ask(t("回车使用默认 {d}，或输入其他模型名: ", d=c('cyan', default)), default)
        else:
            model = _ask(t("该厂商无内置默认模型，请输入模型名: "))
        if model:
            return model
        print(c("red", t("  模型名不能为空。")))


def _step_api_key(provider: str) -> str | None:
    """步骤 3：配置 API Key。返回新输入的 Key；保留现有/跳过时返回 None。"""
    from milu.cli.config import env_key_name

    env_name = env_key_name(provider)
    existing = os.environ.get(env_name)
    url = PROVIDER_KEY_URLS.get(provider)

    print(c("bold", t("\n[3/4] 配置 API Key（环境变量 {env}）", env=env_name)))
    if url:
        print(t("  申请地址: {url}", url=c('cyan', t(url))))
    if existing:
        print(t("  当前已配置: {masked}", masked=c('green', mask_key(existing))))
        key = _ask(t("  回车保留现有，或粘贴新 Key: "))
    else:
        key = _ask(t("  请粘贴 API Key（回车跳过，稍后可再运行 milu setup）: "))
        if not key:
            print(c("yellow", t("  未配置 Key，调用 {p} 时会鉴权失败。", p=provider)))
    return key or None


def _step_search() -> tuple[str | None, str | None, str | None]:
    """步骤 4：选择联网搜索后端。

    :return: (后端标识或 None=跳过, Key 环境变量名或 None, 新 Key 或 None)。
    """
    current = os.environ.get("WEB_SEARCH_PROVIDER", "ddg").strip().lower()
    print(c("bold", t("\n[4/4] 配置联网搜索工具（web_search 后端）")))
    for i, (name, desc, _env, url) in enumerate(SEARCH_BACKENDS, 1):
        url_str = f"  {c('cyan', url)}" if url else ""
        mark = c("yellow", t("  ← 当前")) if name == current else ""
        print(f"  {i}. {c('cyan', f'{name:<7}')} {t(desc)}{url_str}{mark}")
    skip_no = len(SEARCH_BACKENDS) + 1
    print(t("  {n}. 跳过（保持现状: {cur}）", n=skip_no, cur=current))

    backend: str | None = None
    while True:
        raw = _ask(t("请输入编号（回车 = 跳过）: "), str(skip_no)).lower()
        if raw.isdigit() and 1 <= int(raw) <= skip_no:
            idx = int(raw)
            backend = None if idx == skip_no else SEARCH_BACKENDS[idx - 1][0]
            break
        names = [b[0] for b in SEARCH_BACKENDS]
        if raw in names:
            backend = raw
            break
        print(c("red", t("  无效输入：{raw}。", raw=raw)))

    if backend is None:
        return None, None, None

    env_name = next(b[2] for b in SEARCH_BACKENDS if b[0] == backend)
    if env_name is None:  # ddg 免 Key
        try:
            import ddgs  # noqa: F401
        except ImportError:
            print(c("yellow", t('  未检测到 ddgs 库（可选依赖），使用前请执行: pip install "milu[ddg]"')))
        return backend, None, None

    existing = os.environ.get(env_name)
    if existing:
        print(t("  {env} 已配置: {masked}", env=env_name, masked=c('green', mask_key(existing))))
        key = _ask(t("  回车保留现有，或粘贴新 Key: "))
    else:
        key = _ask(t("  请粘贴 {env}（回车跳过）: ", env=env_name))
        if not key:
            print(c("yellow", t("  未配置 {env}，搜索工具运行时会报错提示。", env=env_name)))
    return backend, env_name, key or None


# ── Key 验证（可选，最小请求）──────────────────────────────

async def _verify_api_key(provider: str, model: str) -> tuple[bool, str]:
    """发起一次最小 LLM 请求验证 Key 是否可用（Key 从环境变量读取）。"""
    from milu.llm.base.message import Message, MessageRole
    from milu.llm.providers import ModelRegistry

    llm = ModelRegistry.create(provider, model=model)
    gen = llm.chat([Message(role=MessageRole.USER, content="你好")], max_tokens=8)
    try:
        try:
            await anext(gen)
        except StopAsyncIteration:
            pass  # 空流也说明连通与鉴权成功
        return True, ""
    except Exception as e:
        return False, str(e)
    finally:
        aclose = getattr(gen, "aclose", None)
        if aclose is not None:
            await aclose()
        await llm.aclose()


# ── 主流程 ─────────────────────────────────────────────────

def run_setup_wizard() -> int:
    """交互式初始化引导：选厂商 → 选模型 → API Key → 搜索后端 → 写入并可选验证。"""
    from milu.config import DEFAULT_PROVIDER, load_config, set_user_value
    from milu.llm.providers import ModelRegistry
    from milu.resources import user_data_dir

    config = load_config()
    providers = ModelRegistry.list_providers()
    default_models = config.default_models
    current_provider = config.llm.get("provider") or DEFAULT_PROVIDER
    env_path = user_data_dir() / ".env"

    # 语言选择（即时切换并持久化到用户配置；回车保持当前）
    try:
        lang_raw = _ask(t("语言 / Language [zh/en]（回车 = {cur}）: ", cur=get_lang()), get_lang()).lower()
    except KeyboardInterrupt:
        print(c("dim", t("\n已取消初始化引导，未写入任何配置。")))
        return 130
    chosen_lang = "en" if lang_raw.startswith("e") else "zh"
    set_lang(chosen_lang)
    set_user_value("lang", chosen_lang)

    print(BANNER)
    print(c("bold", c("cyan", t("  milu 初始化引导"))))
    print(BANNER)
    print(t("  共 4 步：选厂商 → 选模型 → API Key → 搜索工具（Ctrl+C 随时退出）"))
    print(t("  密钥写入 {p}，厂商/模型写入用户配置。", p=c('dim', str(env_path))))

    # 收集阶段（Ctrl+C 退出不写任何文件）
    try:
        provider = _step_provider(providers, default_models, current_provider)
        model = _step_model(provider, default_models)
        api_key = _step_api_key(provider)
        backend, search_env, search_key = _step_search()
    except KeyboardInterrupt:
        print(c("dim", t("\n已取消初始化引导，未写入任何配置。")))
        return 130

    # 写入阶段：密钥 → 用户级 .env；厂商/模型 → 用户级 config.json
    from milu.cli.config import env_key_name

    env_updates: dict[str, str] = {}
    if api_key:
        env_updates[env_key_name(provider)] = api_key
    if backend:
        env_updates["WEB_SEARCH_PROVIDER"] = backend
    if search_key and search_env:
        env_updates[search_env] = search_key

    if env_updates:
        update_env_file(env_path, env_updates)
        os.environ.update(env_updates)  # 当前进程即时生效（如引导后直接进入对话）
    config_path, _ = set_user_value("agent.llm.provider", provider)
    set_user_value("agent.llm.model", model)

    # 可选验证（已写入完成，Ctrl+C 只中断验证不影响配置）
    try:
        if os.environ.get(env_key_name(provider)) and _ask_yes_no(
            t("\n是否立即验证 {p} 的 API Key（发送一次最小请求）？", p=provider), default=False
        ):
            print(c("dim", t("  验证中...")))
            ok, err = asyncio.run(_verify_api_key(provider, model))
            if ok:
                print(c("green", t("  验证通过，Key 可用。")))
            else:
                print(c("red", t("  验证失败：{err}", err=err)))
                print(c("dim", t("  请检查 Key 是否正确，可重新运行 milu setup 修改。")))
    except KeyboardInterrupt:
        print(c("dim", t("\n  已跳过验证。")))

    # 总结
    print(f"\n{DIVIDER}")
    print(c("bold", c("green", t("  配置完成！"))))
    print(t("  厂商/模型: {p} / {m}  ", p=c('cyan', provider), m=c('cyan', model))
          + c('dim', f'→ {config_path}'))
    if api_key:
        print(f"  API Key:   {env_key_name(provider)}={mask_key(api_key)}  "
              f"{c('dim', f'→ {env_path}')}")
    if backend:
        key_str = f"（{search_env}={mask_key(search_key)}）" if search_key else ""
        print(t("  搜索后端:  {b}{keystr}  ", b=c('cyan', backend), keystr=key_str)
              + c('dim', f'→ {env_path}'))
    print(t("\n  现在运行 {milu} 即可开始对话；重新配置请再运行 {setup}。",
            milu=c('bold', 'milu'), setup=c('bold', 'milu setup')))
    print(DIVIDER)
    return 0


def offer_first_run_setup(provider: str) -> bool:
    """chat 入口检测不到 Key 时的首次使用询问。

    :param provider: 本次运行解析出的厂商名。
    :return: 是否实际运行了引导（True 时调用方应重新加载配置）。
    """
    from milu.cli.config import env_key_name

    print(c("yellow", t("\n未检测到 {p} 的 API Key（环境变量 {env}）。",
                        p=provider, env=env_key_name(provider))))
    try:
        if _ask_yes_no(t("是否现在进行初始化引导？"), default=True):
            return run_setup_wizard() == 0
    except KeyboardInterrupt:
        print()
    print(c("dim", t("已跳过。稍后可运行 `milu setup`，或在 .env 中设置 {env}。",
                     env=env_key_name(provider))))
    return False
