"""模型成本估算 —— 双轨制（学 Langfuse）：token 一律用 provider 返回的真实
usage，单价查表估算；查不到价格返回 None（绝不静默给错数，展示层标"未配置价格"）。

价格表结构（每百万 token 单价）：
    {"deepseek-chat": {
        "input": 2.0, "cached_input": 0.2, "output": 8.0, "currency": "CNY"
    }, ...}

``cached_input`` 是缓存命中输入 token 的单价。若本次确有缓存命中而价格表未配置
该档价格，返回 None，避免仍按普通输入单价静默高估成本。

匹配规则：模型名小写后先精确命中，再按「最长前缀」匹配
（如 qwen-plus-latest → qwen-plus）。用户表（config.json
observability.price_table）优先于内置表。

⚠️ 内置表仅为出厂示例（2026-06 前后整理），厂商随时调价、新模型层出不穷，
**强烈建议按实际价目在 config 中覆盖**。estimate 结果一律是"估算"。
"""
from __future__ import annotations

# 出厂示例价（每百万 token；CNY=人民币 / USD=美元）。覆盖少量常见基线模型，
# 仅作开箱即有数的示例；生产使用请在 config.json observability.price_table 覆盖。
BUILTIN_PRICE_TABLE: dict[str, dict] = {
    # 国内厂商
    "deepseek-chat": {
        "input": 2.0, "cached_input": 0.2, "output": 8.0, "currency": "CNY",
    },
    "qwen-turbo": {
        "input": 0.3, "cached_input": 0.06, "output": 0.6, "currency": "CNY",
    },
    "qwen-plus": {
        "input": 0.8, "cached_input": 0.16, "output": 2.0, "currency": "CNY",
    },
    "qwen-max": {
        "input": 2.4, "cached_input": 0.48, "output": 9.6, "currency": "CNY",
    },
    # 海外厂商
    "gpt-4o-mini": {
        "input": 0.15, "cached_input": 0.075, "output": 0.6, "currency": "USD",
    },
    "gpt-4o": {
        "input": 2.5, "cached_input": 1.25, "output": 10.0, "currency": "USD",
    },
    "claude-sonnet": {"input": 3.0, "output": 15.0, "currency": "USD"},
}


def resolve_price(model: str, override: dict | None = None) -> dict | None:
    """查模型单价：用户表优先于内置表；精确命中优先于最长前缀。查不到返回 None。"""
    if not model:
        return None
    key = model.lower()
    tables = [t for t in (override, BUILTIN_PRICE_TABLE) if t]
    for table in tables:
        entry = table.get(key) or table.get(model)
        if isinstance(entry, dict):
            return entry
    # 最长前缀匹配（两表合并比较，用户表条目在同长度时优先）
    best: tuple[int, int, dict] | None = None  # (前缀长, 表优先级负值, entry)
    for prio, table in enumerate(tables):
        for k, entry in table.items():
            if not isinstance(entry, dict):
                continue  # 容忍 "//" 注释键等杂项
            kl = k.lower()
            if key.startswith(kl):
                cand = (len(kl), -prio, entry)
                if best is None or cand[:2] > best[:2]:
                    best = cand
    return best[2] if best else None


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    override: dict | None = None,
    *,
    cached_input_tokens: int = 0,
) -> tuple[float, str] | None:
    """估算一次运行的成本。返回 (金额, 币种)；价格缺失/数据异常返回 None。"""
    entry = resolve_price(model, override)
    if entry is None:
        return None
    try:
        input_count = max(int(input_tokens or 0), 0)
        cached_count = min(max(int(cached_input_tokens or 0), 0), input_count)
        cached_price = entry.get("cached_input")
        if cached_count and cached_price is None:
            return None
        cost = (
            (input_count - cached_count) / 1_000_000 * float(entry["input"])
            + cached_count / 1_000_000 * float(cached_price or 0)
            + (output_tokens or 0) / 1_000_000 * float(entry["output"])
        )
        return round(cost, 6), str(entry.get("currency", "CNY"))
    except (KeyError, TypeError, ValueError):
        return None
