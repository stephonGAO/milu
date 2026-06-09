"""统一异常体系 - 所有自定义异常的公共基类，以及厂商错误归类工具。"""


class MiluError(Exception):
    """框架基础异常，所有其他异常的父类"""
    pass


# ── 内容合规 / 安全风控错误识别 ──────────────────────────────
#
# 国产大模型（通义千问 / GLM / 豆包 / Kimi / MiniMax 等）普遍对**输入**与**生成
# 内容**做安全审核，命中时多以 HTTP 400 抛错（data_inspection_failed /
# content_filter / 敏感 / 违规 等关键词）。这类错误与限流、网络抖动不同——
# **重试也不会变好**，应直接给用户清晰提示而非静默吞掉或交给上层 LLM 胡乱解释。

_CONTENT_SAFETY_KEYWORDS = (
    # 通义千问 / DashScope
    "data_inspection_failed", "data inspection",
    "may contain inappropriate", "inappropriate content",
    # OpenAI 兼容通用
    "content_filter", "content filter", "contentfilter",
    "content_policy", "content policy", "content management policy",
    "content moderation", "moderation",
    "responsible ai", "responsibleaipolicy",
    "usage policy", "usage policies", "prohibited content",
    # 智谱 GLM / 豆包 / Kimi / MiniMax 等中文风控提示
    "内容安全", "安全审核", "内容审核", "审核未通过", "命中拦截",
    "敏感", "合规", "违规", "风控", "不安全内容",
)


def content_safety_hint(error: object) -> str | None:
    """若 error 为大模型内容合规/安全风控拦截，返回面向用户的友好中文提示；否则 None。

    :param error: 异常对象或错误字符串。
    :return: 命中返回提示文案（含截断的服务端原始原因，便于排查）；未命中返回 None。
    """
    text = str(error or "")
    low = text.lower()
    if not any(kw in low for kw in _CONTENT_SAFETY_KEYWORDS):
        return None
    raw = " ".join(text.split())          # 去多余空白/换行
    if len(raw) > 160:
        raw = raw[:160] + "…"
    return (
        "⚠️ 内容合规拦截：模型服务商判定本次请求或其生成内容可能涉及敏感/违规信息，"
        "已拒绝响应。这是国产大模型常见的内容安全风控（并非程序故障）——可尝试调整"
        "措辞、更换话题，或改用其他模型/提供商后重试。"
        f"\n（服务端原始提示：{raw}）"
    )
