"""内置工具：日期时间查询与转换

支持获取当前时间、解析日期字符串、时间戳转换等操作。
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Literal, Optional

from agent_framework.tools.decorator import tool

# 自动识别的常见日期格式
_AUTO_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%Y%m%d",
]

_WEEKDAY_NAMES = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


def _datetime_to_dict(dt: datetime) -> dict:
    """将 datetime 对象转为信息字典"""
    return {
        "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "date": dt.strftime("%Y-%m-%d"),
        "time": dt.strftime("%H:%M:%S"),
        "timestamp": int(dt.timestamp()),
        "year": dt.year,
        "month": dt.month,
        "day": dt.day,
        "weekday": _WEEKDAY_NAMES[dt.weekday()],
    }


def _parse_datetime(value: str, fmt: str | None = None) -> datetime:
    """解析日期字符串，支持指定格式或自动识别"""
    if fmt:
        return datetime.strptime(value, fmt)

    for f in _AUTO_FORMATS:
        try:
            return datetime.strptime(value, f)
        except ValueError:
            continue
    raise ValueError(f"无法自动识别日期格式: {value}")


def get_current_time(fmt: str | None = None) -> dict:
    """获取当前日期时间信息"""
    now = datetime.now()
    result = _datetime_to_dict(now)
    if fmt:
        result["datetime"] = now.strftime(fmt)
    return result


@tool(name="datetime_tool", description="日期时间工具：获取当前时间、解析日期字符串、时间戳转换")
async def datetime_tool(
    operation: Literal["now", "parse", "timestamp"],
    value: Optional[str] = None,
    format: Optional[str] = None,
) -> str:
    """
    日期时间操作工具。

    :param operation: 操作类型 - now(当前时间)、parse(解析日期字符串)、timestamp(时间戳转日期)
    :param value: 操作的输入值 - parse时为日期字符串，timestamp时为Unix时间戳
    :param format: 日期格式字符串 - now时自定义输出格式，parse时指定输入格式
    """
    try:
        if operation == "now":
            result = get_current_time(fmt=format)
            return json.dumps(result, ensure_ascii=False)

        elif operation == "parse":
            if not value:
                return "错误: parse 操作需要提供 value 参数"
            dt = _parse_datetime(value, format)
            return json.dumps(_datetime_to_dict(dt), ensure_ascii=False)

        elif operation == "timestamp":
            if not value:
                return "错误: timestamp 操作需要提供 value 参数（Unix时间戳）"
            ts = float(value)
            dt = datetime.fromtimestamp(ts)
            return json.dumps(_datetime_to_dict(dt), ensure_ascii=False)

        else:
            return f"错误: 不支持的操作 '{operation}'，可选: now, parse, timestamp"

    except Exception as e:
        return f"日期时间错误: {e}"
