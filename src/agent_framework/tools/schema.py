"""Python 函数签名 → JSON Schema 转换"""
from __future__ import annotations

import inspect
import re
from typing import get_type_hints, get_origin, get_args, Literal, Optional


def python_type_to_json_schema(py_type) -> dict:
    """将 Python 类型注解转为 JSON Schema 类型"""
    # 基础类型
    if py_type == str:
        return {"type": "string"}
    elif py_type == int:
        return {"type": "integer"}
    elif py_type == float:
        return {"type": "number"}
    elif py_type == bool:
        return {"type": "boolean"}

    # 泛型类型
    origin = get_origin(py_type)
    args = get_args(py_type)

    # Optional[X] -> Union[X, None]
    if origin is type(None) or (origin and str(origin) == "typing.Union" and type(None) in args):
        non_none_args = [a for a in args if a is not type(None)]
        if non_none_args:
            return python_type_to_json_schema(non_none_args[0])
        return {"type": "string"}

    # Literal["a", "b"]
    if origin is Literal:
        return {"type": "string", "enum": list(args)}

    # list[X]
    if origin is list:
        return {"type": "array"}

    # dict
    if origin is dict or py_type == dict:
        return {"type": "object"}

    return {"type": "string"}


def extract_param_descriptions(docstring: str | None) -> dict[str, str]:
    """从 docstring 提取参数描述（:param name: description 格式）"""
    if not docstring:
        return {}
    descriptions = {}
    pattern = r":param\s+(\w+):\s*(.+?)(?=\n\s*:|\n\s*$|$)"
    matches = re.findall(pattern, docstring, re.DOTALL)
    for name, desc in matches:
        descriptions[name] = desc.strip()
    return descriptions


def generate_schema_from_function(func) -> dict:
    """从函数签名生成 OpenAI function calling schema"""
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    param_descriptions = extract_param_descriptions(func.__doc__)

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls") or param_name.startswith("_"):
            continue

        py_type = type_hints.get(param_name, str)
        json_schema = python_type_to_json_schema(py_type)

        if param_name in param_descriptions:
            json_schema["description"] = param_descriptions[param_name]
        else:
            json_schema["description"] = param_name

        if param.default != inspect.Parameter.empty:
            json_schema["default"] = param.default
        else:
            is_optional = get_origin(py_type) and type(None) in get_args(py_type)
            if not is_optional:
                required.append(param_name)

        properties[param_name] = json_schema

    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema
