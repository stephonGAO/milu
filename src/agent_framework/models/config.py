"""模型配置参数定义 - 基础配置和按能力扩展配置"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """
    模型调用的基础参数配置（所有厂商通用）。

    属性:
        model: 模型名称，如 "qwen-max", "kimi-k2.5"
        temperature: 温度参数，控制生成随机性（0.0~2.0）
        top_p: 核采样概率，控制词汇多样性（0.0~1.0）
        max_tokens: 最大输出token数
        stop: 停止词列表，遇到任一停止词则停止生成
        stream: 是否流式输出，本框架固定为True
        frequency_penalty: 频率惩罚，降低重复内容（-2.0~2.0）
        presence_penalty: 存在惩罚，鼓励新话题（-2.0~2.0）
    """
    model: str
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    stream: bool = True
    frequency_penalty: float | None = None
    presence_penalty: float | None = None

    def to_dict(self) -> dict:
        """转换为字典，排除None值，始终包含stream=True"""
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                result[key] = value
        result["stream"] = True
        return result


@dataclass
class WebSearchConfig(ModelConfig):
    """联网搜索扩展参数。适用于支持 web_search 能力的厂商。"""
    web_search: bool = False
    web_search_strategy: str = "auto"


@dataclass
class ThinkingConfig(ModelConfig):
    """
    思考/推理模式扩展参数。
    thinking_level 统一为 low/medium/high 三级，默认 medium。
    """
    enable_thinking: bool = False
    thinking_level: str = "medium"


@dataclass
class FunctionCallingConfig(ModelConfig):
    """函数调用扩展参数。"""
    tools: list[dict] | None = None
    tool_choice: str | dict = "auto"


@dataclass
class ImageGenerationConfig(ModelConfig):
    """图片生成扩展参数。"""
    image_size: str = "1024x1024"
    image_quality: str = "standard"
    num_images: int = 1


@dataclass
class AudioGenerationConfig(ModelConfig):
    """音频生成扩展参数（TTS等）。"""
    voice: str | None = None
    audio_format: str = "mp3"
    speed: float = 1.0
