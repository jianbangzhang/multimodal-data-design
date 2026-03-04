"""标注器抽象基类。"""
from abc import ABC, abstractmethod
from typing import Optional

from src.utils.llm_client import LLMClient


class BaseAnnotator(ABC):
    """
    所有标注器的基类，统一初始化 LLM 客户端和 Prompt 加载。
    子类只需实现 annotate() 和 to_sft_samples() 两个方法。
    """

    def __init__(self, client: LLMClient, prompts: dict):
        self.client = client
        self.prompts = prompts

    @abstractmethod
    def annotate(self, *args, **kwargs) -> Optional[dict]:
        """对单个输入（图片/视频帧列表/音频）进行标注，返回结构化结果。"""
        ...

    @abstractmethod
    def to_sft_samples(self, source_path, annotation: dict) -> list[dict]:
        """将标注结果转换为 SFT 训练样本列表。"""
        ...

    def annotate_and_convert(self, *args, **kwargs) -> list[dict]:
        """便捷方法：标注 + 转换，一步完成。"""
        ann = self.annotate(*args, **kwargs)
        if ann is None:
            return []
        # 第一个位置参数作为 source_path
        source_path = args[0] if args else kwargs.get("path", "")
        return self.to_sft_samples(source_path, ann)
