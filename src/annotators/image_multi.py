"""单图多轮对话标注器：探究型 / 任务型 / 知识扩展 三种风格。"""
from typing import Optional

from src.annotators.base import BaseAnnotator
from src.utils.llm_client import LLMClient
from src.utils.media import encode_image
from src.utils.json_parser import parse_json_response


class ImageMultiTurnAnnotator(BaseAnnotator):
    """
    对单张图片生成多轮对话，每种风格产出一条多轮 SFT 样本。

    三种风格（来自方案文档）：
    - 探究型：从整体到细节逐层追问
    - 任务导向：围绕决策/判断/比较
    - 知识扩展：从图片出发追问背景知识
    """

    def __init__(self, client: LLMClient, prompts: dict, n_styles: int = 2):
        super().__init__(client, prompts)
        self.template = prompts["image_multi_template"]
        self.styles = prompts["multiturn_styles"][:n_styles]

    def annotate(self, image_path: str, style_cfg: Optional[dict] = None) -> Optional[dict]:
        """
        对单张图片生成一种风格的多轮对话。
        如不指定 style_cfg，默认使用第一种风格。
        """
        if style_cfg is None:
            style_cfg = self.styles[0]

        try:
            b64, mt = encode_image(image_path)
        except Exception as e:
            print(f"[ERROR] 读取图片失败 {image_path}: {e}")
            return None

        prompt = self.template.format(**style_cfg)
        messages = [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mt, "data": b64}},
            {"type": "text", "text": prompt},
        ]}]

        raw = self.client.chat(messages, max_tokens=2000)
        if raw is None:
            return None

        result = parse_json_response(raw)
        if result is None:
            print(f"[WARN] JSON 解析失败 {image_path} [{style_cfg['style']}]")
        return result

    def annotate_all_styles(self, image_path: str) -> list[dict]:
        """对同一张图片生成所有已配置风格的多轮对话。"""
        results = []
        for style_cfg in self.styles:
            ann = self.annotate(image_path, style_cfg)
            if ann:
                results.append((style_cfg, ann))
        return results

    def to_sft_samples(self, image_path: str, annotation: dict) -> list[dict]:
        """将单个风格的多轮对话转换为 SFT 样本。"""
        turns = annotation.get("turns", [])
        if len(turns) < 2:
            return []
        return [{"messages": turns, "images": [str(image_path)]}]

    def annotate_and_convert(self, image_path: str) -> list[dict]:
        """对单张图片生成所有风格的多轮对话并转换，一步完成。"""
        samples = []
        for style_cfg, ann in self.annotate_all_styles(image_path):
            samples.extend(self.to_sft_samples(image_path, ann))
        return samples
