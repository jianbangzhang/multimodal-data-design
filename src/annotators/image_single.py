"""单图单轮对话标注器：Caption + 5类 QA + OCR + Tags。"""
from typing import Optional

from src.annotators.base import BaseAnnotator
from src.utils.llm_client import LLMClient
from src.utils.media import encode_image
from src.utils.json_parser import parse_json_response


class ImageSingleAnnotator(BaseAnnotator):
    """
    对单张图片生成：
    - 详细/简洁 Caption
    - 5类 QA（感知/计数/空间/属性/推理）
    - OCR 文字识别
    - 主题 Tags

    一次 API 调用产出 5~8 条 SFT 样本，摊薄 API 成本。
    """

    def __init__(self, client: LLMClient, prompts: dict):
        super().__init__(client, prompts)
        self.prompt = prompts["image_single"]

    def annotate(self, image_path: str) -> Optional[dict]:
        """调用 LLM 对单张图片进行标注，返回解析后的 dict。"""
        try:
            b64, mt = encode_image(image_path)
        except Exception as e:
            print(f"[ERROR] 读取图片失败 {image_path}: {e}")
            return None

        messages = [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mt, "data": b64}},
            {"type": "text", "text": self.prompt},
        ]}]

        raw = self.client.chat(messages, max_tokens=1500)
        if raw is None:
            return None

        result = parse_json_response(raw)
        if result is None:
            print(f"[WARN] JSON 解析失败 {image_path}: {raw[:100]}")
        return result

    def to_sft_samples(self, image_path: str, ann: dict) -> list[dict]:
        """标注结果 → SFT 样本列表。"""
        samples = []
        rel = str(image_path)

        caption = ann.get("caption", {})

        # 详细描述
        if caption.get("detailed"):
            samples.append({"messages": [
                {"role": "user",      "content": "<image>请详细描述这张图片"},
                {"role": "assistant", "content": caption["detailed"]},
            ], "images": [rel]})

        # 简洁描述
        if caption.get("one_sentence"):
            samples.append({"messages": [
                {"role": "user",      "content": "<image>用一句话描述这张图片"},
                {"role": "assistant", "content": caption["one_sentence"]},
            ], "images": [rel]})

        # QA 对（最多5类）
        for qa in ann.get("qa_pairs", []):
            if qa.get("q") and qa.get("a"):
                samples.append({"messages": [
                    {"role": "user",      "content": f"<image>{qa['q']}"},
                    {"role": "assistant", "content": qa["a"]},
                ], "images": [rel]})

        # OCR（有文字才加）
        if ann.get("ocr"):
            samples.append({"messages": [
                {"role": "user",      "content": "<image>请识别图片中的所有文字"},
                {"role": "assistant", "content": ann["ocr"]},
            ], "images": [rel]})

        return samples
