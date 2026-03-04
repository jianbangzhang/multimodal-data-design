"""多图对比标注器：差异分析 + 共同点 + 单图 QA。"""
from pathlib import Path
from typing import Optional
from collections import defaultdict
import json

from src.annotators.base import BaseAnnotator
from src.utils.llm_client import LLMClient
from src.utils.media import encode_image
from src.utils.json_parser import parse_json_response


class MultiImageAnnotator(BaseAnnotator):
    """
    对两张图片生成对比分析数据：
    - 整体差异对比
    - 3类对比 QA（差异/共同点/综合判断）
    - 单图 QA（每张图单独提问）
    """

    def __init__(self, client: LLMClient, prompts: dict):
        super().__init__(client, prompts)
        self.prompt = prompts["multi_image"]

    def annotate(self, path_a: str, path_b: str = None) -> Optional[dict]:  # type: ignore[override]
        """对两张图片进行对比标注。"""
        try:
            b64_a, mt_a = encode_image(path_a)
            b64_b, mt_b = encode_image(path_b)
        except Exception as e:
            print(f"[ERROR] 读取图片失败: {e}")
            return None

        messages = [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mt_a, "data": b64_a}},
            {"type": "image", "source": {"type": "base64", "media_type": mt_b, "data": b64_b}},
            {"type": "text", "text": self.prompt},
        ]}]

        raw = self.client.chat(messages, max_tokens=1500)
        if raw is None:
            return None

        result = parse_json_response(raw)
        if result is None:
            print(f"[WARN] JSON 解析失败 图对 {path_a} / {path_b}")
        return result

    def to_sft_samples(self, path_a: str, annotation: dict,  # type: ignore[override]
                       path_b: str = None) -> list[dict]:
        """标注结果 → SFT 样本列表。"""
        if path_b is None:
            return []

        samples = []
        images = [path_a, path_b]

        # 整体差异描述（取第一条对比 QA 的回答）
        compare_qa = annotation.get("compare_qa", [])
        if compare_qa:
            diff_summary = compare_qa[0].get("a", "")
            samples.append({"messages": [
                {"role": "user",      "content": "<image><image>请比较这两张图片的主要差异"},
                {"role": "assistant", "content": diff_summary},
            ], "images": images})

        # 3条对比 QA
        for qa in compare_qa:
            if qa.get("q") and qa.get("a"):
                samples.append({"messages": [
                    {"role": "user",      "content": f"<image><image>{qa['q']}"},
                    {"role": "assistant", "content": qa["a"]},
                ], "images": images})

        # 单图 QA（<image> 数量必须与图片数量匹配！）
        for qa in annotation.get("individual_qa", []):
            q, a = qa.get("q", ""), qa.get("a", "")
            if not q or not a:
                continue
            # 问图A：只传图A，只放1个<image>
            if "第一张" in q or "左图" in q:
                samples.append({"messages": [
                    {"role": "user",      "content": f"<image>{q}"},
                    {"role": "assistant", "content": a},
                ], "images": [path_a]})
            # 问图B：只传图B，只放1个<image>
            elif "第二张" in q or "右图" in q:
                samples.append({"messages": [
                    {"role": "user",      "content": f"<image>{q}"},
                    {"role": "assistant", "content": a},
                ], "images": [path_b]})

        return samples

    def annotate_and_convert(self, path_a: str, path_b: str) -> list[dict]:  # type: ignore[override]
        """标注 + 转换，一步完成。"""
        ann = self.annotate(path_a, path_b)
        if ann is None:
            return []
        return self.to_sft_samples(path_a, ann, path_b)


# ── 图片对构建工具 ─────────────────────────────────────────────────────────────

def build_pairs_from_coco(coco_ann_file: str, image_dir: str,
                          max_pairs: int = 2000) -> list[tuple[str, str]]:
    """
    从 COCO 标注中按类别分组，构建同类别图片对。
    同类别的两张图有语义关联，对比 QA 质量更高。
    """
    with open(coco_ann_file) as f:
        coco = json.load(f)

    # category_id → image_ids
    cat_images: dict = defaultdict(set)
    for ann in coco["annotations"]:
        cat_images[ann["category_id"]].add(ann["image_id"])

    # image_id → file_name
    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

    pairs: list[tuple[str, str]] = []
    for cat_id, img_ids in cat_images.items():
        imgs = [
            f"{image_dir}/{id_to_file[i]}"
            for i in list(img_ids)[:20]
            if i in id_to_file and Path(f"{image_dir}/{id_to_file[i]}").exists()
        ]
        for i in range(0, len(imgs) - 1, 2):
            pairs.append((imgs[i], imgs[i + 1]))

    return pairs[:max_pairs]


def build_pairs_sequential(image_dir: str, max_pairs: int = 2000) -> list[tuple[str, str]]:
    """
    按文件名排序，构建相邻图片对（适合时序场景）。
    """
    from src.utils.media import get_image_files
    imgs = get_image_files(image_dir)
    pairs = [(imgs[i], imgs[i + 1]) for i in range(0, len(imgs) - 1, 2)]
    return pairs[:max_pairs]
