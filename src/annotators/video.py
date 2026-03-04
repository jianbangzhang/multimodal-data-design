"""视频标注器：均匀帧采样 + 时序理解 + 排序任务。"""
import random
from typing import Optional

from src.annotators.base import BaseAnnotator
from src.utils.llm_client import LLMClient
from src.utils.media import encode_image, extract_uniform_frames
from src.utils.json_parser import parse_json_response


class VideoAnnotator(BaseAnnotator):
    """
    视频标注流程：
    1. 从视频均匀采样 N 帧（均匀采样是业界主流，Video-LLaVA/VideoChat 均使用）
    2. 将帧序列发给 LLM，生成视频理解标注
    3. 转换为 SFT 样本：整体描述 + 5类 QA + 帧排序任务
    """

    def __init__(self, client: LLMClient, prompts: dict,
                 frames_per_video: int = 8,
                 jpeg_quality: int = 90,
                 frames_dir: str = "data/raw/video_frames"):
        super().__init__(client, prompts)
        self.prompt_template = prompts["video"]
        self.frames_per_video = frames_per_video
        self.jpeg_quality = jpeg_quality
        self.frames_dir = frames_dir

    def annotate(self, video_path: str) -> Optional[dict]:  # type: ignore[override]
        """提取帧 + 调用 LLM 标注，返回结构化结果。"""
        frame_paths = extract_uniform_frames(
            video_path, n=self.frames_per_video,
            out_dir=f"{self.frames_dir}/{_stem(video_path)}",
            jpeg_quality=self.jpeg_quality,
        )
        if not frame_paths:
            print(f"[WARN] 无法提取帧: {video_path}")
            return None

        return self._annotate_frames(frame_paths)

    def _annotate_frames(self, frame_paths: list[str]) -> Optional[dict]:
        """对已有帧列表调用 LLM 标注。"""
        content = []
        for fp in frame_paths:
            try:
                b64, mt = encode_image(fp)
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": mt, "data": b64}
                })
            except Exception as e:
                print(f"[WARN] 跳过帧 {fp}: {e}")

        if not content:
            return None

        prompt = self.prompt_template.replace("{N}", str(len(content)))
        content.append({"type": "text", "text": prompt})

        raw = self.client.chat(
            [{"role": "user", "content": content}],
            max_tokens=2000,
        )
        if raw is None:
            return None

        result = parse_json_response(raw)
        if result is None:
            print(f"[WARN] 视频标注 JSON 解析失败")
        # 将帧路径附加到结果中，供 to_sft_samples 使用
        if result:
            result["_frame_paths"] = frame_paths
        return result

    def to_sft_samples(self, video_path: str, annotation: dict) -> list[dict]:
        """标注结果 → SFT 样本列表。"""
        frame_paths = annotation.get("_frame_paths", [])
        if not frame_paths:
            return []

        n = len(frame_paths)
        tags = "<image>" * n
        samples = []

        # 整体描述
        if annotation.get("event_summary"):
            samples.append({"messages": [
                {"role": "user",      "content": f"{tags}请描述这段视频的主要内容"},
                {"role": "assistant", "content": annotation["event_summary"]},
            ], "images": frame_paths})

        # 5类 QA
        for qa in annotation.get("qa_pairs", []):
            if qa.get("q") and qa.get("a"):
                samples.append({"messages": [
                    {"role": "user",      "content": f"{tags}{qa['q']}"},
                    {"role": "assistant", "content": qa["a"]},
                ], "images": frame_paths})

        # 帧排序任务（用打乱的4帧子集）
        rt = annotation.get("reorder_task", {})
        if rt.get("correct_order") and len(frame_paths) >= 4:
            sub = frame_paths[2:6]
            shuffled = sub.copy()
            random.shuffle(shuffled)
            sub_tags = "<image>" * len(shuffled)
            answer = rt["correct_order"]
            if rt.get("explanation"):
                answer += "\n理由：" + rt["explanation"]
            samples.append({"messages": [
                {"role": "user", "content": f"{sub_tags}以上帧来自同一视频但顺序被打乱，请按正确时间顺序排列"},
                {"role": "assistant", "content": answer},
            ], "images": shuffled})

        return samples

    def annotate_and_convert(self, video_path: str) -> list[dict]:  # type: ignore[override]
        """提取帧 + 标注 + 转换，一步完成。"""
        ann = self.annotate(video_path)
        if ann is None:
            return []
        return self.to_sft_samples(video_path, ann)


def _stem(path: str) -> str:
    from pathlib import Path
    return Path(path).stem
