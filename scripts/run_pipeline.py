"""
多模态 SFT 数据生成主流水线。

用法：
    python scripts/run_pipeline.py                          # 运行全部模块
    python scripts/run_pipeline.py --mode image_single      # 只运行单图单轮
    python scripts/run_pipeline.py --mode image_multi
    python scripts/run_pipeline.py --mode multi_image
    python scripts/run_pipeline.py --mode video
    python scripts/run_pipeline.py --mode audio
    python scripts/run_pipeline.py --config configs/my.yaml # 指定配置文件
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml
import traceback

# 确保可以从项目根目录导入
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.llm_client import LLMClient
from src.utils.media import get_image_files, get_video_files
from src.annotators import (
    ImageSingleAnnotator,
    ImageMultiTurnAnnotator,
    MultiImageAnnotator,
    build_pairs_sequential,
    VideoAnnotator,
    AudioAnnotator,
)
from src.filters.quality import filter_dataset
from src.converters.to_llamafactory import register_to_llamafactory
from src.utils.utils import save_final_json, save_jsonl_online


# ── 配置加载 ───────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompts(prompts_path: str) -> dict:
    with open(prompts_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── 各模块运行函数 ──────────────────────────────────────────────────────────────

def run_image_single(cfg: dict, prompts: dict, client: LLMClient) -> list[dict]:
    """单图单轮标注。"""
    print("\n" + "="*50)
    print("[1/5] 单图单轮对话")
    print("="*50)

    annotator = ImageSingleAnnotator(client, prompts)
    images = get_image_files(cfg["data"]["image_dir"])
    target = cfg["targets"]["image_single"]
    # 每张图约产出 7 条，估算需要的图片数
    needed = target // 7 + 1
    images = images[:needed]

    dataset = []
    for i, path in enumerate(images):
        print(f"  [{i+1}/{len(images)}] {Path(path).name}", end=" ... ")
        samples = annotator.annotate_and_convert(path)
        dataset.extend(samples)
        print(f" {len(samples)} 条" if samples else " 跳过")

    print(f"  共 {len(dataset)} 条")
    return dataset


def run_image_multi(cfg: dict, prompts: dict, client: LLMClient) -> list[dict]:
    """单图多轮标注。"""
    print("\n" + "="*50)
    print("[2/5] 单图多轮对话")
    print("="*50)

    n_styles = cfg.get("multiturn", {}).get("n_styles", 2)
    annotator = ImageMultiTurnAnnotator(client, prompts, n_styles=n_styles)
    images = get_image_files(cfg["data"]["image_dir"])
    target = cfg["targets"]["image_multi"]
    needed = target // n_styles + 1
    images = images[:needed]

    dataset = []
    for i, path in enumerate(images):
        print(f"  [{i+1}/{len(images)}] {Path(path).name}", end=" ... ")
        samples = annotator.annotate_and_convert(path)
        dataset.extend(samples)
        print(f" {len(samples)} 条" if samples else " 跳过")

    print(f"  共 {len(dataset)} 条")
    return dataset


def run_multi_image(cfg: dict, prompts: dict, client: LLMClient) -> list[dict]:
    """多图对比标注。"""
    print("\n" + "="*50)
    print("[3/5] 多图对比对话")
    print("="*50)

    annotator = MultiImageAnnotator(client, prompts)
    pairs = build_pairs_sequential(
        cfg["data"]["image_dir"],
        max_pairs=cfg["targets"]["multi_image"] // 4 + 1,
    )

    dataset = []
    for i, (path_a, path_b) in enumerate(pairs):
        print(f"  [{i+1}/{len(pairs)}] {Path(path_a).name} + {Path(path_b).name}", end=" ... ")
        samples = annotator.annotate_and_convert(path_a, path_b)
        dataset.extend(samples)
        print(f" {len(samples)} 条" if samples else " 跳过")

    print(f"  共 {len(dataset)} 条")
    return dataset


def run_video(cfg: dict, prompts: dict, client: LLMClient) -> list[dict]:
    """视频对话标注。"""
    print("\n" + "="*50)
    print("[4/5] 视频对话")
    print("="*50)

    video_cfg = cfg.get("video", {})
    annotator = VideoAnnotator(
        client, prompts,
        frames_per_video=video_cfg.get("frames_per_video", 8),
        jpeg_quality=video_cfg.get("jpeg_quality", 90),
        frames_dir=cfg["data"].get("frames_dir", "data/raw/video_frames"),
    )
    videos = get_video_files(cfg["data"]["video_dir"])
    target = cfg["targets"]["video"]
    # 每个视频约产出 7 条
    needed = target // 7 + 1
    videos = videos[:needed]

    dataset = []
    for i, path in enumerate(videos):
        print(f"  [{i+1}/{len(videos)}] {Path(path).name}", end=" ... ")
        samples = annotator.annotate_and_convert(path)
        dataset.extend(samples)
        print(f" {len(samples)} 条" if samples else " 跳过")

    print(f"  共 {len(dataset)} 条")
    return dataset


def run_audio(cfg: dict, prompts: dict, client: LLMClient) -> list[dict]:
    """音频对话标注（ASR 模式）。"""
    print("\n" + "="*50)
    print("[5/5] 音频对话（ASR）")
    print("="*50)

    audio_cfg = cfg.get("audio", {})
    annotator = AudioAnnotator(
        client, prompts,
        voices_per_sample=audio_cfg.get("voices_per_sample", 2),
        max_tts_text_len=audio_cfg.get("max_tts_text_len", 100),
    )
    dataset = annotator.process_for_asr(cfg["data"]["audio_dir"])
    target = cfg["targets"]["audio"]
    dataset = dataset[:target]

    print(f"  共 {len(dataset)} 条")
    return dataset


# ── 主流水线 ───────────────────────────────────────────────────────────────────

MODE_RUNNERS = {
    "image_single": run_image_single,
    "image_multi":  run_image_multi,
    "multi_image":  run_multi_image,
    "video":        run_video,
    "audio":        run_audio,
}


def run_pipeline(config_path: str = "configs/config.yaml",
                 mode: str = "all"):
    # 1. 加载配置
    cfg = load_config(config_path)
    prompts = load_prompts("configs/prompts.yaml")

    # 2. 初始化 LLM 客户端
    client = LLMClient(
            provider="qwen_local",
            model="ckpt/qwen_vl"
        )

    print(f"模型: qwen vl")
    print(f"输出目录: {cfg['data']['output_dir']}")

    # 3. 运行各模块
    all_data: list[dict] = []

    if mode == "all":
        runners = list(MODE_RUNNERS.values())
    else:
        if mode not in MODE_RUNNERS:
            print(f"[ERROR] 未知 mode: {mode}，可选: {list(MODE_RUNNERS.keys())}")
            sys.exit(1)
        runners = [MODE_RUNNERS[mode]]

    for runner in runners:
        try:
            results = runner(cfg, prompts, client)
            save_jsonl_online(results, f"{cfg['data']['output_dir']}/{runner.__name__}_online.jsonl")
            all_data.extend(results)
        except Exception as e:
            print(f"[ERROR] {runner.__name__} 失败: {e}")
            traceback.print_exc()
            
    # 4. 保存原始结果
    output_dir = cfg["data"]["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    raw_output = f"{output_dir}/multimodal_sft_{mode}_raw.json"

    save_final_json(all_data, raw_output)
    print(f"\n原始数据: {len(all_data)} 条 {raw_output}")

    # 5. 质量过滤
    filtered_output = f"{output_dir}/multimodal_sft.json"
    filter_cfg = cfg.get("filter", {})
    filter_dataset(
        raw_output, filtered_output,
        bad_patterns=filter_cfg.get("bad_patterns"),
    )

    # 6. 注册到 LlamaFactory（可选）
    lf_cfg = cfg.get("llamafactory", {})
    if lf_cfg.get("enabled"):
        register_to_llamafactory(
            filtered_output,
            lf_cfg.get("dataset_name", "my_multimodal_sft"),
            lf_cfg.get("dataset_info_path", "LLaMA-Factory/data/dataset_info.json"),
        )

    print(f"\n全部完成，最终数据 {filtered_output}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多模态 SFT 数据生成流水线")
    parser.add_argument("--config", default="configs/config.yaml", help="配置文件路径")
    parser.add_argument(
        "--mode",
        default="all",
        choices=["all", "image_single", "image_multi", "multi_image", "video", "audio"],
        help="运行模式：all 运行全部，或指定单个模块"
    )
    args = parser.parse_args()

    run_pipeline(config_path=args.config, mode=args.mode)
