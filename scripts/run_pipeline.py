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

模型分配说明：
    图像 / 视频  → Qwen3-VL      (ckpt/qwen_vl)
    音频         → Qwen3-Omni    (ckpt/qwen_omni)   ← 必须用全模态模型
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

import yaml

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
    needed = target // 7 + 1
    images = images[:needed]

    dataset = []
    for i, path in enumerate(images):
        print(f"  [{i+1}/{len(images)}] {Path(path).name}", end=" ... ")
        samples = annotator.annotate_and_convert(path)
        save_jsonl_online(samples, f"{cfg['data']['output_dir']}/image_single_{Path(path).stem}.jsonl")
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
        save_jsonl_online(samples, f"{cfg['data']['output_dir']}/image_multi_{Path(path).stem}.jsonl")
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
        save_jsonl_online(samples, f"{cfg['data']['output_dir']}/multi_image_{Path(path_a).stem}_{Path(path_b).stem}.jsonl")
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
    needed = target // 7 + 1
    videos = videos[:needed]

    dataset = []
    for i, path in enumerate(videos):
        print(f"  [{i+1}/{len(videos)}] {Path(path).name}", end=" ... ")
        samples = annotator.annotate_and_convert(path)
        save_jsonl_online(samples, f"{cfg['data']['output_dir']}/video_{Path(path).stem}.jsonl")
        dataset.extend(samples)
        print(f" {len(samples)} 条" if samples else " 跳过")

    print(f"  共 {len(dataset)} 条")
    return dataset


def run_audio(cfg: dict, prompts: dict, client: LLMClient) -> list[dict]:
    """
    音频对话标注（ASR 模式）。
    必须传入 Qwen3-Omni 客户端，Qwen3-VL 不支持音频输入。
    """
    print("\n" + "="*50)
    print("[5/5] 音频对话（ASR）")
    print("="*50)

    # 安全检查：防止误用 VL 模型跑音频
    if client.provider == "qwen_local":
        print("[WARNING] 音频模块需要 Qwen3-Omni，当前传入的是 Qwen3-VL，跳过音频标注。")
        print("          请在 configs/config.yaml 中配置 audio.model_path，或设置 audio.skip=true")
        return []

    audio_cfg = cfg.get("audio", {})
    annotator = AudioAnnotator(
        client, prompts,
        voices_per_sample=audio_cfg.get("voices_per_sample", 2),
        max_tts_text_len=audio_cfg.get("max_tts_text_len", 100),
    )
    dataset = annotator.process_for_asr(cfg["data"]["audio_dir"])
    target = cfg["targets"]["audio"]
    dataset = dataset[:target]

    save_final_json(dataset, f"{cfg['data']['output_dir']}/audio_asr_raw.json")
    print(f"  共 {len(dataset)} 条")
    return dataset


# ── 主流水线 ───────────────────────────────────────────────────────────────────

# 各模块使用的模型类型：vision 用 Qwen3-VL，audio 用 Qwen3-Omni
MODALITY_MAP = {
    "image_single": "vision",
    "image_multi":  "vision",
    "multi_image":  "vision",
    "video":        "vision",
    "audio":        "audio",   # ← 必须走独立的全模态模型
}

MODE_RUNNERS = {
    "image_single": run_image_single,
    "image_multi":  run_image_multi,
    "multi_image":  run_multi_image,
    "video":        run_video,
    "audio":        run_audio,
}


def _build_clients(cfg: dict) -> dict[str, LLMClient]:
    """
    按需加载模型，避免不必要的显存占用。
    返回 {"vision": LLMClient, "audio": LLMClient | None}
    """
    model_cfg = cfg.get("models", {})

    vision_path = model_cfg.get("vision_model_path", "ckpt/qwen_vl")
    print(f"\n[模型] 视觉模型: {vision_path}")
    vision_client = LLMClient(provider="qwen_local", model=vision_path)

    # 音频模型按需加载
    audio_path = model_cfg.get("audio_model_path", "")
    audio_client = None
    if audio_path:
        print(f"[模型] 音频模型: {audio_path}")
        audio_client = LLMClient(provider="qwen_omni_local", model=audio_path)
    else:
        print("[模型] 未配置 audio_model_path，音频模块将被跳过")
        print("       在 configs/config.yaml 中添加 models.audio_model_path: ckpt/qwen_omni 以启用")

    return {"vision": vision_client, "audio": audio_client}


def run_pipeline(config_path: str = "configs/config.yaml", mode: str = "all"):
    # 1. 加载配置
    cfg = load_config(config_path)
    prompts = load_prompts("configs/prompts.yaml")
    output_dir = cfg["data"]["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 2. 确定需要运行的模块
    if mode == "all":
        runners = list(MODE_RUNNERS.items())
    else:
        if mode not in MODE_RUNNERS:
            print(f"[ERROR] 未知 mode: {mode}，可选: {list(MODE_RUNNERS.keys())}")
            sys.exit(1)
        runners = [(mode, MODE_RUNNERS[mode])]

    # 3. 按需加载模型（只在需要时加载音频模型）
    needs_audio = any(MODALITY_MAP[m] == "audio" for m, _ in runners)
    needs_vision = any(MODALITY_MAP[m] == "vision" for m, _ in runners)

    model_cfg = cfg.get("models", {})
    vision_client = None
    audio_client = None

    if needs_vision:
        vision_path = model_cfg.get("vision_model_path", "ckpt/qwen_vl")
        print(f"\n[模型] 视觉模型加载: {vision_path}")
        vision_client = LLMClient(provider="qwen_local", model=vision_path)

    if needs_audio:
        audio_path = model_cfg.get("audio_model_path", "")
        if audio_path:
            print(f"[模型] 音频模型加载: {audio_path}")
            audio_client = LLMClient(provider="qwen_omni_local", model=audio_path)
        else:
            print("[模型] ⚠️  需要运行音频模块，但 models.audio_model_path 未配置，音频将被跳过")

    print(f"[输出] {output_dir}\n")

    # 4. 运行各模块，按模态选择对应客户端
    all_data: list[dict] = []

    for module_name, runner in runners:
        modality = MODALITY_MAP[module_name]
        client = audio_client if modality == "audio" else vision_client

        # 音频模型未配置时安全跳过
        if client is None:
            print(f"\n[SKIP] {module_name} 跳过（模型未加载）")
            continue

        try:
            results = runner(cfg, prompts, client)
            all_data.extend(results)
        except Exception as e:
            print(f"[ERROR] {runner.__name__} 失败: {e}")
            traceback.print_exc()

    # 5. 保存汇总结果
    raw_output = f"{output_dir}/multimodal_sft_{mode}_raw.json"
    save_final_json(all_data, raw_output)
    print(f"\n原始数据: {len(all_data)} 条 → {raw_output}")

    # 6. 质量过滤
    filtered_output = f"{output_dir}/multimodal_sft.json"
    filter_cfg = cfg.get("filter", {})
    filter_dataset(
        raw_output, filtered_output,
        bad_patterns=filter_cfg.get("bad_patterns"),
    )

    # 7. 注册到 LlamaFactory（可选）
    lf_cfg = cfg.get("llamafactory", {})
    if lf_cfg.get("enabled"):
        register_to_llamafactory(
            filtered_output,
            lf_cfg.get("dataset_name", "my_multimodal_sft"),
            lf_cfg.get("dataset_info_path", "LLaMA-Factory/data/dataset_info.json"),
        )

    print(f"\n全部完成，最终数据 → {filtered_output}")


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
