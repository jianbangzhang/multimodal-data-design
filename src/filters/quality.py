"""质量过滤：结构检查 + 幻觉/拒答模式过滤。"""
import json
from pathlib import Path


def is_valid(sample: dict, bad_patterns: list[str] = None) -> tuple[bool, str]:
    """
    检查单条 SFT 样本是否符合质量要求。
    返回 (是否通过, 原因说明)。
    """
    if bad_patterns is None:
        bad_patterns = [
            "无法查看", "无法分析这张", "作为AI我无法",
            "图片似乎没有", "I cannot", "I'm unable to",
        ]

    msgs = sample.get("messages", [])

    # ── 结构检查 ──────────────────────────────────────────────────────────────
    if len(msgs) < 2:
        return False, "消息不足2条"
    if msgs[0]["role"] != "user":
        return False, "首条消息非user"
    if msgs[-1]["role"] != "assistant":
        return False, "末条消息非assistant"

    # ── <image> 标签数量与图片数量对齐 ───────────────────────────────────────
    imgs = sample.get("images", [])
    n_image_tags = sum(
        m["content"].count("<image>")
        for m in msgs if isinstance(m.get("content"), str)
    )
    if imgs and n_image_tags != len(imgs):
        return False, f"<image>标签数({n_image_tags}) ≠ 图片数({len(imgs)})"

    # ── <audio> 标签与音频数量对齐 ────────────────────────────────────────────
    audios = sample.get("audios", [])
    n_audio_tags = sum(
        m["content"].count("<audio>")
        for m in msgs if isinstance(m.get("content"), str)
    )
    if audios and n_audio_tags != len(audios):
        return False, f"<audio>标签数({n_audio_tags}) ≠ 音频数({len(audios)})"

    # ── 媒体文件存在性检查 ────────────────────────────────────────────────────
    for p in imgs + audios:
        if not Path(p).exists():
            return False, f"文件不存在: {p}"

    # ── 回答质量 ──────────────────────────────────────────────────────────────
    ans = msgs[-1].get("content", "")
    if len(ans) < 8:
        return False, "回答过短（<8字）"

    # ── 拒答/幻觉模式检查 ─────────────────────────────────────────────────────
    for pat in bad_patterns:
        if pat in ans:
            return False, f"疑似拒答: {pat}"

    return True, "ok"


def filter_dataset(input_file: str, output_file: str,
                   bad_patterns: list[str] = None) -> dict:
    """
    对整个数据集进行质量过滤，返回过滤统计。

    Args:
        input_file:   输入 JSON 文件路径
        output_file:  过滤后输出 JSON 文件路径
        bad_patterns: 自定义拒答模式列表

    Returns:
        {total, passed, filtered, filter_reasons}
    """
    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)

    passed = []
    reasons: dict[str, int] = {}

    for sample in data:
        ok, reason = is_valid(sample, bad_patterns)
        if ok:
            passed.append(sample)
        else:
            reasons[reason] = reasons.get(reason, 0) + 1

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(passed, f, ensure_ascii=False, indent=2)

    stats = {
        "total": len(data),
        "passed": len(passed),
        "filtered": len(data) - len(passed),
        "pass_rate": f"{len(passed)/len(data)*100:.1f}%" if data else "N/A",
        "filter_reasons": reasons,
    }

    print(f"质量过滤完成：{stats['passed']}/{stats['total']} 通过 "
          f"({stats['pass_rate']})，{stats['filtered']} 条被过滤")
    if reasons:
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  - {reason}: {count} 条")

    return stats
