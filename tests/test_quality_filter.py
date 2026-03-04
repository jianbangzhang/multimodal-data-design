"""质量过滤器单元测试。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.filters.quality import is_valid


def make_sample(msgs, images=None, audios=None):
    s = {"messages": msgs}
    if images is not None:
        s["images"] = images
    if audios is not None:
        s["audios"] = audios
    return s


def test_valid_text_sample():
    sample = make_sample([
        {"role": "user",      "content": "请描述这张图片"},
        {"role": "assistant", "content": "这是一张风景图，展示了山川河流的壮丽景色。"},
    ])
    ok, reason = is_valid(sample)
    assert ok, f"应该通过，但原因: {reason}"


def test_too_few_messages():
    sample = make_sample([
        {"role": "user", "content": "你好"},
    ])
    ok, reason = is_valid(sample)
    assert not ok
    assert "消息不足" in reason


def test_wrong_role_order():
    sample = make_sample([
        {"role": "assistant", "content": "我先说话"},
        {"role": "user",      "content": "然后用户说"},
    ])
    ok, reason = is_valid(sample)
    assert not ok
    assert "首条" in reason


def test_answer_too_short():
    sample = make_sample([
        {"role": "user",      "content": "描述图片"},
        {"role": "assistant", "content": "好的"},
    ])
    ok, reason = is_valid(sample)
    assert not ok
    assert "过短" in reason


def test_bad_pattern_rejection():
    sample = make_sample([
        {"role": "user",      "content": "<image>描述图片"},
        {"role": "assistant", "content": "作为AI我无法查看或分析这张图片。"},
    ])
    ok, reason = is_valid(sample)
    assert not ok
    assert "疑似拒答" in reason


def test_image_tag_mismatch():
    """<image> 标签数量与图片列表不匹配应该过滤掉（不检查文件存在性）。"""
    import tempfile, os
    # 创建一个真实的临时文件，确保不被文件存在性检查提前拦截
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(b"fake")
        tmp = f.name
    try:
        sample = make_sample(
            msgs=[
                {"role": "user",      "content": "<image><image>比较两图"},
                {"role": "assistant", "content": "两张图都展示了城市风景，但第一张更现代。"},
            ],
            images=[tmp],  # 只有1张图但有2个tag
        )
        ok, reason = is_valid(sample)
        assert not ok, f"应该被过滤，reason={reason}"
        assert "标签数" in reason, f"原因应包含'标签数'，实际: {reason}"
    finally:
        os.unlink(tmp)


def test_multiturn_valid():
    sample = make_sample([
        {"role": "user",      "content": "<image>请描述这张图片"},
        {"role": "assistant", "content": "这是一张城市夜景图，高楼林立，灯火通明。"},
        {"role": "user",      "content": "图中有哪些建筑？"},
        {"role": "assistant", "content": "图中可以看到现代摩天大楼和一座电视塔。"},
    ])
    ok, reason = is_valid(sample)
    assert ok, f"多轮对话应该通过，但原因: {reason}"


if __name__ == "__main__":
    tests = [
        test_valid_text_sample,
        test_too_few_messages,
        test_wrong_role_order,
        test_answer_too_short,
        test_bad_pattern_rejection,
        test_image_tag_mismatch,
        test_multiturn_valid,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {t.__name__}: {e}")

    print(f"\n{passed}/{len(tests)} 通过")
