"""JSON 解析器单元测试。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.json_parser import parse_json_response


def test_plain_json():
    text = '{"key": "value", "num": 42}'
    result = parse_json_response(text)
    assert result == {"key": "value", "num": 42}


def test_markdown_json_block():
    text = '```json\n{"key": "value"}\n```'
    result = parse_json_response(text)
    assert result == {"key": "value"}


def test_markdown_block_no_lang():
    text = '```\n{"key": "value"}\n```'
    result = parse_json_response(text)
    assert result == {"key": "value"}


def test_json_with_surrounding_text():
    text = 'Sure! Here is the result:\n{"key": "value"}\nHope that helps!'
    result = parse_json_response(text)
    assert result == {"key": "value"}


def test_json_array():
    text = '[{"a": 1}, {"a": 2}]'
    result = parse_json_response(text)
    assert result == [{"a": 1}, {"a": 2}]


def test_invalid_returns_none():
    result = parse_json_response("这不是JSON")
    assert result is None


def test_whitespace_handling():
    text = '  \n  {"key": "value"}  \n  '
    result = parse_json_response(text)
    assert result == {"key": "value"}


if __name__ == "__main__":
    tests = [
        test_plain_json,
        test_markdown_json_block,
        test_markdown_block_no_lang,
        test_json_with_surrounding_text,
        test_json_array,
        test_invalid_returns_none,
        test_whitespace_handling,
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
