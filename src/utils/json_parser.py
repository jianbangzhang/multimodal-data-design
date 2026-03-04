"""健壮的 JSON 解析工具，处理模型输出中常见的格式问题。"""
import json
import re
from typing import Optional


def parse_json_response(text: str) -> Optional[dict | list]:
    """
    解析模型返回的 JSON，自动处理：
    - ```json ... ``` markdown 代码块
    - 首尾多余空白
    - 单引号替换（部分模型会输出单引号）
    """
    text = text.strip()

    # 去除 markdown 代码块包裹
    if text.startswith("```"):
        # 匹配 ```json\n...\n``` 或 ```\n...\n```
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        else:
            # 粗暴去除首尾
            text = text.lstrip("`").lstrip("json").rstrip("`").strip()

    # 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试提取第一个 JSON 对象/数组
    for pattern in [r"\{.*\}", r"\[.*\]"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    return None
