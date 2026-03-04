"""将生成的数据集注册到 LlamaFactory 的 dataset_info.json。"""
import json
from pathlib import Path


def register_to_llamafactory(data_path: str, dataset_name: str,
                             info_file: str = "LLaMA-Factory/data/dataset_info.json") -> bool:
    """
    将数据集注册到 LlamaFactory，使其可直接在训练配置中引用。

    Args:
        data_path:    SFT 数据文件的绝对路径
        dataset_name: 在 LlamaFactory 中使用的数据集名称
        info_file:    dataset_info.json 的路径

    Returns:
        True 表示注册成功，False 表示 info_file 不存在（跳过）
    """
    if not Path(info_file).exists():
        print(f"[LlamaFactory] dataset_info.json 不存在，跳过注册: {info_file}")
        return False

    with open(info_file, encoding="utf-8") as f:
        info = json.load(f)

    info[dataset_name] = {
        "file_name": str(Path(data_path).absolute()),
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
            "images": "images",
        },
    }

    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"[LlamaFactory] 已注册数据集: {dataset_name} → {data_path}")
    return True
