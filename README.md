# Multimodal Data Factory

> **核心原则：不造图，造标注。** 图片/视频/音频用已有的，精力花在标注质量和问题多样性上。

用强模型（Claude / GPT-4o）对已有素材自动生成高质量多模态 SFT 训练数据，支持：

-  **单图单轮对话** — Caption + 5类 QA + OCR
-  **单图多轮对话** — 探究型 / 任务型 / 知识扩展 三种风格
-  **多图对比对话** — 差异分析 + 单图 QA
-  **视频对话** — 均匀帧采样 + 时序理解 + 排序任务
-  **音频对话** — ASR + TTS合成 + 情感/意图理解

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp configs/config.example.yaml configs/config.yaml
# 编辑 config.yaml，填入 ANTHROPIC_API_KEY 等

# 3. 放入你的图片
cp path/to/image/*.jpg data/raw/images/

# 4. 运行完整流水线
python scripts/run_pipeline.py

# 5. 只运行单个模块
python scripts/run_pipeline.py --mode image_single
python scripts/run_pipeline.py --mode image_multi
python scripts/run_pipeline.py --mode multi_image
python scripts/run_pipeline.py --mode video
python scripts/run_pipeline.py --mode audio
```

---

## 项目结构

```
multimodal-data-factory/
├── src/
│   ├── annotators/          # 标注引擎（各模态的核心标注逻辑）
│   │   ├── base.py          # 抽象基类
│   │   ├── image_single.py  # 单图单轮标注
│   │   ├── image_multi.py   # 单图多轮标注
│   │   ├── multi_image.py   # 多图对比标注
│   │   ├── video.py         # 视频帧标注
│   │   └── audio.py         # 音频标注（ASR + TTS + 情感）
│   ├── converters/          # 格式转换（标注结果 → SFT 样本）
│   │   ├── to_sft.py        # 各类标注 → ShareGPT SFT 格式
│   │   └── to_llamafactory.py  # 注册到 LlamaFactory
│   ├── filters/             # 质量过滤
│   │   └── quality.py       # 结构检查 + 幻觉过滤
│   └── utils/               # 工具函数
│       ├── media.py         # 图片编码、视频帧提取
│       ├── llm_client.py    # API 客户端（Anthropic / OpenAI 统一封装）
│       └── json_parser.py   # 健壮的 JSON 解析
├── configs/
│   ├── config.example.yaml  # 配置模板
│   └── prompts.yaml         # 所有 Prompt 集中管理
├── scripts/
│   ├── run_pipeline.py      # 主入口
│   ├── download_data.sh     # 公开数据集下载脚本
│   └── register_llamafactory.py  # 单独注册数据集
├── tests/                   # 单元测试
├── data/
│   ├── raw/                 # 原始素材（图片/视频/音频）
│   └── sft/                 # 生成的 SFT 数据
├── requirements.txt
└── README.md
```

---

## 数据集两阶段策略

### 第一阶段：直接复用公开数据集（下载即用）

```bash
bash scripts/download_data.sh
```

| 数据集 | 大小 | 覆盖能力 |
|--------|------|----------|
| LLaVA-Instruct-150K | 150K | 通用图文问答 |
| ShareGPT4V-PT | 1.2M | 高质量图像描述 |
| VideoChat2-IT | 2M | 视频对话 |

### 第二阶段：针对业务场景重标注（本工具的重点）

```
全量数据  →  Qwen2.5-VL-72B 本地粗标（省成本）
                │
                └─ 低置信度样本（约20%）
                          └─  Claude Opus 精标
```

---

## 输出格式

ShareGPT 格式，兼容 LlamaFactory / LLaVA-Next 等主流训练框架：

```json
{
  "messages": [
    {"role": "user",      "content": "<image>请详细描述这张图片"},
    {"role": "assistant", "content": "图片展示了..."}
  ],
  "images": ["data/raw/images/xxx.jpg"]
}
```

---

## 数据比例参考

| 数据类型 | 推荐占比 |
|----------|---------|
| 单图单轮 | 40% |
| 单图多轮 | 25% |
| 多图对话 | 15% |
| 视频对话 | 10% |
| 音频对话 | 10% |
