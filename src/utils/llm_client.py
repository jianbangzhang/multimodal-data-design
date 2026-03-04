import os
import base64
from typing import Optional


class LLMClient:
    """
    统一接口，屏蔽 Anthropic、OpenAI 和本地 Qwen2.5-VL 的差异。

    用法:
        # Anthropic
        client = LLMClient(provider="anthropic", model="claude-opus-4-6")

        # OpenAI
        client = LLMClient(provider="openai", model="gpt-4o")

        # 本地 Qwen2.5-VL
        client = LLMClient(
            provider="qwen_local",
            model="/Users/zhangjianbang/Downloads/minicpmo_training/multimodal-data-factory/ckpt/qwenvl"
        )

        text = client.chat([
            {"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": "描述这张图片"}
            ]}
        ])
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-opus-4-6",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.provider = provider
        self.model = model

        if provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
            )

        elif provider == "openai":
            import openai
            self._client = openai.OpenAI(
                api_key=api_key or os.environ.get("OPENAI_API_KEY"),
                base_url=base_url,
            )

        elif provider == "qwen_local":
            self._load_qwen_local(model)

        else:
            raise ValueError(
                f"不支持的 provider: {provider}，请使用 'anthropic' | 'openai' | 'qwen_local'"
            )

    # ------------------------------------------------------------------
    # 本地 Qwen2.5-VL 加载
    # ------------------------------------------------------------------
    def _load_qwen_local(self, model_path: str):
        """
        加载本地 Qwen2.5-VL 权重。
        - CUDA 可用  → GPU，dtype=bfloat16
        - Apple MPS  → 先加载到 CPU(float32)，再 .to("mps")，规避 MPS 大 buffer 分配 bug
        - 其他        → CPU，dtype=float32
        """
        import platform
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        # ---------- 判断设备 ----------
        if torch.cuda.is_available():
            device     = "cuda"
            torch_dtype = torch.bfloat16
            device_map  = "auto"
            print(f"[LLMClient] 检测到 CUDA，使用 GPU 加载: {model_path}")
        elif platform.system() == "Darwin" and torch.backends.mps.is_available():
            # float16：3B 模型只需 ~6 GB，刚好小于 MPS 显存上限（~6.8 GB）。
            # 设置 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 解除分配器预热限制。
            import os as _os
            _os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
            device      = "mps"
            torch_dtype = torch.float16
            device_map  = None
            print(f"[LLMClient] Apple MPS + float16 加载: {model_path}")
        else:
            device      = "cpu"
            torch_dtype = torch.float32
            device_map  = None
            print(f"[LLMClient] 使用 CPU 加载: {model_path}")

        # ---------- 加载模型 ----------
        load_kwargs = dict(torch_dtype=torch_dtype)
        if device_map:
            load_kwargs["device_map"] = device_map

        self._qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, **load_kwargs
        )

        # MPS / CPU 手动迁移（CUDA 已由 device_map 处理）
        if device_map is None:
            self._qwen_model = self._qwen_model.to(device)

        self._qwen_model.eval()
        self._qwen_device = device

        self._qwen_processor = AutoProcessor.from_pretrained(model_path)
        print(f"[LLMClient] Qwen2.5-VL 加载完成，运行设备: {device}")

    # ------------------------------------------------------------------
    # 统一 chat 入口
    # ------------------------------------------------------------------
    def chat(self, messages: list[dict], max_tokens: int = 2000) -> Optional[str]:
        """发送消息，返回模型文本输出。失败返回 None。"""
        try:
            if self.provider == "anthropic":
                return self._chat_anthropic(messages, max_tokens)
            elif self.provider == "openai":
                return self._chat_openai(messages, max_tokens)
            elif self.provider == "qwen_local":
                return self._chat_qwen_local(messages, max_tokens)
        except Exception as e:
            print(f"[LLMClient ERROR] {e}")
            return None

    # ------------------------------------------------------------------
    # Anthropic
    # ------------------------------------------------------------------
    def _chat_anthropic(self, messages, max_tokens):
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=messages,
        )
        return resp.content[0].text.strip()

    # ------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------
    def _chat_openai(self, messages, max_tokens):
        oai_messages = _convert_to_openai_format(messages)
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=oai_messages,
        )
        return resp.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # 本地 Qwen2.5-VL
    # ------------------------------------------------------------------
    def _chat_qwen_local(self, messages: list[dict], max_tokens: int) -> str:
        """
        将 Anthropic 格式消息转换为 Qwen 格式，执行本地推理，返回文本。
        """
        from qwen_vl_utils import process_vision_info

        qwen_messages = _convert_to_qwen_format(messages)

        # 构建 prompt 文本
        text = self._qwen_processor.apply_chat_template(
            qwen_messages, tokenize=False, add_generation_prompt=True
        )

        # 提取图像 / 视频输入
        image_inputs, video_inputs = process_vision_info(qwen_messages)

        inputs = self._qwen_processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self._qwen_device)

        generated_ids = self._qwen_model.generate(
            **inputs, max_new_tokens=max_tokens
        )

        # 去掉 prompt 部分，只保留生成内容
        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self._qwen_processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0].strip()


# ======================================================================
# 格式转换工具函数
# ======================================================================

def _convert_to_openai_format(messages: list[dict]) -> list[dict]:
    """
    Anthropic 多模态格式  →  OpenAI 格式。
    Anthropic image: {"type":"image","source":{"type":"base64","media_type":"...","data":"..."}}
    OpenAI   image:  {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}
    """
    converted = []
    for msg in messages:
        if isinstance(msg["content"], str):
            converted.append(msg)
            continue

        new_content = []
        for part in msg["content"]:
            if part["type"] == "text":
                new_content.append({"type": "text", "text": part["text"]})
            elif part["type"] == "image":
                src = part["source"]
                if src["type"] == "base64":
                    url = f"data:{src['media_type']};base64,{src['data']}"
                    new_content.append({"type": "image_url", "image_url": {"url": url}})
        converted.append({"role": msg["role"], "content": new_content})

    return converted


def _convert_to_qwen_format(messages: list[dict]) -> list[dict]:
    """
    Anthropic 多模态格式  →  Qwen2.5-VL 格式。

    Anthropic image block:
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "<b64str>"}}

    Qwen image block (base64):
        {"type": "image", "image": "data:image/jpeg;base64,<b64str>"}

    纯文本 content（str）直接透传。
    """
    converted = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        # 纯字符串消息，直接透传
        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue

        new_content = []
        for part in content:
            ptype = part.get("type")

            if ptype == "text":
                new_content.append({"type": "text", "text": part["text"]})

            elif ptype == "image":
                src = part.get("source", {})
                src_type = src.get("type")

                if src_type == "base64":
                    # 转为 data-URI 供 qwen_vl_utils 解析
                    media_type = src.get("media_type", "image/jpeg")
                    data = src.get("data", "")
                    # data 可能是 bytes 或 str
                    if isinstance(data, bytes):
                        data = base64.b64encode(data).decode()
                    data_uri = f"data:{media_type};base64,{data}"
                    new_content.append({"type": "image", "image": data_uri})

                elif src_type == "url":
                    # 直接使用 URL
                    new_content.append({"type": "image", "image": src["url"]})

                elif src_type == "file":
                    # 本地文件路径
                    new_content.append({"type": "image", "image": src["path"]})

            # 如果已经是 Qwen 原生格式（直接包含 "image" 键），原样保留
            elif ptype is None and "image" in part:
                new_content.append(part)

        converted.append({"role": role, "content": new_content})

    return converted