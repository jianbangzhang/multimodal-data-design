import os
import base64
from typing import Optional


class LLMClient:
    """
    统一接口，支持 Anthropic / OpenAI / 本地 Qwen3-VL / 本地 Qwen3-Omni。

    provider 可选值：
        "anthropic"       - Claude API
        "openai"          - OpenAI API（或兼容接口）
        "qwen_local"      - 本地 Qwen3-VL（图像 / 视频）
        "qwen_omni_local" - 本地 Qwen3-Omni（图像 / 视频 / 音频 全模态）

    用法示例：
        # 图像 / 视频
        vision_client = LLMClient(provider="qwen_local", model="ckpt/qwen_vl")

        # 音频（需要 Qwen3-Omni 权重）
        audio_client = LLMClient(provider="qwen_omni_local", model="ckpt/qwen_omni")

        # 发送消息（格式统一使用 Anthropic 多模态格式）
        text = vision_client.chat([
            {"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": "描述这张图片"}
            ]}
        ])

        # 发送音频（audio block）
        text = audio_client.chat([
            {"role": "user", "content": [
                {"type": "audio", "source": {"type": "file", "path": "/path/to/audio.wav"}},
                {"type": "text", "text": "转录这段音频"}
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

        elif provider == "qwen_omni_local":
            self._load_qwen_omni_local(model)

        else:
            raise ValueError(
                f"不支持的 provider: {provider}，"
                f"可选: 'anthropic' | 'openai' | 'qwen_local' | 'qwen_omni_local'"
            )

    # ------------------------------------------------------------------
    # 设备检测（复用）
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_device():
        """返回 (device, torch_dtype, device_map)"""
        import platform
        import torch

        if torch.cuda.is_available():
            return "cuda", torch.bfloat16, "auto"
        elif platform.system() == "Darwin" and torch.backends.mps.is_available():
            os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
            return "mps", torch.float16, None
        else:
            return "cpu", torch.float32, None

    # ------------------------------------------------------------------
    # 本地 Qwen3-VL 加载（图像 / 视频）
    # ------------------------------------------------------------------
    def _load_qwen_local(self, model_path: str):
        import torch
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        device, torch_dtype, device_map = self._detect_device()
        print(f"[LLMClient] Qwen3-VL 加载中 (device={device}): {model_path}")

        load_kwargs = dict(torch_dtype=torch_dtype)
        if device_map:
            load_kwargs["device_map"] = device_map

        self._qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, **load_kwargs
        )
        if device_map is None:
            self._qwen_model = self._qwen_model.to(device)

        self._qwen_model.eval()
        self._qwen_device = device
        self._qwen_processor = AutoProcessor.from_pretrained(model_path)
        print(f"[LLMClient] Qwen3-VL 加载完成，运行设备: {device}")

    # ------------------------------------------------------------------
    # 本地 Qwen3-Omni 加载（图像 / 视频 / 音频 全模态）
    # ------------------------------------------------------------------
    def _load_qwen_omni_local(self, model_path: str):
        """
        Qwen3-Omni 支持音频输入输出，是处理音频标注的正确模型。
        官方推荐 bfloat16 + CUDA，MPS 暂不稳定建议降级到 CPU。
        """
        import torch
        from transformers import Qwen3OmniForConditionalGeneration, AutoProcessor

        device, torch_dtype, device_map = self._detect_device()

        # Qwen3-Omni 在 MPS 上尚不稳定，降级到 CPU
        if device == "mps":
            print("[LLMClient] Qwen3-Omni 暂不支持 MPS，降级到 CPU")
            device, torch_dtype, device_map = "cpu", torch.float32, None

        print(f"[LLMClient] Qwen3-Omni 加载中 (device={device}): {model_path}")

        load_kwargs = dict(torch_dtype=torch_dtype)
        if device_map:
            load_kwargs["device_map"] = device_map

        self._omni_model = Qwen3OmniForConditionalGeneration.from_pretrained(
            model_path, **load_kwargs
        )
        if device_map is None:
            self._omni_model = self._omni_model.to(device)

        self._omni_model.eval()
        self._omni_device = device
        self._omni_processor = AutoProcessor.from_pretrained(model_path)
        print(f"[LLMClient] Qwen3-Omni 加载完成，运行设备: {device}")

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
            elif self.provider == "qwen_omni_local":
                return self._chat_qwen_omni_local(messages, max_tokens)
        except Exception as e:
            print(f"[LLMClient ERROR] ({self.provider}) {e}")
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
    # 本地 Qwen3-VL（图像 / 视频）
    # ------------------------------------------------------------------
    def _chat_qwen_local(self, messages: list[dict], max_tokens: int) -> str:
        qwen_messages = _convert_to_qwen_format(messages)

        inputs = self._qwen_processor.apply_chat_template(
            qwen_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._qwen_device)

        generated_ids = self._qwen_model.generate(**inputs, max_new_tokens=max_tokens)
        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._qwen_processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0].strip()

    # ------------------------------------------------------------------
    # 本地 Qwen3-Omni（图像 / 视频 / 音频）
    # ------------------------------------------------------------------
    def _chat_qwen_omni_local(self, messages: list[dict], max_tokens: int) -> str:
        """
        Qwen3-Omni 推理入口。
        消息格式与 Qwen3-VL 相同，额外支持 audio block：
            {"type": "audio", "source": {"type": "file", "path": "/path/to/audio.wav"}}
            {"type": "audio", "source": {"type": "base64", "media_type": "audio/wav", "data": "<b64>"}}
        """
        omni_messages = _convert_to_omni_format(messages)

        # use_audio_in_video=True 让模型同时处理视频中的音轨；
        # 纯音频文件标注场景设 False
        inputs = self._omni_processor.apply_chat_template(
            omni_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            use_audio_in_video=False,
        )
        inputs = inputs.to(self._omni_device)

        generated_ids = self._omni_model.generate(**inputs, max_new_tokens=max_tokens)
        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._omni_processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0].strip()


# ======================================================================
# 格式转换工具函数
# ======================================================================

def _convert_to_openai_format(messages: list[dict]) -> list[dict]:
    """Anthropic 多模态格式 → OpenAI 格式"""
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
    Anthropic 多模态格式 → Qwen3-VL 格式（图像 / 视频）

    支持的 source 类型：
        base64  → data:image/jpeg;base64,...
        url     → 直接使用 URL
        file    → 本地路径
    """
    converted = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

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
                    media_type = src.get("media_type", "image/jpeg")
                    data = src.get("data", "")
                    if isinstance(data, bytes):
                        data = base64.b64encode(data).decode()
                    new_content.append({"type": "image", "image": f"data:{media_type};base64,{data}"})
                elif src_type == "url":
                    new_content.append({"type": "image", "image": src["url"]})
                elif src_type == "file":
                    new_content.append({"type": "image", "image": src["path"]})

            # 已是 Qwen 原生格式，原样保留
            elif ptype is None and "image" in part:
                new_content.append(part)

        converted.append({"role": role, "content": new_content})
    return converted


def _convert_to_omni_format(messages: list[dict]) -> list[dict]:
    """
    Anthropic 多模态格式 → Qwen3-Omni 格式（图像 / 视频 / 音频）

    在 _convert_to_qwen_format 基础上新增 audio block 支持：
        {"type": "audio", "source": {"type": "file",   "path": "/path/to/audio.wav"}}
        {"type": "audio", "source": {"type": "base64", "media_type": "audio/wav", "data": "<b64>"}}

    Qwen3-Omni audio block 格式：
        {"type": "audio", "audio": "/path/to/audio.wav"}         # 本地路径
        {"type": "audio", "audio": "data:audio/wav;base64,..."}  # base64
    """
    converted = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

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
                    media_type = src.get("media_type", "image/jpeg")
                    data = src.get("data", "")
                    if isinstance(data, bytes):
                        data = base64.b64encode(data).decode()
                    new_content.append({"type": "image", "image": f"data:{media_type};base64,{data}"})
                elif src_type == "url":
                    new_content.append({"type": "image", "image": src["url"]})
                elif src_type == "file":
                    new_content.append({"type": "image", "image": src["path"]})

            elif ptype == "audio":
                # 新增：音频 block 处理
                src = part.get("source", {})
                src_type = src.get("type")
                if src_type == "file":
                    new_content.append({"type": "audio", "audio": src["path"]})
                elif src_type == "base64":
                    media_type = src.get("media_type", "audio/wav")
                    data = src.get("data", "")
                    if isinstance(data, bytes):
                        data = base64.b64encode(data).decode()
                    new_content.append({"type": "audio", "audio": f"data:{media_type};base64,{data}"})
                elif src_type == "url":
                    new_content.append({"type": "audio", "audio": src["url"]})

            # 已是 Omni 原生格式，原样保留
            elif ptype is None and ("image" in part or "audio" in part):
                new_content.append(part)

        converted.append({"role": role, "content": new_content})
    return converted
