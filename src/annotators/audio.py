"""
音频标注器，支持三种模式：
1. ASR 模式：Whisper 转录 → ASR 训练样本
2. TTS 模式：文本 QA → 多音色合成音频 → 音频 QA 样本
3. 理解模式：转录文本 → LLM 分析情感/意图 → 音频理解样本
"""
import asyncio
from pathlib import Path
from typing import Optional

from src.annotators.base import BaseAnnotator
from src.utils.llm_client import LLMClient
from src.utils.json_parser import parse_json_response

# TTS 多音色配置（覆盖不同场景）
TTS_VOICES = [
    "zh-CN-XiaoxiaoNeural",           # 女声，标准普通话
    "zh-CN-YunxiNeural",              # 男声，标准普通话
    "zh-HK-HiuGaaiNeural",            # 粤语
    "zh-CN-liaoning-XiaobeiNeural",   # 东北口音
]


class AudioAnnotator(BaseAnnotator):
    """
    音频标注器。

    ASR 数据（量最大，直接用）：
        WenetSpeech / AISHELL / LibriSpeech → process_for_asr()

    音频理解数据（量少，需构建）：
        - TTS 合成：build_from_tts()
        - 情感/意图：annotate_understanding()
    """

    def __init__(self, client: LLMClient, prompts: dict,
                 whisper_model: str = "large-v3",
                 voices_per_sample: int = 2,
                 max_tts_text_len: int = 100):
        super().__init__(client, prompts)
        self.understanding_prompt = prompts["audio_understanding"]
        self.voices_per_sample = voices_per_sample
        self.max_tts_text_len = max_tts_text_len
        self._whisper = None
        self._whisper_model_name = whisper_model

    def _get_whisper(self):
        """懒加载 Whisper 模型（只在第一次用到时加载）。"""
        if self._whisper is None:
            try:
                import whisper
                print(f"[Audio] 加载 Whisper {self._whisper_model_name}...")
                self._whisper = whisper.load_model(self._whisper_model_name)
            except ImportError:
                raise ImportError("请安装 openai-whisper：pip install openai-whisper")
        return self._whisper

    # ── ASR 模式 ──────────────────────────────────────────────────────────────

    def annotate(self, audio_path: str) -> Optional[dict]:  # type: ignore[override]
        """用 Whisper 转录单个音频文件。"""
        model = self._get_whisper()
        try:
            result = model.transcribe(audio_path, language="zh")
            transcript = result["text"].strip()
            if len(transcript) < 3:
                return None
            return {"transcript": transcript}
        except Exception as e:
            print(f"[ERROR] Whisper 转录失败 {audio_path}: {e}")
            return None

    def to_sft_samples(self, audio_path: str, annotation: dict) -> list[dict]:
        """ASR 标注 → SFT 样本。"""
        transcript = annotation.get("transcript", "")
        if not transcript:
            return []
        return [{"messages": [
            {"role": "user",      "content": "<audio>请转录这段音频"},
            {"role": "assistant", "content": transcript},
        ], "audios": [audio_path]}]

    def process_for_asr(self, audio_dir: str) -> list[dict]:
        """批量处理音频目录，生成 ASR 训练数据。"""
        from src.utils.media import get_audio_files
        dataset = []
        audio_files = get_audio_files(audio_dir)

        for i, path in enumerate(audio_files):
            print(f"[ASR {i+1}/{len(audio_files)}] {Path(path).name}", end=" ... ")
            ann = self.annotate(path)
            if ann:
                dataset.extend(self.to_sft_samples(path, ann))
                print("✓")
            else:
                print("✗ 跳过")

        return dataset

    # ── TTS 模式 ──────────────────────────────────────────────────────────────

    def build_from_tts(self, text_qa_file: str, output_dir: str) -> list[dict]:
        """
        将已有文本 QA 数据转换为语音版：
        每条文本问题用不同音色合成音频，扩充音频数据量。
        这是业界常用的音频数据增强方式。
        """
        import json
        try:
            import edge_tts  # noqa
        except ImportError:
            raise ImportError("TTS 需要安装 edge-tts：pip install edge-tts")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        with open(text_qa_file, encoding="utf-8") as f:
            text_data = json.load(f)

        return asyncio.run(self._build_tts_async(text_data, output_dir))

    async def _build_tts_async(self, text_data: list[dict],
                               output_dir: str) -> list[dict]:
        """异步 TTS 合成（edge-tts 支持异步）。"""
        import edge_tts

        dataset = []
        for i, item in enumerate(text_data):
            # 取 user 的第一条消息作为语音内容
            user_text = next(
                (m["content"].replace("<image>", "").replace("<audio>", "").strip()
                 for m in item.get("messages", []) if m["role"] == "user"),
                None
            )
            if not user_text or len(user_text) > self.max_tts_text_len:
                continue

            answer = next(
                (m["content"] for m in item.get("messages", [])
                 if m["role"] == "assistant"),
                None
            )
            if not answer:
                continue

            voices = TTS_VOICES[:self.voices_per_sample]
            for j, voice in enumerate(voices):
                audio_path = f"{output_dir}/tts_{i:04d}_v{j}.mp3"
                try:
                    comm = edge_tts.Communicate(text=user_text, voice=voice)
                    await comm.save(audio_path)
                    dataset.append({"messages": [
                        {"role": "user",      "content": f"<audio>{user_text}"},
                        {"role": "assistant", "content": answer},
                    ], "audios": [audio_path]})
                except Exception as e:
                    print(f"[TTS ERROR] {e}")

        return dataset

    # ── 理解模式 ──────────────────────────────────────────────────────────────

    def annotate_understanding(self, transcript: str,
                               audio_path: str) -> list[dict]:
        """
        对转录文本进行情感/意图分析，生成音频理解样本。
        适合 IEMOCAP 等有情感标注的数据集，或已转录的音频。
        """
        prompt = self.understanding_prompt.format(transcript=transcript)
        raw = self.client.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=400,
        )
        if raw is None:
            return []

        ann = parse_json_response(raw)
        if not ann:
            return []

        samples = []

        # 情感/意图识别样本
        emotion_ans = (
            f"说话人的情绪是{ann.get('emotion', '未知')}，"
            f"语气{ann.get('formality', '未知')}，"
            f"意图是{ann.get('intent', '未知')}。"
        )
        samples.append({"messages": [
            {"role": "user",      "content": "<audio>请分析这段音频中说话人的情绪"},
            {"role": "assistant", "content": emotion_ans},
        ], "audios": [audio_path]})

        # 语音对话理解样本
        if ann.get("response"):
            samples.append({"messages": [
                {"role": "user",      "content": f"<audio>{transcript}"},
                {"role": "assistant", "content": ann["response"]},
            ], "audios": [audio_path]})

        return samples
