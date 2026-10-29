"""Optional FunASR speech-to-text backend for LifeMate V2.

Only the Audio -> Text stage changes. LLM, tool retrieval, controller, and
ToolEnv receive the same text interface. Never load multiple ASR checkpoints
as a fallback: the combined official parameter budget counts all models.
"""
from __future__ import annotations

from pathlib import Path
import wave

from lifemate_baseline import LocalModels


class FunASRLocalModels(LocalModels):
    """Same MLX LLM planner, experimental Fun-ASR-MLT-Nano-2512 transcriber."""

    def __init__(self, llm_dir: Path, asr_dir: Path, *, device: str = "mps",
                 adapter: Path | None = None, compact_plan: bool = False):
        asr_dir = Path(asr_dir)
        if not asr_dir.is_dir():
            raise ValueError(f"FunASR checkpoint path does not exist: {asr_dir}")
        # LocalModels loads MLX LLM; mlx_whisper import is lightweight and
        # DOES NOT load any Whisper weights until its transcribe() is called.
        super().__init__(llm_dir, asr_dir, adapter=adapter, compact_plan=compact_plan)
        from funasr import AutoModel

        self.funasr_model = AutoModel(
            model=str(asr_dir), device=device, hub="hf",
            trust_remote_code=False, disable_update=True
        )

    def transcribe(self, path: Path) -> str:
        path = Path(path)
        key = str(path)
        if key in self.audio_cache:
            return self.audio_cache[key]
        with wave.open(str(path), "rb") as reader:
            if (reader.getnchannels(), reader.getframerate(), reader.getsampwidth()) != (1, 16000, 2):
                raise ValueError("Expected WAV PCM16 mono 16kHz")
        result = self.funasr_model.generate(
            input=[str(path)], cache={}, batch_size=1,
            language="Vietnamese", itn=True
        )
        if not isinstance(result, list) or not result or not isinstance(result[0], dict):
            raise RuntimeError("Unexpected FunASR transcription result")
        text = result[0].get("text")
        if not isinstance(text, str):
            raise RuntimeError("FunASR result has no text")
        self.audio_cache[key] = text.strip()
        return self.audio_cache[key]
