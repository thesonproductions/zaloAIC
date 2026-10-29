"""Tests for optional experimental ASR-only FunASR backend."""
from __future__ import annotations
import sys, tempfile, unittest, wave
from pathlib import Path
from unittest.mock import Mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from lifemate_v2.asr_backend import FunASRLocalModels


class TestFunASRBackend(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/"clip.wav"
        with wave.open(str(self.path),"wb") as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000)
            w.writeframes(b"\x00\x00"*16000)
        self.obj=FunASRLocalModels.__new__(FunASRLocalModels)
        self.obj.funasr_model=Mock()
        self.obj.funasr_model.generate.return_value=[{"text":" Xin chào. "}]
        self.obj.audio_cache={}

    def test_transcription_text_and_cache(self):
        self.assertEqual(self.obj.transcribe(self.path),"Xin chào.")
        self.assertEqual(self.obj.transcribe(self.path),"Xin chào.")
        self.obj.funasr_model.generate.assert_called_once_with(
            input=[str(self.path)],cache={},batch_size=1,language="Vietnamese",itn=True)

    def test_invalid_wave_format_is_rejected(self):
        with wave.open(str(self.path),"wb") as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(8000)
            w.writeframes(b"\x00\x00"*8000)
        with self.assertRaises(ValueError):
            self.obj.transcribe(self.path)
        self.obj.funasr_model.generate.assert_not_called()

    def test_invalid_model_output_rejected(self):
        self.obj.funasr_model.generate.return_value=[]
        with self.assertRaisesRegex(RuntimeError,"Unexpected FunASR"):
            self.obj.transcribe(self.path)

    def test_missing_text_rejected(self):
        self.obj.funasr_model.generate.return_value=[{"unexpected":123}]
        with self.assertRaisesRegex(RuntimeError,"no text"):
            self.obj.transcribe(self.path)


if __name__=="__main__":unittest.main()
