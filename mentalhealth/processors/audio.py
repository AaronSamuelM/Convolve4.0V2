"""Audio processing module (Whisper Tiny, low-memory)"""

import os
import asyncio
import concurrent.futures
import gc
from typing import Dict

import numpy as np
import librosa

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import eyed3
    EYED3_AVAILABLE = True
except ImportError:
    EYED3_AVAILABLE = False

import torch

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.set_grad_enabled(False)
torch.set_num_threads(1)

executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)


class AudioProcessor:
    _whisper_model = None

    @classmethod
    def _load_whisper(cls):
        if cls._whisper_model is None and WHISPER_AVAILABLE:
            print("[AudioProcessor] Loading Whisper-tiny (CPU)...")
            cls._whisper_model = whisper.load_model("tiny", device="cpu")
            print("[AudioProcessor] Whisper-tiny loaded")

    @classmethod
    async def process_audio_async(cls, audio_path: str) -> Dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, cls.process_audio, audio_path)

    @classmethod
    def process_audio(cls, audio_path: str) -> Dict:
        try:
            y, sr = librosa.load(audio_path, sr=16000, mono=True)
            duration = librosa.get_duration(y=y, sr=sr)

            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            spectral_centroid = np.mean(
                librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            )
            zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])
            rms = np.mean(librosa.feature.rms(y=y)[0])

            mood_from_audio = cls._infer_mood_from_audio(
                tempo, spectral_centroid, rms
            )

            metadata_info = cls._extract_audio_metadata(audio_path)

            transcription = "No transcription available"

            if WHISPER_AVAILABLE and duration <= 30:
                cls._load_whisper()
                try:
                    result = cls._whisper_model.transcribe(
                        audio_path,
                        fp16=False,
                        language="en"
                    )
                    transcription = result.get("text", "").strip()
                except Exception as e:
                    print(f"[AudioProcessor] Whisper failed: {e}")

            emotional_content = cls._analyze_emotional_content(transcription)

            del y
            gc.collect()

            return {
                "duration": float(duration),
                "sample_rate": 16000,
                "tempo": float(tempo),
                "avg_spectral_centroid": float(spectral_centroid),
                "avg_zero_crossing_rate": float(zcr),
                "avg_energy": float(rms),
                "inferred_mood": mood_from_audio,
                "transcription": transcription,
                "emotional_content": emotional_content,
                "title": metadata_info.get("title"),
                "artist": metadata_info.get("artist"),
                "file_size": os.path.getsize(audio_path)
            }

        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _extract_audio_metadata(audio_path: str) -> Dict:
        metadata = {}
        if EYED3_AVAILABLE and audio_path.lower().endswith(".mp3"):
            try:
                audiofile = eyed3.load(audio_path)
                if audiofile and audiofile.tag:
                    metadata["title"] = audiofile.tag.title or "Unknown"
                    metadata["artist"] = audiofile.tag.artist or "Unknown"
            except:
                pass
        return metadata

    @staticmethod
    def _analyze_emotional_content(text: str) -> str:
        if not text:
            return "neutral"

        text = text.lower()
        positive = ["happy", "joy", "love", "hope", "grateful"]
        negative = ["sad", "pain", "hurt", "alone", "fear"]
        calming = ["calm", "relax", "breathe", "peace"]

        pos = sum(w in text for w in positive)
        neg = sum(w in text for w in negative)
        calm = sum(w in text for w in calming)

        if calm:
            return "calming and peaceful"
        if pos > neg:
            return "uplifting and positive"
        if neg > pos:
            return "melancholic or heavy"
        return "neutral"

    @staticmethod
    def _infer_mood_from_audio(tempo: float, spectral_centroid: float, energy: float) -> str:
        if tempo > 120 and energy > 0.1:
            return "energetic/excited"
        if tempo < 80 and energy < 0.05:
            return "calm/relaxed"
        if energy < 0.03:
            return "quiet/peaceful"
        if tempo > 140:
            return "fast-paced/anxious"
        return "moderate/balanced"
