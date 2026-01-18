"""Audio processing module"""
import os
import asyncio
from typing import Dict
import concurrent.futures

try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import eyed3
    EYED3_AVAILABLE = True
except ImportError:
    EYED3_AVAILABLE = False

executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


class AudioProcessor:
    _whisper_model = None
    _whisper_processor = None
    
    @classmethod
    def _load_whisper(cls):
        if cls._whisper_model is None and TRANSFORMERS_AVAILABLE:
            try:
                print("[AudioProcessor] Loading Whisper model...")
                cls._whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
                cls._whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
                print("[AudioProcessor] Model loaded successfully")
            except Exception as e:
                print(f"[AudioProcessor] Failed to load model: {e}")
    
    @classmethod
    async def process_audio_async(cls, audio_path: str) -> Dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, cls.process_audio, audio_path)
    
    @classmethod
    def process_audio(cls, audio_path: str) -> Dict:
        if not LIBROSA_AVAILABLE:
            return {"error": "librosa not available"}
        
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            duration = librosa.get_duration(y=y, sr=sr)
            
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            avg_spectral_centroid = np.mean(spectral_centroids)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            avg_zcr = np.mean(zcr)
            rms = librosa.feature.rms(y=y)[0]
            avg_rms = np.mean(rms)
            
            mood_from_audio = cls._infer_mood_from_audio(tempo, avg_spectral_centroid, avg_rms)
            metadata_info = cls._extract_audio_metadata(audio_path)
            
            transcription = "No transcription available"
            if TRANSFORMERS_AVAILABLE and duration < 60:
                cls._load_whisper()
                if cls._whisper_model is not None:
                    try:
                        if sr != 16000:
                            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                        
                        input_features = cls._whisper_processor(
                            y, sampling_rate=16000, return_tensors="pt"
                        ).input_features
                        
                        predicted_ids = cls._whisper_model.generate(input_features, max_length=100)
                        transcription = cls._whisper_processor.batch_decode(
                            predicted_ids, skip_special_tokens=True
                        )[0]
                    except Exception as e:
                        print(f"[AudioProcessor] Transcription failed: {e}")
            
            emotional_content = cls._analyze_emotional_content(transcription)
            
            return {
                "duration": float(duration),
                "sample_rate": int(sr),
                "tempo": float(tempo),
                "avg_spectral_centroid": float(avg_spectral_centroid),
                "avg_zero_crossing_rate": float(avg_zcr),
                "avg_energy": float(avg_rms),
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
        if EYED3_AVAILABLE and audio_path.lower().endswith('.mp3'):
            try:
                audiofile = eyed3.load(audio_path)
                if audiofile and audiofile.tag:
                    metadata['title'] = audiofile.tag.title or "Unknown"
                    metadata['artist'] = audiofile.tag.artist or "Unknown"
                    metadata['album'] = audiofile.tag.album or "Unknown"
            except:
                pass
        return metadata
    
    @staticmethod
    def _analyze_emotional_content(text: str) -> str:
        if not text or text == "No transcription available":
            return "neutral"
        
        text_lower = text.lower()
        positive_words = ['happy', 'joy', 'love', 'peace', 'hope', 'grateful', 'blessed']
        negative_words = ['sad', 'pain', 'hurt', 'dark', 'alone', 'lost', 'fear']
        calming_words = ['calm', 'relax', 'breathe', 'peace', 'gentle', 'soft']
        
        pos_score = sum(1 for word in positive_words if word in text_lower)
        neg_score = sum(1 for word in negative_words if word in text_lower)
        calm_score = sum(1 for word in calming_words if word in text_lower)
        
        if calm_score > 0:
            return "calming and peaceful"
        elif pos_score > neg_score:
            return "uplifting and positive"
        elif neg_score > pos_score:
            return "melancholic or heavy"
        return "neutral or balanced"
    
    @staticmethod
    def _infer_mood_from_audio(tempo: float, spectral_centroid: float, energy: float) -> str:
        if tempo > 120 and energy > 0.1:
            return "energetic/excited"
        elif tempo < 80 and energy < 0.05:
            return "calm/relaxed"
        elif energy < 0.03:
            return "quiet/peaceful"
        elif tempo > 140:
            return "fast-paced/anxious"
        return "moderate/balanced"
    
    @classmethod
    def generate_description(cls, audio_path: str, metadata: Dict = None) -> str:
        if metadata is None:
            metadata = cls.process_audio(audio_path)
        
        if "error" in metadata:
            return f"Audio file: {os.path.basename(audio_path)}"
        
        desc = ""
        if metadata.get('title') and metadata['title'] != "Unknown":
            desc += f"Song: '{metadata['title']}'"
            if metadata.get('artist'):
                desc += f" by {metadata['artist']}"
            desc += ". "
        
        if metadata.get('transcription') and metadata['transcription'] != "No transcription available":
            desc += f"Content: {metadata['transcription'][:200]}. "
        
        desc += f"Audio with {metadata['inferred_mood']} characteristics. "
        desc += f"Emotional content: {metadata.get('emotional_content', 'neutral')}. "
        desc += f"Duration: {metadata['duration']:.1f}s. "
        desc += f"Tempo: {metadata['tempo']:.0f} BPM"
        
        return desc