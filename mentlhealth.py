import os
import json
import requests
import uuid
import hashlib
import re
import base64
import mimetypes
import asyncio
import concurrent.futures
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict, field
from collections import Counter, defaultdict
from pathlib import Path
from functools import partial

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, 
    Filter, FieldCondition, MatchValue, Range,
    ScoredPoint
)
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

try:
    from PIL import Image
    import numpy as np
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    print("PIL not available. Install: pip install pillow")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("librosa not available. Install: pip install librosa")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("opencv not available. Install: pip install opencv-python")

try:
    import torch
    from transformers import (
        BlipProcessor, BlipForConditionalGeneration,
        Wav2Vec2Processor, Wav2Vec2ForCTC,
        WhisperProcessor, WhisperForConditionalGeneration
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("transformers not available. Install: pip install transformers torch")

try:
    import eyed3
    EYED3_AVAILABLE = True
except ImportError:
    EYED3_AVAILABLE = False
    print("eyed3 not available. Install: pip install eyed3")

# ============================================================================
# CONFIGURATION
# ============================================================================

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 
EMBEDDING_DIM = 384

RESOURCES_COLLECTION = "mental_health_resources"
MEMORY_COLLECTION = "user_memory"
USER_PROFILE_COLLECTION = "user_profiles"
MULTIMODAL_COLLECTION = "multimodal_data"

console = Console()

# Thread pool for async operations
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# ============================================================================
# MULTIMODAL DATA MODELS
# ============================================================================

@dataclass
class MultimodalData:
    """Multimodal data container"""
    id: str
    user_id: str
    data_type: str
    content: str
    metadata: Dict[str, Any]
    embedding_vector: Optional[List[float]] = None
    timestamp: str = None
    processed_data: Optional[Dict] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class MentalHealthResource:
    """Mental health resource"""
    id: str
    title: str
    content: str
    category: str
    resource_type: str
    tags: List[str]
    source: str
    difficulty: str
    duration_minutes: Optional[int] = None

@dataclass
class UserMemory:
    """User interaction memory with multimodal support"""
    id: str
    user_id: str
    session_id: str
    timestamp: str
    query: str
    response_summary: str
    mood: Optional[str]
    resources_used: List[str]
    multimodal_data_ids: List[str] = field(default_factory=list)
    feedback: Optional[str] = None
    notes: Optional[str] = None

@dataclass
class UserProfile:
    """User profile data"""
    user_id: str
    name: Optional[str] = None
    created_at: str = None
    last_active: str = None
    total_interactions: int = 0
    preferences: Dict = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.preferences is None:
            self.preferences = {}

# ============================================================================
# ENHANCED MULTIMODAL PROCESSORS WITH CONTENT ANALYSIS
# ============================================================================

class ImageProcessor:
    """Process and extract features from images with AI vision"""
    
    _model = None
    _processor = None
    
    @classmethod
    def _load_model(cls):
        """Lazy load image captioning model"""
        if cls._model is None and TRANSFORMERS_AVAILABLE:
            try:
                console.print("[cyan]Loading image analysis model...[/cyan]")
                cls._processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                cls._model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                console.print("[green]✓ Image model loaded[/green]")
            except Exception as e:
                console.print(f"[yellow]Could not load image model: {e}[/yellow]")
    
    @classmethod
    async def process_image_async(cls, image_path: str) -> Dict:
        """Async wrapper for image processing"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, cls.process_image, image_path)
    
    @classmethod
    def process_image(cls, image_path: str) -> Dict:
        """Extract features and content from image"""
        if not PILLOW_AVAILABLE:
            return {"error": "PIL not available"}
        
        try:
            img = Image.open(image_path)
            
            # Basic analysis
            width, height = img.size
            mode = img.mode
            
            if mode != 'RGB':
                img = img.convert('RGB')
            
            img_array = np.array(img)
            avg_color = img_array.mean(axis=(0, 1))
            brightness = avg_color.mean()
            red_blue_ratio = avg_color[0] / (avg_color[2] + 1)
            
            mood_from_image = cls._infer_mood_from_colors(brightness, red_blue_ratio, avg_color)
            
            # AI-based image captioning
            caption = "No caption available"
            objects_detected = []
            
            if TRANSFORMERS_AVAILABLE:
                cls._load_model()
                if cls._model is not None:
                    try:
                        inputs = cls._processor(img, return_tensors="pt")
                        out = cls._model.generate(**inputs, max_length=50)
                        caption = cls._processor.decode(out[0], skip_special_tokens=True)
                        
                        # Extract key objects/concepts
                        objects_detected = cls._extract_objects_from_caption(caption)
                    except Exception as e:
                        console.print(f"[yellow]Caption generation failed: {e}[/yellow]")
            
            # Emotional context from image
            emotional_tone = cls._analyze_emotional_context(brightness, red_blue_ratio, caption)
            
            return {
                "width": width,
                "height": height,
                "mode": mode,
                "avg_brightness": float(brightness),
                "avg_color": avg_color.tolist(),
                "red_blue_ratio": float(red_blue_ratio),
                "inferred_mood": mood_from_image,
                "caption": caption,
                "objects_detected": objects_detected,
                "emotional_tone": emotional_tone,
                "file_size": os.path.getsize(image_path)
            }
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def _extract_objects_from_caption(caption: str) -> List[str]:
        """Extract key objects from caption"""
        # Simple extraction - could be enhanced with NLP
        words = caption.lower().split()
        # Filter out common words
        stop_words = {'a', 'an', 'the', 'is', 'are', 'with', 'in', 'on', 'at', 'of'}
        objects = [w for w in words if w not in stop_words and len(w) > 3]
        return objects[:5]
    
    @staticmethod
    def _analyze_emotional_context(brightness: float, ratio: float, caption: str) -> str:
        """Analyze emotional context combining visual and semantic info"""
        caption_lower = caption.lower()
        
        positive_words = ['happy', 'smile', 'joy', 'bright', 'colorful', 'nature', 'peaceful']
        negative_words = ['dark', 'sad', 'alone', 'empty', 'gray', 'gloomy']
        
        positive_score = sum(1 for word in positive_words if word in caption_lower)
        negative_score = sum(1 for word in negative_words if word in caption_lower)
        
        if positive_score > negative_score and brightness > 150:
            return "uplifting and positive"
        elif negative_score > positive_score or brightness < 80:
            return "somber or contemplative"
        else:
            return "neutral and balanced"
    
    @staticmethod
    def _infer_mood_from_colors(brightness: float, rg_ratio: float, avg_color) -> str:
        """Infer mood from image colors"""
        if brightness < 80:
            return "dark/melancholic"
        elif brightness > 180:
            return "bright/energetic"
        elif rg_ratio > 1.2:
            return "warm/calm"
        elif rg_ratio < 0.8:
            return "cool/relaxed"
        else:
            return "balanced/neutral"
    
    @classmethod
    def generate_description(cls, image_path: str, metadata: Dict = None) -> str:
        """Generate comprehensive text description for embedding"""
        if metadata is None:
            metadata = cls.process_image(image_path)
        
        if "error" in metadata:
            return f"Image file: {os.path.basename(image_path)}"
        
        desc = f"Image showing: {metadata.get('caption', 'visual content')}. "
        desc += f"Emotional tone: {metadata.get('emotional_tone', 'neutral')}. "
        desc += f"Visual mood: {metadata['inferred_mood']}. "
        
        if metadata.get('objects_detected'):
            desc += f"Contains: {', '.join(metadata['objects_detected'])}. "
        
        desc += f"Brightness level: {metadata['avg_brightness']:.0f}. "
        desc += f"Dimensions: {metadata['width']}x{metadata['height']}"
        
        return desc


class AudioProcessor:
    """Process and extract features from audio with speech recognition"""
    
    _whisper_model = None
    _whisper_processor = None
    
    @classmethod
    def _load_whisper(cls):
        """Lazy load Whisper model for transcription"""
        if cls._whisper_model is None and TRANSFORMERS_AVAILABLE:
            try:
                console.print("[cyan]Loading audio transcription model...[/cyan]")
                cls._whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
                cls._whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
                console.print("[green]✓ Audio model loaded[/green]")
            except Exception as e:
                console.print(f"[yellow]Could not load whisper: {e}[/yellow]")
    
    @classmethod
    async def process_audio_async(cls, audio_path: str) -> Dict:
        """Async wrapper for audio processing"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, cls.process_audio, audio_path)
    
    @classmethod
    def process_audio(cls, audio_path: str) -> Dict:
        """Extract features and transcribe audio"""
        if not LIBROSA_AVAILABLE:
            return {"error": "librosa not available"}
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)  # 16kHz for Whisper
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Extract acoustic features
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            avg_spectral_centroid = np.mean(spectral_centroids)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            avg_zcr = np.mean(zcr)
            rms = librosa.feature.rms(y=y)[0]
            avg_rms = np.mean(rms)
            
            mood_from_audio = cls._infer_mood_from_audio(tempo, avg_spectral_centroid, avg_rms)
            
            # Extract metadata (title, artist from MP3 tags)
            metadata_info = cls._extract_audio_metadata(audio_path)
            
            # Transcribe speech if available
            transcription = "No transcription available"
            if TRANSFORMERS_AVAILABLE and duration < 60:  # Only transcribe short clips
                cls._load_whisper()
                if cls._whisper_model is not None:
                    try:
                        # Resample if needed
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
                        console.print(f"[yellow]Transcription failed: {e}[/yellow]")
            
            # Analyze emotional content from transcription
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
        """Extract metadata from audio file"""
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
        """Analyze emotional content from transcription"""
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
        else:
            return "neutral or balanced"
    
    @staticmethod
    def _infer_mood_from_audio(tempo: float, spectral_centroid: float, energy: float) -> str:
        """Infer mood from audio features"""
        if tempo > 120 and energy > 0.1:
            return "energetic/excited"
        elif tempo < 80 and energy < 0.05:
            return "calm/relaxed"
        elif energy < 0.03:
            return "quiet/peaceful"
        elif tempo > 140:
            return "fast-paced/anxious"
        else:
            return "moderate/balanced"
    
    @classmethod
    def generate_description(cls, audio_path: str, metadata: Dict = None) -> str:
        """Generate comprehensive text description for embedding"""
        if metadata is None:
            metadata = cls.process_audio(audio_path)
        
        if "error" in metadata:
            return f"Audio file: {os.path.basename(audio_path)}"
        
        desc = ""
        
        # Add title/artist if available
        if metadata.get('title') and metadata['title'] != "Unknown":
            desc += f"Song: '{metadata['title']}'"
            if metadata.get('artist'):
                desc += f" by {metadata['artist']}"
            desc += ". "
        
        # Add transcription
        if metadata.get('transcription') and metadata['transcription'] != "No transcription available":
            desc += f"Content: {metadata['transcription'][:200]}. "
        
        desc += f"Audio with {metadata['inferred_mood']} characteristics. "
        desc += f"Emotional content: {metadata.get('emotional_content', 'neutral')}. "
        desc += f"Duration: {metadata['duration']:.1f}s. "
        desc += f"Tempo: {metadata['tempo']:.0f} BPM"
        
        return desc


class VideoProcessor:
    """Process and extract features from video"""
    
    @staticmethod
    async def process_video_async(video_path: str, sample_frames: int = 5) -> Dict:
        """Async wrapper for video processing"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor, 
            partial(VideoProcessor.process_video, sample_frames=sample_frames),
            video_path
        )
    
    @staticmethod
    def process_video(video_path: str, sample_frames: int = 5) -> Dict:
        """Extract features from video"""
        if not CV2_AVAILABLE:
            return {"error": "opencv not available"}
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            frame_interval = max(1, frame_count // sample_frames)
            brightness_values = []
            frame_descriptions = []
            
            # Process sample frames with image processor
            for i in range(0, min(frame_count, sample_frames * frame_interval), frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    brightness_values.append(np.mean(gray))
                    
                    # Save frame temporarily for analysis
                    if PILLOW_AVAILABLE and TRANSFORMERS_AVAILABLE and len(frame_descriptions) < 3:
                        temp_frame_path = f"/tmp/frame_{i}.jpg"
                        cv2.imwrite(temp_frame_path, frame)
                        frame_data = ImageProcessor.process_image(temp_frame_path)
                        if 'caption' in frame_data:
                            frame_descriptions.append(frame_data['caption'])
                        os.remove(temp_frame_path)
            
            cap.release()
            
            avg_brightness = np.mean(brightness_values) if brightness_values else 0
            mood_from_video = VideoProcessor._infer_mood_from_video(duration, avg_brightness, fps)
            
            # Combine frame descriptions
            video_summary = ". ".join(frame_descriptions) if frame_descriptions else "Video content"
            
            return {
                "duration": float(duration),
                "fps": float(fps),
                "frame_count": frame_count,
                "resolution": f"{width}x{height}",
                "avg_brightness": float(avg_brightness),
                "inferred_mood": mood_from_video,
                "video_summary": video_summary,
                "sample_frame_count": len(frame_descriptions),
                "file_size": os.path.getsize(video_path)
            }
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def _infer_mood_from_video(duration: float, brightness: float, fps: float) -> str:
        """Infer mood from video characteristics"""
        if brightness < 80:
            return "dark/serious"
        elif brightness > 180:
            return "bright/uplifting"
        elif duration < 30:
            return "brief/concise"
        else:
            return "moderate/informative"
    
    @staticmethod
    def generate_description(video_path: str, metadata: Dict = None) -> str:
        """Generate text description for embedding"""
        if metadata is None:
            metadata = VideoProcessor.process_video(video_path)
        
        if "error" in metadata:
            return f"Video file: {os.path.basename(video_path)}"
        
        desc = f"Video showing: {metadata.get('video_summary', 'visual content')}. "
        desc += f"Presentation: {metadata['inferred_mood']}. "
        desc += f"Duration: {metadata['duration']:.1f}s. "
        desc += f"Resolution: {metadata['resolution']}"
        
        return desc


class CodeProcessor:
    """Process and analyze code snippets"""
    
    @staticmethod
    def process_code(code: str, language: str = "python") -> Dict:
        """Analyze code for mental health context"""
        lines = code.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        comment_chars = {'python': '#', 'javascript': '//', 'java': '//', 'c++': '//'}
        comment_char = comment_chars.get(language, '#')
        comments = [l for l in non_empty_lines if l.strip().startswith(comment_char)]
        
        # Extract actual content from comments
        comment_text = " ".join([
            l.strip().lstrip(comment_char).strip() 
            for l in comments
        ])
        
        # Analyze function/class definitions
        functions = len([l for l in non_empty_lines if 'def ' in l or 'function ' in l])
        classes = len([l for l in non_empty_lines if 'class ' in l])
        
        return {
            "language": language,
            "total_lines": len(lines),
            "code_lines": len(non_empty_lines),
            "comment_lines": len(comments),
            "comment_text": comment_text,
            "functions": functions,
            "classes": classes,
            "complexity": "simple" if len(non_empty_lines) < 20 else "moderate" if len(non_empty_lines) < 50 else "complex"
        }
    
    @staticmethod
    def generate_description(code: str, language: str = "python", metadata: Dict = None) -> str:
        """Generate text description for embedding"""
        if metadata is None:
            metadata = CodeProcessor.process_code(code, language)
        
        desc = f"{metadata['language']} code with {metadata['complexity']} complexity. "
        desc += f"{metadata['code_lines']} lines of code, {metadata['functions']} functions. "
        
        if metadata.get('comment_text'):
            desc += f"Comments: {metadata['comment_text'][:200]}"
        
        return desc


class StructuredDataProcessor:
    """Process structured data"""
    
    @staticmethod
    def process_structured(data: Union[Dict, str], data_format: str = "json") -> Dict:
        """Process structured data"""
        if data_format == "json":
            if isinstance(data, str):
                data = json.loads(data)
            
            # Extract summary
            summary = StructuredDataProcessor._summarize_dict(data)
            
            return {
                "format": "json",
                "keys": list(data.keys()) if isinstance(data, dict) else [],
                "depth": StructuredDataProcessor._get_depth(data),
                "size": len(json.dumps(data)),
                "summary": summary
            }
        
        return {"format": data_format, "size": len(str(data))}
    
    @staticmethod
    def _summarize_dict(d: Dict, max_items: int = 5) -> str:
        """Create human-readable summary of dict"""
        if not isinstance(d, dict):
            return str(d)[:100]
        
        summary_parts = []
        for key, value in list(d.items())[:max_items]:
            if isinstance(value, (dict, list)):
                summary_parts.append(f"{key}: {type(value).__name__}")
            else:
                summary_parts.append(f"{key}: {str(value)[:50]}")
        
        return ", ".join(summary_parts)
    
    @staticmethod
    def _get_depth(d: Any, level: int = 0) -> int:
        """Get nesting depth"""
        if not isinstance(d, (dict, list)):
            return level
        if isinstance(d, dict):
            return max([StructuredDataProcessor._get_depth(v, level + 1) for v in d.values()], default=level)
        return max([StructuredDataProcessor._get_depth(item, level + 1) for item in d], default=level)
    
    @staticmethod
    def generate_description(data: Union[Dict, str], data_format: str = "json", metadata: Dict = None) -> str:
        """Generate text description for embedding"""
        if metadata is None:
            metadata = StructuredDataProcessor.process_structured(data, data_format)
        
        desc = f"Structured {metadata['format']} data. "
        
        if metadata.get('summary'):
            desc += f"Contains: {metadata['summary']}. "
        
        if 'keys' in metadata and metadata['keys']:
            desc += f"Fields: {', '.join(metadata['keys'][:5])}"
        
        return desc


# ============================================================================
# MOOD DETECTION ENGINE
# ============================================================================

class MoodDetector:
    """Fast, rule-based mood detection from text"""
    
    MOOD_PATTERNS = {
        "anxious": [
            r'\b(anxious|anxiety|worried|worry|nervous|panic|fear|scared|afraid|uneasy|restless|tense)\b',
            r'\b(can\'t (stop|calm)|racing thoughts|heart racing|on edge)\b',
            r'\b(what if|overthinking)\b'
        ],
        "stressed": [
            r'\b(stressed|stress|overwhelmed|pressure|burden|swamped|exhausted mentally)\b',
            r'\b(too much|can\'t cope|breaking point|burned out|burnout)\b',
            r'\b(deadline|workload|hectic)\b'
        ],
        "depressed": [
            r'\b(depressed|depression|hopeless|empty|numb|worthless|pointless)\b',
            r'\b(no energy|no motivation|giving up|can\'t enjoy)\b',
            r'\b(why bother|what\'s the point)\b'
        ],
        "sad": [
            r'\b(sad|sadness|down|blue|miserable|unhappy|crying|tears)\b',
            r'\b(feel bad|feel awful|feel terrible)\b'
        ],
        "lonely": [
            r'\b(lonely|alone|isolated|disconnected|no one|nobody)\b',
            r'\b(feel abandoned|no friends|by myself)\b'
        ],
        "tired": [
            r'\b(tired|exhausted|fatigued|drained|weary|sleepy|no sleep|insomnia)\b',
            r'\b(can\'t sleep|trouble sleeping|wake up)\b'
        ],
        "angry": [
            r'\b(angry|anger|mad|furious|frustrated|irritated|annoyed|rage)\b',
            r'\b(pissed off|fed up)\b'
        ],
        "calm": [
            r'\b(calm|peaceful|relaxed|serene|tranquil|content)\b',
            r'\b(feeling (good|better|okay|fine))\b'
        ],
        "happy": [
            r'\b(happy|joy|joyful|excited|great|wonderful|fantastic|amazing)\b',
            r'\b(feeling good|doing well)\b'
        ],
        "confused": [
            r'\b(confused|lost|uncertain|don\'t know|unsure|mixed feelings)\b',
            r'\b(not sure|unclear)\b'
        ]
    }
    
    @classmethod
    def detect_mood(cls, text: str) -> Optional[str]:
        """Fast mood detection using regex patterns"""
        if not text:
            return None
        
        text_lower = text.lower()
        mood_scores = defaultdict(int)
        
        for mood, patterns in cls.MOOD_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                mood_scores[mood] += len(matches)
        
        if mood_scores:
            return max(mood_scores.items(), key=lambda x: x[1])[0]
        
        return None


# ============================================================================
# ASYNC MULTIMODAL PROCESSOR
# ============================================================================

class AsyncMultimodalProcessor:
    """Async processor for multimodal data to prevent blocking"""
    
    @staticmethod
    async def process_file(file_path: str, user_id: str) -> Tuple[Optional[str], Dict]:
        """Process file asynchronously and return ID and metadata"""
        
        if not os.path.exists(file_path):
            return None, {"error": "File not found"}
        
        mime_type, _ = mimetypes.guess_type(file_path)
        data_type = "unknown"
        processed_data = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]Processing {os.path.basename(file_path)}...", total=None)
            
            try:
                if mime_type:
                    if mime_type.startswith("image/"):
                        data_type = "image"
                        processed_data = await ImageProcessor.process_image_async(file_path)
                    elif mime_type.startswith("audio/"):
                        data_type = "audio"
                        processed_data = await AudioProcessor.process_audio_async(file_path)
                    elif mime_type.startswith("video/"):
                        data_type = "video"
                        processed_data = await VideoProcessor.process_video_async(file_path)
                
                # Check for code files
                code_extensions = {'.py': 'python', '.js': 'javascript', '.java': 'java', '.cpp': 'c++'}
                ext = Path(file_path).suffix
                if ext in code_extensions:
                    data_type = "code"
                    with open(file_path, 'r') as f:
                        code_content = f.read()
                    processed_data = CodeProcessor.process_code(code_content, code_extensions[ext])
                
                progress.update(task, completed=True)
                
            except Exception as e:
                progress.update(task, completed=True)
                return None, {"error": str(e)}
        
        mm_id = str(uuid.uuid4())
        
        return mm_id, {
            "id": mm_id,
            "user_id": user_id,
            "data_type": data_type,
            "content": file_path,
            "metadata": {
                "filename": os.path.basename(file_path),
                "mime_type": mime_type
            },
            "processed_data": processed_data
        }


# ============================================================================
# ENHANCED QDRANT MANAGER WITH MULTIMODAL SUPPORT
# ============================================================================

class QdrantManager:
    """Manages Qdrant collections with multimodal support"""
    
    def __init__(self, host: str = QDRANT_HOST, port: int = QDRANT_PORT):
        self.client = QdrantClient(host=host, port=port)
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)
        console.print(f"[green]✓[/green] Connected to Qdrant at {host}:{port}")
        console.print(f"[green]✓[/green] Loaded embedding model: {EMBEDDING_MODEL}")
    
    def setup_collections(self):
        """Create all collections"""
        collections = [
            RESOURCES_COLLECTION,
            MEMORY_COLLECTION,
            USER_PROFILE_COLLECTION,
            MULTIMODAL_COLLECTION
        ]
        
        for collection in collections:
            if not self.client.collection_exists(collection):
                self.client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
                )
                console.print(f"[green]✓[/green] Created collection: {collection}")
            else:
                console.print(f"[yellow]![/yellow] Collection exists: {collection}")
    
    def add_multimodal_data(self, multimodal: MultimodalData):
        """Add multimodal data to Qdrant"""
        # Generate embedding based on description
        if multimodal.data_type == "text":
            text_for_embedding = multimodal.content
        elif multimodal.data_type == "image":
            text_for_embedding = ImageProcessor.generate_description(
                multimodal.content, 
                multimodal.processed_data
            )
        elif multimodal.data_type == "audio":
            text_for_embedding = AudioProcessor.generate_description(
                multimodal.content,
                multimodal.processed_data
            )
        elif multimodal.data_type == "video":
            text_for_embedding = VideoProcessor.generate_description(
                multimodal.content,
                multimodal.processed_data
            )
        elif multimodal.data_type == "code":
            text_for_embedding = CodeProcessor.generate_description(
                multimodal.content,
                multimodal.metadata.get("language", "python"),
                multimodal.processed_data
            )
        elif multimodal.data_type == "structured":
            text_for_embedding = StructuredDataProcessor.generate_description(
                multimodal.content,
                multimodal.metadata.get("format", "json"),
                multimodal.processed_data
            )
        else:
            text_for_embedding = str(multimodal.content)
        
        vector = self.encoder.encode(text_for_embedding).tolist()
        
        point = PointStruct(
            id=multimodal.id,
            vector=vector,
            payload={
                "user_id": multimodal.user_id,
                "data_type": multimodal.data_type,
                "content": multimodal.content if multimodal.data_type in ["text", "code"] else multimodal.content[:500],
                "metadata": multimodal.metadata,
                "timestamp": multimodal.timestamp,
                "processed_data": multimodal.processed_data
            }
        )
        
        self.client.upsert(collection_name=MULTIMODAL_COLLECTION, points=[point])
        console.print(f"[green]✓[/green] Added {multimodal.data_type} data to collection")
    
    def search_multimodal(
        self,
        query: str,
        user_id: str,
        data_types: Optional[List[str]] = None,
        limit: int = 5
    ) -> List[Tuple[MultimodalData, float]]:
        """Search multimodal data"""
        query_vector = self.encoder.encode(query).tolist()
        
        must_conditions = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        filter_conditions = Filter(must=must_conditions)
        
        results = self.client.query_points(
            collection_name=MULTIMODAL_COLLECTION,
            query=query_vector,
            query_filter=filter_conditions,
            limit=limit * 2  # Get more to filter by type
        )
        
        multimodal_with_scores = []
        for hit in results.points:
            if data_types and hit.payload["data_type"] not in data_types:
                continue
            
            multimodal = MultimodalData(
                id=hit.id,
                user_id=hit.payload["user_id"],
                data_type=hit.payload["data_type"],
                content=hit.payload["content"],
                metadata=hit.payload["metadata"],
                timestamp=hit.payload["timestamp"],
                processed_data=hit.payload.get("processed_data")
            )
            multimodal_with_scores.append((multimodal, hit.score))
            
            if len(multimodal_with_scores) >= limit:
                break
        
        return multimodal_with_scores
    
    def populate_resources(self, resources: List[MentalHealthResource]):
        """Populate resources collection"""
        points = []
        for resource in resources:
            text = f"{resource.title}. {resource.content}"
            vector = self.encoder.encode(text).tolist()
            
            point = PointStruct(
                id=resource.id,
                vector=vector,
                payload={
                    "title": resource.title,
                    "content": resource.content,
                    "category": resource.category,
                    "resource_type": resource.resource_type,
                    "tags": resource.tags,
                    "source": resource.source,
                    "difficulty": resource.difficulty,
                    "duration_minutes": resource.duration_minutes
                }
            )
            points.append(point)
        
        self.client.upsert(collection_name=RESOURCES_COLLECTION, points=points)
        console.print(f"[green]✓[/green] Uploaded {len(points)} resources to Qdrant")
    
    def search_resources(
        self, 
        query: str, 
        user_id: str,
        category: Optional[str] = None,
        limit: int = 5
    ) -> List[Tuple[MentalHealthResource, float]]:
        """Search resources semantically"""
        query_vector = self.encoder.encode(query).tolist()
        
        filter_conditions = None
        if category:
            filter_conditions = Filter(
                must=[FieldCondition(key="category", match=MatchValue(value=category))]
            )
        
        results = self.client.query_points(
            collection_name=RESOURCES_COLLECTION,
            query=query_vector,
            query_filter=filter_conditions,
            limit=limit
        )
        
        if not results:
            return []
        
        resources_with_scores = []
        for hit in results.points:
            resource = MentalHealthResource(
                id=hit.id,
                title=hit.payload["title"],
                content=hit.payload["content"],
                category=hit.payload["category"],
                resource_type=hit.payload["resource_type"],
                tags=hit.payload["tags"],
                source=hit.payload["source"],
                difficulty=hit.payload["difficulty"],
                duration_minutes=hit.payload.get("duration_minutes")
            )
            resources_with_scores.append((resource, hit.score))
        
        return resources_with_scores
    
    def add_memory(self, memory: UserMemory):
        """Add or update user memory"""
        vector = self.encoder.encode(memory.query).tolist()
        
        point = PointStruct(
            id=memory.id,
            vector=vector,
            payload={
                "user_id": memory.user_id,
                "session_id": memory.session_id,
                "timestamp": memory.timestamp,
                "query": memory.query,
                "response_summary": memory.response_summary,
                "mood": memory.mood,
                "resources_used": memory.resources_used,
                "multimodal_data_ids": memory.multimodal_data_ids,
                "feedback": memory.feedback,
                "notes": memory.notes
            }
        )
        
        self.client.upsert(collection_name=MEMORY_COLLECTION, points=[point])
    
    def get_user_history(self, user_id: str, limit: int = 50) -> List[UserMemory]:
        """Get user's interaction history"""
        results = self.client.scroll(
            collection_name=MEMORY_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
            limit=limit
        )
        
        memories = []
        for point in results[0]:
            memory = UserMemory(
                id=point.id,
                user_id=point.payload["user_id"],
                session_id=point.payload["session_id"],
                timestamp=point.payload["timestamp"],
                query=point.payload["query"],
                response_summary=point.payload.get("response_summary", ""),
                mood=point.payload.get("mood"),
                resources_used=point.payload["resources_used"],
                multimodal_data_ids=point.payload.get("multimodal_data_ids", []),
                feedback=point.payload.get("feedback"),
                notes=point.payload.get("notes")
            )
            memories.append(memory)
        
        return sorted(memories, key=lambda m: m.timestamp, reverse=True)
    
    def search_user_memories(self, user_id: str, query: str, limit: int = 5) -> List[UserMemory]:
        """Search user's past memories semantically"""
        query_vector = self.encoder.encode(query).tolist()
        
        results = self.client.query_points(
            collection_name=MEMORY_COLLECTION,
            query=query_vector,
            query_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
            limit=limit
        )
        
        memories = []
        for hit in results.points:
            memory = UserMemory(
                id=hit.id,
                user_id=hit.payload["user_id"],
                session_id=hit.payload["session_id"],
                timestamp=hit.payload["timestamp"],
                query=hit.payload["query"],
                response_summary=hit.payload.get("response_summary", ""),
                mood=hit.payload.get("mood"),
                resources_used=hit.payload["resources_used"],
                multimodal_data_ids=hit.payload.get("multimodal_data_ids", []),
                feedback=hit.payload.get("feedback"),
                notes=hit.payload.get("notes")
            )
            memories.append(memory)
        
        return memories
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile"""
        try:
            results = self.client.scroll(
                collection_name=USER_PROFILE_COLLECTION,
                scroll_filter=Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                ),
                limit=1
            )
            
            if results[0]:
                point = results[0][0]
                return UserProfile(
                    user_id=point.payload["user_id"],
                    name=point.payload.get("name"),
                    created_at=point.payload.get("created_at"),
                    last_active=point.payload.get("last_active"),
                    total_interactions=point.payload.get("total_interactions", 0),
                    preferences=point.payload.get("preferences", {})
                )
        except Exception as e:
            console.print(f"[yellow]Could not load user profile: {e}[/yellow]")
        
        return None
    
    def upsert_user_profile(self, profile: UserProfile):
        """Create or update user profile"""
        user_id_hash = hashlib.md5(profile.user_id.encode()).hexdigest()
        point_id = str(uuid.UUID(user_id_hash))
        
        vector = self.encoder.encode(f"User {profile.user_id} {profile.name or ''}").tolist()
        
        point = PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "user_id": profile.user_id,
                "name": profile.name,
                "created_at": profile.created_at,
                "last_active": profile.last_active,
                "total_interactions": profile.total_interactions,
                "preferences": profile.preferences
            }
        )
        
        self.client.upsert(collection_name=USER_PROFILE_COLLECTION, points=[point])


# ============================================================================
# LLM RESPONDER (Enhanced for Multimodal)
# ============================================================================

class LLMResponder:
    """Generate responses with multimodal context"""

    def __init__(self, model="llama3"):
        self.model = model
        self.ollama_url = "http://localhost:11434/api/generate"
        console.print(f"[green]✓[/green] LLM initialized: {model}")

    def generate_response(
        self,
        query: str,
        retrieved_resources: List[Tuple[MentalHealthResource, float]],
        user_profile: Optional[UserProfile],
        past_memories: List[UserMemory],
        multimodal_context: List[Tuple[MultimodalData, float]],
        analytics: Dict
    ) -> Tuple[str, str]:
        """Generate response with multimodal context"""
        
        context_parts = []
        
        # User context
        if user_profile and user_profile.name:
            context_parts.append(f"=== USER PROFILE ===")
            context_parts.append(f"Name: {user_profile.name}")
        
        # Multimodal context (ENHANCED)
        if multimodal_context:
            context_parts.append("\n=== MULTIMODAL CONTEXT ===")
            for mm_data, score in multimodal_context:
                context_parts.append(f"\n[{mm_data.data_type.upper()}] Relevance: {score:.3f}")
                
                if mm_data.processed_data:
                    pd = mm_data.processed_data
                    
                    if mm_data.data_type == "image":
                        context_parts.append(f"Caption: {pd.get('caption', 'N/A')}")
                        context_parts.append(f"Emotional tone: {pd.get('emotional_tone', 'N/A')}")
                        context_parts.append(f"Objects: {', '.join(pd.get('objects_detected', []))}")
                    
                    elif mm_data.data_type == "audio":
                        if pd.get('title'):
                            context_parts.append(f"Title: {pd['title']}")
                        if pd.get('artist'):
                            context_parts.append(f"Artist: {pd['artist']}")
                        if pd.get('transcription') and pd['transcription'] != "No transcription available":
                            context_parts.append(f"Transcription: {pd['transcription'][:300]}")
                        context_parts.append(f"Emotional content: {pd.get('emotional_content', 'N/A')}")
                        context_parts.append(f"Mood: {pd.get('inferred_mood', 'N/A')}")
                    
                    elif mm_data.data_type == "video":
                        context_parts.append(f"Summary: {pd.get('video_summary', 'N/A')}")
                        context_parts.append(f"Mood: {pd.get('inferred_mood', 'N/A')}")
                    
                    elif mm_data.data_type == "code":
                        context_parts.append(f"Language: {pd.get('language', 'N/A')}")
                        context_parts.append(f"Complexity: {pd.get('complexity', 'N/A')}")
                        if pd.get('comment_text'):
                            context_parts.append(f"Comments: {pd['comment_text'][:200]}")
        
        # Past memories
        if past_memories:
            context_parts.append("\n=== RELEVANT PAST CONVERSATIONS ===")
            for mem in past_memories[:3]:
                context_parts.append(f"Previous: {mem.query}")
                context_parts.append(f"Solution: {mem.response_summary}")
                if mem.mood:
                    context_parts.append(f"Mood: {mem.mood}")
        
        # Current resources
        context_parts.append("\n=== RELEVANT RESOURCES ===")
        for resource, score in retrieved_resources:
            context_parts.append(f"\nTitle: {resource.title}")
            context_parts.append(f"Content: {resource.content}")
            context_parts.append(f"Source: {resource.source}")
        
        context = "\n".join(context_parts)
        
        system_prompt = f"""You are a assistant.

CRITICAL INSTRUCTIONS:
1. Provide evidence-based, supportive responses
2. ALWAYS cite specific resources by title
3. Consider only most relevant multimodal context (images, audio, video, code) and dont mention word like "from the audio clip" or the media name 
4. When images/audio/video are provided, reference their content in a human readable way
5. Be warm and personalized only whe required
6. Reference past conversations when relevant
7. Keep responses actionable and concise (minimum 20 words and maximum 200 words)

"""

        user_prompt = f"""User query: {query}

{context}

Provide a supportive response that considers only relevant to the question context and multimodal data. name:{user_profile.name}"""

        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": system_prompt + "\n\n" + user_prompt,
                    "stream": False
                },
                timeout=90
            )
            
            full_response = response.json().get("response", "").strip()
            summary = full_response[:200] + "..." if len(full_response) > 200 else full_response
            
            return full_response, summary
            
        except Exception as e:
            console.print(f"[yellow]LLM error: {e}[/yellow]")
            return self._template_response(retrieved_resources, multimodal_context)
    
    def _template_response(
        self, 
        resources: List[Tuple[MentalHealthResource, float]],
        multimodal: List[Tuple[MultimodalData, float]]
    ) -> Tuple[str, str]:
        """Fallback template with multimodal awareness"""
        response = "Here are some evidence-based strategies:\n\n"
        
        if multimodal:
            response += "Based on the content you shared:\n"
            for mm, _ in multimodal[:2]:
                if mm.processed_data:
                    if mm.data_type == "image":
                        response += f"- Image showing: {mm.processed_data.get('caption', 'visual content')}\n"
                    elif mm.data_type == "audio":
                        response += f"- Audio: {mm.processed_data.get('emotional_content', 'content')}\n"
            response += "\n"
        
        for i, (r, _) in enumerate(resources[:3], 1):
            response += f"{i}. **{r.title}**: {r.content}\n"
        
        summary = f"Recommended {len(resources)} resources"
        return response, summary


# ============================================================================
# ANALYTICS
# ============================================================================

class UserAnalytics:
    """Analyze user patterns"""
    
    @staticmethod
    def analyze_patterns(memories: List[UserMemory]) -> Dict:
        """Analyze user patterns"""
        if not memories:
            return {}
        
        moods = [m.mood for m in memories if m.mood]
        mood_counter = Counter(moods)
        
        timestamps = [datetime.fromisoformat(m.timestamp) for m in memories]
        hours = [ts.hour for ts in timestamps]
        days = [ts.strftime('%A') for ts in timestamps]
        
        # Count multimodal usage
        multimodal_count = sum(1 for m in memories if m.multimodal_data_ids)
        
        return {
            "total_interactions": len(memories),
            "mood_distribution": dict(mood_counter),
            "most_common_mood": mood_counter.most_common(1)[0][0] if mood_counter else None,
            "most_active_hour": Counter(hours).most_common(1)[0][0] if hours else None,
            "most_active_day": Counter(days).most_common(1)[0][0] if days else None,
            "engagement_level": "high" if len(memories) > 20 else "moderate" if len(memories) > 5 else "new",
            "recent_moods": moods[:10],
            "unique_moods": len(set(moods)),
            "days_active": len(set(ts.date() for ts in timestamps)),
            "multimodal_usage": multimodal_count,
            "multimodal_percentage": (multimodal_count / len(memories) * 100) if memories else 0
        }


# ============================================================================
# MAIN APPLICATION WITH ASYNC MULTIMODAL SUPPORT
# ============================================================================

class MentalHealthAssistant:
    """Main application with async multimodal capabilities"""
    
    def __init__(self, user_id: str = "6988bc70"):
        self.qdrant = QdrantManager()
        self.llm = LLMResponder()
        self.user_id = user_id
        self.session_id = str(uuid.uuid4())[:8]
        self.user_profile = None
        
        console.print(Panel.fit(
            f"[bold cyan]Enhanced Multimodal Mental Health Support System[/bold cyan]\n"
            f"User ID: {user_id} | Session: {self.session_id}",
            border_style="cyan"
        ))
    
    def initialize(self):
        """Initialize system"""
        console.print("\n[bold]Initializing multimodal system...[/bold]")
        
        self.qdrant.setup_collections()
        
        self.user_profile = self.qdrant.get_user_profile(self.user_id)
        if not self.user_profile:
            self.user_profile = UserProfile(user_id=self.user_id)
            self.qdrant.upsert_user_profile(self.user_profile)
            console.print(f"[green]✓[/green] Created new user profile")
        else:
            console.print(f"[green]✓[/green] Loaded profile for {self.user_profile.name or self.user_id}")
    
    async def process_query_async(
        self, 
        query: str, 
        mood: Optional[str] = None,
        auto_detect_mood: bool = True,
        file_path: Optional[str] = None
    ) -> Dict:
        """Process user query asynchronously with optional multimodal file"""
        
        if not self.user_profile:
            self.user_profile = self.qdrant.get_user_profile(self.user_id)
            if not self.user_profile:
                self.user_profile = UserProfile(user_id=self.user_id)
                self.qdrant.upsert_user_profile(self.user_profile)
        
        # Process multimodal file asynchronously
        multimodal_data_ids = []
        if file_path and os.path.exists(file_path):
            mm_id, mm_data = await AsyncMultimodalProcessor.process_file(file_path, self.user_id)
            if mm_id and 'error' not in mm_data:
                multimodal = MultimodalData(**mm_data)
                self.qdrant.add_multimodal_data(multimodal)
                multimodal_data_ids.append(mm_id)
        
        # Auto-detect mood
        if auto_detect_mood and not mood:
            mood = MoodDetector.detect_mood(query)
            if mood:
                console.print(f"[cyan]Detected mood: {mood}[/cyan]")
        
        # Search resources (async in thread pool)
        console.print(f"[cyan]Searching knowledge base...[/cyan]")
        loop = asyncio.get_event_loop()
        resources = await loop.run_in_executor(
            executor,
            partial(self.qdrant.search_resources, limit=5),
            query,
            self.user_id
        )
        
        # Search past memories
        past_memories = await loop.run_in_executor(
            executor,
            partial(self.qdrant.search_user_memories, limit=3),
            self.user_id,
            query
        )
        
        # Search multimodal context
        multimodal_context = await loop.run_in_executor(
            executor,
            partial(self.qdrant.search_multimodal, limit=3),
            query,
            self.user_id
        )
        
        # Get analytics
        user_history = await loop.run_in_executor(
            executor,
            partial(self.qdrant.get_user_history, limit=100),
            self.user_id
        )
        analytics = UserAnalytics.analyze_patterns(user_history)
        
        # Generate response
        console.print(f"[cyan]Generating personalized response...[/cyan]")
        full_response, summary = await loop.run_in_executor(
            executor,
            self.llm.generate_response,
            query,
            resources,
            self.user_profile,
            past_memories,
            multimodal_context,
            analytics
        )
        
        # Display response
        console.print("\n" + "="*70)
        console.print(Markdown(full_response))
        console.print("="*70)
        
        # Display multimodal context if any
        if multimodal_context:
            self._display_multimodal_context(multimodal_context)
        
        # Save memory
        memory = UserMemory(
            id=str(uuid.uuid4()),
            user_id=self.user_id,
            session_id=self.session_id,
            timestamp=datetime.now().isoformat(),
            query=query,
            response_summary=summary,
            mood=mood,
            resources_used=[r[0].id for r in resources],
            multimodal_data_ids=multimodal_data_ids
        )
        self.qdrant.add_memory(memory)
        
        # Update profile
        self.user_profile.last_active = datetime.now().isoformat()
        self.user_profile.total_interactions += 1
        self.qdrant.upsert_user_profile(self.user_profile)
        
        return {
            "success": True,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "query": query,
            "detected_mood": mood,
            "response": full_response,
            "response_summary": summary,
            "multimodal_data_ids": multimodal_data_ids,
            "timestamp": datetime.now().isoformat()
        }
    
    def process_query(self, query: str, mood: Optional[str] = None, 
                     auto_detect_mood: bool = True, file_path: Optional[str] = None) -> Dict:
        """Synchronous wrapper for async process_query"""
        return asyncio.run(self.process_query_async(query, mood, auto_detect_mood, file_path))
    
    def _display_multimodal_context(self, context: List[Tuple[MultimodalData, float]]):
        """Display detailed multimodal context"""
        console.print("\n[bold cyan]Multimodal Context Analysis:[/bold cyan]")
        
        for mm_data, score in context:
            if mm_data.processed_data:
                pd = mm_data.processed_data
                
                # Create table for this multimodal item
                table = Table(
                    title=f"{mm_data.data_type.upper()} - Relevance: {score:.3f}", 
                    show_header=False, 
                    border_style="cyan"
                )
                table.add_column("Property", style="yellow")
                table.add_column("Value", style="white")
                
                # Add rows based on data type
                if mm_data.data_type == "image":
                    table.add_row("Caption", pd.get('caption', 'N/A'))
                    table.add_row("Emotional Tone", pd.get('emotional_tone', 'N/A'))
                    table.add_row("Mood", pd.get('inferred_mood', 'N/A'))
                    if pd.get('objects_detected'):
                        table.add_row("Objects", ', '.join(pd['objects_detected']))
                    table.add_row("Brightness", f"{pd.get('avg_brightness', 0):.0f}/255")
                
                elif mm_data.data_type == "audio":
                    if pd.get('title'):
                        table.add_row("Title", pd['title'])
                    if pd.get('artist'):
                        table.add_row("Artist", pd['artist'])
                    table.add_row("Duration", f"{pd.get('duration', 0):.1f}s")
                    table.add_row("Mood", pd.get('inferred_mood', 'N/A'))
                    table.add_row("Emotional Content", pd.get('emotional_content', 'N/A'))
                    if pd.get('transcription') and pd['transcription'] != "No transcription available":
                        table.add_row("Transcription", pd['transcription'][:100] + "...")
                
                elif mm_data.data_type == "video":
                    table.add_row("Duration", f"{pd.get('duration', 0):.1f}s")
                    table.add_row("Resolution", pd.get('resolution', 'N/A'))
                    table.add_row("Mood", pd.get('inferred_mood', 'N/A'))
                    if pd.get('video_summary'):
                        table.add_row("Summary", pd['video_summary'][:150])
                
                elif mm_data.data_type == "code":
                    table.add_row("Language", pd.get('language', 'N/A'))
                    table.add_row("Lines", str(pd.get('code_lines', 0)))
                    table.add_row("Complexity", pd.get('complexity', 'N/A'))
                    table.add_row("Functions", str(pd.get('functions', 0)))
                    if pd.get('comment_text'):
                        table.add_row("Comments", pd['comment_text'][:100])
                
                # Print the table
                console.print(table)
                console.print()
    
    def show_analytics(self):
        """Display user analytics"""
        user_history = self.qdrant.get_user_history(self.user_id, limit=100)
        analytics = UserAnalytics.analyze_patterns(user_history)
        
        if not analytics:
            console.print("[yellow]No interaction history yet[/yellow]")
            return
        
        name_display = self.user_profile.name or self.user_id
        
        multimodal_info = ""
        if analytics.get('multimodal_usage', 0) > 0:
            multimodal_info = f"\nMultimodal usage: {analytics['multimodal_usage']} ({analytics['multimodal_percentage']:.1f}%)"
        
        console.print(Panel.fit(
            f"[bold]Mental Health Journey: {name_display}[/bold]\n\n"
            f"Total interactions: {analytics['total_interactions']}\n"
            f"Most common mood: {analytics.get('most_common_mood', 'N/A')}\n"
            f"Most active hour: {analytics.get('most_active_hour', 'N/A')}:00\n"
            f"Most active day: {analytics.get('most_active_day', 'N/A')}\n"
            f"Engagement: {analytics['engagement_level']}\n"
            f"Days active: {analytics.get('days_active', 0)}\n"
            f"Recent moods: {', '.join(analytics.get('recent_moods', [])[:5])}"
            f"{multimodal_info}",
            border_style="magenta"
        ))
    
    def show_history(self):
        """Show recent conversation history"""
        memories = self.qdrant.get_user_history(self.user_id, limit=10)
        
        if not memories:
            console.print("[yellow]No conversation history yet[/yellow]")
            return
        
        console.print(f"\n[bold]Recent Conversations (User: {self.user_profile.name or self.user_id}):[/bold]\n")
        for memory in memories[:5]:
            timestamp = datetime.fromisoformat(memory.timestamp)
            console.print(f"[dim]{timestamp.strftime('%Y-%m-%d %H:%M')}[/dim]")
            console.print(f"Query: {memory.query}")
            console.print(f"Summary: {memory.response_summary}")
            if memory.mood:
                console.print(f"Mood: {memory.mood}")
            if memory.multimodal_data_ids:
                console.print(f"[cyan]📎 Multimodal data: {len(memory.multimodal_data_ids)} items[/cyan]")
            console.print()
    
    def show_help(self):
        """Display help menu"""
        help_text = """
[bold cyan]Available Commands:[/bold cyan]

- Type your question naturally
- /upload <file> - Upload multimodal file (image/audio/video/code)
- /mood <mood> - Set current mood manually
- /analytics - View interaction patterns
- /history - Recent conversations
- /crisis - Emergency resources
- /help - Show this help
- /quit - Exit

[bold yellow]Multimodal Support (Async Processing):[/bold yellow]
✓ Images (.jpg, .png, .gif) - AI captions, mood, objects
✓ Audio (.mp3, .wav) - Speech transcription, lyrics, mood
✓ Video (.mp4, .avi) - Visual analysis, scene detection
✓ Code (.py, .js, .java) - Complexity, comments analysis
✓ Structured data (JSON) - Data insights

[bold green]Enhanced Features:[/bold green]
✓ Non-blocking file processing
✓ Real content analysis (not just metadata)
✓ AI-powered image captions
✓ Speech-to-text transcription
✓ Emotional content detection

[bold green]Example:[/bold green]
"I'm feeling anxious about work"
"/upload my_photo.jpg"  (processes in background)
"""
        console.print(Panel(help_text, border_style="cyan"))
    
    def show_crisis_resources(self):
        """Display crisis resources"""
        crisis_info = """
[bold red]CRISIS RESOURCES - IMMEDIATE HELP[/bold red]

🇺🇸 United States:
- 988 Suicide & Crisis Lifeline: Call/Text 988
- Crisis Text Line: Text HOME to 741741

🇮🇳 India:
- AASRA: 91-9820466726
- Vandrevala Foundation: 1860-2662-345

🌍 International: findahelpline.com

🚨 Emergency: Call local emergency services

[bold yellow]You are not alone. Help is available 24/7.[/bold yellow]
"""
        console.print(Panel(crisis_info, border_style="red"))
    
    def run_interactive(self):
        """Run interactive CLI with async support"""
        self.show_help()
        
        current_mood = None
        
        while True:
            try:
                user_input = Prompt.ask(f"\n[bold green]{self.user_profile.name or 'You'}[/bold green]").strip()
                
                if not user_input:
                    continue
                
                if user_input.startswith("/"):
                    command_parts = user_input.split()
                    command = command_parts[0].lower()
                    
                    if command == "/quit" or command == "/exit":
                        console.print(f"[cyan]Take care, {self.user_profile.name or 'friend'}! 💙[/cyan]")
                        break
                    
                    elif command == "/help":
                        self.show_help()
                    
                    elif command == "/analytics":
                        self.show_analytics()
                    
                    elif command == "/history":
                        self.show_history()
                    
                    elif command == "/crisis":
                        self.show_crisis_resources()
                    
                    elif command == "/mood":
                        if len(command_parts) > 1:
                            current_mood = command_parts[1]
                            console.print(f"[green]✓ Mood set to: {current_mood}[/green]")
                        else:
                            console.print("[yellow]Usage: /mood [anxious|stressed|calm|sad][/yellow]")
                    
                    elif command == "/upload":
                        if len(command_parts) > 1:
                            file_path = " ".join(command_parts[1:])
                            if os.path.exists(file_path):
                                console.print("[cyan]Processing file asynchronously...[/cyan]")
                                self.process_query(
                                    f"I uploaded a file: {os.path.basename(file_path)}", 
                                    mood=current_mood,
                                    file_path=file_path
                                )
                                current_mood = None
                            else:
                                console.print(f"[red]File not found: {file_path}[/red]")
                        else:
                            console.print("[yellow]Usage: /upload <file_path>[/yellow]")
                    
                    else:
                        console.print(f"[yellow]Unknown command. Type /help[/yellow]")
                
                else:
                    self.process_query(user_input, mood=current_mood, auto_detect_mood=True)
                    current_mood = None
            
            except KeyboardInterrupt:
                console.print(f"\n[cyan]Take care! 💙[/cyan]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                import traceback
                traceback.print_exc()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    
    console.print(Panel.fit(
        "[bold cyan]Enhanced Multimodal Mental Health Support System[/bold cyan]\n"
        "Features: AI Image Captions, Speech Transcription, Async Processing\n"
        "Supports: Text, Images, Audio, Video, Code, Structured Data\n\n"
        "[dim]Your mental health matters. Powered by AI.[/dim]",
        border_style="cyan"
    ))
    
    assistant = MentalHealthAssistant(user_id="6988bc70")
    assistant.initialize()
    
    assistant.run_interactive()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())