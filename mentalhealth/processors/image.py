"""Image processing module"""
import os
import asyncio
from typing import Dict, List
import concurrent.futures

try:
    from PIL import Image
    import numpy as np
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


class ImageProcessor:
    _model = None
    _processor = None
    
    @classmethod
    def _load_model(cls):
        if cls._model is None and TRANSFORMERS_AVAILABLE:
            try:
                print("[ImageProcessor] Loading BLIP model...")
                cls._processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                cls._model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                print("[ImageProcessor] Model loaded successfully")
            except Exception as e:
                print(f"[ImageProcessor] Failed to load model: {e}")
    
    @classmethod
    async def process_image_async(cls, image_path: str) -> Dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, cls.process_image, image_path)
    
    @classmethod
    def process_image(cls, image_path: str) -> Dict:
        if not PILLOW_AVAILABLE:
            return {"error": "PIL not available"}
        
        try:
            img = Image.open(image_path)
            width, height = img.size
            mode = img.mode
            
            if mode != 'RGB':
                img = img.convert('RGB')
            
            img_array = np.array(img)
            avg_color = img_array.mean(axis=(0, 1))
            brightness = avg_color.mean()
            red_blue_ratio = avg_color[0] / (avg_color[2] + 1)
            
            mood_from_image = cls._infer_mood_from_colors(brightness, red_blue_ratio, avg_color)
            
            caption = "No caption available"
            objects_detected = []
            
            if TRANSFORMERS_AVAILABLE:
                cls._load_model()
                if cls._model is not None:
                    try:
                        inputs = cls._processor(img, return_tensors="pt")
                        out = cls._model.generate(**inputs, max_length=50)
                        caption = cls._processor.decode(out[0], skip_special_tokens=True)
                        objects_detected = cls._extract_objects_from_caption(caption)
                    except Exception as e:
                        print(f"[ImageProcessor] Caption generation failed: {e}")
            
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
        words = caption.lower().split()
        stop_words = {'a', 'an', 'the', 'is', 'are', 'with', 'in', 'on', 'at', 'of'}
        return [w for w in words if w not in stop_words and len(w) > 3][:5]
    
    @staticmethod
    def _analyze_emotional_context(brightness: float, ratio: float, caption: str) -> str:
        caption_lower = caption.lower()
        positive_words = ['happy', 'smile', 'joy', 'bright', 'colorful', 'nature', 'peaceful']
        negative_words = ['dark', 'sad', 'alone', 'empty', 'gray', 'gloomy']
        
        positive_score = sum(1 for word in positive_words if word in caption_lower)
        negative_score = sum(1 for word in negative_words if word in caption_lower)
        
        if positive_score > negative_score and brightness > 150:
            return "uplifting and positive"
        elif negative_score > positive_score or brightness < 80:
            return "somber or contemplative"
        return "neutral and balanced"
    
    @staticmethod
    def _infer_mood_from_colors(brightness: float, rg_ratio: float, avg_color) -> str:
        if brightness < 80:
            return "dark/melancholic"
        elif brightness > 180:
            return "bright/energetic"
        elif rg_ratio > 1.2:
            return "warm/calm"
        elif rg_ratio < 0.8:
            return "cool/relaxed"
        return "balanced/neutral"
    
    @classmethod
    def generate_description(cls, image_path: str, metadata: Dict = None) -> str:
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