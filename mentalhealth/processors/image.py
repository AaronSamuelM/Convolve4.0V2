"""Low-memory image processing module"""

import os
import asyncio
from typing import Dict, List
import concurrent.futures
import gc

from PIL import Image
import numpy as np

executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)


class ImageProcessor:
    @classmethod
    async def process_image_async(cls, image_path: str) -> Dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, cls.process_image, image_path)

    @classmethod
    def process_image(cls, image_path: str) -> Dict:
        try:
            img = Image.open(image_path).convert("RGB")
            width, height = img.size

            img_small = img.resize((160, 160))
            arr = np.asarray(img_small, dtype=np.uint8)

            avg_color = arr.mean(axis=(0, 1))
            brightness = avg_color.mean()
            red_blue_ratio = avg_color[0] / (avg_color[2] + 1)

            inferred_mood = cls._infer_mood_from_colors(brightness, red_blue_ratio)
            caption = cls._generate_caption(brightness, avg_color)
            objects = cls._infer_objects(avg_color)
            emotional_tone = cls._analyze_emotional_context(brightness, caption)

            result = {
                "width": width,
                "height": height,
                "mode": "RGB",
                "avg_brightness": float(brightness),
                "avg_color": avg_color.tolist(),
                "red_blue_ratio": float(red_blue_ratio),
                "inferred_mood": inferred_mood,
                "caption": caption,
                "objects_detected": objects,
                "emotional_tone": emotional_tone,
                "file_size": os.path.getsize(image_path)
            }

            del img, img_small, arr
            gc.collect()

            return result

        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _generate_caption(brightness: float, color) -> str:
        if brightness < 70:
            return "a dark and low-light scene"
        if brightness > 180:
            return "a bright and well-lit scene"

        if color[1] > color[0] and color[1] > color[2]:
            return "a scene dominated by natural or green tones"
        if color[2] > color[0]:
            return "a cool-toned scene with calm atmosphere"

        return "a visually balanced indoor or outdoor scene"

    @staticmethod
    def _infer_objects(color) -> List[str]:
        objects = []
        if color[1] > 120:
            objects.append("nature")
        if color[0] > 150:
            objects.append("warm lighting")
        if color[2] > 130:
            objects.append("cool environment")
        return objects[:3]

    @staticmethod
    def _analyze_emotional_context(brightness: float, caption: str) -> str:
        if brightness < 80:
            return "somber or introspective"
        if brightness > 170:
            return "uplifting and positive"
        if "nature" in caption:
            return "calm and grounding"
        return "neutral and balanced"

    @staticmethod
    def _infer_mood_from_colors(brightness: float, ratio: float) -> str:
        if brightness < 80:
            return "dark/melancholic"
        if brightness > 180:
            return "bright/energetic"
        if ratio > 1.2:
            return "warm/calm"
        if ratio < 0.8:
            return "cool/relaxed"
        return "balanced/neutral"

    @classmethod
    def generate_description(cls, image_path: str, metadata: Dict = None) -> str:
        if metadata is None:
            metadata = cls.process_image(image_path)

        if "error" in metadata:
            return f"Image file: {os.path.basename(image_path)}"

        desc = f"Image shows {metadata['caption']}. "
        desc += f"Emotional tone appears {metadata['emotional_tone']}. "
        desc += f"Visual mood is {metadata['inferred_mood']}. "

        if metadata["objects_detected"]:
            desc += f"Likely elements include {', '.join(metadata['objects_detected'])}. "

        desc += f"Brightness level {metadata['avg_brightness']:.0f}. "
        desc += f"Resolution {metadata['width']}x{metadata['height']}."

        return desc
