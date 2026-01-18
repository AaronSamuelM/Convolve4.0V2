import os
import asyncio
import numpy as np
from typing import Dict
from functools import partial
import concurrent.futures
import cv2
from .image import ImageProcessor
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
class VideoProcessor:
    @staticmethod
    async def process_video_async(video_path: str, sample_frames: int = 5) -> Dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor, 
            partial(VideoProcessor.process_video, sample_frames=sample_frames),
            video_path
        )
    
    @staticmethod
    def process_video(video_path: str, sample_frames: int = 5) -> Dict:
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
            
            for i in range(0, min(frame_count, sample_frames * frame_interval), frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    brightness_values.append(np.mean(gray))
                    
                    if len(frame_descriptions) < 3:
                        temp_frame_path = f"/tmp/frame_{i}.jpg"
                        cv2.imwrite(temp_frame_path, frame)
                        frame_data = ImageProcessor.process_image(temp_frame_path)
                        if 'caption' in frame_data:
                            frame_descriptions.append(frame_data['caption'])
                        os.remove(temp_frame_path)
            
            cap.release()
            
            avg_brightness = np.mean(brightness_values) if brightness_values else 0
            mood_from_video = VideoProcessor._infer_mood_from_video(duration, avg_brightness, fps)
            
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
        if metadata is None:
            metadata = VideoProcessor.process_video(video_path)
        
        if "error" in metadata:
            return f"Video file: {os.path.basename(video_path)}"
        
        desc = f"Video showing: {metadata.get('video_summary', 'visual content')}. "
        desc += f"Presentation: {metadata['inferred_mood']}. "
        desc += f"Duration: {metadata['duration']:.1f}s. "
        desc += f"Resolution: {metadata['resolution']}"
        
        return desc
