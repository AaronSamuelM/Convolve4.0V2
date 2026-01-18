import os
import asyncio
from typing import Dict
from functools import partial
import concurrent.futures
from moviepy.editor import VideoFileClip
from PIL import Image
import numpy as np


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
    def process_video(video_path: str, sample_frames: int = 3) -> Dict:
        """
        Process video using moviepy (lighter alternative to OpenCV).
        Reduced default sample_frames from 5 to 3 for speed.
        """
        try:
            # Load video
            clip = VideoFileClip(video_path)
            
            # Get basic metadata
            duration = clip.duration
            fps = clip.fps
            width, height = clip.size
            frame_count = int(duration * fps)
            
            # Sample frames at intervals
            brightness_values = []
            frame_descriptions = []
            
            # Calculate time intervals for sampling
            time_interval = duration / sample_frames if sample_frames > 0 else duration
            
            for i in range(sample_frames):
                timestamp = min(i * time_interval, duration - 0.1)
                
                # Extract frame at timestamp
                frame = clip.get_frame(timestamp)
                
                # Calculate brightness (convert to grayscale)
                # frame is already numpy array (RGB)
                gray = np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140])
                brightness_values.append(np.mean(gray))
                
                # Process first 2 frames for descriptions (reduced from 3)
                if len(frame_descriptions) < 2:
                    # Save frame temporarily
                    temp_frame_path = f"/tmp/frame_{i}.jpg"
                    Image.fromarray(frame).save(temp_frame_path)
                    
                    # Process with ImageProcessor
                    frame_data = ImageProcessor.process_image(temp_frame_path)
                    if 'caption' in frame_data:
                        frame_descriptions.append(frame_data['caption'])
                    
                    # Cleanup
                    if os.path.exists(temp_frame_path):
                        os.remove(temp_frame_path)
            
            # Close clip to free resources
            clip.close()
            
            # Calculate metrics
            avg_brightness = np.mean(brightness_values) if brightness_values else 0
            mood_from_video = VideoProcessor._infer_mood_from_video(
                duration, avg_brightness, fps
            )
            
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
        """Infer mood/tone from video metrics"""
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
        """Generate text description from video metadata"""
        if metadata is None:
            metadata = VideoProcessor.process_video(video_path)
        
        if "error" in metadata:
            return f"Video file: {os.path.basename(video_path)}"
        
        desc = f"Video showing: {metadata.get('video_summary', 'visual content')}. "
        desc += f"Presentation: {metadata['inferred_mood']}. "
        desc += f"Duration: {metadata['duration']:.1f}s. "
        desc += f"Resolution: {metadata['resolution']}"
        
        return desc
