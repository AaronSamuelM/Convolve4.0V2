"""Async multimodal file processor"""
import os
import uuid
import mimetypes
from typing import Tuple, Dict, Optional
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn
from .processors.image import ImageProcessor
from .processors.audio import AudioProcessor
from .processors.video import VideoProcessor
from .processors.code import CodeProcessor


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
            
            code_extensions = {'.py': 'python', '.js': 'javascript', '.java': 'java', '.cpp': 'c++'}
            ext = Path(file_path).suffix
            if ext in code_extensions:
                data_type = "code"
                with open(file_path, 'r') as f:
                    code_content = f.read()
                processed_data = CodeProcessor.process_code(code_content, code_extensions[ext])
            
        except Exception as e:
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