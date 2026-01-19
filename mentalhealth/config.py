"""Configuration and constants for Mental Health Assistant"""
import os
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

RESOURCES_COLLECTION = "mental_health_resources"
MEMORY_COLLECTION = "user_memory"
USER_PROFILE_COLLECTION = "user_profiles"
MULTIMODAL_COLLECTION = "multimodal_data"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

MAX_WORKERS = int(os.getenv("MAX_WORKERS", 4))
MAX_AUDIO_DURATION = int(os.getenv("MAX_AUDIO_DURATION", 60))  
VIDEO_SAMPLE_FRAMES = int(os.getenv("VIDEO_SAMPLE_FRAMES", 5))