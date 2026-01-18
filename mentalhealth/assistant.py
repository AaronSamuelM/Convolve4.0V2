"""Main Mental Health Assistant"""
import os
import uuid
import asyncio
from datetime import datetime
from typing import Dict, Optional
from functools import partial
import concurrent.futures

from .config import MAX_WORKERS
from .models import UserMemory, UserProfile, MultimodalData
from .core.qdrant_manager import QdrantManager
from .core.mood import MoodDetector
from .core.analytics import UserAnalytics
from .multimodal import AsyncMultimodalProcessor

executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)


class MentalHealthAssistant:
    def __init__(self, user_id: str, is_guest: bool = False):
        self.is_guest = is_guest
        self.qdrant = QdrantManager()
        self.llm = None 
        self.user_id = user_id
        self.session_id = str(uuid.uuid4())[:8]
        self.user_profile = None
        
        print(f"[Assistant] Initialized for user: {user_id}, Session: {self.session_id}")
    
    def get_llm(self):
        """Lazy load LLM only when needed"""
        if self.llm is None:
            from .core.llm import LLMResponder
            self.llm = LLMResponder()
            print("[Assistant] LLM loaded on-demand")
        return self.llm
    
    def create_user_profile(self, name: str, email: str, password_hash: str):
        if self.is_guest:
            return

        self.user_profile = UserProfile(
            user_id=self.user_id,
            name=name,
            email=email,
            password_hash=password_hash,
            created_at=datetime.now().isoformat()
        )
        self.qdrant.upsert_user_profile(self.user_profile)
        print(f"[Assistant] Created user profile: {email}")

    def initialize(self):
        print("[Assistant] Initializing...")
        
        self.user_profile = self.qdrant.get_user_profile(self.user_id)
        if not self.user_profile:
            self.user_profile = UserProfile(user_id=self.user_id)
            self.qdrant.upsert_user_profile(self.user_profile)
            print("[Assistant] Created new user profile")
        else:
            print(f"[Assistant] Loaded profile for {self.user_profile.name or self.user_id}")
    
    async def process_query_async(
        self,
        query: str,
        mood: Optional[str] = None,
        auto_detect_mood: bool = True,
        file_path: Optional[str] = None
    ) -> Dict:
        if not self.is_guest:
            if not self.user_profile:
                self.user_profile = self.qdrant.get_user_profile(self.user_id)
                if not self.user_profile:
                    self.user_profile = UserProfile(user_id=self.user_id)
                    self.qdrant.upsert_user_profile(self.user_profile)
        
        multimodal_data_ids = []
        if file_path and os.path.exists(file_path):
            mm_id, mm_data = await AsyncMultimodalProcessor.process_file(file_path, self.user_id)
            if mm_id and 'error' not in mm_data:
                multimodal = MultimodalData(**mm_data)
                if not self.is_guest:
                    self.qdrant.add_multimodal_data(multimodal)
                multimodal_data_ids.append(mm_id)
        
        if auto_detect_mood and not mood:
            mood = MoodDetector.detect_mood(query)
            if mood:
                print(f"[Assistant] Detected mood: {mood}")
        
        print("[Assistant] Searching knowledge base...")
        loop = asyncio.get_event_loop()
        
        resources = await loop.run_in_executor(
            executor,
            partial(self.qdrant.search_resources, limit=5),
            query,
            self.user_id
        )
        
        past_memories = await loop.run_in_executor(
            executor,
            partial(self.qdrant.search_user_memories, limit=3),
            self.user_id,
            query
        )
        
        multimodal_context = await loop.run_in_executor(
            executor,
            partial(self.qdrant.search_multimodal, limit=3),
            query,
            self.user_id
        )
        
        user_history = await loop.run_in_executor(
            executor,
            partial(self.qdrant.get_user_history, limit=100),
            self.user_id
        )
        
        analytics = UserAnalytics.analyze_patterns(user_history)
        
        print("[Assistant] Generating response...")
        # Use lazy-loaded LLM
        llm = self.get_llm()
        full_response, summary = await loop.run_in_executor(
            executor,
            llm.generate_response,
            query,
            resources,
            self.user_profile,
            past_memories,
            multimodal_context,
            analytics
        )
        
        print(f"[Assistant] Response generated ({len(full_response)} chars)")
        
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
        
        if not self.is_guest:
            self.qdrant.add_memory(memory)
            self.user_profile.last_active = datetime.now().isoformat()
            self.user_profile.total_interactions += 1
            self.qdrant.upsert_user_profile(self.user_profile)
        
        return {
            "success": True,
            "user_id": self.user_id,
            "is_guest": self.is_guest,
            "query": query,
            "detected_mood": mood,
            "response": full_response,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
    
    def process_query(
        self,
        query: str,
        mood: Optional[str] = None,
        auto_detect_mood: bool = True,
        file_path: Optional[str] = None
    ) -> Dict:
        return asyncio.run(self.process_query_async(query, mood, auto_detect_mood, file_path))