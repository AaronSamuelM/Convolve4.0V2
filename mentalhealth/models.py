"""Data models for Mental Health Assistant"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any


@dataclass
class MultimodalData:
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
    user_id: str
    email: Optional[str] = None
    password_hash: Optional[str] = None
    name: Optional[str] = None
    created_at: str = None
    last_active: str = None
    total_interactions: int = 0
    preferences: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.preferences is None:
            self.preferences = {}