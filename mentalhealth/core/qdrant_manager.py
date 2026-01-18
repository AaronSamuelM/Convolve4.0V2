"""Qdrant database manager with FastEmbed"""
import os
import uuid
import hashlib
from typing import List, Tuple, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue
)
from fastembed import TextEmbedding

from ..config import (
    QDRANT_URL, QDRANT_API_KEY, EMBEDDING_DIM,
    RESOURCES_COLLECTION, MEMORY_COLLECTION, USER_PROFILE_COLLECTION,
    MULTIMODAL_COLLECTION
)
from ..models import (
    MentalHealthResource, UserMemory, UserProfile, MultimodalData
)
from ..processors.image import ImageProcessor
from ..processors.audio import AudioProcessor


class QdrantManager:
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.encoder = None  # Lazy load
        self.embedding_model = "BAAI/bge-small-en-v1.5"  # 384 dimensions
        
        print(f"[QdrantManager] Connected to Qdrant")
        print(f"[QdrantManager] Using FastEmbed model: {self.embedding_model}")

    def _get_encoder(self):
        """Lazy load FastEmbed encoder"""
        if self.encoder is None:
            self.encoder = TextEmbedding(model_name=self.embedding_model)
            print(f"[QdrantManager] FastEmbed model loaded")
        return self.encoder
    
    def _encode_text(self, text: str) -> List[float]:
        """Encode text using FastEmbed"""
        encoder = self._get_encoder()
        # FastEmbed returns generator, get first embedding and convert to list
        embedding = list(encoder.embed([text]))[0]
        return embedding.tolist()

    def setup_collections(self):
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
                print(f"[QdrantManager] Created collection: {collection}")
    
    def get_user_by_email(self, email: str) -> Optional[UserProfile]:
        res, _ = self.client.scroll(
            USER_PROFILE_COLLECTION,
            Filter(must=[FieldCondition(key="email", match=MatchValue(value=email))]),
            limit=1
        )
        if not res:
            return None
        return UserProfile(**res[0].payload)
    
    def add_multimodal_data(self, multimodal: MultimodalData):
        # Generate text for embedding based on data type
        if multimodal.data_type == "image":
            text_for_embedding = ImageProcessor.generate_description(
                multimodal.content, multimodal.processed_data
            )
        elif multimodal.data_type == "audio":
            text_for_embedding = AudioProcessor.generate_description(
                multimodal.content, multimodal.processed_data
            )
        else:
            text_for_embedding = str(multimodal.content)
        
        vector = self._encode_text(text_for_embedding)
        
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
        print(f"[QdrantManager] Added {multimodal.data_type} data")
    
    def search_multimodal(
        self,
        query: str,
        user_id: str,
        data_types: Optional[List[str]] = None,
        limit: int = 2
    ) -> List[Tuple[MultimodalData, float]]:
        query_vector = self._encode_text(query)
        filter_conditions = Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        )
        
        results = self.client.query_points(
            collection_name=MULTIMODAL_COLLECTION,
            query=query_vector,
            query_filter=filter_conditions,
            limit=limit * 2
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
        points = []
        for resource in resources:
            text = f"{resource.title}. {resource.content}"
            vector = self._encode_text(text)
            
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
        print(f"[QdrantManager] Uploaded {len(points)} resources")
    
    def search_resources(
        self,
        query: str,
        user_id: str,
        category: Optional[str] = None,
        limit: int = 5
    ) -> List[Tuple[MentalHealthResource, float]]:
        query_vector = self._encode_text(query)
        
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
        vector = self._encode_text(memory.query)
        
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
        query_vector = self._encode_text(query)
        
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
                    email=point.payload.get("email"),
                    password_hash=point.payload.get("password_hash"),
                    name=point.payload.get("name"),
                    created_at=point.payload.get("created_at"),
                    last_active=point.payload.get("last_active"),
                    total_interactions=point.payload.get("total_interactions", 0),
                    preferences=point.payload.get("preferences", {})
                )
        except Exception as e:
            print(f"[QdrantManager] Could not load user profile: {e}")
        
        return None
    
    def upsert_user_profile(self, profile: UserProfile):
        user_id_hash = hashlib.md5(profile.user_id.encode()).hexdigest()
        point_id = str(uuid.UUID(user_id_hash))
        
        vector = self._encode_text(f"User {profile.user_id} {profile.name or ''}")
        
        point = PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "user_id": profile.user_id,
                "email": profile.email,
                "password_hash": profile.password_hash,
                "name": profile.name,
                "created_at": profile.created_at,
                "last_active": profile.last_active,
                "total_interactions": profile.total_interactions,
                "preferences": profile.preferences
            }
        )
        
        self.client.upsert(collection_name=USER_PROFILE_COLLECTION, points=[point])
