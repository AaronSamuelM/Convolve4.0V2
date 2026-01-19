"""LLM Responder with OpenAI for both completions and embeddings"""
import requests
from typing import List, Tuple, Dict, Optional
from openai import OpenAI

from ..config import GROQ_API_KEY, DEFAULT_LLM_MODEL, GROQ_API_URL, OPENAI_API_KEY
from ..models import MentalHealthResource, UserProfile, UserMemory, MultimodalData


class LLMResponder:
    """
    LLM Responder using:
    - Groq for chat completions (fast, cheap)
    - OpenAI for embeddings (zero server memory, $0.13/1M tokens)
    """
    
    def __init__(self, model: str = DEFAULT_LLM_MODEL):
        self.api_key = GROQ_API_KEY
        self.model = model
        self.url = GROQ_API_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # OpenAI client for embeddings (lazy loaded)
        self._openai_client = None
        self.embedding_model = "text-embedding-3-small"  # $0.02/1M tokens
        
        print(f"[LLM] Initialized: {model}")
        print(f"[LLM] Embeddings: OpenAI {self.embedding_model} (cloud-based, zero memory)")
    
    def _get_openai_client(self):
        """Lazy load OpenAI client"""
        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=OPENAI_API_KEY)
            print("[LLM] OpenAI client loaded")
        return self._openai_client
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding from OpenAI (cloud-based, no local memory).
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats (embedding vector, 1536 dimensions)
        """
        try:
            client = self._get_openai_client()
            response = client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"[LLM] Embedding error: {e}")
            # Fallback: return zero vector
            return [0.0] * 1536

    def generate_response(
        self,
        query: str,
        retrieved_resources: List[Tuple[MentalHealthResource, float]],
        user_profile: Optional[UserProfile],
        past_memories: List[UserMemory],
        multimodal_context: List[Tuple[MultimodalData, float]],
        analytics: Dict
    ) -> Tuple[str, str]:
        print("In generate Response")
        context_parts = []
        
        if user_profile and user_profile.name:
            context_parts.append(f"=== USER PROFILE ===")
            context_parts.append(f"Name: {user_profile.name}")
        
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
        
        if past_memories:
            context_parts.append("\n=== RELEVANT PAST CONVERSATIONS ===")
            for mem in past_memories[:3]:
                context_parts.append(f"Previous: {mem.query}")
                context_parts.append(f"Solution: {mem.response_summary}")
                if mem.mood:
                    context_parts.append(f"Mood: {mem.mood}")
        
        context_parts.append("\n=== RELEVANT RESOURCES ===")
        for resource, score in retrieved_resources:
            context_parts.append(f"\nTitle: {resource.title}")
            context_parts.append(f"Content: {resource.content}")
            context_parts.append(f"Source: {resource.source}")
        
        context = "\n".join(context_parts)
        system_prompt = """You are a mental health assistant.

CRITICAL INSTRUCTIONS:
1. Provide evidence-based, supportive responses
2. ALWAYS cite specific resources by title
3. Consider only most relevant multimodal context (images, audio, video, code) and dont mention word like "from the audio clip" or the media name
4. When images/audio/video are provided, reference their content in a human readable way
5. Be warm and personalized only when required
6. Reference past conversations when relevant
7. Keep responses actionable and concise (minimum 20 words and maximum 200 words)
"""

        user_prompt = f"""User query: {query}

{context}

Provide a supportive response that considers only relevant to the question context and multimodal data."""

        try:
            r = requests.post(
                self.url,
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.6,
                    "max_completion_tokens": 300
                }
            )
            
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"].strip()
            summary = text[:200]
            return text, summary
            
        except Exception as e:
            print(f"[LLM] Error: {e}")
            return self._template_response(retrieved_resources, multimodal_context)
    
    def _template_response(
        self,
        resources: List[Tuple[MentalHealthResource, float]],
        multimodal: List[Tuple[MultimodalData, float]]
    ) -> Tuple[str, str]:
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