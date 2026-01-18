"""User analytics and pattern analysis"""
from datetime import datetime
from collections import Counter
from typing import List, Dict

from ..models import UserMemory


class UserAnalytics:
    @staticmethod
    def analyze_patterns(memories: List[UserMemory]) -> Dict:
        if not memories:
            return {}
        
        moods = [m.mood for m in memories if m.mood]
        mood_counter = Counter(moods)
        
        timestamps = [datetime.fromisoformat(m.timestamp) for m in memories]
        hours = [ts.hour for ts in timestamps]
        days = [ts.strftime('%A') for ts in timestamps]
        
        multimodal_count = sum(1 for m in memories if m.multimodal_data_ids)
        
        return {
            "total_interactions": len(memories),
            "mood_distribution": dict(mood_counter),
            "most_common_mood": mood_counter.most_common(1)[0][0] if mood_counter else None,
            "most_active_hour": Counter(hours).most_common(1)[0][0] if hours else None,
            "most_active_day": Counter(days).most_common(1)[0][0] if days else None,
            "engagement_level": "high" if len(memories) > 20 else "moderate" if len(memories) > 5 else "new",
            "recent_moods": moods[:10],
            "unique_moods": len(set(moods)),
            "days_active": len(set(ts.date() for ts in timestamps)),
            "multimodal_usage": multimodal_count,
            "multimodal_percentage": (multimodal_count / len(memories) * 100) if memories else 0
        }