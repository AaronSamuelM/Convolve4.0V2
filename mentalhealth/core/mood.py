"""Mood detection from text"""
import re
from collections import defaultdict
from typing import Optional


class MoodDetector:
    MOOD_PATTERNS = {
        "anxious": [
            r'\b(anxious|anxiety|worried|worry|nervous|panic|fear|scared|afraid|uneasy|restless|tense)\b',
            r'\b(can\'t (stop|calm)|racing thoughts|heart racing|on edge)\b',
            r'\b(what if|overthinking)\b'
        ],
        "stressed": [
            r'\b(stressed|stress|overwhelmed|pressure|burden|swamped|exhausted mentally)\b',
            r'\b(too much|can\'t cope|breaking point|burned out|burnout)\b',
            r'\b(deadline|workload|hectic)\b'
        ],
        "depressed": [
            r'\b(depressed|depression|hopeless|empty|numb|worthless|pointless)\b',
            r'\b(no energy|no motivation|giving up|can\'t enjoy)\b',
            r'\b(why bother|what\'s the point)\b'
        ],
        "sad": [
            r'\b(sad|sadness|down|blue|miserable|unhappy|crying|tears)\b',
            r'\b(feel bad|feel awful|feel terrible)\b'
        ],
        "lonely": [
            r'\b(lonely|alone|isolated|disconnected|no one|nobody)\b',
            r'\b(feel abandoned|no friends|by myself)\b'
        ],
        "tired": [
            r'\b(tired|exhausted|fatigued|drained|weary|sleepy|no sleep|insomnia)\b',
            r'\b(can\'t sleep|trouble sleeping|wake up)\b'
        ],
        "angry": [
            r'\b(angry|anger|mad|furious|frustrated|irritated|annoyed|rage)\b',
            r'\b(pissed off|fed up)\b'
        ],
        "calm": [
            r'\b(calm|peaceful|relaxed|serene|tranquil|content)\b',
            r'\b(feeling (good|better|okay|fine))\b'
        ],
        "happy": [
            r'\b(happy|joy|joyful|excited|great|wonderful|fantastic|amazing)\b',
            r'\b(feeling good|doing well)\b'
        ],
        "confused": [
            r'\b(confused|lost|uncertain|don\'t know|unsure|mixed feelings)\b',
            r'\b(not sure|unclear)\b'
        ]
    }
    
    @classmethod
    def detect_mood(cls, text: str) -> Optional[str]:
        if not text:
            return None
        
        text_lower = text.lower()
        mood_scores = defaultdict(int)
        
        for mood, patterns in cls.MOOD_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                mood_scores[mood] += len(matches)
        
        if mood_scores:
            return max(mood_scores.items(), key=lambda x: x[1])[0]
        
        return None