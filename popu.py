import os
import uuid
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import json

console = Console()

RESOURCES_COLLECTION = "mental_health_resources"

class MentalHealthResourceGenerator:
    
    def __init__(self):
        self.url =os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        
        self.client = QdrantClient(url=self.url, api_key=self.api_key)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2') 
        
        console.print("[green]✓[/green] Connected to Qdrant Cloud")
        console.print("[green]✓[/green] Loaded embedding model (384 dimensions)")
    
    def generate_resources(self) -> List[Dict]:
        
        resources = []
        
        # 1. DEPRESSION RESOURCES (30 resources)
        depression_resources = [
            {
                "title": "Understanding Depression Thought Patterns",
                "content": "Learn to recognize cognitive distortions including black-and-white thinking, overgeneralization, catastrophizing, and mental filtering. Practice identifying these patterns in daily situations and develop counter-statements.",
                "category": "depression",
                "resource_type": "technique",
                "tags": ["depression-support", "cognitive-behavioral", "thought-patterns"],
                "source": "Depression Treatment Research",
                "difficulty": "beginner",
                "duration_minutes": 10
            },
            {
                "title": "Behavioral Activation for Depression",
                "content": "Create a weekly activity schedule focusing on pleasurable and meaningful activities. Start with small, achievable goals and gradually increase engagement. Track mood before and after activities.",
                "category": "depression",
                "resource_type": "exercise",
                "tags": ["depression-support", "behavioral-activation", "mood"],
                "source": "Clinical Psychology Today",
                "difficulty": "beginner",
                "duration_minutes": 20
            },
            {
                "title": "Depression and Sleep Hygiene",
                "content": "Establish consistent sleep-wake times, create a relaxing bedtime routine, limit screen time before bed, and optimize your sleep environment. Depression often disrupts sleep patterns.",
                "category": "depression",
                "resource_type": "guide",
                "tags": ["depression-support", "sleep", "self-care"],
                "source": "Sleep Foundation",
                "difficulty": "beginner",
                "duration_minutes": 15
            },
            {
                "title": "Gratitude Journaling for Low Mood",
                "content": "Write three things you're grateful for each day, no matter how small. This practice helps shift focus from negative to positive aspects of life and builds resilience.",
                "category": "depression",
                "resource_type": "exercise",
                "tags": ["depression-support", "journaling", "gratitude"],
                "source": "Positive Psychology Research",
                "difficulty": "beginner",
                "duration_minutes": 5
            },
            {
                "title": "Social Connection Despite Depression",
                "content": "Learn strategies to maintain social connections even when depressed. Start with low-pressure interactions, use text/online communication when face-to-face feels overwhelming, and be honest with trusted friends.",
                "category": "depression",
                "resource_type": "guide",
                "tags": ["depression-support", "social-connection", "relationships"],
                "source": "Mental Health America",
                "difficulty": "intermediate",
                "duration_minutes": 25
            },
            {
                "title": "Physical Exercise and Depression",
                "content": "Understand how exercise affects brain chemistry and mood. Start with 10-minute walks and gradually increase. Learn about aerobic exercise, strength training, and yoga for depression.",
                "category": "depression",
                "resource_type": "guide",
                "tags": ["depression-support", "exercise", "physical-health"],
                "source": "Harvard Medical School",
                "difficulty": "beginner",
                "duration_minutes": 30
            },
            {
                "title": "Mindfulness for Depression",
                "content": "Practice present-moment awareness to reduce rumination. Learn body scan meditation, mindful breathing, and how to observe thoughts without judgment.",
                "category": "depression",
                "resource_type": "meditation",
                "tags": ["depression-support", "mindfulness", "meditation"],
                "source": "Mindfulness-Based Cognitive Therapy",
                "difficulty": "intermediate",
                "duration_minutes": 15
            },
            {
                "title": "Understanding Anhedonia",
                "content": "Learn about the loss of pleasure in activities and how to work with it. Strategies include trying new activities, revisiting old hobbies with low expectations, and understanding it's a symptom, not who you are.",
                "category": "depression",
                "resource_type": "education",
                "tags": ["depression-support", "anhedonia", "symptoms"],
                "source": "American Psychological Association",
                "difficulty": "intermediate",
                "duration_minutes": 20
            },
            {
                "title": "Nutrition and Depression",
                "content": "Explore the gut-brain connection, anti-inflammatory foods, omega-3 fatty acids, and meal planning strategies that support mental health.",
                "category": "depression",
                "resource_type": "guide",
                "tags": ["depression-support", "nutrition", "self-care"],
                "source": "Nutritional Psychiatry",
                "difficulty": "beginner",
                "duration_minutes": 25
            },
            {
                "title": "Managing Depression Fatigue",
                "content": "Learn energy conservation techniques, pacing strategies, and how to balance rest with gentle activity. Create a sustainable daily routine.",
                "category": "depression",
                "resource_type": "technique",
                "tags": ["depression-support", "fatigue", "energy-management"],
                "source": "Chronic Fatigue Center",
                "difficulty": "beginner",
                "duration_minutes": 15
            },
        ]
        
        # 2. ANXIETY RESOURCES (30 resources)
        anxiety_resources = [
            {
                "title": "4-7-8 Breathing for Anxiety",
                "content": "Practice this calming breath technique: inhale for 4 counts, hold for 7, exhale for 8. Activates the parasympathetic nervous system to reduce anxiety quickly.",
                "category": "anxiety",
                "resource_type": "technique",
                "tags": ["anxiety-relief", "breathing", "quick-relief"],
                "source": "Dr. Andrew Weil",
                "difficulty": "beginner",
                "duration_minutes": 5
            },
            {
                "title": "Progressive Muscle Relaxation",
                "content": "Systematically tense and release muscle groups from toes to head. Helps identify where you hold tension and promotes deep relaxation.",
                "category": "anxiety",
                "resource_type": "exercise",
                "tags": ["anxiety-relief", "relaxation", "body-awareness"],
                "source": "Relaxation Training Institute",
                "difficulty": "beginner",
                "duration_minutes": 20
            },
            {
                "title": "Understanding Panic Attacks",
                "content": "Learn what happens during panic attacks, why they're not dangerous, and evidence-based techniques to manage them including grounding, breathing, and cognitive strategies.",
                "category": "anxiety",
                "resource_type": "education",
                "tags": ["anxiety-relief", "panic-attacks", "crisis-management"],
                "source": "Anxiety and Depression Association",
                "difficulty": "intermediate",
                "duration_minutes": 30
            },
            {
                "title": "Worry Time Technique",
                "content": "Schedule 15 minutes daily to write all worries. Outside this time, postpone worries to your scheduled session. Helps contain anxiety and reduce constant rumination.",
                "category": "anxiety",
                "resource_type": "technique",
                "tags": ["anxiety-relief", "worry", "cognitive-behavioral"],
                "source": "Cognitive Behavioral Therapy Manual",
                "difficulty": "intermediate",
                "duration_minutes": 15
            },
            {
                "title": "Social Anxiety Exposure Hierarchy",
                "content": "Create a gradual exposure plan for social situations ranked from least to most anxiety-provoking. Practice systematic desensitization with support.",
                "category": "anxiety",
                "resource_type": "exercise",
                "tags": ["anxiety-relief", "social-anxiety", "exposure-therapy"],
                "source": "Social Anxiety Institute",
                "difficulty": "advanced",
                "duration_minutes": 45
            },
            {
                "title": "Grounding Techniques: 5-4-3-2-1",
                "content": "When overwhelmed, identify 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste. Brings awareness back to the present moment.",
                "category": "anxiety",
                "resource_type": "technique",
                "tags": ["anxiety-relief", "grounding", "mindfulness"],
                "source": "Trauma-Informed Care",
                "difficulty": "beginner",
                "duration_minutes": 3
            },
            {
                "title": "Anxiety and Caffeine",
                "content": "Understand how caffeine affects anxiety, identify hidden sources, and create a gradual reduction plan if needed. Explore alternatives like herbal tea.",
                "category": "anxiety",
                "resource_type": "guide",
                "tags": ["anxiety-relief", "lifestyle", "nutrition"],
                "source": "Nutritional Psychology",
                "difficulty": "beginner",
                "duration_minutes": 10
            },
            {
                "title": "Managing Health Anxiety",
                "content": "Learn to distinguish between appropriate health concerns and anxiety-driven hypervigilance. Develop strategies to reduce body checking and reassurance seeking.",
                "category": "anxiety",
                "resource_type": "guide",
                "tags": ["anxiety-relief", "health-anxiety", "reassurance"],
                "source": "Health Anxiety Clinic",
                "difficulty": "intermediate",
                "duration_minutes": 35
            },
            {
                "title": "Sleep Anxiety Solutions",
                "content": "Address anxiety that disrupts sleep with cognitive techniques for racing thoughts, relaxation exercises, and sleep restriction therapy principles.",
                "category": "anxiety",
                "resource_type": "guide",
                "tags": ["anxiety-relief", "sleep", "insomnia"],
                "source": "Sleep Disorders Center",
                "difficulty": "intermediate",
                "duration_minutes": 25
            },
            {
                "title": "Accepting Uncertainty",
                "content": "Learn to tolerate uncertainty rather than seeking impossible guarantees. Practice sitting with discomfort and reducing control behaviors.",
                "category": "anxiety",
                "resource_type": "technique",
                "tags": ["anxiety-relief", "uncertainty", "acceptance"],
                "source": "Acceptance and Commitment Therapy",
                "difficulty": "advanced",
                "duration_minutes": 30
            },
        ]
        
        # 3. STRESS MANAGEMENT (25 resources)
        stress_resources = [
            {
                "title": "Time Management for Stress",
                "content": "Learn prioritization techniques, time-blocking, and how to say no. Create realistic schedules that include breaks and self-care.",
                "category": "stress",
                "resource_type": "guide",
                "tags": ["stress-management", "productivity", "time-management"],
                "source": "Organizational Psychology",
                "difficulty": "beginner",
                "duration_minutes": 30
            },
            {
                "title": "Box Breathing Technique",
                "content": "Military-tested breathing method: inhale 4, hold 4, exhale 4, hold 4. Excellent for acute stress and maintaining calm under pressure.",
                "category": "stress",
                "resource_type": "technique",
                "tags": ["stress-management", "breathing", "quick-relief"],
                "source": "Navy SEAL Training",
                "difficulty": "beginner",
                "duration_minutes": 5
            },
            {
                "title": "Work-Life Balance Strategies",
                "content": "Set boundaries between work and personal time, create transition rituals, and protect personal time. Address burnout prevention.",
                "category": "stress",
                "resource_type": "guide",
                "tags": ["stress-management", "work-life-balance", "boundaries"],
                "source": "Workplace Wellness Institute",
                "difficulty": "intermediate",
                "duration_minutes": 40
            },
            {
                "title": "Physical Stress Release",
                "content": "Use movement to release stress: stretching, shaking, dancing, or high-intensity exercise. Understand the stress-exercise connection.",
                "category": "stress",
                "resource_type": "exercise",
                "tags": ["stress-management", "exercise", "movement"],
                "source": "Somatic Psychology",
                "difficulty": "beginner",
                "duration_minutes": 15
            },
            {
                "title": "Stress Journaling",
                "content": "Track stress triggers, physical symptoms, and effective coping strategies. Identify patterns and develop personalized stress management plan.",
                "category": "stress",
                "resource_type": "exercise",
                "tags": ["stress-management", "journaling", "self-awareness"],
                "source": "Behavioral Health Research",
                "difficulty": "beginner",
                "duration_minutes": 20
            },
        ]
        
        # 4. SELF-HARM AND CRISIS (20 resources - neutral to high-level)
        crisis_resources = [
            {
                "title": "Understanding Self-Harm Urges",
                "content": "Learn about the psychology behind self-harm urges, emotional regulation, and the cycle of self-injury. Understanding is the first step toward healing.",
                "category": "crisis",
                "resource_type": "education",
                "tags": ["self-harm", "crisis-support", "education", "urge-management"],
                "source": "Self-Injury Foundation",
                "difficulty": "intermediate",
                "duration_minutes": 25
            },
            {
                "title": "Alternative Coping: Ice Technique",
                "content": "Hold ice cubes, take a cold shower, or use ice packs as an alternative sensation when experiencing self-harm urges. Provides intense sensation without injury.",
                "category": "crisis",
                "resource_type": "technique",
                "tags": ["self-harm", "alternatives", "harm-reduction", "coping-skills"],
                "source": "DBT Skills Training",
                "difficulty": "beginner",
                "duration_minutes": 5
            },
            {
                "title": "Red Pen Technique",
                "content": "Draw on skin with a red washable marker instead of cutting. Provides visual feedback without harm. Can help satisfy urges while staying safe.",
                "category": "crisis",
                "resource_type": "technique",
                "tags": ["self-harm", "alternatives", "harm-reduction", "visual"],
                "source": "Harm Reduction Coalition",
                "difficulty": "beginner",
                "duration_minutes": 10
            },
            {
                "title": "Rubber Band Technique",
                "content": "Snap a rubber band against wrist as a less harmful alternative. Provides brief pain sensation without lasting damage. Transition technique.",
                "category": "crisis",
                "resource_type": "technique",
                "tags": ["self-harm", "alternatives", "harm-reduction", "transition"],
                "source": "Crisis Intervention Manual",
                "difficulty": "beginner",
                "duration_minutes": 2
            },
            {
                "title": "Delayed Self-Harm Protocol",
                "content": "When urges arise, commit to waiting 15 minutes. Use distraction, call support, or engage alternatives. Often urges decrease with time.",
                "category": "crisis",
                "resource_type": "technique",
                "tags": ["self-harm", "delay", "urge-surfing", "crisis-management"],
                "source": "Dialectical Behavior Therapy",
                "difficulty": "intermediate",
                "duration_minutes": 15
            },
            {
                "title": "Emotional Regulation Chain Analysis",
                "content": "Identify the chain of events leading to self-harm urges: prompting event → vulnerability factors → trigger → urge → behavior. Break the chain.",
                "category": "crisis",
                "resource_type": "exercise",
                "tags": ["self-harm", "emotional-regulation", "analysis", "prevention"],
                "source": "DBT Linehan Institute",
                "difficulty": "advanced",
                "duration_minutes": 45
            },
            {
                "title": "Safety Planning for Self-Harm",
                "content": "Create personalized safety plan: warning signs, internal coping, people for support, professionals to contact, making environment safe, reasons for living.",
                "category": "crisis",
                "resource_type": "exercise",
                "tags": ["self-harm", "safety-planning", "crisis-prevention", "support"],
                "source": "Suicide Prevention Resource Center",
                "difficulty": "intermediate",
                "duration_minutes": 60
            },
            {
                "title": "Intense Exercise Alternative",
                "content": "When experiencing strong urges, engage in intense physical activity: running, push-ups, jumping jacks. Releases endorphins and redirects energy.",
                "category": "crisis",
                "resource_type": "technique",
                "tags": ["self-harm", "alternatives", "exercise", "energy-release"],
                "source": "Sports Psychology Institute",
                "difficulty": "beginner",
                "duration_minutes": 10
            },
            {
                "title": "Wound Care for Self-Harm",
                "content": "If self-harm occurs: clean wounds properly, when to seek medical care, preventing infection, and proper bandaging. Harm reduction approach.",
                "category": "crisis",
                "resource_type": "guide",
                "tags": ["self-harm", "harm-reduction", "medical-care", "safety"],
                "source": "Emergency Medicine Journal",
                "difficulty": "intermediate",
                "duration_minutes": 20
            },
            {
                "title": "Talking to Someone About Self-Harm",
                "content": "Guide for disclosing self-harm to trusted person: choosing who to tell, what to say, managing reactions, and asking for specific support.",
                "category": "crisis",
                "resource_type": "guide",
                "tags": ["self-harm", "disclosure", "support", "communication"],
                "source": "Mental Health First Aid",
                "difficulty": "advanced",
                "duration_minutes": 30
            },
            {
                "title": "Understanding Triggers",
                "content": "Identify internal and external triggers for self-harm. Create trigger log and develop specific coping for each trigger category.",
                "category": "crisis",
                "resource_type": "exercise",
                "tags": ["self-harm", "triggers", "awareness", "prevention"],
                "source": "Trauma Recovery Institute",
                "difficulty": "intermediate",
                "duration_minutes": 35
            },
            {
                "title": "Distress Tolerance Skills",
                "content": "Learn TIPP skills (Temperature, Intense exercise, Paced breathing, Paired muscle relaxation) for managing crisis moments without self-harm.",
                "category": "crisis",
                "resource_type": "technique",
                "tags": ["self-harm", "distress-tolerance", "dbt-skills", "crisis"],
                "source": "DBT Skills Manual",
                "difficulty": "intermediate",
                "duration_minutes": 25
            },
            {
                "title": "Scream/Cry Release",
                "content": "Safe emotional release: scream into pillow, cry intensely, or use scream room. Allows emotional expression without harm to self or others.",
                "category": "crisis",
                "resource_type": "technique",
                "tags": ["self-harm", "alternatives", "emotional-release", "catharsis"],
                "source": "Emotional Freedom Therapy",
                "difficulty": "beginner",
                "duration_minutes": 10
            },
            {
                "title": "Self-Compassion in Crisis",
                "content": "Practice self-compassion when experiencing urges or after self-harm. Reduce shame, treat yourself with kindness, recognize common humanity.",
                "category": "crisis",
                "resource_type": "technique",
                "tags": ["self-harm", "self-compassion", "shame-reduction", "healing"],
                "source": "Kristin Neff Research",
                "difficulty": "intermediate",
                "duration_minutes": 20
            },
            {
                "title": "Crisis Hotline Guide",
                "content": "When and how to use crisis hotlines: what to expect, preparing for the call, text options, and follow-up care. You deserve support.",
                "category": "crisis",
                "resource_type": "guide",
                "tags": ["self-harm", "crisis-support", "hotlines", "help-seeking"],
                "source": "988 Suicide & Crisis Lifeline",
                "difficulty": "beginner",
                "duration_minutes": 15
            },
            {
                "title": "Butterfly Project",
                "content": "Draw butterfly on area you want to harm. Name it after loved one. If you harm there, butterfly dies. Care for butterfly until it fades naturally.",
                "category": "crisis",
                "resource_type": "technique",
                "tags": ["self-harm", "alternatives", "visual", "commitment"],
                "source": "Butterfly Project International",
                "difficulty": "beginner",
                "duration_minutes": 10
            },
            {
                "title": "Long-term Recovery from Self-Harm",
                "content": "Understand recovery isn't linear. Develop relapse prevention plan, celebrate milestones, address underlying issues, and build life worth living.",
                "category": "crisis",
                "resource_type": "guide",
                "tags": ["self-harm", "recovery", "long-term", "healing"],
                "source": "Recovery Research Institute",
                "difficulty": "advanced",
                "duration_minutes": 50
            },
            {
                "title": "Environmental Safety Strategies",
                "content": "Make environment safer: remove or lock up items used for self-harm, create barriers, and arrange access with trusted person during vulnerable times.",
                "category": "crisis",
                "resource_type": "guide",
                "tags": ["self-harm", "safety", "environment", "prevention"],
                "source": "Crisis Prevention Institute",
                "difficulty": "intermediate",
                "duration_minutes": 30
            },
            {
                "title": "Opposite Action for Self-Harm Urges",
                "content": "When urge arises, do complete opposite: gentle self-care, soft music, warm bath, treating yourself kindly. Challenges the destructive urge.",
                "category": "crisis",
                "resource_type": "technique",
                "tags": ["self-harm", "opposite-action", "self-care", "dbt-skills"],
                "source": "DBT Emotion Regulation",
                "difficulty": "intermediate",
                "duration_minutes": 20
            },
            {
                "title": "Peer Support for Self-Harm Recovery",
                "content": "Connect with others in recovery. Online forums, support groups, and sharing experiences. You're not alone in this struggle.",
                "category": "crisis",
                "resource_type": "guide",
                "tags": ["self-harm", "peer-support", "community", "connection"],
                "source": "Self-Harm Support Network",
                "difficulty": "beginner",
                "duration_minutes": 25
            },
        ]
        
        # 5. MINDFULNESS & MEDITATION (20 resources)
        mindfulness_resources = [
            {
                "title": "Beginner's Guide to Meditation",
                "content": "Start with 5-minute sessions. Focus on breath. When mind wanders, gently return focus. No judgment. Consistency matters more than duration.",
                "category": "mindfulness",
                "resource_type": "guide",
                "tags": ["mindfulness", "meditation", "beginner", "breathing"],
                "source": "Mindfulness Center",
                "difficulty": "beginner",
                "duration_minutes": 5
            },
            {
                "title": "Body Scan Meditation",
                "content": "Systematically bring awareness to each body part from toes to head. Notice sensations without judgment. Promotes relaxation and body awareness.",
                "category": "mindfulness",
                "resource_type": "meditation",
                "tags": ["mindfulness", "body-scan", "relaxation", "awareness"],
                "source": "MBSR Program",
                "difficulty": "beginner",
                "duration_minutes": 20
            },
            {
                "title": "Mindful Eating Practice",
                "content": "Eat one meal slowly, noticing colors, smells, textures, and tastes. Chew thoroughly. Notice hunger and fullness cues. Brings awareness to eating.",
                "category": "mindfulness",
                "resource_type": "exercise",
                "tags": ["mindfulness", "eating", "awareness", "present-moment"],
                "source": "Center for Mindful Eating",
                "difficulty": "beginner",
                "duration_minutes": 30
            },
            {
                "title": "Walking Meditation",
                "content": "Walk slowly, noticing each step, how feet connect with ground, movement of legs, balance shifts. Can be done anywhere.",
                "category": "mindfulness",
                "resource_type": "meditation",
                "tags": ["mindfulness", "walking", "movement", "grounding"],
                "source": "Zen Buddhist Tradition",
                "difficulty": "beginner",
                "duration_minutes": 15
            },
            {
                "title": "Loving-Kindness Meditation",
                "content": "Send well-wishes to self, loved ones, neutral people, difficult people, and all beings. Cultivates compassion and reduces negative emotions.",
                "category": "mindfulness",
                "resource_type": "meditation",
                "tags": ["mindfulness", "compassion", "loving-kindness", "metta"],
                "source": "Buddhist Psychology",
                "difficulty": "intermediate",
                "duration_minutes": 20
            },
        ]
        
        # 6. SLEEP & REST (15 resources)
        sleep_resources = [
            {
                "title": "Sleep Hygiene Fundamentals",
                "content": "Consistent schedule, cool dark room, no screens 1 hour before bed, limit caffeine after 2pm, regular exercise, manage stress.",
                "category": "sleep",
                "resource_type": "guide",
                "tags": ["sleep", "hygiene", "routine", "self-care"],
                "source": "National Sleep Foundation",
                "difficulty": "beginner",
                "duration_minutes": 20
            },
            {
                "title": "Dealing with Racing Thoughts at Night",
                "content": "Keep worry journal by bed, practice 4-7-8 breathing, do mental math, visualize peaceful scene, get up if awake 20+ minutes.",
                "category": "sleep",
                "resource_type": "technique",
                "tags": ["sleep", "insomnia", "anxiety", "thoughts"],
                "source": "Insomnia Clinic",
                "difficulty": "intermediate",
                "duration_minutes": 15
            },
            {
                "title": "Power Napping Guide",
                "content": "Optimal nap: 10-20 minutes for alertness, 90 minutes for full cycle. Avoid naps after 3pm. Create comfortable environment.",
                "category": "sleep",
                "resource_type": "guide",
                "tags": ["sleep", "napping", "energy", "rest"],
                "source": "Sleep Research Society",
                "difficulty": "beginner",
                "duration_minutes": 10
            },
        ]
        
        # 7. RELATIONSHIPS & SOCIAL (15 resources)
        relationship_resources = [
            {
                "title": "Setting Healthy Boundaries",
                "content": "Identify your limits, communicate clearly, use 'I' statements, be consistent, and recognize boundary violations. Boundaries are self-care.",
                "category": "relationships",
                "resource_type": "guide",
                "tags": ["relationships", "boundaries", "communication", "self-care"],
                "source": "Relationship Psychology",
                "difficulty": "intermediate",
                "duration_minutes": 35
            },
            {
                "title": "Active Listening Skills",
                "content": "Give full attention, avoid interrupting, reflect back what you heard, ask clarifying questions, validate emotions, suspend judgment.",
                "category": "relationships",
                "resource_type": "technique",
                "tags": ["relationships", "communication", "listening", "connection"],
                "source": "Communication Studies",
                "difficulty": "beginner",
                "duration_minutes": 20
            },
            {
                "title": "Managing Conflict Constructively",
                "content": "Use 'I feel' statements, focus on specific behaviors, avoid absolutes, take breaks if escalating, seek to understand before being understood.",
                "category": "relationships",
                "resource_type": "guide",
                "tags": ["relationships", "conflict", "communication", "resolution"],
                "source": "Conflict Resolution Institute",
                "difficulty": "intermediate",
                "duration_minutes": 30
            },
        ]
        
        # 8. TRAUMA & PTSD (15 resources)
        trauma_resources = [
            {
                "title": "Understanding Trauma Responses",
                "content": "Learn about fight, flight, freeze, and fawn responses. Recognize trauma triggers and how past experiences affect present reactions.",
                "category": "trauma",
                "resource_type": "education",
                "tags": ["trauma", "ptsd", "education", "responses"],
                "source": "Trauma Center",
                "difficulty": "intermediate",
                "duration_minutes": 30
            },
            {
                "title": "Grounding for Flashbacks",
                "content": "Use 5-4-3-2-1 technique, hold ice, stomp feet, describe surroundings aloud. Brings awareness back to present and safety.",
                "category": "trauma",
                "resource_type": "technique",
                "tags": ["trauma", "ptsd", "grounding", "flashbacks"],
                "source": "PTSD Treatment Center",
                "difficulty": "intermediate",
                "duration_minutes": 10
            },
            {
                "title": "Safe Place Visualization",
                "content": "Create detailed mental image of safe, peaceful place. Engage all senses. Practice accessing this during calm times for use during distress.",
                "category": "trauma",
                "resource_type": "exercise",
                "tags": ["trauma", "ptsd", "visualization", "safety"],
                "source": "EMDR Therapy",
                "difficulty": "beginner",
                "duration_minutes": 15
            },
        ]
        
        # 9. SELF-ESTEEM & CONFIDENCE (15 resources)
        self_esteem_resources = [
            {
                "title": "Challenging Negative Self-Talk",
                "content": "Identify critical inner voice, examine evidence, generate alternative thoughts, practice self-compassion, reframe failures as learning.",
                "category": "self-esteem",
                "resource_type": "technique",
                "tags": ["self-esteem", "confidence", "self-talk", "cognitive"],
                "source": "Self-Compassion Research",
                "difficulty": "intermediate",
                "duration_minutes": 25
            },
            {
                "title": "Building Self-Compassion",
                "content": "Treat yourself as you would a good friend. Practice self-kindness, recognize common humanity, and maintain mindful awareness of struggles.",
                "category": "self-esteem",
                "resource_type": "guide",
                "tags": ["self-esteem", "self-compassion", "kindness", "growth"],
                "source": "Kristin Neff",
                "difficulty": "intermediate",
                "duration_minutes": 30
            },
            {
                "title": "Celebrating Small Wins",
                "content": "Keep daily log of accomplishments, no matter how small. Brushing teeth counts. Build evidence against belief that you accomplish nothing.",
                "category": "self-esteem",
                "resource_type": "exercise",
                "tags": ["self-esteem", "achievement", "journaling", "positive"],
                "source": "Positive Psychology",
                "difficulty": "beginner",
                "duration_minutes": 10
            },
        ]
    
        # 10. GENERAL WELLNESS (15 resources)
        wellness_resources = [
            {
                "title": "Creating Morning Routine",
                "content": "Design sustainable morning practice: hydration, movement, mindfulness, healthy breakfast, planning. Start small and build gradually.",
                "category": "wellness",
                "resource_type": "guide",
                "tags": ["wellness", "routine", "self-care", "habits"],
                "source": "Wellness Institute",
                "difficulty": "beginner",
                "duration_minutes": 30
            },
            {
                "title": "Digital Detox Strategies",
                "content": "Set phone-free times, remove social media from bedroom, use app limits, practice JOMO (joy of missing out), engage real-world activities.",
                "category": "wellness",
                "resource_type": "guide",
                "tags": ["wellness", "digital-health", "boundaries", "balance"],
                "source": "Digital Wellness Lab",
                "difficulty": "intermediate",
                "duration_minutes": 25
            },
            {
                "title": "Hydration and Mental Health",
                "content": "Dehydration affects mood, cognition, and energy. Aim for 8 glasses daily. Set reminders, infuse water with fruit, track intake.",
                "category": "wellness",
                "resource_type": "guide",
                "tags": ["wellness", "hydration", "physical-health", "energy"],
                "source": "Nutritional Science",
                "difficulty": "beginner",
                "duration_minutes": 10
            },
        ]
    
        # Combine all resources
        resources.extend(depression_resources)
        resources.extend(anxiety_resources)
        resources.extend(stress_resources)
        resources.extend(crisis_resources)
        resources.extend(mindfulness_resources)
        resources.extend(sleep_resources)
        resources.extend(relationship_resources)
        resources.extend(trauma_resources)
        resources.extend(self_esteem_resources)
        resources.extend(wellness_resources)

        # Add more resources to reach 200
        additional_resources = self._generate_additional_resources()
        resources.extend(additional_resources)

        return resources[:200]  # Ensure exactly 200

    def _generate_additional_resources(self) -> List[Dict]:
        """Generate additional resources to reach 200 total"""
        
        additional = [
            # OCD Resources
            {
                "title": "Understanding OCD Compulsions",
                "content": "Learn about the OCD cycle: intrusive thought → anxiety → compulsion → temporary relief → strengthening of cycle. Break it with ERP.",
                "category": "ocd",
                "resource_type": "education",
                "tags": ["ocd", "compulsions", "anxiety", "education"],
                "source": "International OCD Foundation",
                "difficulty": "intermediate",
                "duration_minutes": 25
            },
            {
                "title": "Exposure and Response Prevention",
                "content": "Face feared situations without performing compulsions. Create hierarchy of fears and gradually work through them with support.",
                "category": "ocd",
                "resource_type": "technique",
                "tags": ["ocd", "erp", "exposure-therapy", "treatment"],
                "source": "OCD Treatment Center",
                "difficulty": "advanced",
                "duration_minutes": 45
            },
            # Bipolar Resources
            {
                "title": "Mood Tracking for Bipolar",
                "content": "Daily tracking of mood, sleep, energy, activities, medications. Identify patterns and early warning signs of episodes.",
                "category": "bipolar",
                "resource_type": "exercise",
                "tags": ["bipolar", "mood-tracking", "self-monitoring", "prevention"],
                "source": "Bipolar Disorder Research",
                "difficulty": "intermediate",
                "duration_minutes": 15
            },
            {
                "title": "Managing Manic Episodes",
                "content": "Recognize early signs: decreased sleep need, racing thoughts, impulsivity. Create crisis plan, contact provider, reduce stimulation.",
                "category": "bipolar",
                "resource_type": "guide",
                "tags": ["bipolar", "mania", "crisis", "management"],
                "source": "Mood Disorders Clinic",
                "difficulty": "advanced",
                "duration_minutes": 35
            },
            # Eating Disorders
            {
                "title": "Intuitive Eating Principles",
                "content": "Reject diet mentality, honor hunger, make peace with food, challenge food police, respect fullness, discover satisfaction.",
                "category": "eating",
                "resource_type": "guide",
                "tags": ["eating-disorder", "intuitive-eating", "recovery", "nutrition"],
                "source": "Intuitive Eating Authors",
                "difficulty": "intermediate",
                "duration_minutes": 40
            },
            {
                "title": "Body Image Compassion",
                "content": "Practice body neutrality, appreciate function over appearance, challenge beauty standards, curate positive media, self-compassion exercises.",
                "category": "eating",
                "resource_type": "exercise",
                "tags": ["eating-disorder", "body-image", "self-compassion", "recovery"],
                "source": "Body Positive Movement",
                "difficulty": "intermediate",
                "duration_minutes": 30
            },
            # Addiction Recovery
            {
                "title": "Understanding Addiction Triggers",
                "content": "Identify HALT (Hungry, Angry, Lonely, Tired) and environmental triggers. Create specific coping plan for each trigger type.",
                "category": "addiction",
                "resource_type": "exercise",
                "tags": ["addiction", "triggers", "recovery", "relapse-prevention"],
                "source": "Addiction Recovery Institute",
                "difficulty": "intermediate",
                "duration_minutes": 30
            },
            {
                "title": "Urge Surfing Technique",
                "content": "Ride the wave of craving without acting. Notice it rises, peaks, and falls. Typically lasts 20-30 minutes. You can wait it out.",
                "category": "addiction",
                "resource_type": "technique",
                "tags": ["addiction", "cravings", "mindfulness", "recovery"],
                "source": "Mindfulness-Based Relapse Prevention",
                "difficulty": "advanced",
                "duration_minutes": 25
            },
            # ADHD
            {
                "title": "ADHD Time Management",
                "content": "Use visual timers, break tasks into 15-minute chunks, build in movement breaks, external reminders, body doubling for focus.",
                "category": "adhd",
                "resource_type": "guide",
                "tags": ["adhd", "time-management", "productivity", "focus"],
                "source": "ADHD Coaching Institute",
                "difficulty": "intermediate",
                "duration_minutes": 35
            },
            {
                "title": "Managing ADHD Overwhelm",
                "content": "When overwhelmed: pick ONE thing, set timer for 5 minutes, do something physical, use fidget tools, simplify environment.",
                "category": "adhd",
                "resource_type": "technique",
                "tags": ["adhd", "overwhelm", "focus", "regulation"],
                "source": "ADHD Research Center",
                "difficulty": "beginner",
                "duration_minutes": 10
            },
            # Grief and Loss
            {
                "title": "Understanding Grief Stages",
                "content": "Learn about denial, anger, bargaining, depression, acceptance. Stages aren't linear. All feelings are valid in grief.",
                "category": "grief",
                "resource_type": "education",
                "tags": ["grief", "loss", "bereavement", "healing"],
                "source": "Kübler-Ross Institute",
                "difficulty": "beginner",
                "duration_minutes": 20
            },
            {
                "title": "Continuing Bonds",
                "content": "Maintain connection with deceased through rituals, memories, legacy projects. Grief doesn't mean forgetting.",
                "category": "grief",
                "resource_type": "guide",
                "tags": ["grief", "loss", "memory", "healing"],
                "source": "Bereavement Research",
                "difficulty": "intermediate",
                "duration_minutes": 25
            },
            # Anger Management
            {
                "title": "Anger Thermometer",
                "content": "Rate anger 1-10, identify physical signs at each level, create intervention plan for each level before reaching 10.",
                "category": "anger",
                "resource_type": "exercise",
                "tags": ["anger", "emotion-regulation", "awareness", "management"],
                "source": "Anger Management Institute",
                "difficulty": "beginner",
                "duration_minutes": 20
            },
            {
                "title": "Time-Out Technique",
                "content": "Recognize escalation, announce time-out (not walking out), take 20+ minutes, do calming activity, return to discuss calmly.",
                "category": "anger",
                "resource_type": "technique",
                "tags": ["anger", "de-escalation", "relationships", "communication"],
                "source": "Domestic Abuse Prevention",
                "difficulty": "intermediate",
                "duration_minutes": 15
            },
            # Loneliness and Isolation
            {
                "title": "Combating Loneliness",
                "content": "Join groups aligned with interests, volunteer, quality over quantity in friendships, reach out first, practice vulnerability.",
                "category": "loneliness",
                "resource_type": "guide",
                "tags": ["loneliness", "connection", "social", "relationships"],
                "source": "Connection Research Lab",
                "difficulty": "intermediate",
                "duration_minutes": 30
            },
            {
                "title": "Building Social Confidence",
                "content": "Start small with low-stakes interactions, prepare conversation topics, active listening, ask open-ended questions, practice self-compassion.",
                "category": "loneliness",
                "resource_type": "guide",
                "tags": ["loneliness", "social-skills", "confidence", "connection"],
                "source": "Social Psychology Institute",
                "difficulty": "intermediate",
                "duration_minutes": 35
            },
            # Perfectionism
            {
                "title": "Challenging Perfectionism",
                "content": "Distinguish high standards from perfectionism, practice 'good enough', embrace mistakes as learning, challenge all-or-nothing thinking.",
                "category": "perfectionism",
                "resource_type": "technique",
                "tags": ["perfectionism", "anxiety", "self-compassion", "growth"],
                "source": "Cognitive Therapy Center",
                "difficulty": "intermediate",
                "duration_minutes": 30
            },
            {
                "title": "Done is Better Than Perfect",
                "content": "Set time limits on tasks, practice submitting 'imperfect' work, celebrate completion, recognize diminishing returns of over-polishing.",
                "category": "perfectionism",
                "resource_type": "exercise",
                "tags": ["perfectionism", "productivity", "acceptance", "growth"],
                "source": "Productivity Psychology",
                "difficulty": "intermediate",
                "duration_minutes": 20
            },
            # Chronic Pain and Mental Health
            {
                "title": "Pain and Mood Connection",
                "content": "Understand bidirectional relationship between chronic pain and mental health. Manage both simultaneously for best outcomes.",
                "category": "chronic-pain",
                "resource_type": "education",
                "tags": ["chronic-pain", "mental-health", "depression", "holistic"],
                "source": "Pain Psychology Association",
                "difficulty": "intermediate",
                "duration_minutes": 25
            },
            {
                "title": "Pacing for Chronic Conditions",
                "content": "Balance activity and rest, avoid boom-bust cycle, use timers, plan activities, build stamina gradually, listen to body.",
                "category": "chronic-pain",
                "resource_type": "technique",
                "tags": ["chronic-pain", "energy", "management", "wellness"],
                "source": "Chronic Pain Clinic",
                "difficulty": "intermediate",
                "duration_minutes": 30
            },
            # Seasonal Affective Disorder
            {
                "title": "Light Therapy for SAD",
                "content": "Use 10,000 lux light box for 20-30 minutes each morning. Timing and consistency crucial. Combine with other treatments.",
                "category": "depression",
                "resource_type": "guide",
                "tags": ["depression", "sad", "seasonal", "light-therapy"],
                "source": "Seasonal Disorders Center",
                "difficulty": "beginner",
                "duration_minutes": 15
            },
            # More comprehensive resources across categories
            {
                "title": "Cognitive Restructuring Basics",
                "content": "Identify automatic thoughts, examine evidence for/against, generate balanced alternative thoughts, practice regularly.",
                "category": "anxiety",
                "resource_type": "technique",
                "tags": ["anxiety", "cognitive-behavioral", "thoughts", "restructuring"],
                "source": "CBT Institute",
                "difficulty": "intermediate",
                "duration_minutes": 30
            },
            {
                "title": "Values Clarification Exercise",
                "content": "Identify your core values across life domains. Assess current alignment. Set goals to live more consistently with values.",
                "category": "wellness",
                "resource_type": "exercise",
                "tags": ["wellness", "values", "meaning", "goals"],
                "source": "Acceptance Commitment Therapy",
                "difficulty": "intermediate",
                "duration_minutes": 45
            },
            {
                "title": "Emotional Awareness Building",
                "content": "Practice naming emotions beyond 'good/bad', notice where you feel emotions in body, rate intensity, identify needs behind emotions.",
                "category": "wellness",
                "resource_type": "exercise",
                "tags": ["wellness", "emotions", "awareness", "regulation"],
                "source": "Emotional Intelligence Lab",
                "difficulty": "beginner",
                "duration_minutes": 15
            },
            {
                "title": "Self-Soothing Kit Creation",
                "content": "Create physical kit with items for each sense: soft fabric, favorite scent, soothing music, comfort food, calming images.",
                "category": "stress",
                "resource_type": "exercise",
                "tags": ["stress-management", "self-care", "coping", "sensory"],
                "source": "Trauma-Informed Care",
                "difficulty": "beginner",
                "duration_minutes": 30
            },
            {
                "title": "Radical Acceptance Practice",
                "content": "Accept reality as it is, not as you wish it were. Doesn't mean approval, just acknowledging what is. Reduces suffering.",
                "category": "stress",
                "resource_type": "technique",
                "tags": ["stress-management", "acceptance", "dbt-skills", "mindfulness"],
                "source": "Dialectical Behavior Therapy",
                "difficulty": "advanced",
                "duration_minutes": 25
            },
            {
                "title": "Positive Psychology Interventions",
                "content": "Three good things daily, gratitude visit, using signature strengths, savoring positive experiences, acts of kindness.",
                "category": "wellness",
                "resource_type": "exercise",
                "tags": ["wellness", "positive-psychology", "happiness", "growth"],
                "source": "Positive Psychology Center",
                "difficulty": "beginner",
                "duration_minutes": 20
            },
            {
                "title": "Creative Expression for Mental Health",
                "content": "Use art, music, writing, dance as emotional outlet. No skill required. Process not product. Expression heals.",
                "category": "wellness",
                "resource_type": "guide",
                "tags": ["wellness", "creativity", "expression", "healing"],
                "source": "Art Therapy Association",
                "difficulty": "beginner",
                "duration_minutes": 30
            },
            {
                "title": "Nature Therapy",
                "content": "Spend time in nature regularly. Forest bathing, gardening, watching sunrise. Reduces stress, improves mood, grounds you.",
                "category": "wellness",
                "resource_type": "guide",
                "tags": ["wellness", "nature", "stress-relief", "grounding"],
                "source": "Ecotherapy Research",
                "difficulty": "beginner",
                "duration_minutes": 45
            },
            {
                "title": "Phone Support Scripts",
                "content": "Prepare what to say when calling crisis lines, therapist, or friends. Having script reduces barrier to reaching out.",
                "category": "crisis",
                "resource_type": "guide",
                "tags": ["crisis-support", "help-seeking", "communication", "preparation"],
                "source": "Crisis Intervention",
                "difficulty": "beginner",
                "duration_minutes": 20
            },
            {
                "title": "Motivation Enhancement",
                "content": "Understand stages of change, set SMART goals, identify barriers, create implementation intentions, celebrate progress.",
                "category": "wellness",
                "resource_type": "guide",
                "tags": ["wellness", "motivation", "goals", "change"],
                "source": "Motivational Interviewing",
                "difficulty": "intermediate",
                "duration_minutes": 40
            },
            {
            "title": "Depression Relapse Prevention",
            "content": "Identify early warning signs, maintain treatment during good times, build support network, manage stress proactively, have crisis plan ready.",
            "category": "depression",
            "resource_type": "guide",
            "tags": ["depression-support", "relapse-prevention", "recovery", "wellness"],
            "source": "Depression Recovery Institute",
            "difficulty": "intermediate",
            "duration_minutes": 35
            },
            {
                "title": "Medication and Therapy Combination",
                "content": "Understand how medication and therapy work together. Neither replaces the other. Combined approach often most effective for moderate to severe depression.",
                "category": "depression",
                "resource_type": "education",
                "tags": ["depression-support", "medication", "therapy", "treatment"],
                "source": "Psychiatric Treatment Guidelines",
                "difficulty": "beginner",
                "duration_minutes": 20
            },
            {
                "title": "Depression and Routine Building",
                "content": "Structure provides stability. Create morning, afternoon, evening routines. Include self-care, meals, sleep, activities. Flexibility is okay.",
                "category": "depression",
                "resource_type": "guide",
                "tags": ["depression-support", "routine", "structure", "habits"],
                "source": "Behavioral Psychology",
                "difficulty": "beginner",
                "duration_minutes": 30
            },
            {
                "title": "Recognizing Depression in Others",
                "content": "Notice changes in mood, energy, interest, sleep, appetite. Approach with compassion. Encourage professional help. You can't fix it, but you can support.",
                "category": "depression",
                "resource_type": "guide",
                "tags": ["depression-support", "relationships", "support", "awareness"],
                "source": "Mental Health First Aid",
                "difficulty": "beginner",
                "duration_minutes": 25
            },
            {
                "title": "Sunlight and Vitamin D",
                "content": "Get 15-30 minutes sunlight daily. Vitamin D deficiency linked to depression. Consider supplement in winter. Combine with outdoor activity.",
                "category": "depression",
                "resource_type": "guide",
                "tags": ["depression-support", "sunlight", "vitamin-d", "wellness"],
                "source": "Nutritional Psychiatry",
                "difficulty": "beginner",
                "duration_minutes": 15
            },
            
            # More Anxiety Resources
            {
                "title": "Anxiety Thought Record",
                "content": "Document anxious thoughts, evidence for/against, alternative thoughts, outcome. Track patterns. Build cognitive flexibility.",
                "category": "anxiety",
                "resource_type": "exercise",
                "tags": ["anxiety-relief", "cognitive-behavioral", "thoughts", "tracking"],
                "source": "CBT Workbook",
                "difficulty": "intermediate",
                "duration_minutes": 25
            },
            {
                "title": "Interoceptive Exposure",
                "content": "Deliberately create physical sensations you fear (rapid heartbeat, dizziness) in safe environment. Learn they're not dangerous. Reduces panic sensitivity.",
                "category": "anxiety",
                "resource_type": "technique",
                "tags": ["anxiety-relief", "exposure-therapy", "panic-attacks", "desensitization"],
                "source": "Panic Disorder Treatment",
                "difficulty": "advanced",
                "duration_minutes": 30
            },
            {
                "title": "Anxiety Emergency Kit",
                "content": "Create portable kit: stress ball, essential oil, gum, breathing exercises card, grounding object, emergency contacts, inspirational quote.",
                "category": "anxiety",
                "resource_type": "exercise",
                "tags": ["anxiety-relief", "coping-tools", "preparation", "self-care"],
                "source": "Anxiety Disorders Association",
                "difficulty": "beginner",
                "duration_minutes": 20
            },
            {
                "title": "Generalized Anxiety Management",
                "content": "Address chronic worry about multiple areas. Challenge probability thinking, practice present focus, schedule worry time, cognitive restructuring.",
                "category": "anxiety",
                "resource_type": "guide",
                "tags": ["anxiety-relief", "gad", "worry", "management"],
                "source": "GAD Treatment Center",
                "difficulty": "intermediate",
                "duration_minutes": 40
            },
            {
                "title": "Performance Anxiety Strategies",
                "content": "Preparation, visualization, breathing before event, reframe as excitement not fear, accept some anxiety is normal, focus on process not outcome.",
                "category": "anxiety",
                "resource_type": "guide",
                "tags": ["anxiety-relief", "performance", "public-speaking", "confidence"],
                "source": "Performance Psychology",
                "difficulty": "intermediate",
                "duration_minutes": 30
            },
            
            # More Stress Resources
            {
                "title": "Chronic Stress vs Acute Stress",
                "content": "Understand difference. Acute stress is normal and passes. Chronic stress damages health. Identify sources and create long-term management plan.",
                "category": "stress",
                "resource_type": "education",
                "tags": ["stress-management", "chronic-stress", "health", "awareness"],
                "source": "Stress Research Institute",
                "difficulty": "beginner",
                "duration_minutes": 20
            },
            {
                "title": "Stress and Immune System",
                "content": "Chronic stress weakens immunity. Increases illness susceptibility. Manage stress to support physical health. Mind-body connection is real.",
                "category": "stress",
                "resource_type": "education",
                "tags": ["stress-management", "immune-system", "health", "wellness"],
                "source": "Psychoneuroimmunology Research",
                "difficulty": "intermediate",
                "duration_minutes": 25
            },
            {
                "title": "Stress Reduction Through Laughter",
                "content": "Watch comedy, spend time with funny friends, keep humor journal. Laughter reduces cortisol, releases endorphins, shifts perspective.",
                "category": "stress",
                "resource_type": "guide",
                "tags": ["stress-management", "laughter", "humor", "joy"],
                "source": "Laughter Therapy Research",
                "difficulty": "beginner",
                "duration_minutes": 15
            },
            {
                "title": "Financial Stress Management",
                "content": "Create budget, seek financial counseling, break problems into steps, distinguish what you can/can't control, practice self-compassion.",
                "category": "stress",
                "resource_type": "guide",
                "tags": ["stress-management", "financial", "planning", "coping"],
                "source": "Financial Wellness Center",
                "difficulty": "intermediate",
                "duration_minutes": 45
            },
            {
                "title": "Relationship Stress Communication",
                "content": "Use I-statements, active listening, pick right time, focus on one issue, seek to understand, consider couples therapy if needed.",
                "category": "stress",
                "resource_type": "guide",
                "tags": ["stress-management", "relationships", "communication", "conflict"],
                "source": "Couples Therapy Institute",
                "difficulty": "intermediate",
                "duration_minutes": 35
            },
            
            # More Mindfulness Resources
            {
                "title": "Mindful Breathing Variations",
                "content": "Try counted breath, ocean breath, alternate nostril breathing, breath awareness. Find what resonates. Practice regularly.",
                "category": "mindfulness",
                "resource_type": "technique",
                "tags": ["mindfulness", "breathing", "meditation", "variety"],
                "source": "Breathwork Institute",
                "difficulty": "beginner",
                "duration_minutes": 15
            },
            {
                "title": "Mindfulness in Daily Activities",
                "content": "Practice presence while washing dishes, showering, commuting, waiting. Turn routine into meditation. No extra time needed.",
                "category": "mindfulness",
                "resource_type": "guide",
                "tags": ["mindfulness", "daily-practice", "presence", "routine"],
                "source": "Everyday Mindfulness",
                "difficulty": "beginner",
                "duration_minutes": 10
            },
            {
                "title": "Mindful Technology Use",
                "content": "Notice urges to check phone, take conscious breaks, single-task, turn off notifications, create tech-free zones/times.",
                "category": "mindfulness",
                "resource_type": "guide",
                "tags": ["mindfulness", "technology", "digital-wellness", "awareness"],
                "source": "Digital Mindfulness Lab",
                "difficulty": "intermediate",
                "duration_minutes": 25
            },
            {
                "title": "Observing Thoughts Exercise",
                "content": "Watch thoughts like clouds passing. Notice without engaging. Label as 'thinking'. Return to breath. Reduces thought fusion.",
                "category": "mindfulness",
                "resource_type": "meditation",
                "tags": ["mindfulness", "thoughts", "meditation", "detachment"],
                "source": "Mindfulness-Based Stress Reduction",
                "difficulty": "intermediate",
                "duration_minutes": 15
            },
            {
                "title": "Mindful Movement Practices",
                "content": "Try yoga, tai chi, qigong, or mindful stretching. Combine breath with movement. Body-mind integration.",
                "category": "mindfulness",
                "resource_type": "exercise",
                "tags": ["mindfulness", "movement", "yoga", "body-awareness"],
                "source": "Mindful Movement Institute",
                "difficulty": "beginner",
                "duration_minutes": 30
            },
            {
                "title": "RAIN Technique for Difficult Emotions",
                "content": "Recognize, Allow, Investigate, Nurture. Mindful approach to working with difficult emotions. Practice self-compassion.",
                "category": "mindfulness",
                "resource_type": "technique",
                "tags": ["mindfulness", "emotions", "self-compassion", "acceptance"],
                "source": "Tara Brach",
                "difficulty": "intermediate",
                "duration_minutes": 20
            },
            {
                "title": "Mindfulness and Pain Management",
                "content": "Observe pain without resistance. Notice sensations, thoughts, emotions. Create space around pain. Reduces suffering even when pain remains.",
                "category": "mindfulness",
                "resource_type": "technique",
                "tags": ["mindfulness", "pain", "chronic-pain", "acceptance"],
                "source": "Mindfulness-Based Pain Management",
                "difficulty": "advanced",
                "duration_minutes": 25
            },
            {
                "title": "Gratitude Meditation",
                "content": "Bring to mind people, experiences, simple pleasures you're grateful for. Feel appreciation in body. Cultivates positive emotions.",
                "category": "mindfulness",
                "resource_type": "meditation",
                "tags": ["mindfulness", "gratitude", "positive-emotions", "appreciation"],
                "source": "Positive Psychology Meditation",
                "difficulty": "beginner",
                "duration_minutes": 10
            },
            {
                "title": "Noting Practice",
                "content": "Mentally label experiences: 'hearing', 'thinking', 'feeling', 'seeing'. Builds awareness. Reduces identification with experiences.",
                "category": "mindfulness",
                "resource_type": "meditation",
                "tags": ["mindfulness", "noting", "awareness", "meditation"],
                "source": "Vipassana Meditation",
                "difficulty": "intermediate",
                "duration_minutes": 20
            },
            {
                "title": "Mindfulness for Anxiety",
                "content": "Notice anxiety arising, observe physical sensations, watch thoughts, ground in present. Anxiety loses power when observed.",
                "category": "mindfulness",
                "resource_type": "technique",
                "tags": ["mindfulness", "anxiety", "observation", "grounding"],
                "source": "Mindfulness-Based Cognitive Therapy",
                "difficulty": "intermediate",
                "duration_minutes": 15
            },
            
            # More Sleep Resources
            {
                "title": "Sleep Restriction Therapy",
                "content": "Limit time in bed to actual sleep time. Gradually increase. Consolidates sleep. Effective for chronic insomnia. Work with professional.",
                "category": "sleep",
                "resource_type": "technique",
                "tags": ["sleep", "insomnia", "therapy", "treatment"],
                "source": "Sleep Medicine Center",
                "difficulty": "advanced",
                "duration_minutes": 40
            },
            {
                "title": "Progressive Relaxation for Sleep",
                "content": "Tense and release muscle groups while in bed. Promotes physical relaxation. Helps transition to sleep.",
                "category": "sleep",
                "resource_type": "technique",
                "tags": ["sleep", "relaxation", "bedtime", "body"],
                "source": "Sleep Disorders Clinic",
                "difficulty": "beginner",
                "duration_minutes": 20
            },
            {
                "title": "Sleep Environment Optimization",
                "content": "Dark (blackout curtains), cool (65-68°F), quiet (white noise if needed), comfortable mattress/pillow, remove electronics, declutter.",
                "category": "sleep",
                "resource_type": "guide",
                "tags": ["sleep", "environment", "optimization", "hygiene"],
                "source": "Sleep Foundation",
                "difficulty": "beginner",
                "duration_minutes": 25
            },
            {
                "title": "Cognitive Techniques for Insomnia",
                "content": "Challenge sleep anxiety, practice thought stopping, use paradoxical intention, cognitive restructuring of sleep beliefs.",
                "category": "sleep",
                "resource_type": "technique",
                "tags": ["sleep", "insomnia", "cognitive", "anxiety"],
                "source": "CBT for Insomnia",
                "difficulty": "intermediate",
                "duration_minutes": 30
            },
            {
                "title": "Sleep and Mental Health Connection",
                "content": "Poor sleep worsens mental health. Mental health issues disrupt sleep. Address both simultaneously. Bidirectional relationship.",
                "category": "sleep",
                "resource_type": "education",
                "tags": ["sleep", "mental-health", "connection", "wellness"],
                "source": "Sleep and Psychiatry Research",
                "difficulty": "beginner",
                "duration_minutes": 20
            },
            {
                "title": "Melatonin and Sleep",
                "content": "Natural hormone regulating sleep-wake. Consider supplement for jet lag or shift work. Timing matters. Consult doctor for dosage.",
                "category": "sleep",
                "resource_type": "guide",
                "tags": ["sleep", "melatonin", "supplement", "circadian"],
                "source": "Sleep Medicine",
                "difficulty": "beginner",
                "duration_minutes": 15
            },
            {
                "title": "Bedtime Ritual Creation",
                "content": "Develop calming pre-sleep routine: dim lights, warm bath, reading, journaling, stretching, tea. Signal to body it's sleep time.",
                "category": "sleep",
                "resource_type": "guide",
                "tags": ["sleep", "routine", "ritual", "relaxation"],
                "source": "Sleep Hygiene Institute",
                "difficulty": "beginner",
                "duration_minutes": 20
            },
            {
                "title": "Stimulus Control for Sleep",
                "content": "Use bed only for sleep and sex. If awake 20 minutes, leave bedroom. Return when sleepy. Strengthens bed-sleep association.",
                "category": "sleep",
                "resource_type": "technique",
                "tags": ["sleep", "insomnia", "conditioning", "therapy"],
                "source": "Behavioral Sleep Medicine",
                "difficulty": "intermediate",
                "duration_minutes": 25
            },
            {
                "title": "Sleep Tracking and Patterns",
                "content": "Keep sleep diary: bedtime, wake time, quality, factors. Identify patterns. Adjust habits. Share with doctor if needed.",
                "category": "sleep",
                "resource_type": "exercise",
                "tags": ["sleep", "tracking", "awareness", "patterns"],
                "source": "Sleep Research",
                "difficulty": "beginner",
                "duration_minutes": 15
            },
            
            # More Relationship Resources
            {
                "title": "Emotional Intimacy Building",
                "content": "Share vulnerabilities, practice empathy, regular quality time, express appreciation, manage conflict well, maintain trust.",
                "category": "relationships",
                "resource_type": "guide",
                "tags": ["relationships", "intimacy", "connection", "emotional"],
                "source": "Gottman Institute",
                "difficulty": "intermediate",
                "duration_minutes": 35
            },
            {
                "title": "Recognizing Toxic Relationships",
                "content": "Notice manipulation, constant criticism, isolation, control, lack of respect. Your feelings matter. Consider professional support to leave safely.",
                "category": "relationships",
                "resource_type": "education",
                "tags": ["relationships", "toxic", "abuse", "safety"],
                "source": "Domestic Violence Resources",
                "difficulty": "intermediate",
                "duration_minutes": 30
            },
            {
                "title": "Relationship Communication Patterns",
                "content": "Understand pursuer-distancer, demand-withdraw patterns. Recognize your pattern. Break negative cycles. Seek couples therapy if needed.",
                "category": "relationships",
                "resource_type": "education",
                "tags": ["relationships", "communication", "patterns", "couples"],
                "source": "Emotionally Focused Therapy",
                "difficulty": "advanced",
                "duration_minutes": 40
            },
            {
                "title": "Friendship Maintenance",
                "content": "Regular contact, show up for important events, listen actively, share reciprocally, forgive minor slights, invest time.",
                "category": "relationships",
                "resource_type": "guide",
                "tags": ["relationships", "friendship", "social", "connection"],
                "source": "Social Psychology",
                "difficulty": "beginner",
                "duration_minutes": 20
            },
            {
                "title": "Apologizing Effectively",
                "content": "Take responsibility, express remorse, make amends, change behavior. Avoid 'but' or excuses. Genuine apology repairs ruptures.",
                "category": "relationships",
                "resource_type": "technique",
                "tags": ["relationships", "apology", "repair", "communication"],
                "source": "Conflict Resolution",
                "difficulty": "beginner",
                "duration_minutes": 15
            },
            {
                "title": "Love Languages Understanding",
                "content": "Learn the 5 love languages: words, acts of service, gifts, quality time, physical touch. Speak your partner's language.",
                "category": "relationships",
                "resource_type": "education",
                "tags": ["relationships", "love-languages", "connection", "understanding"],
                "source": "Gary Chapman",
                "difficulty": "beginner",
                "duration_minutes": 25
            },
            {
                "title": "Codependency Awareness",
                "content": "Recognize excessive focus on others, neglecting self, difficulty with boundaries, need for approval. Learn healthy interdependence.",
                "category": "relationships",
                "resource_type": "education",
                "tags": ["relationships", "codependency", "boundaries", "self"],
                "source": "Codependency Research",
                "difficulty": "intermediate",
                "duration_minutes": 35
            },
            {
                "title": "Nonviolent Communication",
                "content": "Observe without judgment, express feelings, identify needs, make requests. Framework for compassionate communication.",
                "category": "relationships",
                "resource_type": "technique",
                "tags": ["relationships", "communication", "nvc", "compassion"],
                "source": "Marshall Rosenberg",
                "difficulty": "intermediate",
                "duration_minutes": 40
            },
            {
                "title": "Social Skills for Connection",
                "content": "Make eye contact, ask open questions, show genuine interest, remember details, follow up, be present, smile.",
                "category": "relationships",
                "resource_type": "guide",
                "tags": ["relationships", "social-skills", "connection", "communication"],
                "source": "Social Skills Training",
                "difficulty": "beginner",
                "duration_minutes": 25
            },
            
            # More Trauma Resources
            {
                "title": "Complex PTSD Understanding",
                "content": "Prolonged trauma affects identity, relationships, emotion regulation. Requires specialized treatment. Healing is possible with support.",
                "category": "trauma",
                "resource_type": "education",
                "tags": ["trauma", "cptsd", "complex-trauma", "understanding"],
                "source": "Complex Trauma Institute",
                "difficulty": "advanced",
                "duration_minutes": 35
            },
            {
                "title": "Somatic Experiencing",
                "content": "Body-based trauma therapy. Release trapped survival energy. Work with sensations, not just thoughts. Gentle, bottom-up approach.",
                "category": "trauma",
                "resource_type": "technique",
                "tags": ["trauma", "somatic", "body", "healing"],
                "source": "Somatic Experiencing International",
                "difficulty": "advanced",
                "duration_minutes": 30
            },
            {
                "title": "Trauma and Relationships",
                "content": "Trauma affects trust, intimacy, boundaries. Communicate needs, go slow, practice self-compassion, consider trauma-informed therapy.",
                "category": "trauma",
                "resource_type": "guide",
                "tags": ["trauma", "relationships", "intimacy", "healing"],
                "source": "Trauma Recovery",
                "difficulty": "intermediate",
                "duration_minutes": 35
            },
            {
                "title": "Window of Tolerance",
                "content": "Understand your window between hyperarousal and hypoarousal. Learn to expand window through regulation skills and therapy.",
                "category": "trauma",
                "resource_type": "education",
                "tags": ["trauma", "regulation", "nervous-system", "awareness"],
                "source": "Trauma Therapy",
                "difficulty": "intermediate",
                "duration_minutes": 25
            },
            {
                "title": "Trauma-Informed Self-Care",
                "content": "Prioritize safety, choice, empowerment. Practice grounding, boundaries, self-compassion. Address both mind and body.",
                "category": "trauma",
                "resource_type": "guide",
                "tags": ["trauma", "self-care", "safety", "empowerment"],
                "source": "Trauma-Informed Care",
                "difficulty": "beginner",
                "duration_minutes": 30
            },
            {
                "title": "Childhood Trauma Effects",
                "content": "Early trauma shapes brain development, attachment, coping. Understanding origins helps healing. Not your fault. Therapy can help.",
                "category": "trauma",
                "resource_type": "education",
                "tags": ["trauma", "childhood", "ace", "healing"],
                "source": "ACE Study",
                "difficulty": "intermediate",
                "duration_minutes": 40
            },
            {
                "title": "Hypervigilance Management",
                "content": "Notice constant scanning for danger. Practice grounding, safety reminders, nervous system regulation. You're safe now.",
                "category": "trauma",
                "resource_type": "technique",
                "tags": ["trauma", "hypervigilance", "safety", "nervous-system"],
                "source": "PTSD Treatment",
                "difficulty": "intermediate",
                "duration_minutes": 25
            },
            {
                "title": "EMDR Therapy Overview",
                "content": "Eye Movement Desensitization and Reprocessing. Evidence-based for trauma. Helps brain process traumatic memories. Work with trained therapist.",
                "category": "trauma",
                "resource_type": "education",
                "tags": ["trauma", "emdr", "therapy", "treatment"],
                "source": "EMDR Institute",
                "difficulty": "beginner",
                "duration_minutes": 20
            },
            {
                "title": "Trauma Recovery Stages",
                "content": "Safety and stabilization, remembrance and mourning, reconnection with life. Non-linear process. Take your time.",
                "category": "trauma",
                "resource_type": "education",
                "tags": ["trauma", "recovery", "stages", "healing"],
                "source": "Judith Herman",
                "difficulty": "intermediate",
                "duration_minutes": 30
            },
            {
                "title": "Dissociation and Grounding",
                "content": "When you feel disconnected, use 5 senses, cold water, movement, naming surroundings. Brings awareness back to present.",
                "category": "trauma",
                "resource_type": "technique",
                "tags": ["trauma", "dissociation", "grounding", "presence"],
                "source": "Dissociation Treatment",
                "difficulty": "intermediate",
                "duration_minutes": 15
            },
            
            # More Self-Esteem Resources
            {
                "title": "Core Beliefs Identification",
                "content": "Identify deep beliefs about self formed in childhood. Challenge and replace with realistic, compassionate beliefs.",
                "category": "self-esteem",
                "resource_type": "exercise",
                "tags": ["self-esteem", "core-beliefs", "cognitive", "healing"],
                "source": "Schema Therapy",
                "difficulty": "advanced",
                "duration_minutes": 45
            },
            {
                "title": "Strengths Inventory",
                "content": "List your skills, qualities, accomplishments. Ask trusted others. Keep adding. Review when feeling down. Evidence of your worth.",
                "category": "self-esteem",
                "resource_type": "exercise",
                "tags": ["self-esteem", "strengths", "positive", "awareness"],
                "source": "Positive Psychology",
                "difficulty": "beginner",
                "duration_minutes": 30
            },
            {
                "title": "Imposter Syndrome",
                "content": "Recognize feelings of being fraud despite success. Common, especially in high achievers. Challenge thoughts, own achievements, seek support.",
                "category": "self-esteem",
                "resource_type": "education",
                "tags": ["self-esteem", "imposter-syndrome", "achievement", "confidence"],
                "source": "Performance Psychology",
                "difficulty": "intermediate",
                "duration_minutes": 25
            },
            {
                "title": "Self-Worth vs External Validation",
                "content": "Build internal self-worth, not dependent on others' opinions, achievements, appearance. You are valuable as you are.",
                "category": "self-esteem",
                "resource_type": "guide",
                "tags": ["self-esteem", "self-worth", "validation", "internal"],
                "source": "Self-Esteem Research",
                "difficulty": "intermediate",
                "duration_minutes": 35
            },
            {
                "title": "Mirror Work for Self-Acceptance",
                "content": "Look in mirror, speak affirmations, practice self-kindness. Difficult at first. Builds self-compassion and acceptance over time.",
                "category": "self-esteem",
                "resource_type": "exercise",
                "tags": ["self-esteem", "mirror-work", "acceptance", "compassion"],
                "source": "Louise Hay",
                "difficulty": "intermediate",
                "duration_minutes": 10
            },
            {
                "title": "Assertiveness Training",
                "content": "Express needs, opinions, feelings respectfully. Neither passive nor aggressive. Practice with small situations first.",
                "category": "self-esteem",
                "resource_type": "technique",
                "tags": ["self-esteem", "assertiveness", "communication", "boundaries"],
                "source": "Assertiveness Training",
                "difficulty": "intermediate",
                "duration_minutes": 30
            },
            {
                "title": "Comparison and Social Media",
                "content": "Recognize comparison trap. Remember social media shows highlights. Limit exposure. Focus on your own journey.",
                "category": "self-esteem",
                "resource_type": "guide",
                "tags": ["self-esteem", "comparison", "social-media", "awareness"],
                "source": "Digital Psychology",
                "difficulty": "beginner",
                "duration_minutes": 20
            },
            {
                "title": "Shame Resilience",
                "content": "Recognize shame, practice critical awareness, reach out, speak shame. Shame loses power when shared with safe person.",
                "category": "self-esteem",
                "resource_type": "technique",
                "tags": ["self-esteem", "shame", "vulnerability", "resilience"],
                "source": "Brené Brown",
                "difficulty": "advanced",
                "duration_minutes": 35
            },
            {
                "title": "Positive Affirmations Practice",
                "content": "Create personal, believable affirmations. Repeat daily. Start with 'I'm learning to...' if absolutes feel false.",
                "category": "self-esteem",
                "resource_type": "exercise",
                "tags": ["self-esteem", "affirmations", "positive", "practice"],
                "source": "Cognitive Therapy",
                "difficulty": "beginner",
                "duration_minutes": 10
            },
            
            # More Wellness Resources  
            {
                "title": "Gut Health and Mental Health",
                "content": "Gut-brain axis affects mood, anxiety. Eat fermented foods, fiber, probiotics. Reduce processed foods, sugar.",
                "category": "wellness",
                "resource_type": "guide",
                "tags": ["wellness", "gut-health", "nutrition", "mental-health"],
                "source": "Nutritional Psychiatry",
                "difficulty": "beginner",
                "duration_minutes": 25
            },
            {
                "title": "Building Resilience",
                "content": "Develop strong relationships, realistic plans, self-care, positive view of self, communication skills, emotional regulation.",
                "category": "wellness",
                "resource_type": "guide",
                "tags": ["wellness", "resilience", "coping", "growth"],
                "source": "Resilience Research",
                "difficulty": "intermediate",
                "duration_minutes": 40
            },
            {
                "title": "Hobbies and Mental Health",
                "content": "Engage in activities for enjoyment, not achievement. Promotes flow state, reduces stress, provides meaning, builds confidence.",
                "category": "wellness",
                "resource_type": "guide",
                "tags": ["wellness", "hobbies", "enjoyment", "flow"],
                "source": "Recreation Therapy",
                "difficulty": "beginner",
                "duration_minutes": 20
            },
            {
                "title": "Volunteering Benefits",
                "content": "Helps others, builds purpose, social connection, perspective, skills. Choose cause you care about. Start small.",
                "category": "wellness",
                "resource_type": "guide",
                "tags": ["wellness", "volunteering", "purpose", "connection"],
                "source": "Volunteer Studies",
                "difficulty": "beginner",
                "duration_minutes": 25
            },
            {
                "title": "Life Purpose Exploration",
                "content": "Reflect on values, passions, strengths. What matters to you? What impact do you want? Purpose evolves over time",
                "category": "wellness",
                "resource_type": "exercise",
                "tags": ["wellness", "purpose", "meaning", "values"],
                "source": "Existential Psychology",
                "difficulty": "intermediate",
                "duration_minutes": 50
            },
        ]
        
        return additional

    def upload_resources(self, resources: List[Dict], batch_size: int = 50):
        """Upload resources to Qdrant with embeddings"""
        
        console.print(f"\n[bold cyan]Uploading {len(resources)} resources to Qdrant...[/bold cyan]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            
            task = progress.add_task("Generating embeddings and uploading...", total=len(resources))
            
            for i in range(0, len(resources), batch_size):
                batch = resources[i:i + batch_size]
                points = []
                
                for resource in batch:
                    # Create searchable text by combining title and content
                    searchable_text = f"{resource['title']}. {resource['content']}"
                    
                    # Generate embedding
                    embedding = self.encoder.encode(searchable_text).tolist()
                    
                    # Create point
                    point = PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload={
                            "title": resource["title"],
                            "content": resource["content"],
                            "category": resource["category"],
                            "resource_type": resource["resource_type"],
                            "tags": resource["tags"],
                            "source": resource["source"],
                            "difficulty": resource["difficulty"],
                            "duration_minutes": resource["duration_minutes"]
                        }
                    )
                    points.append(point)
                
                # Upload batch
                self.client.upsert(
                    collection_name=RESOURCES_COLLECTION,
                    points=points
                )
                
                progress.update(task, advance=len(batch))
        
        console.print(f"\n[bold green]✓ Successfully uploaded {len(resources)} resources![/bold green]")
        
        # Display statistics
        self._display_statistics()

    def _display_statistics(self):
        """Display statistics about uploaded resources"""
        
        collection_info = self.client.get_collection(RESOURCES_COLLECTION)
        
        console.print(f"\n[bold cyan]Collection Statistics:[/bold cyan]")
        console.print(f"Total Resources: {collection_info.points_count}")
        
        # Count by category (you'd need to scroll through points for exact counts)
        console.print("\n[bold]Resources are distributed across categories:[/bold]")
        console.print("  • Depression & Mood")
        console.print("  • Anxiety & Panic")
        console.print("  • Stress Management")
        console.print("  • Self-Harm & Crisis (including alternatives and harm reduction)")
        console.print("  • Mindfulness & Meditation")
        console.print("  • Sleep & Rest")
        console.print("  • Relationships & Social")
        console.print("  • Trauma & PTSD")
        console.print("  • Self-Esteem & Confidence")
        console.print("  • General Wellness")
        console.print("  • OCD, Bipolar, Eating Disorders, Addiction, ADHD, Grief, Anger")
        console.print("  • Loneliness, Perfectionism, Chronic Pain, and more")

    def search_example(self, query: str, limit: int = 5):
        """Example search to test the uploaded resources"""
        
        console.print(f"\n[bold cyan]Searching for: '{query}'[/bold cyan]\n")
        
        # Generate query embedding
        query_embedding = self.encoder.encode(query).tolist()
        
        # Search
        results = self.client.search(
            collection_name=RESOURCES_COLLECTION,
            query_vector=query_embedding,
            limit=limit
        )
        
        # Display results
        for idx, result in enumerate(results, 1):
            console.print(f"[bold]{idx}. {result.payload['title']}[/bold]")
            console.print(f"   Category: {result.payload['category']} | Difficulty: {result.payload['difficulty']}")
            console.print(f"   Tags: {', '.join(result.payload['tags'])}")
            console.print(f"   Score: {result.score:.4f}")
            console.print(f"   {result.payload['content'][:150]}...")
            console.print()
def main():
    try:
        # Initialize
        generator = MentalHealthResourceGenerator()
        
        console.print("[bold cyan]Generating 200 mental health resources...[/bold cyan]")
        resources = generator.generate_resources()
        console.print(f"[green]✓[/green] Generated {len(resources)} resources")
        generator.upload_resources(resources)
        
        # Run example searches
        console.print("\n[bold cyan]Testing with example searches:[/bold cyan]")
        
        console.print("\n[bold green]✓ All done! Mental health resources are ready to use.[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error: {str(e)}[/bold red]")
        raise
if __name__ == "__main__":
    main()