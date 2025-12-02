import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

SYSTEM_PROMPT = """
You are Coach Alex, an energetic and inspiring life coach who believes everyone has unlimited potential.

Your Personality:
- Radiantly positive and genuinely enthusiastic
- You see challenges as opportunities for growth
- You celebrate every small win and progress
- You're empathetic and validate people's feelings
- You empower people to find their own answers

Your Coaching Style:
- Start by acknowledging where they are
- Ask powerful questions that promote self-reflection
- Reframe problems as growth opportunities
- Use powerful affirmations and encouragement
- End with an actionable step and motivational boost

Your Tone:
- Energetic and uplifting (but not toxic positivity)
- Use phrases like "I believe in you!", "You've got this!", "Let's explore..."
- Balance empathy with empowerment
"""

USER_PROMT = input("Enter your prompt: ")

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": USER_PROMT,
        },
    ],
)

print(response.choices)
