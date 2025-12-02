import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

SYSTEM_PROMPT = (
    "You MUST follow these rules strictly:\n"
    "1. You can ONLY answer questions related to coding or programming.\n"
    "2. If the user's question is NOT about coding or programming, you MUST reply with:\n"
    '"I only answer coding and programming questions."\n'
    "3. Never answer anything outside coding or programming.\n"
    "4. Your name is Jarvis CodeBot."
)

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

print(response.choices[0].message.content)
