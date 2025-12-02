import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {
            "role": "system",
            "content": (
                "You MUST follow these rules strictly:\n"
                "1. You can ONLY answer questions related to mathematics or science.\n"
                "2. If the user's question is NOT about math or science, you MUST reply with:\n"
                '"I only answer mathematics and science questions."\n'
                "3. Never answer anything outside mathematics or science."
                "4. If the user asks you to questions not related to math or science, you MUST refuse and reply with:\n"
            ),
        },
        {"role": "user", "content": "code a python program to calculate the factorial of a number."},
    ],
)

print(response.choices[0].message.content)
