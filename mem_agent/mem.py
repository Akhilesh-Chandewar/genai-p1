from dotenv import load_dotenv
from mem0 import Memory
import os
import json
from openai import OpenAI

load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

client = OpenAI(
    api_key=GOOGLE_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

config = {
    "version": "v1.1",
    "embedder": {
        "type": "gemini",
        "config": {"api_key": GOOGLE_API_KEY, "model": "models/text-embedding-004"},
    },
   "llm": {
        "provider": "gemini",
        "config": {
            "model": "gemini-2.0-flash-001",
            "temperature": 0.2,
            "max_tokens": 2000,
            "top_p": 1.0
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {"host": "localhost", "port": 6333},
    },
}

memory_client = Memory.from_config(config)

while True:

    user_query = input("> ")

    search_memory = memory_client.search(
        query=user_query,
        user_id="akhil",
    )

    memories = [
        f"ID: {mem.get("id")}\nMemory: {mem.get("memory")}"
        for mem in search_memory.get("results")
    ]

    print("Found Memories", memories)

    SYSTEM_PROMPT = f"""
        Here is the context about the user:
        {json.dumps(memories)}
    """

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ],
    )

    print("Response:", response.choices[0].message.content)

    memory_client.add_memory(
        user_id="akhil",
        memory=user_query,
        response=response.choices[0].message.content,
    )

    print("Memory added successfully.")
