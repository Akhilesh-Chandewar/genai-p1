#!/usr/bin/env python3
"""
mem_gemini_qdrant.py

Gemini-only memory chat loop using mem0 + Qdrant.
Handles dimension mismatch by manually creating collection with correct dimensions.
"""

import json
import os
import time
from dotenv import load_dotenv

# mem0
from mem0 import Memory

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse

# Gemini (Google GenAI)
try:
    from google import genai
except Exception:
    import google.generativeai as genai  # type: ignore

load_dotenv()

# -------------------------
# Config / env vars
# -------------------------
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "mem_agent_collection_gemini")
USER_ID = os.getenv("MEM_USER_ID", "akhil")

# Gemini text-embedding-004 produces 768-dimensional vectors
EMBEDDING_DIMENSION = 768

if not GOOGLE_API_KEY:
    raise RuntimeError("Please set GEMINI_API_KEY in your .env file")

# Configure genai client
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception:
    try:
        genai_client = genai.Client(api_key=GOOGLE_API_KEY)  # type: ignore
        genai = genai_client  # type: ignore
    except Exception:
        raise RuntimeError(
            "Could not configure Google GenAI client. Ensure google-genai is installed."
        )

# mem0 config
config = {
    "version": "v1",
    "embedder": {
        "provider": "gemini",
        "config": {"api_key": GOOGLE_API_KEY, "model": "models/text-embedding-004"},
    },
    "llm": {
        "provider": "gemini",
        "config": {"api_key": GOOGLE_API_KEY, "model": "gemini-2.5-flash"},
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {"url": QDRANT_URL, "collection_name": QDRANT_COLLECTION},
    },
}


def ensure_collection_with_correct_dimensions():
    """
    Ensure Qdrant collection exists with correct dimensions (768).
    Deletes and recreates if dimensions don't match.
    """
    print(
        f"[qdrant] Checking/creating collection '{QDRANT_COLLECTION}' with dimension {EMBEDDING_DIMENSION}..."
    )
    qc = QdrantClient(url=QDRANT_URL)

    try:
        # Check if collection exists
        collections = qc.get_collections().collections
        collection_exists = any(c.name == QDRANT_COLLECTION for c in collections)

        if collection_exists:
            print(
                f"[qdrant] Collection '{QDRANT_COLLECTION}' exists. Checking dimensions..."
            )

            # Get collection info to check dimensions
            try:
                collection_info = qc.get_collection(QDRANT_COLLECTION)
                # Access the vector params
                if hasattr(collection_info.config, "params"):
                    current_dim = collection_info.config.params.vectors.size
                elif hasattr(collection_info, "config") and hasattr(
                    collection_info.config, "vector_size"
                ):
                    current_dim = collection_info.config.vector_size
                else:
                    # Try to get from vectors config
                    current_dim = (
                        collection_info.config.params.vectors.size
                        if hasattr(collection_info.config.params, "vectors")
                        else None
                    )

                print(f"[qdrant] Current dimension: {current_dim}")

                if current_dim != EMBEDDING_DIMENSION:
                    print(
                        f"[qdrant] Dimension mismatch! Expected {EMBEDDING_DIMENSION}, got {current_dim}. Deleting collection..."
                    )
                    qc.delete_collection(collection_name=QDRANT_COLLECTION)
                    time.sleep(1)
                    collection_exists = False
                else:
                    print(f"[qdrant] Dimensions match. Collection is ready.")
                    return qc
            except Exception as e:
                print(
                    f"[qdrant] Could not verify dimensions: {e}. Recreating collection..."
                )
                qc.delete_collection(collection_name=QDRANT_COLLECTION)
                time.sleep(1)
                collection_exists = False

        if not collection_exists:
            print(
                f"[qdrant] Creating collection with dimension {EMBEDDING_DIMENSION}..."
            )
            qc.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSION, distance=Distance.COSINE
                ),
            )
            print(f"[qdrant] Collection '{QDRANT_COLLECTION}' created successfully!")
            time.sleep(1)

        return qc

    except Exception as e:
        print(f"[error] Failed to ensure collection: {e}")
        raise


def create_mem_client():
    """Create a new Memory client from config."""
    print("[mem] Initializing Memory client...")
    return Memory.from_config(config)


def run_chat_loop():
    # Ensure collection is properly set up before starting
    print("\n=== Initializing Qdrant collection ===")
    try:
        ensure_collection_with_correct_dimensions()
    except Exception as e:
        print(f"[error] Failed to initialize Qdrant: {e}")
        return

    # Now create mem0 client - it should use the existing collection
    mem_client = create_mem_client()

    print("\n🤖 Chatbot with Memory (Gemini + Qdrant) is ready!")
    print("Type 'exit' to quit.\n")

    retry_count = 0
    MAX_RETRIES = 1

    while True:
        user_query = input("> ").strip()
        if not user_query:
            continue
        if user_query.lower() in ("exit", "quit", "bye"):
            print("👋 Goodbye!")
            break

        # Search for relevant memories
        try:
            search_memory = mem_client.search(query=user_query, user_id=USER_ID)
            retry_count = 0  # Reset on success
        except UnexpectedResponse as e:
            errmsg = str(e)
            print(f"[error] Qdrant search error: {errmsg}")

            if "Vector dimension error" in errmsg or "expected dim" in errmsg:
                if retry_count < MAX_RETRIES:
                    retry_count += 1
                    print(
                        f"[fix] Dimension mismatch detected. Recreating collection (attempt {retry_count}/{MAX_RETRIES})..."
                    )

                    try:
                        ensure_collection_with_correct_dimensions()
                        mem_client = create_mem_client()

                        # Retry search
                        search_memory = mem_client.search(
                            query=user_query, user_id=USER_ID
                        )
                        retry_count = 0  # Reset on success
                    except Exception as e2:
                        print(f"[error] Retry failed: {e2}")
                        if retry_count >= MAX_RETRIES:
                            print(
                                "[error] Max retries reached. Please check your configuration."
                            )
                            print(
                                f"[error] Ensure mem0 is using Gemini embeddings (768 dim), not OpenAI (1536 dim)"
                            )
                        continue
                else:
                    print("[error] Max retries reached. Skipping this query.")
                    retry_count = 0
                    continue
            else:
                print("[error] Unexpected Qdrant error.")
                continue
        except Exception as e:
            print(f"[error] Search failed: {e}")
            continue

        # Process search results
        results = (
            search_memory.get("results", [])
            if isinstance(search_memory, dict)
            else getattr(search_memory, "get", lambda k, d=None: d)("results", [])
        )
        if results is None:
            results = []

        memories = [f"ID: {m.get('id')}\nMemory: {m.get('memory')}" for m in results]

        if memories:
            print("\n[Found Memories]")
            for m in memories:
                print(m)
        else:
            print("\n[Found Memories] None")

        SYSTEM_PROMPT = f"""
You are an assistant that uses short user memories to help answer queries.
Here are relevant memories for this user (if any):
{json.dumps(memories, indent=2)}
"""

        # Use Gemini to answer
        ai_response_text = None
        try:
            # Build the prompt combining system context and user query
            full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {user_query}\n\nAssistant:"

            # Try different Gemini API patterns
            try:
                # Pattern 1: GenerativeModel (most common for google-genai)
                model = genai.GenerativeModel("gemini-2.5-flash")  # type: ignore
                resp = model.generate_content(full_prompt)
                ai_response_text = resp.text
            except AttributeError:
                try:
                    # Pattern 2: Direct generate_content on genai module
                    resp = genai.generate_content(
                        model="gemini-2.5-flash", contents=full_prompt
                    )
                    ai_response_text = resp.text
                except Exception:
                    # Pattern 3: Using google.generativeai (older pattern)
                    import google.generativeai as genai_alt

                    genai_alt.configure(api_key=GOOGLE_API_KEY)
                    model = genai_alt.GenerativeModel("gemini-2.5-flash")
                    resp = model.generate_content(full_prompt)
                    ai_response_text = resp.text

        except Exception as e:
            print(f"[error] Gemini call failed: {e}")
            ai_response_text = "Sorry — I had trouble contacting the LLM."

        print(f"\nAI: {ai_response_text}")

        # Save memory
        try:
            mem_client.add(
                user_id=USER_ID,
                messages=[
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": ai_response_text},
                ],
            )
            print("[mem] Memory saved.\n")
        except Exception as e:
            print(f"[error] Failed to save memory: {e}")
            if "dimension" in str(e).lower():
                print("[fix] Dimension error while saving. Recreating collection...")
                try:
                    ensure_collection_with_correct_dimensions()
                    mem_client = create_mem_client()
                    mem_client.add(
                        user_id=USER_ID,
                        messages=[
                            {"role": "user", "content": user_query},
                            {"role": "assistant", "content": ai_response_text},
                        ],
                    )
                    print("[mem] Memory saved after recreation.\n")
                except Exception as e2:
                    print(f"[error] Retry save failed: {e2}")


if __name__ == "__main__":
    run_chat_loop()
