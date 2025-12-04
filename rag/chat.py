from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

# Set API key for Gemini embeddings
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

# Load existing Qdrant vector collection
vector_store = QdrantVectorStore.from_existing_collection(
    url=QDRANT_URL,
    collection_name="learning_rag_gemini",
    embedding=embedding_model,
)

# Gemini OpenAI-Compatible Client
client = OpenAI(
    api_key=GOOGLE_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Take user input
user_query = input("Ask something: ")

# Search relevant chunks
search_results = vector_store.similarity_search(query=user_query)

context = "\n\n\n".join([
    f"Page Content: {r.page_content}\n"
    f"Page Number: {r.metadata.get('page_label')}\n"
    f"File Location: {r.metadata.get('source')}"
    for r in search_results
])

SYSTEM_PROMPT = f"""
You are a helpful AI Assistant who answers user queries based ONLY on the context retrieved from the PDF.

Always cite the page number.

Context:
{context}
"""

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ],
)

print("\n🤖:", response.choices[0].message.content, "\n")
