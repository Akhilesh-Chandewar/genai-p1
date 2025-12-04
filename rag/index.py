from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Fetch values from .env
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

# Set API key for Gemini
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

pdf_path = Path(__file__).parent / "nodejs.pdf"

# Load PDF
loader = PyPDFLoader(file_path=str(pdf_path))
docs = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)
chunks = text_splitter.split_documents(docs)

# Gemini Embedding Model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

# Qdrant Vector Store
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url=QDRANT_URL,
    collection_name="learning_rag_gemini",
)

print("Indexing of documents done with Gemini....")
