# --- Python RAG Microservice with FastAPI ---
# This microservice receives a fileId from the Node.js backend,
# retrieves the file from MongoDB GridFS, processes it, and indexes
# the chunks in Qdrant for retrieval.

# Requirements:
# Install the following packages:
# pip install fastapi uvicorn "python-dotenv[extra]" "pymongo[srv]" PyMuPDF tiktoken langchain-text-splitters qdrant-client sentence-transformers

import os
import json
import io
from typing import List, Dict, Any
from dotenv import load_dotenv
import uvicorn

from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel

# MongoDB specific imports
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from gridfs import GridFS
from bson.objectid import ObjectId

# Qdrant specific imports
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

# RAG Pipeline imports (adapted from previous notebook)
import fitz
import tiktoken
import math
import re
from collections import Counter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Environment Variables ---
# Load environment variables from .env file
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") # This is for a hosted Qdrant. Using :memory: below.
# The URL for this service, used by the Node.js backend.
# PYTHON_SERVICE_URL = os.getenv("PYTHON_SERVICE_URL")

# --- FastAPI App Initialization ---
app = FastAPI(title="RAG Microservice", version="1.0.0")

# --- Global Services (database connections, models) ---
db_client = None
db_instance = None
fs = None
qdrant_client = None
embedding_model = None

# --- RAG Pipeline Configuration (same as previous script) ---
CHUNK_SIZE_TOKENS = 512
OVERLAP_PERCENTAGE = 0.15
ENCODING_NAME = "cl100k_base"
MAX_LINES_TO_CHECK = 5
REPETITION_THRESHOLD_PERCENT = 70

ENCODER = tiktoken.get_encoding(ENCODING_NAME)
COLLECTION_NAME = "policy-documents"

# --- RAG Pipeline Functions (adapted for in-memory file processing) ---

def count_tokens(text: str) -> int:
    """Counts tokens using the global tiktoken encoder."""
    return len(ENCODER.encode(text))

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> List[Dict]:
    """
    Extracts text content page by page from PDF bytes (in-memory).

    Args:
        pdf_bytes (bytes): The raw bytes of the PDF document.

    Returns:
        list[dict]: A list of dictionaries, each containing 'page_num' and 'text'.
    """
    pages_content = []
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text = page.get_text("text")
            pages_content.append({"page_num": page_num + 1, "text": text})
        document.close()
    except Exception as e:
        print(f"Error reading PDF from bytes: {e}")
    return pages_content

# The following functions (identify_common_page_elements, remove_identified_elements,
# and chunk_text_with_metadata) are identical to the ones in the previous notebook.
# They are included here for completeness.

def identify_common_page_elements(all_pages_content: dict[str, list[dict]],
                                   max_lines: int = MAX_LINES_TO_CHECK,
                                   repetition_threshold_percent: int = REPETITION_THRESHOLD_PERCENT) -> tuple[set, set]:
    """Analyzes text from multiple pages (excluding first pages) to identify common header and footer lines."""
    header_candidates = Counter()
    footer_candidates = Counter()
    total_non_first_pages = 0
    for doc_id, pages_data in all_pages_content.items():
        for page_data in pages_data:
            page_num = page_data['page_num']
            page_text = page_data['text']
            if page_num == 1: continue
            total_non_first_pages += 1
            lines = [line.strip() for line in page_text.split('\n') if line.strip()]
            for i in range(min(max_lines, len(lines))):
                header_candidates[lines[i]] += 1
            for i in range(max(0, len(lines) - max_lines), len(lines)):
                footer_candidates[lines[i]] += 1
    common_header_lines = set()
    common_footer_lines = set()
    if total_non_first_pages == 0: return common_header_lines, common_footer_lines
    threshold_count = math.ceil(total_non_first_pages * (repetition_threshold_percent / 100))
    for line, count in header_candidates.items():
        if count >= threshold_count: common_header_lines.add(line)
    for line, count in footer_candidates.items():
        if count >= threshold_count and (re.fullmatch(r'\s*\d+\s*', line) or re.fullmatch(r'Page\s+\d+\s*(of\s+\d+)?', line, re.IGNORECASE)):
            common_footer_lines.add(line)
    return common_header_lines, common_footer_lines

def remove_identified_elements(page_text: str, page_num: int,
                               common_header_lines: set, common_footer_lines: set) -> str:
    """Removes identified common header and footer lines from a page's text."""
    if page_num == 1: return page_text
    lines = [line.strip() for line in page_text.split('\n')]
    final_lines = []
    temp_lines = []
    for i, line in enumerate(lines):
        if line in common_header_lines and i < MAX_LINES_TO_CHECK: continue
        else: temp_lines.append(line)
    footer_check_start_index = max(0, len(temp_lines) - MAX_LINES_TO_CHECK)
    for i, line in enumerate(temp_lines):
        if line in common_footer_lines and i >= footer_check_start_index: continue
        else: final_lines.append(line)
    return "\n".join(line for line in final_lines if line.strip() != "")

def chunk_text_with_metadata(text: str, chunk_size_tokens: int, overlap_percentage: float,
                             doc_id: str, page: int, base_clause_id_prefix: str):
    """Splits a given text into chunks using RecursiveCharacterTextSplitter and adds metadata."""
    avg_chars_per_token = 4
    chunk_size_chars = chunk_size_tokens * avg_chars_per_token
    overlap_chars = math.floor(chunk_size_chars * overlap_percentage)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_chars,
        chunk_overlap=overlap_chars,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    raw_chunks = text_splitter.split_text(text)
    processed_chunks = []
    for i, chunk_content in enumerate(raw_chunks):
        token_length = count_tokens(chunk_content)
        clause_id = f"{base_clause_id_prefix}-{doc_id}-p{page}-c{i + 1}"
        metadata = {
            "doc_id": doc_id,
            "page": page,
            "clause_id": clause_id,
            "chunk_length_tokens": token_length,
            "chunk_length_chars": len(chunk_content)
        }
        processed_chunks.append({"content": chunk_content, "metadata": metadata})
    return processed_chunks

def process_single_pdf_bytes(pdf_bytes: bytes, doc_id: str) -> List[Dict]:
    """
    Processes a single PDF from bytes, cleans it, and chunks the content.
    This function combines the logic from the notebook into a single, reusable step.
    """
    pages_content = extract_text_from_pdf_bytes(pdf_bytes)
    if not pages_content:
        raise ValueError("Failed to extract content from PDF.")
        
    all_docs_pages_content = {doc_id: pages_content}
    common_header_lines, common_footer_lines = identify_common_page_elements(all_docs_pages_content)
    
    all_processed_chunks = []
    for page_data in pages_content:
        cleaned_page_text = remove_identified_elements(
            page_data['text'], page_data['page_num'], common_header_lines, common_footer_lines
        )
        if cleaned_page_text.strip():
            page_chunks = chunk_text_with_metadata(
                text=cleaned_page_text,
                chunk_size_tokens=CHUNK_SIZE_TOKENS,
                overlap_percentage=OVERLAP_PERCENTAGE,
                doc_id=doc_id,
                page=page_data['page_num'],
                base_clause_id_prefix="Clause"
            )
            all_processed_chunks.extend(page_chunks)
            
    return all_processed_chunks

# --- Database & Model Initialization ---

async def connect_db():
    """Initializes MongoDB and Qdrant clients."""
    global db_client, db_instance, fs, qdrant_client, embedding_model
    try:
        if MONGO_URI is None:
            raise ValueError("MONGO_URI not set in environment variables.")

        db_client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
        db_instance = db_client.get_database("bajaj_documents") # Get the database instance with specific database name
        fs = GridFS(db_instance)
        print("✅ MongoDB connected successfully.")

        # Initialize Qdrant client (in-memory for this example)
        qdrant_client = QdrantClient(":memory:")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
        
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=embedding_model.get_sentence_embedding_dimension(), distance=models.Distance.COSINE),
        )
        print("✅ Qdrant client and embedding model initialized.")
    except Exception as e:
        print(f"❌ Failed to connect to databases or load models: {e}")
        db_client = None
        raise HTTPException(status_code=500, detail="Database connection or model loading failed")

def get_db():
    """Dependency injection for MongoDB connection."""
    if db_client is None:
        raise HTTPException(status_code=500, detail="Database client not initialized")
    return db_client

def get_qdrant_client():
    """Dependency injection for Qdrant client."""
    if qdrant_client is None:
        raise HTTPException(status_code=500, detail="Qdrant client not initialized")
    return qdrant_client

def get_embedding_model():
    """Dependency injection for SentenceTransformer model."""
    if embedding_model is None:
        raise HTTPException(status_code=500, detail="Embedding model not initialized")
    return embedding_model

# --- API Endpoints ---

@app.get("/")
async def root():
    """Test endpoint to verify the service is running."""
    return {"message": "🚀 FastAPI RAG Microservice is running!", "endpoints": {
        "process_file": "/process-file",
        "docs": "/docs"
    }}

class ProcessFileRequest(BaseModel):
    fileId: str

@app.post("/process-file")
async def process_file(request: ProcessFileRequest, db=Depends(get_db), qdrant=Depends(get_qdrant_client), embed_model=Depends(get_embedding_model)):
    """
    Endpoint to process a file by its ID from MongoDB GridFS.
    It retrieves, chunks, embeds, and indexes the document.
    """
    file_id = request.fileId
    print(f"Received request to process file with ID: {file_id}")
    
    try:
        # 1. Retrieve file from GridFS
        file_document = fs.find_one({"_id": ObjectId(file_id)})
        if not file_document:
            raise HTTPException(status_code=404, detail="File not found in GridFS")
            
        file_bytes_stream = fs.get(ObjectId(file_id))
        file_bytes = file_bytes_stream.read()
        file_name = file_document.filename

        print(f"Successfully retrieved file '{file_name}' from GridFS.")

        # 2. Process the PDF content
        processed_chunks = process_single_pdf_bytes(file_bytes, doc_id=file_id)

        if not processed_chunks:
            raise HTTPException(status_code=500, detail="No chunks were generated from the document.")

        print(f"Successfully generated {len(processed_chunks)} chunks.")

        # 3. Embed and index the chunks in Qdrant
        points = []
        for chunk in processed_chunks:
            vector = embed_model.encode(chunk['content']).tolist()
            points.append(
                models.PointStruct(
                    id=chunk['metadata']['clause_id'],
                    vector=vector,
                    payload=chunk['metadata'] | {"content": chunk['content']}
                )
            )
        
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        
        print("Chunks have been embedded and indexed in Qdrant.")

        return {"status": "success", "message": "File processed and indexed successfully.", "fileId": file_id, "indexed_chunks": len(points)}
    
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


# --- Startup Event Handler ---
@app.on_event("startup")
async def startup_event():
    """Connect to databases on application startup."""
    try:
        await connect_db()
        print("✅ FastAPI service startup completed successfully!")
    except Exception as e:
        print(f"❌ Startup error: {e}")
        # Don't raise the exception, just log it

# --- Main Entry Point for Uvicorn ---
# This part is for local development. In a production environment,
# you would run this with a command like:
# uvicorn main:app --host 0.0.0.0 --port 8001
# The 'main' refers to the filename, and 'app' is the FastAPI instance.

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)