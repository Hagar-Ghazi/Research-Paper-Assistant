"""
Single source of truth for every setting in the project
All other files import from here change a value once it updates everywhere 
"""

from pathlib import Path

# Path
BASE_DIR       = Path(__file__).parent          
CACHE_DIR      = BASE_DIR / "cache"             
PDF_DIR        = BASE_DIR / "pdfs"             
INGESTED_FILE  = BASE_DIR / "ingested.txt"      # one paper_id per line


# arXiv topics to fetch (https://arxiv.org/category_taxonomy)
ARXIV_TOPICS = [
    "cs.AI",    # AI
    "cs.LG",    # ML
    "cs.CL",    # NLP
]

MAX_PAPERS_PER_TOPIC = 5 

# Chunking
CHUNK_SIZE    = 600    # target characters per chunk
CHUNK_OVERLAP = 80     # overlap between consecutive chunks


# Embedding model
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# Ollama
OLLAMA_MODEL = "aya-expanse"         
OLLAMA_EMBED_MODEL  = "nomic-embed-text"        # used if i switch from sentence-transformers
OLLAMA_BASE_URL     = "http://localhost:11434"


# Retrieval 
TOP_K            = 6      # number of chunks returned per query
SCORE_THRESHOLD  = 0.40   # minimum cosine similarity to keep a FAISS result
RRF_K            = 60     # RRF constant higher = softer rank penalty


# Streamlit UI 
APP_TITLE    = "Research Paper Assistant"
APP_ICON     = "📄"
