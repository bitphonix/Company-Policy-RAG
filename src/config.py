import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
TESTS_DIR = BASE_DIR / "tests"

for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, CHROMA_DB_DIR, TESTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  


CHUNK_SIZE = 800  
CHUNK_OVERLAP = 200  

# Alternative chunking strategies we'll implement:
CHUNKING_STRATEGIES = {
    "recursive": {  # Default - splits on paragraphs, then sentences
        "chunk_size": 800,
        "chunk_overlap": 200,
    },
    "semantic": {  # Splits based on semantic similarity (advanced)
        "buffer_size": 1,  # Number of sentences to group
        "breakpoint_threshold": 0.5,  # Similarity threshold
    },
    "fixed": {  # Simple fixed-size chunks (baseline)
        "chunk_size": 500,
        "chunk_overlap": 100,
    }
}

# ============================================================================
# VECTOR STORE CONFIGURATION (CHROMA)
# ============================================================================
COLLECTION_NAME = "company_policies"  # Main collection name
CHROMA_PERSIST_DIRECTORY = str(CHROMA_DB_DIR)

# Distance metric for similarity search
# Options: "cosine" (default), "l2", "ip" (inner product)
# Cosine is best for normalized embeddings
DISTANCE_METRIC = "cosine"

# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================
# How many chunks to retrieve initially
# Why 5-10? More chunks = more context but also more noise
TOP_K_RETRIEVAL = 8

# MMR (Maximal Marginal Relevance) parameters
# MMR balances relevance with diversity
# lambda=1: Only relevance, lambda=0: Only diversity
MMR_LAMBDA = 0.7  # 70% relevance, 30% diversity
MMR_FETCH_K = 20  # Fetch more, then re-rank with MMR

# Hybrid search weights
# How much to weight semantic vs keyword search
SEMANTIC_WEIGHT = 0.7  # 70% semantic
KEYWORD_WEIGHT = 0.3   # 30% keyword (BM25)

# Similarity threshold - minimum similarity score to consider
# Range: 0-1, higher = more strict
SIMILARITY_THRESHOLD = 0.5

# ============================================================================
# LLM CONFIGURATION (OPENAI)
# ============================================================================
# GPT-4 for high-quality answers
# For cost savings, could use "gpt-3.5-turbo" 
LLM_MODEL_NAME = "gpt-4o-mini"  # Good balance of cost/quality
LLM_TEMPERATURE = 0.1  # Low temperature for factual answers
LLM_MAX_TOKENS = 1000  # Max response length

# Streaming for better UX
STREAM_RESPONSES = True

# ============================================================================
# RERANKING CONFIGURATION
# ============================================================================
# Cross-encoder for reranking retrieved chunks
# Why rerank? Initial retrieval might miss nuances
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_K = 5  # Rerank top results to this many

# ============================================================================
# PROMPT TEMPLATES
# ============================================================================
# System prompt for the RAG assistant
SYSTEM_PROMPT = """You are a helpful HR assistant for our company. Your role is to answer questions about company policies accurately and helpfully.

Guidelines:
1. ONLY use information from the provided context (policy documents)
2. If you're not sure or the information isn't in the context, say "I don't have that information in the policy documents"
3. Always cite which document/section you're referencing
4. Be friendly and professional
5. If policies conflict or are unclear, mention this
6. For sensitive topics (termination, legal issues), remind users to consult HR directly

When answering:
- Be concise but complete
- Use bullet points for lists
- Quote exact policy text when important
- Provide context and examples when helpful
"""

# Template for answering with sources
QA_PROMPT_TEMPLATE = """Context from company policies:
{context}

Question: {question}

Please provide a helpful answer based on the policy documents above. Include:
1. Direct answer to the question
2. Which document(s) you're referencing
3. Any important caveats or related information

Answer:"""

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
# For testing retrieval quality
EVAL_METRICS = {
    "precision_at_k": [1, 3, 5],  # Precision at different K values
    "mrr": True,  # Mean Reciprocal Rank
    "ndcg": True,  # Normalized Discounted Cumulative Gain
}

# ============================================================================
# CACHE CONFIGURATION
# ============================================================================
# Cache frequent queries to save API calls
ENABLE_CACHE = True
CACHE_TTL = 3600  # Cache time-to-live in seconds (1 hour)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = BASE_DIR / "app.log"

# ============================================================================
# UI CONFIGURATION (STREAMLIT)
# ============================================================================
APP_TITLE = "🏢 Company Policy Assistant"
APP_ICON = "🤖"
PAGE_CONFIG = {
    "page_title": APP_TITLE,
    "page_icon": APP_ICON,
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_pdf_files():
    """Get all PDF files from raw data directory"""
    return list(RAW_DATA_DIR.glob("*.pdf"))

def print_config_summary():
    """Print configuration summary for debugging"""
    print("=" * 60)
    print("COMPANY POLICY RAG - CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"PDF Files Found: {len(get_pdf_files())}")
    print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
    print(f"Chunk Size: {CHUNK_SIZE} chars (overlap: {CHUNK_OVERLAP})")
    print(f"LLM Model: {LLM_MODEL_NAME}")
    print(f"Top-K Retrieval: {TOP_K_RETRIEVAL}")
    print(f"Chroma DB Location: {CHROMA_PERSIST_DIRECTORY}")
    print("=" * 60)

if __name__ == "__main__":
    print_config_summary()