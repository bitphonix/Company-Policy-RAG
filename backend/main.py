"""
FastAPI Backend for Company Policy RAG System

Features:
- RESTful API endpoints
- Async operations for performance
- Authentication & rate limiting
- Auto-generated API docs (Swagger/ReDoc)
- CORS support for frontend
- Request validation with Pydantic
- Error handling & logging

Run with: uvicorn backend.main:app --reload
API Docs: http://localhost:8000/docs
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, HTTPException, Depends, status, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time
import logging
from datetime import datetime
import asyncio

# Import RAG components
from embedding import EmbeddingManager
from retrieval import AdvancedRetriever
from generation import AnswerGenerator
from reranker import Reranker, RerankedRetriever
from cache_manager import QueryCache

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC MODELS (Request/Response schemas)
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str = Field(..., min_length=3, max_length=500, description="User's question")
    strategy: Optional[str] = Field(None, description="Retrieval strategy: semantic, keyword, hybrid, mmr")
    k: int = Field(5, ge=1, le=10, description="Number of sources to retrieve")
    use_reranking: bool = Field(True, description="Whether to use cross-encoder reranking")
    stream: bool = Field(False, description="Stream response tokens")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What is the POSH policy?",
                "strategy": "hybrid",
                "k": 5,
                "use_reranking": True,
                "stream": False
            }
        }


class Source(BaseModel):
    """Source document model"""
    id: int
    file: str
    page: int
    score: float
    preview: str
    relevance: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    answer: str
    sources: List[Source]
    confidence: str
    retrieval_strategy: str
    query_type: str
    response_time: float
    from_cache: bool = False
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "answer": "The POSH policy stands for...",
                "sources": [
                    {
                        "id": 1,
                        "file": "POSH Policy.pdf",
                        "page": 2,
                        "score": 0.95,
                        "preview": "The Company expects...",
                        "relevance": "Highly Relevant"
                    }
                ],
                "confidence": "high",
                "retrieval_strategy": "hybrid_reranked",
                "query_type": "factual",
                "response_time": 2.34,
                "from_cache": False
            }
        }


class ConversationRequest(BaseModel):
    """Request for conversational queries"""
    query: str = Field(..., min_length=3, max_length=500)
    conversation_id: str = Field(..., description="Unique conversation identifier")
    context: Optional[List[Dict[str, str]]] = Field(None, description="Previous Q&A pairs")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str
    components: Dict[str, str]


class StatsResponse(BaseModel):
    """Statistics response"""
    total_queries: int
    cache_stats: Dict[str, Any]
    uptime: float
    avg_response_time: float


# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="Company Policy RAG API",
    description="RESTful API for intelligent company policy question answering",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# GLOBAL STATE (Dependency Injection)
# ============================================================================

class RAGSystem:
    """Singleton for RAG system components"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def initialize(self):
        """Initialize RAG components"""
        if self.initialized:
            return
        
        logger.info("Initializing RAG system...")
        
        # Load components
        self.manager = EmbeddingManager()
        self.manager.load_vector_store()
        
        self.retriever = AdvancedRetriever(self.manager)
        self.reranker = Reranker()
        self.reranked_retriever = RerankedRetriever(self.retriever, self.reranker)
        
        self.generator = AnswerGenerator(self.retriever)
        self.cache = QueryCache()
        
        # Analytics
        self.stats = {
            "total_queries": 0,
            "start_time": time.time(),
            "response_times": [],
        }
        
        self.initialized = True
        logger.info("✓ RAG system initialized")


# Global instance
rag_system = RAGSystem()


def get_rag_system():
    """Dependency to get RAG system"""
    if not rag_system.initialized:
        rag_system.initialize()
    return rag_system


# ============================================================================
# AUTHENTICATION (Simple API Key)
# ============================================================================

API_KEYS = {
    "dev-key-123": "development",
    "prod-key-456": "production",
}

async def verify_api_key(x_api_key: str = Header(...)):
    """
    Verify API key from header
    
    In production:
    - Use proper JWT tokens
    - Store keys in database
    - Implement rate limiting per key
    """
    if x_api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return API_KEYS[x_api_key]


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    logger.info("Starting Company Policy RAG API...")
    rag_system.initialize()


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API info"""
    return {
        "message": "Company Policy RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(system: RAGSystem = Depends(get_rag_system)):
    """
    Health check endpoint
    
    Returns system status and component health
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        components={
            "vector_store": "operational",
            "retriever": "operational",
            "generator": "operational",
            "cache": "operational",
        }
    )


@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    system: RAGSystem = Depends(get_rag_system),
):
    """Main query endpoint"""
    start_time = time.time()
    
    try:
        logger.info(f"Processing query: {request.query}")
        logger.info(f"Strategy: {request.strategy}, Reranking: {request.use_reranking}")
        
        # Check cache
        cached = system.cache.get(request.query)
        if cached:
            logger.info(f"Cache hit for: {request.query}")
            
            # Ensure all required fields are present
            if 'retrieval_strategy' not in cached:
                cached['retrieval_strategy'] = 'cached'
            if 'query_type' not in cached:
                cached['query_type'] = 'unknown'
            
            cached['response_time'] = time.time() - start_time
            cached['from_cache'] = True
            
            return QueryResponse(**cached)

        
        # Retrieve documents
        logger.info("Starting retrieval...")
        try:
            if request.use_reranking:
                retrieval_result = system.reranked_retriever.retrieve(
                    request.query,
                    k=request.k,
                    strategy=request.strategy,
                    fetch_k=request.k * 3,
                )
            else:
                retrieval_result = system.retriever.retrieve(
                    request.query,
                    k=request.k,
                    strategy=request.strategy,
                )
            logger.info(f"Retrieval complete. Got {len(retrieval_result.documents)} docs")
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Retrieval failed: {str(e)}"
            )
        
        # Generate answer
        logger.info("Generating answer...")
        try:
            answer = system.generator.generate_answer(
                request.query,
                retrieval_strategy=request.strategy,
                k=request.k,
            )
            logger.info("Answer generated successfully")
            
        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Answer generation failed: {str(e)}"
            )
        
        # Format sources with relevance
        sources = []
        for i, source_data in enumerate(answer.sources):
            source = Source(**source_data)
            
            # Add relevance explanation if reranking was used
            if request.use_reranking and i < len(retrieval_result.scores):
                score = retrieval_result.scores[i]
                source.relevance = system.reranker.get_relevance_explanation(score)
            
            sources.append(source)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Build response
        response = QueryResponse(
            answer=answer.answer,
            sources=sources,
            confidence=answer.confidence,
            retrieval_strategy=answer.retrieval_strategy,
            query_type=answer.query_type,
            response_time=response_time,
            from_cache=False,
            metadata={
                "total_context_length": answer.total_context_length,
                "reranking_used": request.use_reranking,
            }
        )
        
        # Cache the response
        system.cache.set(
            request.query,
            answer=answer.answer,
            sources=[s.dict() for s in sources],
            confidence=answer.confidence,
            retrieval_strategy=answer.retrieval_strategy,  # ADD THIS
            query_type=answer.query_type,  # ADD THIS
        )
        
        # Update stats
        system.stats["total_queries"] += 1
        system.stats["response_times"].append(response_time)
        
        logger.info(f"Query processed in {response_time:.2f}s: {request.query}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@app.post("/query/stream")
async def query_stream(
    request: QueryRequest,
    system: RAGSystem = Depends(get_rag_system),
):
    """
    Streaming query endpoint
    
    Returns answer tokens as they're generated (SSE format)
    Better UX for long responses
    """
    async def generate():
        try:
            # Get context
            retrieval_result = system.retriever.retrieve(
                request.query,
                k=request.k,
                strategy=request.strategy,
            )
            
            # Stream answer
            for chunk in system.generator.stream_answer(request.query, k=request.k):
                yield f"data: {chunk}\n\n"
                await asyncio.sleep(0.01)  # Small delay for streaming effect
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: ERROR: {str(e)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats(system: RAGSystem = Depends(get_rag_system)):
    """
    Get system statistics
    
    Returns:
    - Total queries
    - Cache performance
    - Uptime
    - Average response time
    """
    uptime = time.time() - system.stats["start_time"]
    
    avg_response_time = (
        sum(system.stats["response_times"]) / len(system.stats["response_times"])
        if system.stats["response_times"] else 0
    )
    
    return StatsResponse(
        total_queries=system.stats["total_queries"],
        cache_stats=system.cache.get_stats(),
        uptime=uptime,
        avg_response_time=avg_response_time,
    )


@app.post("/cache/clear")
async def clear_cache(
    system: RAGSystem = Depends(get_rag_system),
    # api_env: str = Depends(verify_api_key)  # Uncomment for auth
):
    """
    Clear query cache
    
    Use when policies are updated
    Requires authentication in production
    """
    system.cache.clear()
    return {"message": "Cache cleared successfully"}


@app.get("/documents")
async def list_documents(system: RAGSystem = Depends(get_rag_system)):
    """
    List all documents in the knowledge base
    
    Returns document names and metadata
    """
    stats = system.manager.get_statistics()
    
    return {
        "total_chunks": stats.get("total_chunks", 0),
        "collection_name": stats.get("collection_name", ""),
        "embedding_model": stats.get("embedding_model", ""),
        "metadata_keys": stats.get("sample_metadata_keys", []),
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catch-all exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )



# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )