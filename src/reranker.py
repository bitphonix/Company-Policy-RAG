"""
Cross-Encoder Reranking for RAG

Why Reranking?
- Initial retrieval (bi-encoder) is fast but less accurate
- Reranking (cross-encoder) is slower but much more accurate
- Two-stage pipeline: Retrieve many → Rerank to best

Performance Impact:
- 15-30% improvement in precision@5
- Minimal latency increase (~200ms for 20 candidates)

How it works:
1. Retrieve 20-30 candidates with bi-encoder (fast)
2. Score each candidate with cross-encoder (accurate)
3. Return top-K reranked results
"""

import logging
from typing import List, Tuple
import numpy as np

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL, RERANK_TOP_K

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Reranker:
    """
    Cross-encoder reranker for improved retrieval precision
    
    Architecture:
    - Bi-encoder (embedding): Query and docs encoded separately → fast
    - Cross-encoder (reranker): Query+doc encoded together → accurate
    
    Use case:
    retrieval.retrieve(k=20) → reranker.rerank(top_k=5) → final results
    """
    
    def __init__(self, model_name: str = RERANKER_MODEL):
        """
        Initialize cross-encoder model
        
        Args:
            model_name: HuggingFace cross-encoder model
        
        Popular models:
        - ms-marco-MiniLM-L-6-v2: Fast, good quality
        - ms-marco-TinyBERT-L-2-v2: Faster, lower quality
        - ms-marco-MiniLM-L-12-v2: Slower, higher quality
        """
        logger.info(f"Loading cross-encoder: {model_name}")
        
        self.model = CrossEncoder(
            model_name,
            max_length=512,  # Max input length
            device='mps',  # Use 'cuda' if GPU, 'cpu' if no GPU/MPS
        )
        
        logger.info("✓ Cross-encoder loaded")
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = RERANK_TOP_K,
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents using cross-encoder
        
        Process:
        1. Create (query, doc) pairs
        2. Score each pair with cross-encoder
        3. Sort by score
        4. Return top-K
        
        Args:
            query: User's question
            documents: Retrieved documents (usually 20-30)
            top_k: Number of results to return after reranking
            
        Returns:
            List of (Document, score) tuples, sorted by score
        """
        if not documents:
            logger.warning("No documents to rerank")
            return []
        
        logger.info(f"Reranking {len(documents)} documents → top {top_k}")
        
        # Create (query, document) pairs
        pairs = [
            [query, doc.page_content]
            for doc in documents
        ]
        
        # Score all pairs
        # This is where the magic happens - cross-encoder sees query+doc together
        scores = self.model.predict(pairs)
        
        # Combine documents with scores
        doc_score_pairs = list(zip(documents, scores))
        
        # Sort by score (descending)
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-K
        reranked = doc_score_pairs[:top_k]
        
        logger.info(f"✓ Reranked. Top score: {reranked[0][1]:.3f}, "
                   f"Bottom score: {reranked[-1][1]:.3f}")
        
        return reranked
    
    def rerank_with_threshold(
        self,
        query: str,
        documents: List[Document],
        top_k: int = RERANK_TOP_K,
        threshold: float = 0.5,
    ) -> List[Tuple[Document, float]]:
        """
        Rerank with score threshold
        
        Returns fewer than top_k if scores are below threshold
        Useful for filtering out low-quality matches
        
        Args:
            query: User's question
            documents: Retrieved documents
            top_k: Max number of results
            threshold: Minimum score to include
            
        Returns:
            List of (Document, score) tuples above threshold
        """
        reranked = self.rerank(query, documents, top_k=len(documents))
        
        # Filter by threshold
        filtered = [
            (doc, score) for doc, score in reranked
            if score >= threshold
        ]
        
        # Limit to top_k
        return filtered[:top_k]
    
    def get_relevance_explanation(self, score: float) -> str:
        """
        Convert score to human-readable relevance
        
        Cross-encoder scores are typically in range [-10, 10]
        But can vary by model
        
        Args:
            score: Cross-encoder score
            
        Returns:
            Relevance description
        """
        if score > 5:
            return "Highly Relevant"
        elif score > 2:
            return "Relevant"
        elif score > 0:
            return "Somewhat Relevant"
        elif score > -2:
            return "Low Relevance"
        else:
            return "Not Relevant"


# ============================================================================
# INTEGRATION WITH EXISTING RETRIEVAL
# ============================================================================

class RerankedRetriever:
    """
    Wrapper that adds reranking to any retriever
    
    Usage:
    retriever = AdvancedRetriever(manager)
    reranked_retriever = RerankedRetriever(retriever, reranker)
    results = reranked_retriever.retrieve(query, k=5)  # Automatically reranked!
    """
    
    def __init__(self, base_retriever, reranker: Reranker):
        """
        Initialize with base retriever and reranker
        
        Args:
            base_retriever: Any retriever with .retrieve() method
            reranker: Reranker instance
        """
        self.base_retriever = base_retriever
        self.reranker = reranker
        logger.info("✓ Reranked retriever initialized")
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        strategy: str = None,
        fetch_k: int = 20,
    ):
        """
        Retrieve with automatic reranking
        
        Process:
        1. Retrieve fetch_k candidates (e.g., 20)
        2. Rerank to top k (e.g., 5)
        
        Args:
            query: User's question
            k: Final number of results
            strategy: Retrieval strategy
            fetch_k: Number to retrieve before reranking
            
        Returns:
            RetrievalResult with reranked documents
        """
        # Get more candidates than needed
        initial_result = self.base_retriever.retrieve(
            query,
            k=fetch_k,
            strategy=strategy,
        )
        
        # Rerank
        reranked_docs_scores = self.reranker.rerank(
            query,
            initial_result.documents,
            top_k=k,
        )
        
        # Unpack
        reranked_docs = [doc for doc, _ in reranked_docs_scores]
        reranked_scores = [score for _, score in reranked_docs_scores]
        
        # Update result
        from retrieval import RetrievalResult
        return RetrievalResult(
            documents=reranked_docs,
            scores=reranked_scores,
            strategy_used=f"{initial_result.strategy_used}_reranked",
            query_type=initial_result.query_type,
            total_retrieved=len(reranked_docs),
        )


# ============================================================================
# TESTING
# ============================================================================

def test_reranking():
    """
    Compare retrieval with and without reranking
    """
    from embedding import EmbeddingManager
    from retrieval import AdvancedRetriever
    
    print("\n" + "="*70)
    print("TESTING CROSS-ENCODER RERANKING")
    print("="*70)
    
    # Initialize components
    manager = EmbeddingManager()
    manager.load_vector_store()
    
    base_retriever = AdvancedRetriever(manager)
    reranker = Reranker()
    reranked_retriever = RerankedRetriever(base_retriever, reranker)
    
    # Test query
    query = "How do I claim expenses?"
    
    # Without reranking
    print(f"\nQuery: {query}")
    print("\n" + "-"*70)
    print("WITHOUT RERANKING:")
    print("-"*70)
    
    base_result = base_retriever.retrieve(query, k=5)
    for i, (doc, score) in enumerate(zip(base_result.documents, base_result.scores), 1):
        print(f"{i}. Score: {score:.3f}")
        print(f"   Source: {doc.metadata.get('source_file')}")
        print(f"   Preview: {doc.page_content[:100]}...")
        print()
    
    # With reranking
    print("-"*70)
    print("WITH RERANKING:")
    print("-"*70)
    
    reranked_result = reranked_retriever.retrieve(query, k=5, fetch_k=15)
    for i, (doc, score) in enumerate(zip(reranked_result.documents, reranked_result.scores), 1):
        relevance = reranker.get_relevance_explanation(score)
        print(f"{i}. Score: {score:.3f} ({relevance})")
        print(f"   Source: {doc.metadata.get('source_file')}")
        print(f"   Preview: {doc.page_content[:100]}...")
        print()
    
    print("="*70)
    print("\nKEY OBSERVATIONS:")
    print("1. Reranking may change the order of results")
    print("2. Scores are different (cross-encoder scores)")
    print("3. Most relevant docs should rise to the top")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_reranking()