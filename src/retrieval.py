"""
Advanced Retrieval Strategies for Company Policy RAG

This module implements multiple retrieval approaches:
1. Semantic Search - Vector similarity (what we just built)
2. Keyword Search (BM25) - Traditional search engine approach
3. Hybrid Search - Combines semantic + keyword
4. Query Routing - Different strategies for different questions
5. Reranking - Improve results with cross-encoders

Key Learning:
- No single retrieval method is perfect
- Hybrid approaches outperform pure semantic or keyword
- Query analysis helps choose the right strategy
- Reranking can dramatically improve precision
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import numpy as np

from embedding import EmbeddingManager
from config import (
    TOP_K_RETRIEVAL,
    SEMANTIC_WEIGHT,
    KEYWORD_WEIGHT,
    MMR_LAMBDA,
    MMR_FETCH_K,
    SIMILARITY_THRESHOLD,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryType(Enum):
    """
    Different types of queries require different retrieval strategies
    
    Why categorize?
    - Factual: "What is X?" → Needs precise match
    - Procedural: "How do I X?" → Needs step-by-step
    - Comparative: "X vs Y?" → Needs multiple sources
    - Open-ended: "Tell me about X" → Needs comprehensive coverage
    """
    FACTUAL = "factual"          # "What is the POSH policy?"
    PROCEDURAL = "procedural"    # "How do I claim expenses?"
    COMPARATIVE = "comparative"  # "Compare WFH vs office attendance"
    OPEN_ENDED = "open_ended"    # "Tell me about benefits"
    UNKNOWN = "unknown"


@dataclass
class RetrievalResult:
    """
    Enhanced result with metadata about retrieval
    
    Why track this?
    - Debug retrieval quality
    - Show confidence to users
    - A/B test strategies
    """
    documents: List[Document]
    scores: List[float]
    strategy_used: str
    query_type: QueryType
    total_retrieved: int


class QueryAnalyzer:
    """
    Analyzes queries to determine optimal retrieval strategy
    
    This is a simple rule-based approach
    Production systems use ML models for this
    """
    
    FACTUAL_KEYWORDS = ["what is", "define", "definition", "who is", "when is"]
    PROCEDURAL_KEYWORDS = ["how to", "how do i", "steps to", "process for", "procedure"]
    COMPARATIVE_KEYWORDS = ["compare", "difference between", "vs", "versus", "better"]
    
    @classmethod
    def analyze_query(cls, query: str) -> QueryType:
        """
        Determine query type from text
        
        Args:
            query: User's question
            
        Returns:
            QueryType enum
        """
        query_lower = query.lower()
        
        # Check each category
        if any(kw in query_lower for kw in cls.FACTUAL_KEYWORDS):
            return QueryType.FACTUAL
        
        if any(kw in query_lower for kw in cls.PROCEDURAL_KEYWORDS):
            return QueryType.PROCEDURAL
        
        if any(kw in query_lower for kw in cls.COMPARATIVE_KEYWORDS):
            return QueryType.COMPARATIVE
        
        # Default to open-ended
        return QueryType.OPEN_ENDED
    
    @classmethod
    def expand_query(cls, query: str, query_type: QueryType) -> List[str]:
        """
        Generate query variations for better retrieval
        
        Example:
        "POSH policy" → ["POSH policy", "sexual harassment policy", 
                         "workplace harassment prevention"]
        
        Args:
            query: Original query
            query_type: Type of query
            
        Returns:
            List of query variations
        """
        variations = [query]  # Always include original
        
        # Add common expansions based on company context
        expansions = {
            "posh": ["sexual harassment", "workplace harassment prevention"],
            "pto": ["paid time off", "leave", "vacation"],
            "wfh": ["work from home", "remote work"],
            "expense": ["reimbursement", "claim"],
        }
        
        query_lower = query.lower()
        for abbr, full_forms in expansions.items():
            if abbr in query_lower:
                for full in full_forms:
                    variations.append(query.lower().replace(abbr, full))
        
        return variations[:3]  # Limit to avoid noise


class HybridRetriever:
    """
    Combines semantic (vector) and keyword (BM25) search
    
    Why hybrid?
    - Semantic: Great for meaning ("POSH" → "sexual harassment")
    - Keyword: Great for exact terms ("Form 16", "Section 80C")
    - Together: Best of both worlds!
    """
    
    def __init__(self, embedding_manager: EmbeddingManager):
        """
        Initialize with vector store
        
        Args:
            embedding_manager: Manager with loaded vector store
        """
        self.embedding_manager = embedding_manager
        self.bm25_index = None
        self.documents = None
        self._initialize_bm25()
    
    def _initialize_bm25(self):
        """
        Build BM25 index from all documents in vector store
        
        BM25 = Best Match 25 (classic IR algorithm)
        - TF-IDF with improvements
        - Fast keyword matching
        - Handles document length normalization
        """
        logger.info("Building BM25 index...")
        
        # Get all documents from vector store
        vector_store = self.embedding_manager.vector_store
        collection = vector_store._collection
        
        # Fetch all documents
        results = collection.get(include=['documents', 'metadatas'])
        
        if not results or not results['documents']:
            logger.warning("No documents found in vector store")
            return
        
        # Store documents
        self.documents = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(results['documents'], results['metadatas'])
        ]
        
        # Tokenize for BM25
        tokenized_docs = [doc.page_content.lower().split() for doc in self.documents]
        
        # Build BM25 index
        self.bm25_index = BM25Okapi(tokenized_docs)
        
        logger.info(f"✓ BM25 index built with {len(self.documents)} documents")
    
    def semantic_search(
        self,
        query: str,
        k: int = TOP_K_RETRIEVAL,
    ) -> List[Tuple[Document, float]]:
        """
        Pure semantic (vector) search
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of (Document, score) tuples
        """
        return self.embedding_manager.search_with_scores(query, k=k)
    
    def keyword_search(
        self,
        query: str,
        k: int = TOP_K_RETRIEVAL,
    ) -> List[Tuple[Document, float]]:
        """
        Pure keyword (BM25) search
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of (Document, score) tuples
        """
        if not self.bm25_index or not self.documents:
            logger.warning("BM25 index not initialized")
            return []
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        # Return documents with scores
        results = [
            (self.documents[i], float(scores[i]))
            for i in top_indices
        ]
        
        return results
    
    def hybrid_search(
        self,
        query: str,
        k: int = TOP_K_RETRIEVAL,
        semantic_weight: float = SEMANTIC_WEIGHT,
        keyword_weight: float = KEYWORD_WEIGHT,
    ) -> List[Tuple[Document, float]]:
        """
        Hybrid search using Reciprocal Rank Fusion (RRF)
        
        RRF combines rankings from different retrieval methods
        Formula: score(d) = Σ 1/(k + rank_i(d))
        Where k=60 is a constant, rank_i is rank in method i
        
        Why RRF?
        - Doesn't require score normalization
        - Proven to work well in practice
        - Simple and robust
        
        Args:
            query: Search query
            k: Number of final results
            semantic_weight: Weight for semantic scores
            keyword_weight: Weight for keyword scores
            
        Returns:
            List of (Document, combined_score) tuples
        """
        # Get results from both methods
        semantic_results = self.semantic_search(query, k=k*2)  # Get more for fusion
        keyword_results = self.keyword_search(query, k=k*2)
        
        # Reciprocal Rank Fusion
        rrf_k = 60  # Standard RRF constant
        doc_scores = {}
        
        # Add semantic scores
        for rank, (doc, score) in enumerate(semantic_results, 1):
            doc_id = doc.page_content  # Use content as ID
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {'doc': doc, 'score': 0}
            doc_scores[doc_id]['score'] += semantic_weight / (rrf_k + rank)
        
        # Add keyword scores
        for rank, (doc, score) in enumerate(keyword_results, 1):
            doc_id = doc.page_content
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {'doc': doc, 'score': 0}
            doc_scores[doc_id]['score'] += keyword_weight / (rrf_k + rank)
        
        # Sort by combined score
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        # Return top-k
        results = [
            (item['doc'], item['score'])
            for item in sorted_docs[:k]
        ]
        
        return results
    
    def mmr_search(
        self,
        query: str,
        k: int = TOP_K_RETRIEVAL,
        fetch_k: int = MMR_FETCH_K,
        lambda_mult: float = MMR_LAMBDA,
    ) -> List[Document]:
        """
        Maximal Marginal Relevance search
        
        MMR balances relevance with diversity
        - Avoids retrieving very similar documents
        - Good for comprehensive answers
        
        Formula: MMR = λ * Sim(q,d) - (1-λ) * max Sim(d, D_selected)
        
        Args:
            query: Search query
            k: Number of final results
            fetch_k: Initial retrieval size
            lambda_mult: 0-1, higher = more relevance, less diversity
            
        Returns:
            List of diverse documents
        """
        vector_store = self.embedding_manager.vector_store
        
        results = vector_store.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )
        
        return results


class AdvancedRetriever:
    """
    Main retrieval orchestrator
    
    Decides which retrieval strategy to use based on query analysis
    """
    
    def __init__(self, embedding_manager: EmbeddingManager):
        """
        Initialize with all retrieval methods
        
        Args:
            embedding_manager: Manager with loaded vector store
        """
        self.embedding_manager = embedding_manager
        self.hybrid_retriever = HybridRetriever(embedding_manager)
        self.query_analyzer = QueryAnalyzer()
        
        logger.info("✓ Advanced retriever initialized")
    
    def retrieve(
        self,
        query: str,
        k: int = TOP_K_RETRIEVAL,
        strategy: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Smart retrieval with automatic strategy selection
        
        Args:
            query: User's question
            k: Number of results
            strategy: Force specific strategy (None = auto)
            
        Returns:
            RetrievalResult with documents and metadata
        """
        # Analyze query
        query_type = self.query_analyzer.analyze_query(query)
        
        # Determine strategy
        if strategy is None:
            strategy = self._select_strategy(query_type)
        
        logger.info(f"Query type: {query_type.value}, Strategy: {strategy}")
        
        # Execute retrieval
        if strategy == "semantic":
            results = self.hybrid_retriever.semantic_search(query, k=k)
            docs = [doc for doc, _ in results]
            scores = [score for _, score in results]
        
        elif strategy == "keyword":
            results = self.hybrid_retriever.keyword_search(query, k=k)
            docs = [doc for doc, _ in results]
            scores = [score for _, score in results]
        
        elif strategy == "hybrid":
            results = self.hybrid_retriever.hybrid_search(query, k=k)
            docs = [doc for doc, _ in results]
            scores = [score for _, score in results]
        
        elif strategy == "mmr":
            docs = self.hybrid_retriever.mmr_search(query, k=k)
            scores = [1.0] * len(docs)  # MMR doesn't return scores
        
        else:
            # Default to hybrid
            results = self.hybrid_retriever.hybrid_search(query, k=k)
            docs = [doc for doc, _ in results]
            scores = [score for _, score in results]
        
        # Filter by similarity threshold if applicable
        if strategy in ["semantic", "hybrid"] and scores:
            filtered = [
                (doc, score) for doc, score in zip(docs, scores)
                if score >= SIMILARITY_THRESHOLD
            ]
            if filtered:
                docs = [doc for doc, _ in filtered]
                scores = [score for _, score in filtered]
        
        return RetrievalResult(
            documents=docs,
            scores=scores,
            strategy_used=strategy,
            query_type=query_type,
            total_retrieved=len(docs),
        )
    
    def _select_strategy(self, query_type: QueryType) -> str:
        """
        Choose retrieval strategy based on query type
        
        Strategy mapping:
        - Factual: Hybrid (exact terms matter)
        - Procedural: Semantic (understand "how to")
        - Comparative: MMR (need diverse sources)
        - Open-ended: Hybrid (comprehensive)
        
        Args:
            query_type: Type of query
            
        Returns:
            Strategy name
        """
        strategy_map = {
            QueryType.FACTUAL: "hybrid",
            QueryType.PROCEDURAL: "semantic",
            QueryType.COMPARATIVE: "mmr",
            QueryType.OPEN_ENDED: "hybrid",
            QueryType.UNKNOWN: "hybrid",
        }
        
        return strategy_map.get(query_type, "hybrid")
    
    def retrieve_with_metadata_filter(
        self,
        query: str,
        filter_dict: Dict[str, Any],
        k: int = TOP_K_RETRIEVAL,
    ) -> RetrievalResult:
        """
        Retrieve with metadata filtering
        
        Example use cases:
        - "Show only from POSH policy" → {"source_file": "POSH Policy.pdf"}
        - "2024 policies only" → {"year": "2024"}
        
        Args:
            query: Search query
            filter_dict: Chroma metadata filter
            k: Number of results
            
        Returns:
            Filtered retrieval results
        """
        # Use vector store directly for metadata filtering
        results = self.embedding_manager.search_with_scores(
            query,
            k=k,
            filter_metadata=filter_dict,
        )
        
        docs = [doc for doc, _ in results]
        scores = [score for _, score in results]
        
        return RetrievalResult(
            documents=docs,
            scores=scores,
            strategy_used="semantic_filtered",
            query_type=QueryType.UNKNOWN,
            total_retrieved=len(docs),
        )


# ============================================================================
# TESTING & COMPARISON
# ============================================================================

def compare_retrieval_strategies(retriever: AdvancedRetriever, query: str):
    """
    Compare all retrieval strategies for a query
    
    Use this to understand trade-offs
    """
    print("\n" + "="*70)
    print(f"COMPARING STRATEGIES FOR: '{query}'")
    print("="*70)
    
    strategies = ["semantic", "keyword", "hybrid", "mmr"]
    
    for strategy in strategies:
        print(f"\n{strategy.upper()} SEARCH:")
        print("-" * 70)
        
        result = retriever.retrieve(query, k=3, strategy=strategy)
        
        for i, (doc, score) in enumerate(zip(result.documents, result.scores), 1):
            print(f"\n  {i}. Score: {score:.3f}")
            print(f"     Source: {doc.metadata.get('source_file', 'unknown')}")
            print(f"     Preview: {doc.page_content[:100]}...")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    """
    Test advanced retrieval strategies
    
    Run: python src/retrieval.py
    """
    from embedding import EmbeddingManager
    
    print("\n" + "="*70)
    print("ADVANCED RETRIEVAL TESTING")
    print("="*70)
    
    # Load vector store
    manager = EmbeddingManager()
    manager.load_vector_store()
    
    # Initialize retriever
    retriever = AdvancedRetriever(manager)
    
    # Test queries
    test_queries = [
        "What is the POSH policy?",  # Factual
        "How do I claim expenses?",  # Procedural
        "What are the exit procedures?",  # Procedural
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"QUERY: {query}")
        print('='*70)
        
        result = retriever.retrieve(query, k=3)
        
        print(f"Query Type: {result.query_type.value}")
        print(f"Strategy: {result.strategy_used}")
        print(f"Results: {result.total_retrieved}")
        print("\nTop 3 Results:")
        
        for i, (doc, score) in enumerate(zip(result.documents, result.scores), 1):
            print(f"\n{i}. Score: {score:.3f}")
            print(f"   Source: {doc.metadata.get('source_file')}")
            print(f"   Page: {doc.metadata.get('page')}")
            print(f"   Preview: {doc.page_content[:150]}...")
    
    # Compare strategies for one query
    compare_retrieval_strategies(retriever, "How do I claim expenses?")