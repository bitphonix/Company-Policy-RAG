"""
Query Caching for RAG Performance Optimization

Benefits:
1. Faster responses for repeated queries
2. Reduced API costs (OpenAI calls)
3. Better user experience
4. Analytics on common questions

Simple in-memory cache with TTL (Time To Live)
For production: Use Redis or Memcached
"""

import time
import hashlib
import json
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
import logging

from config import CACHE_TTL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CachedAnswer:
    """Cached answer with metadata"""
    answer: str
    sources: list
    confidence: str
    retrieval_strategy: str = "unknown"  # ADD THIS
    query_type: str = "unknown"  # ADD THIS
    timestamp: float = 0
    hit_count: int = 0


class QueryCache:
    """
    Simple in-memory cache for RAG queries
    
    Features:
    - TTL-based expiration
    - Query normalization (case-insensitive, trimmed)
    - Hit counting for analytics
    - LRU-like behavior (most accessed stay longer)
    """
    
    def __init__(self, ttl: int = CACHE_TTL):
        """
        Initialize cache
        
        Args:
            ttl: Time to live in seconds (default: 1 hour)
        """
        self.cache: Dict[str, CachedAnswer] = {}
        self.ttl = ttl
        logger.info(f"Cache initialized with TTL={ttl}s")
    
    def _normalize_query(self, query: str) -> str:
        """
        Normalize query for better cache hits
        
        Examples:
        "What is POSH?" → "what is posh"
        "  Tell me about POSH  " → "tell me about posh"
        
        Args:
            query: Original query
            
        Returns:
            Normalized query
        """
        # Lowercase and strip whitespace
        normalized = query.lower().strip()
        
        # Remove extra spaces
        normalized = " ".join(normalized.split())
        
        return normalized
    
    def _hash_query(self, query: str) -> str:
        """
        Create hash of normalized query
        
        Using hash instead of direct string allows:
        - Faster lookups
        - Consistent key format
        - Privacy (cache keys don't expose queries)
        
        Args:
            query: Normalized query
            
        Returns:
            SHA256 hash
        """
        return hashlib.sha256(query.encode()).hexdigest()
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get cached answer if available and not expired
        
        Args:
            query: User's query
            
        Returns:
            Cached answer dict or None
        """
        normalized = self._normalize_query(query)
        key = self._hash_query(normalized)
        
        if key not in self.cache:
            logger.debug(f"Cache MISS: {query}")
            return None
        
        cached = self.cache[key]
        
        # Check if expired
        age = time.time() - cached.timestamp
        if age > self.ttl:
            logger.debug(f"Cache EXPIRED: {query} (age: {age:.0f}s)")
            del self.cache[key]
            return None
        
        # Update hit count
        cached.hit_count += 1
        
        logger.info(f"Cache HIT: {query} (hits: {cached.hit_count})")
        
        return {
            "answer": cached.answer,
            "sources": cached.sources,
            "confidence": cached.confidence,
            "from_cache": True,
        }
    
    def set(
        self,
        query: str,
        answer: str,
        sources: list,
        confidence: str,
        retrieval_strategy: str = "unknown",  # ADD THIS
        query_type: str = "unknown",  # ADD THIS
    ):
        """
        Cache an answer
        
        Args:
            query: User's query
            answer: Generated answer
            sources: Source documents
            confidence: Confidence level
            retrieval_strategy: Strategy used
            query_type: Type of query
        """
        normalized = self._normalize_query(query)
        key = self._hash_query(normalized)
        
        cached = CachedAnswer(
            answer=answer,
            sources=sources,
            confidence=confidence,
            retrieval_strategy=retrieval_strategy,  # ADD THIS
            query_type=query_type,  # ADD THIS
            timestamp=time.time(),
            hit_count=0,
        )
        
        self.cache[key] = cached
        logger.info(f"Cached answer for: {query}")

    
    def invalidate(self, query: str):
        """
        Invalidate specific query
        
        Use when policy documents are updated
        
        Args:
            query: Query to invalidate
        """
        normalized = self._normalize_query(query)
        key = self._hash_query(normalized)
        
        if key in self.cache:
            del self.cache[key]
            logger.info(f"Invalidated cache for: {query}")
    
    def clear(self):
        """Clear entire cache"""
        count = len(self.cache)
        self.cache.clear()
        logger.info(f"Cache cleared ({count} entries removed)")
    
    def cleanup_expired(self):
        """
        Remove expired entries
        
        Call periodically to prevent memory growth
        """
        now = time.time()
        expired_keys = [
            key for key, cached in self.cache.items()
            if (now - cached.timestamp) > self.ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dict with cache metrics
        """
        if not self.cache:
            return {
                "total_entries": 0,
                "total_hits": 0,
                "hit_rate": 0.0,
            }
        
        total_hits = sum(c.hit_count for c in self.cache.values())
        total_queries = total_hits + len(self.cache)  # Approximation
        
        stats = {
            "total_entries": len(self.cache),
            "total_hits": total_hits,
            "hit_rate": total_hits / total_queries if total_queries > 0 else 0,
            "avg_hits_per_entry": total_hits / len(self.cache),
            "oldest_entry_age": max(
                time.time() - c.timestamp for c in self.cache.values()
            ),
        }
        
        return stats
    
    def get_popular_queries(self, top_n: int = 5) -> list:
        """
        Get most frequently asked questions
        
        Useful for:
        - Understanding user needs
        - Creating FAQ section
        - Improving documentation
        
        Args:
            top_n: Number of top queries to return
            
        Returns:
            List of (query_hash, hit_count) tuples
        """
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].hit_count,
            reverse=True
        )
        
        return [
            (key, cached.hit_count)
            for key, cached in sorted_entries[:top_n]
        ]


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Test cache functionality
    """
    print("\n" + "="*60)
    print("TESTING QUERY CACHE")
    print("="*60)
    
    # Create cache
    cache = QueryCache(ttl=10)  # 10 second TTL for testing
    
    # Test 1: Cache miss
    result = cache.get("What is POSH policy?")
    print(f"\nTest 1 - Initial query: {result}")
    
    # Test 2: Cache set
    cache.set(
        query="What is POSH policy?",
        answer="POSH stands for...",
        sources=[{"file": "POSH.pdf", "page": 1}],
        confidence="high",
    )
    print("Test 2 - Cached answer")
    
    # Test 3: Cache hit (exact match)
    result = cache.get("What is POSH policy?")
    print(f"\nTest 3 - Exact match: {'HIT' if result else 'MISS'}")
    
    # Test 4: Cache hit (normalized match)
    result = cache.get("  WHAT IS POSH POLICY?  ")
    print(f"Test 4 - Normalized match: {'HIT' if result else 'MISS'}")
    
    # Test 5: Cache miss (different query)
    result = cache.get("How to claim expenses?")
    print(f"Test 5 - Different query: {'MISS' if not result else 'HIT'}")
    
    # Test 6: Multiple hits
    for i in range(5):
        cache.get("What is POSH policy?")
    
    stats = cache.get_stats()
    print(f"\nTest 6 - Stats after multiple hits:")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Total hits: {stats['total_hits']}")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    
    # Test 7: Expiration
    print("\nTest 7 - Waiting for expiration (11 seconds)...")
    time.sleep(11)
    result = cache.get("What is POSH policy?")
    print(f"  After expiration: {'HIT' if result else 'MISS (Expired)'}")
    
    print("\n" + "="*60)