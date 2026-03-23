"""
Embedding Generation and Vector Store Management

This module handles:
1. Converting text chunks into vector embeddings
2. Storing vectors in Chroma DB
3. Batch processing for efficiency
4. Incremental updates (add new docs without rebuilding)

Key Learning:
- Embeddings = numerical representations of text meaning
- Similar text = close vectors in embedding space
- Chroma = persistent vector database (survives restarts)
- Batch processing = much faster than one-by-one
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from tqdm import tqdm

# Updated imports for langchain
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import (
    EMBEDDING_MODEL_NAME,
    CHROMA_PERSIST_DIRECTORY,
    COLLECTION_NAME,
    PROCESSED_DATA_DIR,
)
from ingestion import DocumentProcessor, ProcessedDocument

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages embedding generation and vector store operations
    
    Why use a class?
    - Encapsulates embedding model (expensive to load)
    - Reuse model across operations
    - Clean interface for different operations
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        """
        Initialize embedding model and vector store
        
        Args:
            model_name: HuggingFace model for embeddings
        """
        logger.info(f"Initializing EmbeddingManager with model: {model_name}")
        
        # Initialize embedding model
        # Why HuggingFaceEmbeddings?
        # - Free and open source
        # - Runs locally (no API costs)
        # - all-MiniLM-L6-v2 is fast and good quality
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'mps'}, 
            encode_kwargs={
                'normalize_embeddings': True,  
                'batch_size': 32, 
            }
        )
        
        logger.info("✓ Embedding model loaded")
        
        # Vector store (will be initialized when needed)
        self.vector_store: Optional[Chroma] = None
        self.collection_name = COLLECTION_NAME
    
    def create_vector_store(
        self, 
        chunks: List[ProcessedDocument],
        persist_directory: str = CHROMA_PERSIST_DIRECTORY,
    ) -> Chroma:
        """
        Create new Chroma vector store from chunks
        
        This is the main ingestion pipeline:
        1. Convert ProcessedDocuments to LangChain Documents
        2. Generate embeddings (batched for speed)
        3. Store in Chroma with metadata
        4. Persist to disk
        
        Args:
            chunks: List of processed document chunks
            persist_directory: Where to save the database
            
        Returns:
            Chroma vector store instance
        """
        logger.info(f"Creating vector store with {len(chunks)} chunks")
        
        # Convert ProcessedDocument to LangChain Document format
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk.content,
                metadata=chunk.metadata
            )
            documents.append(doc)
        
        logger.info("Converting chunks to vectors (this may take 1-2 minutes)...")
        
        # Create Chroma vector store
        # Why from_documents?
        # - Handles embedding generation automatically
        # - Batches for efficiency
        # - Stores metadata
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=persist_directory,
        )
        
        logger.info(f"✓ Vector store created with {len(chunks)} chunks")
        logger.info(f"✓ Persisted to: {persist_directory}")
        
        self.vector_store = vector_store
        return vector_store
    
    def load_vector_store(
        self,
        persist_directory: str = CHROMA_PERSIST_DIRECTORY,
    ) -> Chroma:
        """
        Load existing vector store from disk
        
        Use this to avoid re-embedding on every run
        Much faster than create_vector_store
        
        Args:
            persist_directory: Where the database is saved
            
        Returns:
            Chroma vector store instance
        """
        logger.info(f"Loading vector store from: {persist_directory}")
        
        vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )
        
        # Verify it loaded correctly
        collection = vector_store._collection
        count = collection.count()
        logger.info(f"✓ Loaded vector store with {count} chunks")
        
        self.vector_store = vector_store
        return vector_store
    
    def add_documents(self, chunks: List[ProcessedDocument]):
        """
        Add new documents to existing vector store
        
        Use case: You get new policy documents
        Instead of rebuilding everything, just add the new ones
        
        Args:
            chunks: New chunks to add
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Load or create first.")
        
        logger.info(f"Adding {len(chunks)} new chunks to vector store")
        
        # Convert to LangChain Documents
        documents = [
            Document(page_content=chunk.content, metadata=chunk.metadata)
            for chunk in chunks
        ]
        
        # Add to existing store
        self.vector_store.add_documents(documents)
        
        logger.info(f"✓ Added {len(chunks)} chunks successfully")
    
    def search_similar(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Search for similar chunks using semantic similarity
        
        This is the core retrieval function
        
        Args:
            query: Question or search query
            k: Number of results to return
            filter_metadata: Filter by metadata (e.g., {"source_file": "POSH Policy.pdf"})
            
        Returns:
            List of most similar documents
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        # Perform similarity search
        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter_metadata,  # Chroma supports metadata filtering
        )
        
        return results
    
    def search_with_scores(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[tuple[Document, float]]:
        """
        Search with similarity scores
        
        Scores help you:
        - Filter low-confidence results
        - Understand retrieval quality
        - Decide if you need more context
        
        Args:
            query: Search query
            k: Number of results
            filter_metadata: Optional metadata filter
            
        Returns:
            List of (Document, score) tuples
            Score range: 0-1 for cosine similarity (higher = more similar)
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_metadata,
        )
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store
        
        Useful for monitoring and debugging
        """
        if not self.vector_store:
            return {"error": "Vector store not initialized"}
        
        collection = self.vector_store._collection
        count = collection.count()
        
        # Get sample of metadata to understand structure
        sample = collection.peek(limit=5)
        
        stats = {
            "total_chunks": count,
            "collection_name": self.collection_name,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "sample_metadata_keys": list(sample['metadatas'][0].keys()) if sample['metadatas'] else [],
        }
        
        return stats
    
    def delete_collection(self):
        """
        Delete the entire collection (use with caution!)
        
        Use case: Starting fresh with new chunking strategy
        """
        if self.vector_store:
            self.vector_store.delete_collection()
            logger.info(f"✓ Deleted collection: {self.collection_name}")
            self.vector_store = None


def build_vector_store_from_scratch():
    """
    Complete pipeline: Load chunks → Embed → Store
    
    Run this once to create your vector database
    """
    logger.info("="*60)
    logger.info("BUILDING VECTOR STORE FROM SCRATCH")
    logger.info("="*60)
    
    # 1. Load processed chunks
    chunks_file = PROCESSED_DATA_DIR / "processed_chunks.json"
    
    if not chunks_file.exists():
        logger.error(f"Chunks file not found: {chunks_file}")
        logger.error("Run ingestion.py first!")
        return None
    
    logger.info(f"Loading chunks from: {chunks_file}")
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    # Convert back to ProcessedDocument objects
    chunks = [
        ProcessedDocument(
            content=c['content'],
            metadata=c['metadata'],
            chunk_id=c['chunk_id'],
        )
        for c in chunks_data
    ]
    
    logger.info(f"✓ Loaded {len(chunks)} chunks")
    
    # 2. Create embedding manager
    manager = EmbeddingManager()
    
    # 3. Create vector store
    vector_store = manager.create_vector_store(chunks)
    
    # 4. Print statistics
    stats = manager.get_statistics()
    logger.info("="*60)
    logger.info("VECTOR STORE STATISTICS")
    logger.info("="*60)
    logger.info(f"Total chunks: {stats['total_chunks']}")
    logger.info(f"Collection: {stats['collection_name']}")
    logger.info(f"Embedding model: {stats['embedding_model']}")
    logger.info(f"Metadata keys: {', '.join(stats['sample_metadata_keys'])}")
    logger.info("="*60)
    
    return manager


def test_retrieval(manager: EmbeddingManager):
    """
    Test retrieval with sample queries
    
    This helps verify everything works before building the full app
    """
    logger.info("\n" + "="*60)
    logger.info("TESTING RETRIEVAL")
    logger.info("="*60)
    
    test_queries = [
        "What is the POSH policy?",
        "How do I claim expenses?",
        "What is the attendance policy?",
        "Tell me about salary advance",
        "What are the exit procedures?",
    ]
    
    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        logger.info("-" * 60)
        
        # Search with scores
        results = manager.search_with_scores(query, k=3)
        
        for i, (doc, score) in enumerate(results, 1):
            logger.info(f"\nResult {i} (Score: {score:.3f}):")
            logger.info(f"  Source: {doc.metadata.get('source_file', 'unknown')}")
            logger.info(f"  Page: {doc.metadata.get('page', '?')}")
            logger.info(f"  Content: {doc.page_content[:150]}...")
    
    logger.info("\n" + "="*60)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    """
    Build vector store and test retrieval
    
    Run this after ingestion.py:
    python src/embeddings.py
    """
    
    # Build vector store
    manager = build_vector_store_from_scratch()
    
    if manager:
        # Test retrieval
        test_retrieval(manager)
        
        logger.info("\n✓ Vector store ready!")
        logger.info(f"✓ Saved to: {CHROMA_PERSIST_DIRECTORY}")
        logger.info("\nNext time, load with: manager.load_vector_store()")