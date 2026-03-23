"""
Document Ingestion Pipeline for Company Policy RAG

This module handles:
1. PDF loading and text extraction
2. Intelligent chunking with multiple strategies
3. Metadata extraction (file name, page numbers, sections)
4. Document preprocessing and cleaning

Key Learning:
- Why chunking matters: LLMs have context limits, embeddings work best on focused text
- Overlap importance: Prevents splitting mid-thought
- Metadata enrichment: Enables filtering and better citations
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

# PDF processing
from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_core.documents import Document

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNKING_STRATEGIES,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessedDocument:
    """
    Structured representation of a processed document chunk
    
    Why use dataclass?
    - Type safety
    - Easy serialization
    - Clear schema for what we're storing
    """
    content: str  # The actual text
    metadata: Dict[str, Any]  # File name, page, section, etc.
    chunk_id: str  # Unique identifier
    embedding: Optional[List[float]] = None  # Will be added later


class DocumentProcessor:
    """
    Handles all document processing logic
    
    Design pattern: Strategy pattern for different chunking methods
    This makes it easy to A/B test different approaches
    """
    
    def __init__(self, strategy: str = "recursive"):
        """
        Initialize processor with chunking strategy
        
        Args:
            strategy: "recursive", "semantic", or "fixed"
        """
        self.strategy = strategy
        self.text_splitter = self._create_text_splitter(strategy)
        logger.info(f"Initialized DocumentProcessor with '{strategy}' strategy")
    
    def _create_text_splitter(self, strategy: str):
        """
        Factory method for creating appropriate text splitter
        
        Why different strategies?
        - Recursive: Smart splitting (paragraphs → sentences → words)
        - Fixed: Simple baseline
        - Semantic: Advanced - splits by meaning (we'll add this later)
        """
        if strategy == "recursive":
            # Default and most versatile
            # Tries to split on: \n\n, \n, space, ""
            return RecursiveCharacterTextSplitter(
                chunk_size=CHUNKING_STRATEGIES["recursive"]["chunk_size"],
                chunk_overlap=CHUNKING_STRATEGIES["recursive"]["chunk_overlap"],
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len,
            )
        
        elif strategy == "fixed":
            # Simple baseline for comparison
            return CharacterTextSplitter(
                chunk_size=CHUNKING_STRATEGIES["fixed"]["chunk_size"],
                chunk_overlap=CHUNKING_STRATEGIES["fixed"]["chunk_overlap"],
                separator=" ",
            )
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def load_pdf(self, file_path: Path) -> List[Document]:
        """
        Load PDF and extract text with metadata
        
        Why PDFPlumber over PyPDF?
        - Better table extraction
        - More accurate text positioning
        - Handles complex layouts better
        
        Falls back to PyPDF if PDFPlumber fails
        """
        logger.info(f"Loading PDF: {file_path.name}")
        
        try:
            # Try PDFPlumber first (better quality)
            loader = PDFPlumberLoader(str(file_path))
            documents = loader.load()
            logger.info(f"✓ Loaded {len(documents)} pages with PDFPlumber")
        except Exception as e:
            logger.warning(f"PDFPlumber failed, trying PyPDF: {e}")
            try:
                # Fallback to PyPDF
                loader = PyPDFLoader(str(file_path))
                documents = loader.load()
                logger.info(f"✓ Loaded {len(documents)} pages with PyPDF")
            except Exception as e2:
                logger.error(f"✗ Failed to load {file_path.name}: {e2}")
                return []
        
        # Enrich metadata
        for i, doc in enumerate(documents):
            doc.metadata.update({
                "source_file": file_path.name,
                "page": i + 1,
                "total_pages": len(documents),
            })
        
        return documents
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text
        
        Common PDF issues:
        - Multiple spaces
        - Weird line breaks
        - Special characters
        - Headers/footers repeated
        """
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove multiple newlines
        text = re.sub(r'\n+', '\n', text)
        
        # Remove page numbers (common pattern: "Page 1 of 10")
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_section_title(self, text: str) -> Optional[str]:
        """
        Try to extract section title from chunk
        
        Heuristic: First line if it's short and title-cased
        This helps with metadata enrichment
        """
        lines = text.split('\n')
        if not lines:
            return None
        
        first_line = lines[0].strip()
        
        # Check if it looks like a title
        if (len(first_line) < 100 and 
            len(first_line) > 5 and
            (first_line.isupper() or first_line.istitle())):
            return first_line
        
        return None
    
    def chunk_documents(self, documents: List[Document]) -> List[ProcessedDocument]:
        """
        Split documents into chunks with rich metadata
        
        This is where the magic happens:
        1. Split text intelligently
        2. Preserve metadata
        3. Add chunk-specific info
        4. Clean and validate
        """
        all_chunks = []
        
        for doc in documents:
            # Clean the text first
            cleaned_text = self.clean_text(doc.page_content)
            
            if not cleaned_text:
                logger.warning(f"Empty content in {doc.metadata.get('source_file', 'unknown')}")
                continue
            
            # Create temporary document for splitting
            temp_doc = Document(page_content=cleaned_text, metadata=doc.metadata)
            
            # Split into chunks
            chunks = self.text_splitter.split_documents([temp_doc])
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                section = self.extract_section_title(chunk.page_content)
                
                # Create unique chunk ID
                chunk_id = f"{doc.metadata['source_file']}_{doc.metadata['page']}_{i}"
                
                # Enrich metadata
                chunk_metadata = {
                    **chunk.metadata,
                    "chunk_index": i,
                    "section": section,
                    "chunk_length": len(chunk.page_content),
                }
                
                processed_chunk = ProcessedDocument(
                    content=chunk.page_content,
                    metadata=chunk_metadata,
                    chunk_id=chunk_id,
                )
                
                all_chunks.append(processed_chunk)
        
        logger.info(f"✓ Created {len(all_chunks)} chunks from {len(documents)} pages")
        return all_chunks
    
    def process_directory(self, directory: Path = RAW_DATA_DIR) -> List[ProcessedDocument]:
        """
        Process all PDFs in directory
        
        Main entry point for ingestion pipeline
        """
        pdf_files = list(directory.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        if not pdf_files:
            logger.error(f"No PDF files found in {directory}")
            return []
        
        all_chunks = []
        
        for pdf_file in pdf_files:
            try:
                # Load PDF
                documents = self.load_pdf(pdf_file)
                
                if not documents:
                    continue
                
                # Chunk documents
                chunks = self.chunk_documents(documents)
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
                continue
        
        logger.info(f"✓ Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def save_processed_chunks(self, chunks: List[ProcessedDocument], filename: str = "processed_chunks.json"):
        """
        Save processed chunks to disk for inspection/debugging
        
        Useful for:
        - Debugging chunking quality
        - Caching processed documents
        - Analytics on document structure
        """
        output_path = PROCESSED_DATA_DIR / filename
        
        # Convert to JSON-serializable format
        chunks_data = [
            {
                "content": chunk.content,
                "metadata": chunk.metadata,
                "chunk_id": chunk.chunk_id,
            }
            for chunk in chunks
        ]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Saved {len(chunks)} chunks to {output_path}")
    
    def get_statistics(self, chunks: List[ProcessedDocument]) -> Dict[str, Any]:
        """
        Generate statistics about processed documents
        
        Useful for understanding your corpus:
        - Average chunk size
        - Document distribution
        - Metadata coverage
        """
        if not chunks:
            return {}
        
        chunk_lengths = [len(chunk.content) for chunk in chunks]
        sources = [chunk.metadata.get('source_file', 'unknown') for chunk in chunks]
        
        stats = {
            "total_chunks": len(chunks),
            "unique_documents": len(set(sources)),
            "avg_chunk_length": sum(chunk_lengths) / len(chunk_lengths),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "chunks_per_document": {
                source: sources.count(source) 
                for source in set(sources)
            }
        }
        
        return stats


def compare_chunking_strategies(sample_pdf: Path):
    """
    Utility function to compare different chunking strategies
    
    Use this to find optimal settings for your documents
    """
    strategies = ["recursive", "fixed"]
    results = {}
    
    for strategy in strategies:
        processor = DocumentProcessor(strategy=strategy)
        docs = processor.load_pdf(sample_pdf)
        chunks = processor.chunk_documents(docs)
        stats = processor.get_statistics(chunks)
        results[strategy] = stats
    
    print("\n" + "="*60)
    print("CHUNKING STRATEGY COMPARISON")
    print("="*60)
    for strategy, stats in results.items():
        print(f"\n{strategy.upper()}:")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Avg length: {stats['avg_chunk_length']:.0f} chars")
        print(f"  Range: {stats['min_chunk_length']}-{stats['max_chunk_length']} chars")
    print("="*60 + "\n")
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    """
    Test the ingestion pipeline
    
    Run this to process your documents:
    python src/ingestion.py
    """
    print("\n" + "="*60)
    print("COMPANY POLICY DOCUMENT INGESTION")
    print("="*60 + "\n")
    
    # Initialize processor
    processor = DocumentProcessor(strategy="recursive")
    
    # Process all documents
    chunks = processor.process_directory()
    
    if chunks:
        # Save processed chunks
        processor.save_processed_chunks(chunks)
        
        # Print statistics
        stats = processor.get_statistics(chunks)
        print("\n" + "="*60)
        print("PROCESSING STATISTICS")
        print("="*60)
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Unique documents: {stats['unique_documents']}")
        print(f"Average chunk length: {stats['avg_chunk_length']:.0f} characters")
        print(f"\nChunks per document:")
        for doc, count in stats['chunks_per_document'].items():
            print(f"  {doc}: {count} chunks")
        print("="*60 + "\n")
        
        # Show sample chunk
        print("SAMPLE CHUNK:")
        print("-" * 60)
        sample = chunks[0]
        print(f"Source: {sample.metadata['source_file']}")
        print(f"Page: {sample.metadata['page']}")
        print(f"Content preview: {sample.content[:200]}...")
        print("-" * 60)
    else:
        print("⚠️ No chunks created. Check if PDFs are in data/raw/")