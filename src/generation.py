"""
Answer Generation with LLM and Source Citations

This module handles:
1. Formatting retrieved context for LLM
2. Prompt engineering for accurate answers
3. Source citation and attribution
4. Confidence scoring
5. Conversation memory

Key Learning:
- Context formatting matters for LLM performance
- Prompt engineering is critical for accuracy
- Source citations build trust
- Conversation history enables follow-up questions
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from config import (
    OPENAI_API_KEY,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    SYSTEM_PROMPT,
    QA_PROMPT_TEMPLATE,
)
from retrieval import AdvancedRetriever, RetrievalResult

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Answer:
    """
    Structured answer with metadata
    
    Why structure it?
    - Shows sources to user
    - Enables feedback collection
    - Tracks confidence
    - Supports analytics
    """
    answer: str
    sources: List[Dict[str, Any]]
    confidence: str  # "high", "medium", "low"
    retrieval_strategy: str
    query_type: str
    total_context_length: int


class AnswerGenerator:
    """
    Generates answers using LLM and retrieved context
    
    Responsibilities:
    - Format context from retrieval
    - Engineer prompts for accuracy
    - Call OpenAI API
    - Extract and format sources
    - Handle conversation history
    """
    
    def __init__(self, retriever: AdvancedRetriever):
        """
        Initialize with retriever and LLM
        
        Args:
            retriever: Advanced retriever for document search
        """
        self.retriever = retriever
        
        # Initialize OpenAI LLM
        self.llm = ChatOpenAI(
            model=LLM_MODEL_NAME,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            openai_api_key=OPENAI_API_KEY,
        )
        
        # Conversation history
        self.conversation_history: List[Any] = []
        
        logger.info(f"✓ Answer generator initialized with {LLM_MODEL_NAME}")
    
    def format_context(self, retrieval_result: RetrievalResult) -> str:
        """
        Format retrieved documents into context for LLM
        
        Good formatting:
        - Clear source attribution
        - Numbered for reference
        - Includes metadata (file, page)
        - Not too verbose
        
        Args:
            retrieval_result: Results from retriever
            
        Returns:
            Formatted context string
        """
        if not retrieval_result.documents:
            return "No relevant information found in the policy documents."
        
        context_parts = []
        
        for i, doc in enumerate(retrieval_result.documents, 1):
            source_file = doc.metadata.get('source_file', 'Unknown')
            page = doc.metadata.get('page', '?')
            
            # Format each chunk
            context_parts.append(
                f"[Source {i}] {source_file} (Page {page}):\n{doc.page_content}\n"
            )
        
        return "\n".join(context_parts)
    
    def extract_sources(self, retrieval_result: RetrievalResult) -> List[Dict[str, Any]]:
        """
        Extract source information for citation
        
        Returns list of sources with metadata
        Users can click to see original document
        
        Args:
            retrieval_result: Results from retriever
            
        Returns:
            List of source dictionaries
        """
        sources = []
        
        for i, (doc, score) in enumerate(
            zip(retrieval_result.documents, retrieval_result.scores), 1
        ):
            source = {
                "id": i,
                "file": doc.metadata.get('source_file', 'Unknown'),
                "page": doc.metadata.get('page', '?'),
                "score": float(score),
                "preview": doc.page_content[:200] + "...",
            }
            sources.append(source)
        
        return sources
    
    def calculate_confidence(
        self,
        retrieval_result: RetrievalResult,
    ) -> str:
        """
        Calculate confidence in answer based on retrieval quality
        
        Heuristics:
        - High: Multiple relevant docs with high scores
        - Medium: Some relevant docs
        - Low: Few or low-scoring docs
        
        Args:
            retrieval_result: Results from retriever
            
        Returns:
            "high", "medium", or "low"
        """
        if not retrieval_result.documents or not retrieval_result.scores:
            return "low"
        
        avg_score = sum(retrieval_result.scores) / len(retrieval_result.scores)
        num_docs = len(retrieval_result.documents)
        
        # High confidence: good scores and multiple docs
        if avg_score > 0.7 and num_docs >= 3:
            return "high"
        
        # Medium confidence: decent scores or some docs
        elif avg_score > 0.5 or num_docs >= 2:
            return "medium"
        
        # Low confidence: poor scores or few docs
        else:
            return "low"
    
    def generate_answer(
        self,
        query: str,
        retrieval_strategy: Optional[str] = None,
        k: int = 5,
    ) -> Answer:
        """
        Main function to generate answer
        
        Pipeline:
        1. Retrieve relevant documents
        2. Format context
        3. Create prompt
        4. Call LLM
        5. Extract sources
        6. Return structured answer
        
        Args:
            query: User's question
            retrieval_strategy: Force specific strategy (None = auto)
            k: Number of documents to retrieve
            
        Returns:
            Answer object with response and metadata
        """
        logger.info(f"Generating answer for: {query}")
        
        # 1. Retrieve relevant documents
        retrieval_result = self.retriever.retrieve(
            query,
            k=k,
            strategy=retrieval_strategy,
        )
        
        # 2. Format context
        context = self.format_context(retrieval_result)
        
        # 3. Create prompt
        prompt = QA_PROMPT_TEMPLATE.format(
            context=context,
            question=query,
        )
        
        # 4. Call LLM
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        
        response = self.llm.invoke(messages)
        answer_text = response.content
        
        # 5. Extract sources
        sources = self.extract_sources(retrieval_result)
        
        # 6. Calculate confidence
        confidence = self.calculate_confidence(retrieval_result)
        
        # 7. Create answer object
        answer = Answer(
            answer=answer_text,
            sources=sources,
            confidence=confidence,
            retrieval_strategy=retrieval_result.strategy_used,
            query_type=retrieval_result.query_type.value,
            total_context_length=len(context),
        )
        
        logger.info(f"✓ Answer generated (confidence: {confidence})")
        
        return answer
    
    def generate_with_history(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Answer:
        """
        Generate answer with conversation context
        
        Enables follow-up questions:
        - "What is POSH policy?"
        - "When was it updated?" ← needs context
        
        Args:
            query: Current question
            conversation_history: List of previous Q&A pairs
            
        Returns:
            Answer with conversation context
        """
        # Retrieve documents
        retrieval_result = self.retriever.retrieve(query, k=5)
        context = self.format_context(retrieval_result)
        
        # Build conversation messages
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        
        # Add conversation history
        if conversation_history:
            for turn in conversation_history[-3:]:  # Last 3 turns
                messages.append(HumanMessage(content=turn['question']))
                messages.append(AIMessage(content=turn['answer']))
        
        # Add current query with context
        current_prompt = QA_PROMPT_TEMPLATE.format(
            context=context,
            question=query,
        )
        messages.append(HumanMessage(content=current_prompt))
        
        # Generate answer
        response = self.llm.invoke(messages)
        
        # Package response
        answer = Answer(
            answer=response.content,
            sources=self.extract_sources(retrieval_result),
            confidence=self.calculate_confidence(retrieval_result),
            retrieval_strategy=retrieval_result.strategy_used,
            query_type=retrieval_result.query_type.value,
            total_context_length=len(context),
        )
        
        return answer
    
    def stream_answer(self, query: str, k: int = 5):
        """
        Stream answer for better UX (tokens appear gradually)
        
        Use this in production for responsiveness
        
        Args:
            query: User's question
            k: Number of documents to retrieve
            
        Yields:
            Chunks of answer text
        """
        # Retrieve documents
        retrieval_result = self.retriever.retrieve(query, k=k)
        context = self.format_context(retrieval_result)
        
        # Create prompt
        prompt = QA_PROMPT_TEMPLATE.format(context=context, question=query)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        
        # Stream response
        for chunk in self.llm.stream(messages):
            if chunk.content:
                yield chunk.content


class ConversationManager:
    """
    Manages multi-turn conversations
    
    Handles:
    - Conversation history
    - Follow-up questions
    - Context window management
    """
    
    def __init__(self, answer_generator: AnswerGenerator):
        """
        Initialize with answer generator
        
        Args:
            answer_generator: Generator for answers
        """
        self.generator = answer_generator
        self.history: List[Dict[str, str]] = []
    
    def ask(self, query: str) -> Answer:
        """
        Ask a question with conversation context
        
        Args:
            query: User's question
            
        Returns:
            Answer object
        """
        # Generate answer with history
        answer = self.generator.generate_with_history(query, self.history)
        
        # Add to history
        self.history.append({
            "question": query,
            "answer": answer.answer,
        })
        
        return answer
    
    def clear_history(self):
        """Clear conversation history"""
        self.history = []
        logger.info("Conversation history cleared")
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.history


# ============================================================================
# TESTING
# ============================================================================

def format_answer_display(answer: Answer):
    """
    Pretty print answer for testing
    
    Args:
        answer: Answer object to display
    """
    print("\n" + "="*70)
    print("ANSWER")
    print("="*70)
    print(answer.answer)
    print("\n" + "-"*70)
    print("METADATA")
    print("-"*70)
    print(f"Confidence: {answer.confidence}")
    print(f"Strategy: {answer.retrieval_strategy}")
    print(f"Query Type: {answer.query_type}")
    print(f"Context Length: {answer.total_context_length} chars")
    print("\n" + "-"*70)
    print("SOURCES")
    print("-"*70)
    for source in answer.sources:
        print(f"\n{source['id']}. {source['file']} (Page {source['page']})")
        print(f"   Score: {source['score']:.3f}")
        print(f"   Preview: {source['preview'][:100]}...")
    print("="*70 + "\n")


if __name__ == "__main__":
    """
    Test answer generation
    
    Run: python src/generation.py
    """
    from embedding import EmbeddingManager
    from retrieval import AdvancedRetriever
    
    print("\n" + "="*70)
    print("TESTING ANSWER GENERATION")
    print("="*70)
    
    # Initialize components
    manager = EmbeddingManager()
    manager.load_vector_store()
    retriever = AdvancedRetriever(manager)
    generator = AnswerGenerator(retriever)
    
    # Test queries
    test_queries = [
        "What is the POSH policy?",
        "How do I claim expenses?",
        "What are the exit procedures?",
        "Tell me about salary advance policy",
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"QUERY: {query}")
        answer = generator.generate_answer(query)
        format_answer_display(answer)
    
    # Test conversation
    print("\n" + "="*70)
    print("TESTING CONVERSATION (Follow-up)")
    print("="*70)
    
    conv_manager = ConversationManager(generator)
    
    # First question
    print("\nQ1: What is the POSH policy?")
    answer1 = conv_manager.ask("What is the POSH policy?")
    print(f"A1: {answer1.answer[:200]}...")
    
    # Follow-up (should understand context)
    print("\nQ2: Who should I contact for it?")
    answer2 = conv_manager.ask("Who should I contact for it?")
    print(f"A2: {answer2.answer[:200]}...")