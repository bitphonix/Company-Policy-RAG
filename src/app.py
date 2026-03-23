"""
Streamlit Web Interface for Company Policy RAG

Features:
1. Beautiful chat interface
2. Source document viewer
3. Strategy comparison mode
4. Conversation history
5. Performance analytics

Run with: streamlit run src/app.py
"""

import streamlit as st
import time
from datetime import datetime
import json
from pathlib import Path

# Import our RAG components
from embedding import EmbeddingManager
from retrieval import AdvancedRetriever, QueryType
from generation import AnswerGenerator, ConversationManager

from config import (
    APP_TITLE,
    APP_ICON,
    PAGE_CONFIG,
    RAW_DATA_DIR,
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(**PAGE_CONFIG)

# Custom CSS for better UI
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .source-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #1f77b4;
    }
    .metadata-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        background-color: #e0e0e0;
        font-size: 0.85rem;
        margin: 0.25rem;
    }
    .confidence-high {
        background-color: #d4edda;
        color: #155724;
    }
    .confidence-medium {
        background-color: #fff3cd;
        color: #856404;
    }
    .confidence-low {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

@st.cache_resource
def initialize_rag_system():
    """
    Initialize RAG system (cached to avoid reloading)
    
    This runs once and is cached across sessions
    """
    with st.spinner("🔄 Loading RAG system..."):
        # Initialize components
        manager = EmbeddingManager()
        manager.load_vector_store()
        
        retriever = AdvancedRetriever(manager)
        generator = AnswerGenerator(retriever)
        
        return generator, retriever

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'show_sources' not in st.session_state:
    st.session_state.show_sources = True

if 'analytics' not in st.session_state:
    st.session_state.analytics = {
        'total_queries': 0,
        'by_query_type': {},
        'by_strategy': {},
        'avg_confidence': [],
    }

# Load RAG system
try:
    generator, retriever = initialize_rag_system()
    st.session_state.rag_ready = True
except Exception as e:
    st.error(f"❌ Failed to initialize RAG system: {e}")
    st.session_state.rag_ready = False

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.markdown("---")
    
    # Settings
    st.subheader("⚙️ Settings")
    
    # Retrieval strategy
    strategy_option = st.selectbox(
        "Retrieval Strategy",
        ["Auto (Recommended)", "Semantic", "Keyword", "Hybrid", "MMR"],
        help="Auto selects the best strategy based on query type"
    )
    
    strategy_map = {
        "Auto (Recommended)": None,
        "Semantic": "semantic",
        "Keyword": "keyword",
        "Hybrid": "hybrid",
        "MMR": "mmr",
    }
    selected_strategy = strategy_map[strategy_option]
    
    # Number of sources
    num_sources = st.slider(
        "Number of Sources",
        min_value=3,
        max_value=10,
        value=5,
        help="How many documents to retrieve"
    )
    
    # Show sources toggle
    st.session_state.show_sources = st.checkbox(
        "Show Sources",
        value=True,
        help="Display source documents with each answer"
    )
    
    st.markdown("---")
    
    # Statistics
    st.subheader("📊 Statistics")
    st.metric("Total Queries", st.session_state.analytics['total_queries'])
    
    if st.session_state.analytics['avg_confidence']:
        avg_conf = sum(st.session_state.analytics['avg_confidence']) / len(st.session_state.analytics['avg_confidence'])
        st.metric("Avg Confidence", f"{avg_conf:.1%}")
    
    # Clear conversation
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.rerun()
    
    st.markdown("---")
    
    # Document info
    st.subheader("📚 Knowledge Base")
    pdf_files = list(RAW_DATA_DIR.glob("*.pdf"))
    st.info(f"**{len(pdf_files)}** policy documents loaded")
    
    with st.expander("View Documents"):
        for pdf in sorted(pdf_files):
            st.text(f"• {pdf.name}")

# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================

st.title(f"{APP_ICON} Company Policy Assistant")
st.markdown("Ask me anything about company policies, benefits, procedures, and more!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources if available
        if message["role"] == "assistant" and "sources" in message and st.session_state.show_sources:
            with st.expander(f"📄 Sources ({len(message['sources'])} documents)", expanded=False):
                for source in message['sources']:
                    st.markdown(f"""
                    <div class="source-card">
                        <strong>📄 {source['file']}</strong> (Page {source['page']})<br>
                        <small>Relevance Score: {source['score']:.3f}</small><br>
                        <small>{source['preview']}</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Show metadata if available
        if message["role"] == "assistant" and "metadata" in message:
            meta = message["metadata"]
            
            # Confidence badge
            confidence_class = f"confidence-{meta.get('confidence', 'medium')}"
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <span class="metadata-badge {confidence_class}">
                    🎯 {meta.get('confidence', 'medium').title()} Confidence
                </span>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <span class="metadata-badge">
                    🔍 {meta.get('strategy', 'unknown').title()}
                </span>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <span class="metadata-badge">
                    💬 {meta.get('query_type', 'unknown').title()}
                </span>
                """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask about policies, benefits, procedures..."):
    if not st.session_state.rag_ready:
        st.error("❌ RAG system is not ready. Please check the error above.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Searching policy documents..."):
                try:
                    # Generate answer
                    answer = generator.generate_answer(
                        prompt,
                        retrieval_strategy=selected_strategy,
                        k=num_sources,
                    )
                    
                    # Display answer with typing effect
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    # Simulate typing (or use real streaming)
                    for chunk in answer.answer.split():
                        full_response += chunk + " "
                        time.sleep(0.02)
                        message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(answer.answer)
                    
                    # Store message with metadata
                    assistant_message = {
                        "role": "assistant",
                        "content": answer.answer,
                        "sources": answer.sources,
                        "metadata": {
                            "confidence": answer.confidence,
                            "strategy": answer.retrieval_strategy,
                            "query_type": answer.query_type,
                        }
                    }
                    st.session_state.messages.append(assistant_message)
                    
                    # Update analytics
                    st.session_state.analytics['total_queries'] += 1
                    
                    query_type = answer.query_type
                    st.session_state.analytics['by_query_type'][query_type] = \
                        st.session_state.analytics['by_query_type'].get(query_type, 0) + 1
                    
                    strategy = answer.retrieval_strategy
                    st.session_state.analytics['by_strategy'][strategy] = \
                        st.session_state.analytics['by_strategy'].get(strategy, 0) + 1
                    
                    # Map confidence to number
                    conf_map = {"high": 1.0, "medium": 0.6, "low": 0.3}
                    st.session_state.analytics['avg_confidence'].append(
                        conf_map.get(answer.confidence, 0.6)
                    )
                    
                    # Update conversation history
                    st.session_state.conversation_history.append({
                        "question": prompt,
                        "answer": answer.answer,
                    })
                    
                    # Show sources
                    if st.session_state.show_sources:
                        with st.expander(f"📄 Sources ({len(answer.sources)} documents)", expanded=False):
                            for source in answer.sources:
                                st.markdown(f"""
                                <div class="source-card">
                                    <strong>📄 {source['file']}</strong> (Page {source['page']})<br>
                                    <small>Relevance Score: {source['score']:.3f}</small><br>
                                    <small>{source['preview']}</small>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Show metadata badges
                    confidence_class = f"confidence-{answer.confidence}"
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                        <span class="metadata-badge {confidence_class}">
                            🎯 {answer.confidence.title()} Confidence
                        </span>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <span class="metadata-badge">
                            🔍 {answer.retrieval_strategy.title()}
                        </span>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <span class="metadata-badge">
                            💬 {answer.query_type.title()}
                        </span>
                        """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Error generating answer: {e}")
                    import traceback
                    st.code(traceback.format_exc())

# ============================================================================
# EXAMPLE QUESTIONS
# ============================================================================

if not st.session_state.messages:
    st.markdown("### 💡 Try asking:")
    
    example_questions = [
        "What is the POSH policy?",
        "How do I claim expenses?",
        "What are the exit procedures?",
        "Tell me about salary advance",
        "What is the attendance policy?",
        "How many days notice period?",
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(question, key=f"example_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()

# ============================================================================
# ANALYTICS TAB (Optional)
# ============================================================================

if st.session_state.analytics['total_queries'] > 0:
    with st.expander("📈 Advanced Analytics", expanded=False):
        st.subheader("Query Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**By Query Type**")
            if st.session_state.analytics['by_query_type']:
                import plotly.express as px
                fig = px.pie(
                    values=list(st.session_state.analytics['by_query_type'].values()),
                    names=list(st.session_state.analytics['by_query_type'].keys()),
                    title="Query Types"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**By Strategy**")
            if st.session_state.analytics['by_strategy']:
                fig = px.bar(
                    x=list(st.session_state.analytics['by_strategy'].keys()),
                    y=list(st.session_state.analytics['by_strategy'].values()),
                    title="Retrieval Strategies Used"
                )
                st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    🤖 Powered by LangChain, Chroma, OpenAI GPT-4o-mini<br>
    Built with ❤️ for Adda247 Team
</div>
""", unsafe_allow_html=True)