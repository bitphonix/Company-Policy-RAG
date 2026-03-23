"""
Streamlit Frontend with FastAPI Backend Integration

This demonstrates clean separation of concerns:
- Backend (FastAPI): Business logic, RAG operations
- Frontend (Streamlit): User interface only

Run Backend First:
    uvicorn backend.main:app --reload

Then Frontend:
    streamlit run frontend/app_with_api.py
"""

import streamlit as st
import requests
import time
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

API_BASE_URL = "http://localhost:8000"
API_KEY = None  # Set if authentication is enabled

st.set_page_config(
    page_title="🏢 Company Policy Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .api-status-ok {
        color: #28a745;
    }
    .api-status-error {
        color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# API CLIENT FUNCTIONS
# ============================================================================

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def query_api(query, strategy=None, k=5, use_reranking=True):
    """Send query to API"""
    headers = {}
    if API_KEY:
        headers['X-API-Key'] = API_KEY
    
    payload = {
        "query": query,
        "strategy": strategy,
        "k": k,
        "use_reranking": use_reranking
    }
    
    response = requests.post(
        f"{API_BASE_URL}/query",
        json=payload,
        headers=headers
    )
    
    response.raise_for_status()
    return response.json()


def get_api_stats():
    """Get API statistics"""
    response = requests.get(f"{API_BASE_URL}/stats")
    response.raise_for_status()
    return response.json()


def get_documents():
    """Get document list"""
    response = requests.get(f"{API_BASE_URL}/documents")
    response.raise_for_status()
    return response.json()


# ============================================================================
# SESSION STATE
# ============================================================================

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'api_healthy' not in st.session_state:
    st.session_state.api_healthy = check_api_health()

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("🏢 Policy Assistant")
    
    # API Status
    st.markdown("---")
    st.subheader("🔌 API Status")
    
    if st.button("🔄 Refresh Status"):
        st.session_state.api_healthy = check_api_health()
        st.rerun()
    
    if st.session_state.api_healthy:
        st.markdown("**Status:** <span class='api-status-ok'>● Online</span>", 
                   unsafe_allow_html=True)
    else:
        st.markdown("**Status:** <span class='api-status-error'>● Offline</span>", 
                   unsafe_allow_html=True)
        st.error("⚠️ API is not running. Start with:\n```\nuvicorn backend.main:app --reload\n```")
    
    st.markdown("---")
    
    # Settings
    st.subheader("⚙️ Settings")
    
    strategy_option = st.selectbox(
        "Retrieval Strategy",
        ["Auto", "Semantic", "Keyword", "Hybrid", "MMR"],
        help="Auto selects best strategy based on query type"
    )
    
    strategy_map = {
        "Auto": None,
        "Semantic": "semantic",
        "Keyword": "keyword",
        "Hybrid": "hybrid",
        "MMR": "mmr",
    }
    selected_strategy = strategy_map[strategy_option]
    
    num_sources = st.slider(
        "Number of Sources",
        min_value=3,
        max_value=10,
        value=5,
        help="How many documents to retrieve"
    )
    
    use_reranking = st.checkbox(
        "Use Reranking",
        value=True,
        help="Use cross-encoder reranking for better precision"
    )
    
    show_sources = st.checkbox(
        "Show Sources",
        value=True,
        help="Display source documents with each answer"
    )
    
    st.markdown("---")
    
    # Statistics
    st.subheader("📊 Statistics")
    
    if st.session_state.api_healthy:
        try:
            stats = get_api_stats()
            st.metric("Total Queries", stats['total_queries'])
            st.metric("Avg Response Time", f"{stats['avg_response_time']:.2f}s")
            
            cache_stats = stats.get('cache_stats', {})
            if cache_stats.get('total_entries', 0) > 0:
                st.metric("Cache Hit Rate", f"{cache_stats.get('hit_rate', 0):.1%}")
        except:
            st.info("Stats unavailable")
    
    # Clear conversation
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    
    # Document info
    st.subheader("📚 Knowledge Base")
    
    if st.session_state.api_healthy:
        try:
            docs = get_documents()
            st.info(f"**{docs.get('total_chunks', 0)}** chunks indexed")
            st.caption(f"Model: {docs.get('embedding_model', 'N/A')}")
        except:
            st.info("Document info unavailable")

# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================

st.title("🤖 Company Policy Assistant")
st.markdown("Ask me anything about company policies, benefits, procedures, and more!")

if not st.session_state.api_healthy:
    st.error("⚠️ Backend API is not running. Please start it first.")
    st.code("uvicorn backend.main:app --reload", language="bash")
    st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources if available
        if message["role"] == "assistant" and "sources" in message and show_sources:
            with st.expander(f"📄 Sources ({len(message['sources'])} documents)", expanded=False):
                for source in message['sources']:
                    relevance = source.get('relevance', '')
                    relevance_badge = f" **({relevance})**" if relevance else ""
                    
                    st.markdown(f"""
                    <div class="source-card">
                        <strong>📄 {source['file']}</strong> (Page {source['page']}){relevance_badge}<br>
                        <small>Relevance Score: {source['score']:.3f}</small><br>
                        <small>{source['preview']}</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Show metadata if available
        if message["role"] == "assistant" and "metadata" in message:
            meta = message["metadata"]
            
            confidence_class = f"confidence-{meta.get('confidence', 'medium')}"
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <span class="metadata-badge {confidence_class}">
                    🎯 {meta.get('confidence', 'medium').title()}
                </span>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <span class="metadata-badge">
                    🔍 {meta.get('retrieval_strategy', 'unknown').title()}
                </span>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <span class="metadata-badge">
                    💬 {meta.get('query_type', 'unknown').title()}
                </span>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <span class="metadata-badge">
                    ⚡ {meta.get('response_time', 0):.2f}s
                </span>
                """, unsafe_allow_html=True)
            
            if meta.get('from_cache'):
                st.caption("⚡ Served from cache")

# Chat input
if prompt := st.chat_input("Ask about policies, benefits, procedures..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching policy documents..."):
            try:
                # Call API
                response = query_api(
                    prompt,
                    strategy=selected_strategy,
                    k=num_sources,
                    use_reranking=use_reranking,
                )
                
                # Display answer with typing effect
                message_placeholder = st.empty()
                full_response = ""
                
                # Simulate typing
                for chunk in response['answer'].split():
                    full_response += chunk + " "
                    time.sleep(0.02)
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(response['answer'])
                
                # Store message
                assistant_message = {
                    "role": "assistant",
                    "content": response['answer'],
                    "sources": response['sources'],
                    "metadata": {
                        "confidence": response['confidence'],
                        "retrieval_strategy": response['retrieval_strategy'],
                        "query_type": response['query_type'],
                        "response_time": response['response_time'],
                        "from_cache": response.get('from_cache', False),
                    }
                }
                st.session_state.messages.append(assistant_message)
                
                # Show sources
                if show_sources:
                    with st.expander(f"📄 Sources ({len(response['sources'])} documents)", expanded=False):
                        for source in response['sources']:
                            relevance = source.get('relevance', '')
                            relevance_badge = f" **({relevance})**" if relevance else ""
                            
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>📄 {source['file']}</strong> (Page {source['page']}){relevance_badge}<br>
                                <small>Relevance Score: {source['score']:.3f}</small><br>
                                <small>{source['preview']}</small>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Show metadata badges
                confidence_class = f"confidence-{response['confidence']}"
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <span class="metadata-badge {confidence_class}">
                        🎯 {response['confidence'].title()}
                    </span>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <span class="metadata-badge">
                        🔍 {response['retrieval_strategy'].title()}
                    </span>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <span class="metadata-badge">
                        💬 {response['query_type'].title()}
                    </span>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <span class="metadata-badge">
                        ⚡ {response['response_time']:.2f}s
                    </span>
                    """, unsafe_allow_html=True)
                
                if response.get('from_cache'):
                    st.caption("⚡ Served from cache")
                
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure it's running.")
            except requests.exceptions.HTTPError as e:
                st.error(f"❌ API Error: {e}")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# Example questions
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

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    🤖 Backend: FastAPI + LangChain + ChromaDB | Frontend: Streamlit<br>
    Built with ❤️ for Production RAG Systems
</div>
""", unsafe_allow_html=True)