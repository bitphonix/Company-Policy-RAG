"""
Python Client for Company Policy RAG API

Examples of how to interact with the FastAPI backend
"""

import requests
import json
from typing import Optional, Dict, Any
import sseclient  # For streaming


class PolicyRAGClient:
    """
    Python client for Policy RAG API
    
    Usage:
        client = PolicyRAGClient("http://localhost:8000")
        response = client.query("What is the POSH policy?")
        print(response['answer'])
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        """
        Initialize client
        
        Args:
            base_url: API base URL
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.headers = {}
        
        if api_key:
            self.headers['X-API-Key'] = api_key
    
    def query(
        self,
        query: str,
        strategy: Optional[str] = None,
        k: int = 5,
        use_reranking: bool = True,
    ) -> Dict[str, Any]:
        """
        Send a query to the API
        
        Args:
            query: User's question
            strategy: Retrieval strategy (semantic, keyword, hybrid, mmr)
            k: Number of sources to retrieve
            use_reranking: Whether to use reranking
            
        Returns:
            API response as dictionary
        """
        url = f"{self.base_url}/query"
        
        payload = {
            "query": query,
            "strategy": strategy,
            "k": k,
            "use_reranking": use_reranking,
        }
        
        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        
        return response.json()
    
    def query_stream(self, query: str, k: int = 5):
        """
        Stream query response (SSE)
        
        Args:
            query: User's question
            k: Number of sources
            
        Yields:
            Answer tokens as they're generated
        """
        url = f"{self.base_url}/query/stream"
        
        payload = {
            "query": query,
            "k": k,
            "stream": True,
        }
        
        response = requests.post(url, json=payload, headers=self.headers, stream=True)
        response.raise_for_status()
        
        client = sseclient.SSEClient(response)
        
        for event in client.events():
            if event.data == "[DONE]":
                break
            yield event.data
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        url = f"{self.base_url}/health"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get API statistics"""
        url = f"{self.base_url}/stats"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def list_documents(self) -> Dict[str, Any]:
        """List documents in knowledge base"""
        url = f"{self.base_url}/documents"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def clear_cache(self) -> Dict[str, str]:
        """Clear query cache (requires auth)"""
        url = f"{self.base_url}/cache/clear"
        response = requests.post(url, headers=self.headers)
        response.raise_for_status()
        return response.json()


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_basic_query():
    """Example: Basic query"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Query")
    print("="*60)
    
    client = PolicyRAGClient()
    
    response = client.query("What is the POSH policy?")
    
    print(f"\nAnswer: {response['answer']}\n")
    print(f"Confidence: {response['confidence']}")
    print(f"Strategy: {response['retrieval_strategy']}")
    print(f"Response time: {response['response_time']:.2f}s")
    print(f"\nSources ({len(response['sources'])}):")
    for source in response['sources']:
        print(f"  - {source['file']} (Page {source['page']}, Score: {source['score']:.3f})")


def example_with_reranking():
    """Example: Query with reranking comparison"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Reranking Comparison")
    print("="*60)
    
    client = PolicyRAGClient()
    query = "How do I claim expenses?"
    
    # Without reranking
    print("\nWithout reranking:")
    response1 = client.query(query, use_reranking=False)
    print(f"Top source: {response1['sources'][0]['file']}")
    print(f"Score: {response1['sources'][0]['score']:.3f}")
    
    # With reranking
    print("\nWith reranking:")
    response2 = client.query(query, use_reranking=True)
    print(f"Top source: {response2['sources'][0]['file']}")
    print(f"Score: {response2['sources'][0]['score']:.3f}")
    print(f"Relevance: {response2['sources'][0].get('relevance', 'N/A')}")


def example_streaming():
    """Example: Streaming response"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Streaming Response")
    print("="*60)
    
    client = PolicyRAGClient()
    
    print("\nAnswer (streaming):")
    for chunk in client.query_stream("What are the exit procedures?"):
        print(chunk, end='', flush=True)
    print("\n")


def example_stats():
    """Example: Get statistics"""
    print("\n" + "="*60)
    print("EXAMPLE 4: API Statistics")
    print("="*60)
    
    client = PolicyRAGClient()
    
    stats = client.get_stats()
    
    print(f"\nTotal queries: {stats['total_queries']}")
    print(f"Average response time: {stats['avg_response_time']:.2f}s")
    print(f"Uptime: {stats['uptime']/60:.1f} minutes")
    
    cache_stats = stats['cache_stats']
    print(f"\nCache statistics:")
    print(f"  Total entries: {cache_stats.get('total_entries', 0)}")
    print(f"  Total hits: {cache_stats.get('total_hits', 0)}")
    print(f"  Hit rate: {cache_stats.get('hit_rate', 0):.1%}")


def example_javascript():
    """Example JavaScript/fetch code for frontend"""
    print("\n" + "="*60)
    print("EXAMPLE 5: JavaScript/TypeScript Code")
    print("="*60)
    
    js_code = """
// JavaScript example for frontend integration

const API_BASE_URL = 'http://localhost:8000';

async function queryPolicy(question) {
    const response = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            // 'X-API-Key': 'your-api-key-here'  // If auth enabled
        },
        body: JSON.stringify({
            query: question,
            strategy: 'hybrid',
            k: 5,
            use_reranking: true
        })
    });
    
    if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
    }
    
    const data = await response.json();
    return data;
}

// Usage
queryPolicy("What is the POSH policy?")
    .then(result => {
        console.log('Answer:', result.answer);
        console.log('Sources:', result.sources);
        console.log('Confidence:', result.confidence);
    })
    .catch(error => console.error('Error:', error));

// Streaming example
async function queryPolicyStream(question) {
    const response = await fetch(`${API_BASE_URL}/query/stream`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            query: question,
            k: 5
        })
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\\n');
        
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = line.substring(6);
                if (data === '[DONE]') return;
                console.log(data);  // Or append to UI
            }
        }
    }
}
"""
    
    print(js_code)


def example_curl():
    """Example cURL commands"""
    print("\n" + "="*60)
    print("EXAMPLE 6: cURL Commands")
    print("="*60)
    
    curl_commands = """
# Basic query
curl -X POST http://localhost:8000/query \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "What is the POSH policy?",
    "strategy": "hybrid",
    "k": 5,
    "use_reranking": true
  }'

# Health check
curl http://localhost:8000/health

# Get statistics
curl http://localhost:8000/stats

# List documents
curl http://localhost:8000/documents

# Clear cache (with API key)
curl -X POST http://localhost:8000/cache/clear \\
  -H "X-API-Key: dev-key-123"
"""
    
    print(curl_commands)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("COMPANY POLICY RAG API - CLIENT EXAMPLES")
    print("="*60)
    print("\nMake sure the API is running:")
    print("  uvicorn backend.main:app --reload")
    print("\n" + "="*60)
    
    # Check if API is running
    try:
        client = PolicyRAGClient()
        health = client.health_check()
        print(f"\n✓ API is running: {health['status']}")
        
        # Run examples
        example_basic_query()
        example_with_reranking()
        # example_streaming()  # Uncomment if sseclient is installed
        example_stats()
        example_javascript()
        example_curl()
        
    except requests.exceptions.ConnectionError:
        print("\n❌ API is not running!")
        print("Start it with: uvicorn backend.main:app --reload")
        print("\nYou can still see the code examples:")
        example_javascript()
        example_curl()