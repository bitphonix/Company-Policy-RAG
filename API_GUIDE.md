# 📚 API Guide - Company Policy RAG

Complete guide to using the FastAPI backend for the Company Policy RAG system.

---

## 🚀 Getting Started

### Start the API

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Access Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

---

## 📡 Endpoints

### 1. Health Check

**GET** `/health`

Check if the API is running and all components are operational.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-18T12:00:00",
  "version": "1.0.0",
  "components": {
    "vector_store": "operational",
    "retriever": "operational",
    "generator": "operational",
    "cache": "operational"
  }
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

---

### 2. Query (Main Endpoint)

**POST** `/query`

Submit a question and get an AI-generated answer with source citations.

**Request Body:**
```json
{
  "query": "What is the POSH policy?",
  "strategy": "hybrid",
  "k": 5,
  "use_reranking": true,
  "stream": false
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | Yes | - | User's question (3-500 chars) |
| `strategy` | string | No | `null` (auto) | Retrieval strategy: `semantic`, `keyword`, `hybrid`, `mmr` |
| `k` | integer | No | `5` | Number of sources (1-10) |
| `use_reranking` | boolean | No | `true` | Enable cross-encoder reranking |
| `stream` | boolean | No | `false` | Stream response tokens (use `/query/stream` instead) |

**Response:**
```json
{
  "answer": "The POSH (Prevention of Sexual Harassment) policy...",
  "sources": [
    {
      "id": 1,
      "file": "Adda247 - Posh Policy.pdf",
      "page": 2,
      "score": 0.964,
      "preview": "The Company expects its employees...",
      "relevance": "Highly Relevant"
    }
  ],
  "confidence": "high",
  "retrieval_strategy": "hybrid_reranked",
  "query_type": "factual",
  "response_time": 2.34,
  "from_cache": false,
  "metadata": {
    "total_context_length": 3292,
    "reranking_used": true
  }
}
```

**Examples:**

**Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "How do I claim expenses?",
        "use_reranking": True,
        "k": 5
    }
)

data = response.json()
print(data['answer'])
```

**JavaScript:**
```javascript
const response = await fetch('http://localhost:8000/query', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        query: 'What are the exit procedures?',
        strategy: 'hybrid',
        k: 5
    })
});

const data = await response.json();
console.log(data.answer);
```

**cURL:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the salary advance policy?",
    "use_reranking": true,
    "k": 5
  }'
```

---

### 3. Streaming Query

**POST** `/query/stream`

Stream the answer token-by-token for better UX with long responses.

**Request:** Same as `/query`

**Response:** Server-Sent Events (SSE)

```
data: The
data: POSH
data: policy
...
data: [DONE]
```

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/query/stream",
    json={"query": "What is the POSH policy?"},
    stream=True
)

for line in response.iter_lines():
    if line:
        decoded = line.decode('utf-8')
        if decoded.startswith('data: '):
            token = decoded[6:]
            if token == '[DONE]':
                break
            print(token, end='', flush=True)
```

**JavaScript Example:**
```javascript
const response = await fetch('http://localhost:8000/query/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: 'What is the POSH policy?' })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value);
    // Process SSE format
    console.log(chunk);
}
```

---

### 4. Statistics

**GET** `/stats`

Get API usage statistics and performance metrics.

**Response:**
```json
{
  "total_queries": 150,
  "cache_stats": {
    "total_entries": 45,
    "total_hits": 89,
    "hit_rate": 0.372
  },
  "uptime": 3600.5,
  "avg_response_time": 2.45
}
```

**Example:**
```bash
curl http://localhost:8000/stats
```

---

### 5. List Documents

**GET** `/documents`

Get information about the indexed document collection.

**Response:**
```json
{
  "total_chunks": 196,
  "collection_name": "company_policies",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "metadata_keys": [
    "source_file",
    "page",
    "total_pages",
    "chunk_index",
    "chunk_length"
  ]
}
```

---

### 6. Clear Cache

**POST** `/cache/clear`

Clear the query cache. Useful after updating policy documents.

**Response:**
```json
{
  "message": "Cache cleared successfully"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/cache/clear
```

---

## 🔐 Authentication

### Enable API Key Authentication

**1. Set API keys in environment:**
```bash
export API_KEY_DEV="dev-key-123"
export API_KEY_PROD="prod-key-456"
```

**2. Include in requests:**

**Header:**
```
X-API-Key: dev-key-123
```

**Python:**
```python
headers = {"X-API-Key": "dev-key-123"}
response = requests.post(
    "http://localhost:8000/query",
    json={"query": "..."},
    headers=headers
)
```

**JavaScript:**
```javascript
const response = await fetch('http://localhost:8000/query', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'X-API-Key': 'dev-key-123'
    },
    body: JSON.stringify({...})
});
```

**cURL:**
```bash
curl -X POST http://localhost:8000/query \
  -H "X-API-Key: dev-key-123" \
  -H "Content-Type: application/json" \
  -d '{"query": "..."}'
```

---

## 📊 Query Strategies

### Semantic Search
Best for: Understanding meaning and intent

**Example:**
```json
{
  "query": "How to report workplace harassment?",
  "strategy": "semantic"
}
```

Uses: Vector similarity with embeddings

### Keyword Search (BM25)
Best for: Exact terms and abbreviations

**Example:**
```json
{
  "query": "Form 16 tax document",
  "strategy": "keyword"
}
```

Uses: Traditional search engine approach

### Hybrid Search (Recommended)
Best for: Balanced accuracy

**Example:**
```json
{
  "query": "PTO policy details",
  "strategy": "hybrid"
}
```

Uses: Combines semantic + keyword with RRF fusion

### MMR (Maximal Marginal Relevance)
Best for: Diverse perspectives

**Example:**
```json
{
  "query": "Compare different leave types",
  "strategy": "mmr"
}
```

Uses: Balances relevance with diversity

---

## 🎯 Reranking

Cross-encoder reranking improves precision by 15-30%.

**Enable (recommended):**
```json
{
  "query": "...",
  "use_reranking": true
}
```

**Disable (faster but less accurate):**
```json
{
  "query": "...",
  "use_reranking": false
}
```

**How it works:**
1. Retrieve 15-20 candidates (fast bi-encoder)
2. Rerank with cross-encoder (slow but accurate)
3. Return top-K results

---

## ⚡ Rate Limiting

Default limits (can be configured):
- **10 requests/minute** per IP
- **100 requests/hour** per API key

**Response when limited:**
```json
{
  "error": "Rate limit exceeded",
  "status_code": 429,
  "retry_after": 60
}
```

---

## 🐛 Error Handling

### Common Errors

**400 Bad Request**
```json
{
  "error": "Validation error",
  "detail": [
    {
      "loc": ["body", "query"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**401 Unauthorized**
```json
{
  "error": "Invalid API key",
  "status_code": 401
}
```

**500 Internal Server Error**
```json
{
  "error": "Internal server error",
  "detail": "OpenAI API timeout",
  "status_code": 500,
  "timestamp": "2025-01-18T12:00:00"
}
```

### Error Handling in Code

**Python:**
```python
try:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    data = response.json()
except requests.exceptions.HTTPError as e:
    print(f"API Error: {e}")
    print(e.response.json())
except requests.exceptions.ConnectionError:
    print("Cannot connect to API")
```

**JavaScript:**
```javascript
try {
    const response = await fetch(url, options);
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail);
    }
    
    const data = await response.json();
} catch (error) {
    console.error('API Error:', error);
}
```

---

## 🧪 Testing the API

### Using Python

```python
# test_api.py
import requests

def test_health():
    response = requests.get("http://localhost:8000/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_query():
    response = requests.post(
        "http://localhost:8000/query",
        json={
            "query": "What is the POSH policy?",
            "k": 5
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["sources"]) > 0

if __name__ == "__main__":
    test_health()
    test_query()
    print("✓ All tests passed")
```

### Using Postman

1. **Import OpenAPI spec**: http://localhost:8000/openapi.json
2. **Create collection** from spec
3. **Test endpoints** with different parameters

---

## 📈 Performance Tips

### 1. Use Caching
- Repeated queries are served from cache (~50ms vs 2-3s)
- Cache automatically expires after 1 hour

### 2. Optimize K Value
- `k=3`: Fastest, less context
- `k=5`: Balanced (recommended)
- `k=10`: Most context, slower

### 3. Choose Strategy Wisely
- **semantic**: Fastest
- **keyword**: Fast
- **hybrid**: Medium (recommended)
- **mmr**: Slower but diverse

### 4. Disable Reranking for Speed
- Reranking adds ~200-300ms
- Use `use_reranking: false` for real-time apps

---

## 🔄 Integration Examples

### React Application

```typescript
// api.ts
const API_BASE_URL = 'http://localhost:8000';

export async function queryPolicy(query: string): Promise<QueryResponse> {
    const response = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-API-Key': process.env.REACT_APP_API_KEY
        },
        body: JSON.stringify({
            query,
            use_reranking: true,
            k: 5
        })
    });
    
    if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
    }
    
    return response.json();
}

// Component usage
const [answer, setAnswer] = useState('');

const handleQuery = async (question: string) => {
    try {
        const result = await queryPolicy(question);
        setAnswer(result.answer);
    } catch (error) {
        console.error('Query failed:', error);
    }
};
```

### Slack Bot

```python
# slack_bot.py
from slack_bolt import App
import requests

app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

@app.message("policy")
def handle_policy_question(message, say):
    question = message['text'].replace('policy', '').strip()
    
    # Query API
    response = requests.post(
        "http://localhost:8000/query",
        json={"query": question, "k": 3}
    )
    
    data = response.json()
    
    # Format response
    say(
        blocks=[
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": data['answer']}
            },
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"Confidence: {data['confidence']}"}
                ]
            }
        ]
    )

if __name__ == "__main__":
    app.start(port=3000)
```

---

## 📞 Support

**API Issues:**
- Check `/health` endpoint
- Review logs: `tail -f app.log`
- Check `/stats` for metrics

**Documentation:**
- Interactive docs: http://localhost:8000/docs
- This guide: `API_GUIDE.md`
- Deployment: `DEPLOYMENT.md`

**Contact:**
- GitHub Issues
- Email: support@yourcompany.com