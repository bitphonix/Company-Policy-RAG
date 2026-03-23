# 🚀 Production Deployment Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     Users/Clients                        │
└────────────┬────────────────────────────┬────────────────┘
             │                            │
             ▼                            ▼
    ┌────────────────┐          ┌────────────────┐
    │   Streamlit    │          │   Direct API   │
    │   Frontend     │          │    Clients     │
    │   (Port 8501)  │          │                │
    └────────┬───────┘          └────────┬───────┘
             │                            │
             └────────────┬───────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   FastAPI Backend     │
              │   (Port 8000)         │
              │   - Authentication    │
              │   - Rate Limiting     │
              │   - Query Processing  │
              └───────────┬───────────┘
                          │
             ┌────────────┼────────────┐
             │            │            │
             ▼            ▼            ▼
    ┌──────────────┐ ┌──────────┐ ┌─────────┐
    │   ChromaDB   │ │  OpenAI  │ │  Cache  │
    │  Vector DB   │ │    API   │ │ (Redis) │
    └──────────────┘ └──────────┘ └─────────┘
```

---

## 📋 Deployment Options

### Option 1: Docker Deployment (Recommended)

**Create `Dockerfile`:**

```dockerfile
# Backend Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY backend/ ./backend/
COPY data/ ./data/
COPY chroma_db/ ./chroma_db/

# Environment
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Create `docker-compose.yml`:**

```yaml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./chroma_db:/app/chroma_db
      - ./data:/app/data
    restart: unless-stopped
    
  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "8501:8501"
    environment:
      - API_BASE_URL=http://backend:8000
    depends_on:
      - backend
    restart: unless-stopped
    
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
```

**Run:**
```bash
docker-compose up -d
```

---

### Option 2: Cloud Deployment

#### **A. AWS Deployment**

**Using AWS Elastic Beanstalk:**

1. **Install EB CLI:**
```bash
pip install awsebcli
```

2. **Initialize:**
```bash
eb init -p python-3.11 company-policy-rag
```

3. **Create environment:**
```bash
eb create production-env
```

4. **Deploy:**
```bash
eb deploy
```

**Environment Variables in AWS:**
```bash
eb setenv OPENAI_API_KEY=your_key_here
```

#### **B. Google Cloud Run**

1. **Build container:**
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/policy-rag-backend
```

2. **Deploy:**
```bash
gcloud run deploy policy-rag-backend \
  --image gcr.io/PROJECT_ID/policy-rag-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=your_key_here
```

#### **C. Heroku**

1. **Create `Procfile`:**
```
web: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

2. **Deploy:**
```bash
heroku create company-policy-rag
heroku config:set OPENAI_API_KEY=your_key_here
git push heroku main
```

---

### Option 3: VPS/Server Deployment

**Using Nginx + Systemd:**

1. **Create systemd service (`/etc/systemd/system/policy-rag.service`):**

```ini
[Unit]
Description=Company Policy RAG API
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/var/www/policy-rag
Environment="PATH=/var/www/policy-rag/venv/bin"
EnvironmentFile=/var/www/policy-rag/.env
ExecStart=/var/www/policy-rag/venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8000 --workers 4

[Install]
WantedBy=multi-user.target
```

2. **Nginx config (`/etc/nginx/sites-available/policy-rag`):**

```nginx
server {
    listen 80;
    server_name policy-rag.yourcompany.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

3. **Enable and start:**
```bash
sudo systemctl enable policy-rag
sudo systemctl start policy-rag
sudo nginx -s reload
```

---

## 🔒 Security Hardening

### 1. Enable Authentication

**Update `backend/main.py`:**

```python
# Uncomment in endpoints
# api_env: str = Depends(verify_api_key)

# Store keys securely (use environment variables)
API_KEYS = {
    os.getenv("API_KEY_DEV"): "development",
    os.getenv("API_KEY_PROD"): "production",
}
```

### 2. Add Rate Limiting

**Install:**
```bash
pip install slowapi
```

**Add to `backend/main.py`:**
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

@app.post("/query")
@limiter.limit("10/minute")  # 10 queries per minute
async def query(...):
    ...
```

### 3. HTTPS/SSL

**Using Let's Encrypt (Certbot):**

```bash
sudo certbot --nginx -d policy-rag.yourcompany.com
```

### 4. CORS Configuration

**Restrict origins in production:**

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-frontend.com",
        "https://app.yourcompany.com"
    ],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
```

---

## ⚡ Performance Optimization

### 1. Use Redis for Caching

**Install:**
```bash
pip install redis aioredis
```

**Update `cache_manager.py` to use Redis:**

```python
import redis

class RedisCache:
    def __init__(self):
        self.redis_client = redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=True
        )
    
    def get(self, key):
        data = self.redis_client.get(key)
        return json.loads(data) if data else None
    
    def set(self, key, value, ttl=3600):
        self.redis_client.setex(
            key,
            ttl,
            json.dumps(value)
        )
```

### 2. Database Connection Pooling

**For production ChromaDB:**

```python
# Use persistent connection
vector_store = Chroma(
    persist_directory=CHROMA_PERSIST_DIRECTORY,
    embedding_function=embeddings,
)
```

### 3. Async Operations

Backend is already async-ready. For heavy operations:

```python
@app.post("/query")
async def query(...):
    # Run retrieval in thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        system.retriever.retrieve,
        request.query
    )
```

### 4. Load Balancing

**Using Nginx:**

```nginx
upstream backend {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}

server {
    location / {
        proxy_pass http://backend;
    }
}
```

**Run multiple workers:**
```bash
uvicorn backend.main:app --workers 4 --port 8000
```

---

## 📊 Monitoring & Logging

### 1. Add Logging

**Structured logging:**

```python
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "default",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["file"],
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
```

### 2. Metrics with Prometheus

```bash
pip install prometheus-fastapi-instrumentator
```

```python
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

### 3. Health Checks

Already implemented at `/health`

**Monitor with:**
```bash
curl http://localhost:8000/health
```

---

## 🔄 CI/CD Pipeline

### GitHub Actions Example

**`.github/workflows/deploy.yml`:**

```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        python tests/evaluation.py
    
    - name: Deploy to production
      env:
        DEPLOY_KEY: ${{ secrets.DEPLOY_KEY }}
      run: |
        # Your deployment script
        ./deploy.sh
```

---

## 📈 Scaling Strategies

### 1. Horizontal Scaling

- **Multiple API instances** behind load balancer
- **Shared Redis cache** across instances
- **Separate vector DB** server

### 2. Vertical Scaling

- **Increase worker count**: `--workers 8`
- **More RAM** for embedding model
- **GPU** for faster inference

### 3. Database Scaling

- **Chroma Cloud** for managed vector DB
- **Pinecone** as alternative
- **Sharding** for large document sets

---

## ✅ Pre-Deployment Checklist

- [ ] Environment variables set
- [ ] API keys secured
- [ ] Authentication enabled
- [ ] Rate limiting configured
- [ ] CORS properly restricted
- [ ] HTTPS/SSL enabled
- [ ] Logging configured
- [ ] Error monitoring setup
- [ ] Backups automated
- [ ] Health checks working
- [ ] Load testing completed
- [ ] Documentation updated

---

## 🆘 Troubleshooting

### API Not Responding

```bash
# Check if running
ps aux | grep uvicorn

# Check logs
tail -f app.log

# Test locally
curl http://localhost:8000/health
```

### High Memory Usage

```bash
# Monitor
htop

# Solution: Reduce workers or enable swap
```

### Slow Responses

- Check OpenAI API limits
- Enable Redis caching
- Increase worker count
- Use GPU for embeddings

---

## 📞 Support

For issues:
1. Check logs: `tail -f app.log`
2. Test health: `curl /health`
3. Review metrics: `curl /stats`

---

**Next Steps:**
1. Choose deployment option
2. Follow security hardening
3. Set up monitoring
4. Configure CI/CD
5. Load test
6. Deploy! 🚀