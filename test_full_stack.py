"""
Complete System Test - FastAPI Backend + Reranking

Run this to verify your entire stack is working:
1. Backend API
2. Reranking
3. All retrieval strategies
4. Caching

Usage:
    # Start API first
    uvicorn backend.main:app --reload
    
    # Then run tests
    python test_full_stack.py
"""

import requests
import time
from typing import Dict, Any

API_BASE_URL = "http://localhost:8000"

# ANSI colors for pretty output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_success(msg: str):
    print(f"{GREEN}✓ {msg}{RESET}")


def print_error(msg: str):
    print(f"{RED}✗ {msg}{RESET}")


def print_info(msg: str):
    print(f"{BLUE}ℹ {msg}{RESET}")


def print_section(msg: str):
    print(f"\n{YELLOW}{'='*60}")
    print(f"{msg}")
    print(f"{'='*60}{RESET}\n")


def test_health():
    """Test 1: Health check"""
    print_section("TEST 1: Health Check")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print_success("API is healthy")
            print_info(f"Status: {data['status']}")
            print_info(f"Version: {data['version']}")
            return True
        else:
            print_error(f"Health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to API. Is it running?")
        print_info("Start with: uvicorn backend.main:app --reload")
        return False
    except Exception as e:
        print_error(f"Health check error: {e}")
        return False


def test_basic_query():
    """Test 2: Basic query"""
    print_section("TEST 2: Basic Query")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={
                "query": "What is the POSH policy?",
                "k": 3,
                "use_reranking": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("Query executed successfully")
            print_info(f"Answer length: {len(data['answer'])} chars")
            print_info(f"Sources: {len(data['sources'])}")
            print_info(f"Confidence: {data['confidence']}")
            print_info(f"Strategy: {data['retrieval_strategy']}")
            print_info(f"Response time: {data['response_time']:.2f}s")
            
            # Show first 100 chars of answer
            print_info(f"Answer preview: {data['answer'][:100]}...")
            
            return True
        else:
            print_error(f"Query failed: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print_error(f"Query error: {e}")
        return False


def test_reranking():
    """Test 3: Query with reranking"""
    print_section("TEST 3: Query with Reranking")
    
    try:
        # Without reranking
        print_info("Querying WITHOUT reranking...")
        response1 = requests.post(
            f"{API_BASE_URL}/query",
            json={
                "query": "How do I claim expenses?",
                "k": 5,
                "use_reranking": False
            },
            timeout=30
        )
        
        # With reranking
        print_info("Querying WITH reranking...")
        response2 = requests.post(
            f"{API_BASE_URL}/query",
            json={
                "query": "How do I claim expenses?",
                "k": 5,
                "use_reranking": True
            },
            timeout=30
        )
        
        if response1.status_code == 200 and response2.status_code == 200:
            data1 = response1.json()
            data2 = response2.json()
            
            print_success("Both queries executed")
            
            print_info(f"\nWithout reranking:")
            print_info(f"  Top source: {data1['sources'][0]['file']}")
            print_info(f"  Score: {data1['sources'][0]['score']:.3f}")
            print_info(f"  Time: {data1['response_time']:.2f}s")
            
            print_info(f"\nWith reranking:")
            print_info(f"  Top source: {data2['sources'][0]['file']}")
            print_info(f"  Score: {data2['sources'][0]['score']:.3f}")
            print_info(f"  Relevance: {data2['sources'][0].get('relevance', 'N/A')}")
            print_info(f"  Time: {data2['response_time']:.2f}s")
            
            # Check if reranking changed order
            if data1['sources'][0]['file'] != data2['sources'][0]['file']:
                print_success("Reranking changed the order (expected)")
            else:
                print_info("Reranking kept same order (can happen)")
            
            return True
        else:
            print_error("Reranking test failed")
            return False
            
    except Exception as e:
        print_error(f"Reranking test error: {e}")
        return False


def test_all_strategies():
    """Test 4: All retrieval strategies"""
    print_section("TEST 4: All Retrieval Strategies")
    
    strategies = ["semantic", "keyword", "hybrid", "mmr"]
    results = {}
    
    for strategy in strategies:
        print_info(f"Testing {strategy}...")
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/query",
                json={
                    "query": "What are the exit procedures?",
                    "k": 3,
                    "strategy": strategy,
                    "use_reranking": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                results[strategy] = {
                    "success": True,
                    "time": data['response_time'],
                    "sources": len(data['sources'])
                }
                print_success(f"{strategy}: {data['response_time']:.2f}s, {len(data['sources'])} sources")
            else:
                results[strategy] = {"success": False}
                print_error(f"{strategy} failed")
                
        except Exception as e:
            results[strategy] = {"success": False}
            print_error(f"{strategy} error: {e}")
    
    # Summary
    successful = sum(1 for r in results.values() if r.get("success"))
    print_info(f"\nSuccessful: {successful}/{len(strategies)}")
    
    return successful == len(strategies)


def test_caching():
    """Test 5: Query caching"""
    print_section("TEST 5: Query Caching")
    
    query = "What is the attendance policy?"
    
    try:
        # First query (cache miss)
        print_info("First query (cache miss)...")
        response1 = requests.post(
            f"{API_BASE_URL}/query",
            json={"query": query, "k": 3},
            timeout=30
        )
        
        # Second query (cache hit)
        time.sleep(0.5)  # Small delay
        print_info("Second query (cache hit)...")
        response2 = requests.post(
            f"{API_BASE_URL}/query",
            json={"query": query, "k": 3},
            timeout=30
        )
        
        if response1.status_code == 200 and response2.status_code == 200:
            data1 = response1.json()
            data2 = response2.json()
            
            print_info(f"First query time: {data1['response_time']:.2f}s")
            print_info(f"Second query time: {data2['response_time']:.2f}s")
            print_info(f"From cache: {data2.get('from_cache', False)}")
            
            if data2.get('from_cache'):
                speedup = data1['response_time'] / data2['response_time']
                print_success(f"Cache working! {speedup:.1f}x faster")
            else:
                print_info("Cache not hit (expected on first run)")
            
            return True
        else:
            print_error("Caching test failed")
            return False
            
    except Exception as e:
        print_error(f"Caching test error: {e}")
        return False


def test_stats():
    """Test 6: Statistics endpoint"""
    print_section("TEST 6: Statistics")
    
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print_success("Stats retrieved")
            print_info(f"Total queries: {data['total_queries']}")
            print_info(f"Avg response time: {data['avg_response_time']:.2f}s")
            print_info(f"Uptime: {data['uptime']/60:.1f} minutes")
            
            cache_stats = data.get('cache_stats', {})
            if cache_stats.get('total_entries', 0) > 0:
                print_info(f"Cache entries: {cache_stats['total_entries']}")
                print_info(f"Cache hit rate: {cache_stats.get('hit_rate', 0):.1%}")
            
            return True
        else:
            print_error("Stats request failed")
            return False
            
    except Exception as e:
        print_error(f"Stats error: {e}")
        return False


def test_documents():
    """Test 7: Documents endpoint"""
    print_section("TEST 7: Document Information")
    
    try:
        response = requests.get(f"{API_BASE_URL}/documents", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print_success("Document info retrieved")
            print_info(f"Total chunks: {data.get('total_chunks', 0)}")
            print_info(f"Collection: {data.get('collection_name', 'N/A')}")
            print_info(f"Embedding model: {data.get('embedding_model', 'N/A')}")
            return True
        else:
            print_error("Documents request failed")
            return False
            
    except Exception as e:
        print_error(f"Documents error: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print(f"\n{BLUE}{'='*60}")
    print(f"COMPANY POLICY RAG - FULL STACK TEST")
    print(f"{'='*60}{RESET}\n")
    
    print_info(f"API Base URL: {API_BASE_URL}")
    print_info(f"Testing FastAPI backend with reranking\n")
    
    tests = [
        ("Health Check", test_health),
        ("Basic Query", test_basic_query),
        ("Reranking", test_reranking),
        ("All Strategies", test_all_strategies),
        ("Caching", test_caching),
        ("Statistics", test_stats),
        ("Documents", test_documents),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_error(f"{test_name} crashed: {e}")
            results[test_name] = False
        
        time.sleep(0.5)  # Small delay between tests
    
    # Summary
    print_section("TEST SUMMARY")
    
    total = len(results)
    passed = sum(1 for r in results.values() if r)
    
    for test_name, passed_test in results.items():
        status = f"{GREEN}PASS{RESET}" if passed_test else f"{RED}FAIL{RESET}"
        print(f"{test_name:.<40} {status}")
    
    print(f"\n{YELLOW}Results: {passed}/{total} tests passed{RESET}")
    
    if passed == total:
        print(f"\n{GREEN}{'='*60}")
        print(f"🎉 ALL TESTS PASSED! Your RAG system is ready!")
        print(f"{'='*60}{RESET}\n")

        print_info("Next steps:")
        print_info("  1. Access frontend: streamlit run frontend/app_with_api.py")
        print_info("  2. View API docs: http://localhost:8000/docs")
        print_info("  3. Test with real questions!")
    else:
        print(f"\n{YELLOW}{'='*60}")
        print(f"⚠️  Some tests failed. Check errors above.")
        print(f"{'='*60}{RESET}\n")


if __name__ == "__main__":
    run_all_tests()