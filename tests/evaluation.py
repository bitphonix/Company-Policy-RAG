"""
Evaluation Framework for RAG System

Tests:
1. Retrieval quality (precision, recall)
2. Answer accuracy
3. Source attribution
4. Strategy comparison

Run: python tests/evaluation.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import json
from typing import List, Dict, Any
from dataclasses import dataclass
import time

from embedding import EmbeddingManager
from retrieval import AdvancedRetriever
from generation import AnswerGenerator


@dataclass
class TestCase:
    """Single test case"""
    query: str
    expected_sources: List[str]  # Expected document names
    category: str  # POSH, Expenses, Exit, etc.
    difficulty: str  # easy, medium, hard


# ============================================================================
# TEST CASES
# ============================================================================

TEST_CASES = [
    # POSH Policy
    TestCase(
        query="What is the POSH policy?",
        expected_sources=["Adda247 - Posh Policy.pdf"],
        category="POSH",
        difficulty="easy"
    ),
    TestCase(
        query="Who should I report sexual harassment to?",
        expected_sources=["Adda247 - Posh Policy.pdf"],
        category="POSH",
        difficulty="medium"
    ),
    
    # Expenses
    TestCase(
        query="How do I claim expenses?",
        expected_sources=["Expense Claim Guidline & Policy.pdf", "Flexi Allowance"],
        category="Expenses",
        difficulty="easy"
    ),
    TestCase(
        query="Can I claim food expenses?",
        expected_sources=["Flexi Allowance"],
        category="Expenses",
        difficulty="medium"
    ),
    
    # Exit
    TestCase(
        query="What are the exit procedures?",
        expected_sources=["HR03-FAQ-Exit.pdf", "Employee exit Check list.pdf"],
        category="Exit",
        difficulty="easy"
    ),
    TestCase(
        query="How many days is the notice period?",
        expected_sources=["HR03-FAQ-Exit.pdf"],
        category="Exit",
        difficulty="medium"
    ),
    
    # Salary
    TestCase(
        query="Can I get salary advance?",
        expected_sources=["Salary Advance Policy"],
        category="Salary",
        difficulty="easy"
    ),
    TestCase(
        query="What is the maximum salary advance amount?",
        expected_sources=["Salary Advance Policy"],
        category="Salary",
        difficulty="medium"
    ),
    
    # Attendance
    TestCase(
        query="What is the attendance policy?",
        expected_sources=["Metis_HR_Attendance SOP.pdf"],
        category="Attendance",
        difficulty="easy"
    ),
    
    # Hard cases
    TestCase(
        query="Compare POSH and Anti-Bribery policies",
        expected_sources=["Adda247 - Posh Policy.pdf", "Anti Bribery"],
        category="Comparative",
        difficulty="hard"
    ),
]


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_retrieval(
    test_case: TestCase,
    retriever: AdvancedRetriever,
) -> Dict[str, Any]:
    """
    Evaluate retrieval quality for a test case
    
    Metrics:
    - Precision@K: How many retrieved docs are relevant?
    - Recall: Did we get all expected sources?
    - MRR: Mean Reciprocal Rank
    """
    result = retriever.retrieve(test_case.query, k=5)
    
    retrieved_sources = [
        doc.metadata.get('source_file', '')
        for doc in result.documents
    ]
    
    # Calculate metrics
    relevant_retrieved = sum(
        1 for source in retrieved_sources
        if any(expected in source for expected in test_case.expected_sources)
    )
    
    precision = relevant_retrieved / len(retrieved_sources) if retrieved_sources else 0
    
    recall = sum(
        1 for expected in test_case.expected_sources
        if any(expected in source for source in retrieved_sources)
    ) / len(test_case.expected_sources)
    
    # Find rank of first relevant document (for MRR)
    first_relevant_rank = None
    for i, source in enumerate(retrieved_sources, 1):
        if any(expected in source for expected in test_case.expected_sources):
            first_relevant_rank = i
            break
    
    mrr = 1 / first_relevant_rank if first_relevant_rank else 0
    
    return {
        "query": test_case.query,
        "category": test_case.category,
        "difficulty": test_case.difficulty,
        "precision@5": precision,
        "recall": recall,
        "mrr": mrr,
        "strategy": result.strategy_used,
        "num_retrieved": len(result.documents),
        "retrieved_sources": retrieved_sources[:3],  # Top 3
    }


def evaluate_answer_quality(
    test_case: TestCase,
    generator: AnswerGenerator,
) -> Dict[str, Any]:
    """
    Evaluate answer quality
    
    Metrics:
    - Has answer (not "I don't know")
    - Has sources
    - Confidence level
    - Response time
    """
    start_time = time.time()
    
    answer = generator.generate_answer(test_case.query, k=5)
    
    response_time = time.time() - start_time
    
    # Simple quality checks
    has_answer = len(answer.answer) > 50 and "don't have" not in answer.answer.lower()
    has_sources = len(answer.sources) > 0
    
    # Check if expected sources are cited
    cited_sources = [s['file'] for s in answer.sources]
    correct_sources = sum(
        1 for expected in test_case.expected_sources
        if any(expected in cited for cited in cited_sources)
    )
    
    source_accuracy = correct_sources / len(test_case.expected_sources)
    
    return {
        "query": test_case.query,
        "category": test_case.category,
        "difficulty": test_case.difficulty,
        "has_answer": has_answer,
        "has_sources": has_sources,
        "num_sources": len(answer.sources),
        "source_accuracy": source_accuracy,
        "confidence": answer.confidence,
        "response_time": response_time,
        "answer_length": len(answer.answer),
    }


def compare_strategies(
    test_case: TestCase,
    retriever: AdvancedRetriever,
) -> Dict[str, Any]:
    """
    Compare all retrieval strategies for a query
    """
    strategies = ["semantic", "keyword", "hybrid", "mmr"]
    results = {}
    
    for strategy in strategies:
        result = retriever.retrieve(test_case.query, k=5, strategy=strategy)
        
        retrieved_sources = [
            doc.metadata.get('source_file', '')
            for doc in result.documents
        ]
        
        relevant = sum(
            1 for source in retrieved_sources
            if any(expected in source for expected in test_case.expected_sources)
        )
        
        results[strategy] = {
            "relevant_docs": relevant,
            "total_docs": len(result.documents),
            "precision": relevant / len(result.documents) if result.documents else 0,
        }
    
    # Find best strategy
    best_strategy = max(results.items(), key=lambda x: x[1]['precision'])
    
    return {
        "query": test_case.query,
        "strategies": results,
        "best_strategy": best_strategy[0],
        "best_precision": best_strategy[1]['precision'],
    }


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def run_evaluation():
    """Run complete evaluation suite"""
    print("\n" + "="*70)
    print("RAG SYSTEM EVALUATION")
    print("="*70 + "\n")
    
    # Initialize system
    print("Loading RAG system...")
    manager = EmbeddingManager()
    manager.load_vector_store()
    retriever = AdvancedRetriever(manager)
    generator = AnswerGenerator(retriever)
    print("✓ System loaded\n")
    
    # Run evaluations
    retrieval_results = []
    answer_results = []
    strategy_comparisons = []
    
    print(f"Running {len(TEST_CASES)} test cases...\n")
    
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"[{i}/{len(TEST_CASES)}] Testing: {test_case.query}")
        
        # Evaluate retrieval
        ret_result = evaluate_retrieval(test_case, retriever)
        retrieval_results.append(ret_result)
        
        # Evaluate answer
        ans_result = evaluate_answer_quality(test_case, generator)
        answer_results.append(ans_result)
        
        # Compare strategies (only for first 5 to save time)
        if i <= 5:
            comp_result = compare_strategies(test_case, retriever)
            strategy_comparisons.append(comp_result)
        
        print(f"  ✓ P@5: {ret_result['precision@5']:.2f}, "
              f"Confidence: {ans_result['confidence']}, "
              f"Time: {ans_result['response_time']:.2f}s\n")
    
    # ========================================================================
    # PRINT SUMMARY
    # ========================================================================
    
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70 + "\n")
    
    # Retrieval metrics
    avg_precision = sum(r['precision@5'] for r in retrieval_results) / len(retrieval_results)
    avg_recall = sum(r['recall'] for r in retrieval_results) / len(retrieval_results)
    avg_mrr = sum(r['mrr'] for r in retrieval_results) / len(retrieval_results)
    
    print("RETRIEVAL PERFORMANCE:")
    print(f"  Average Precision@5: {avg_precision:.2%}")
    print(f"  Average Recall:      {avg_recall:.2%}")
    print(f"  Average MRR:         {avg_mrr:.3f}")
    
    # Answer metrics
    answer_rate = sum(1 for r in answer_results if r['has_answer']) / len(answer_results)
    source_rate = sum(1 for r in answer_results if r['has_sources']) / len(answer_results)
    avg_source_acc = sum(r['source_accuracy'] for r in answer_results) / len(answer_results)
    avg_time = sum(r['response_time'] for r in answer_results) / len(answer_results)
    
    print("\nANSWER GENERATION:")
    print(f"  Answer Rate:         {answer_rate:.2%}")
    print(f"  Source Citation:     {source_rate:.2%}")
    print(f"  Source Accuracy:     {avg_source_acc:.2%}")
    print(f"  Avg Response Time:   {avg_time:.2f}s")
    
    # Confidence distribution
    confidence_dist = {}
    for r in answer_results:
        conf = r['confidence']
        confidence_dist[conf] = confidence_dist.get(conf, 0) + 1
    
    print("\nCONFIDENCE DISTRIBUTION:")
    for conf, count in sorted(confidence_dist.items()):
        print(f"  {conf.title()}: {count} ({count/len(answer_results):.1%})")
    
    # By category
    print("\nPERFORMANCE BY CATEGORY:")
    categories = set(r['category'] for r in retrieval_results)
    for category in sorted(categories):
        cat_results = [r for r in retrieval_results if r['category'] == category]
        cat_precision = sum(r['precision@5'] for r in cat_results) / len(cat_results)
        print(f"  {category}: {cat_precision:.2%} precision")
    
    # Strategy comparison
    if strategy_comparisons:
        print("\nSTRATEGY COMPARISON (First 5 queries):")
        strategy_wins = {}
        for comp in strategy_comparisons:
            best = comp['best_strategy']
            strategy_wins[best] = strategy_wins.get(best, 0) + 1
        
        for strategy, wins in sorted(strategy_wins.items(), key=lambda x: -x[1]):
            print(f"  {strategy.title()}: {wins} wins")
    
    print("\n" + "="*70 + "\n")
    
    # Save results
    results_file = Path(__file__).parent / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "retrieval_results": retrieval_results,
            "answer_results": answer_results,
            "strategy_comparisons": strategy_comparisons,
            "summary": {
                "avg_precision": avg_precision,
                "avg_recall": avg_recall,
                "avg_mrr": avg_mrr,
                "answer_rate": answer_rate,
                "source_accuracy": avg_source_acc,
                "avg_response_time": avg_time,
            }
        }, f, indent=2)
    
    print(f"✓ Results saved to {results_file}")
    
    return {
        "retrieval_results": retrieval_results,
        "answer_results": answer_results,
        "summary": {
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "answer_rate": answer_rate,
        }
    }


if __name__ == "__main__":
    run_evaluation()