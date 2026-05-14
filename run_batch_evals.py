import time, asyncio, sys
from rag_langgraph import query_graph, write_log
from ragas_eval import evaluate_response, format_scores

# The 10 highly technical evaluation queries
EVAL_QUERIES = [
    "Explain the architectural differences between naive RAG and agentic self-corrective RAG.",
    "How does FlashRank improve the precision of document retrieval in a RAG pipeline?",
    "What are the primary causes of hallucination in Large Language Models?",
    "Describe the Multi-Vector Retrieval technique and its advantages over standard chunking.",
    "How do quantized vision models process and extract data from technical diagrams?",
    "What are the limitations of relying solely on zero-shot prompting for complex reasoning tasks?",
    "How can LangGraph be utilized to build cyclic, stateful workflows for language agents?",
    "What is the role of a 'Critic' or 'Grounding Checker' in a self-healing LLM architecture?",
    "Provide an overview of the performance trade-offs when using smaller parameter models (like 3B vs 8B) for evaluation.",
    "What is Alternating Refined Binarizations (ARB-LLM) and how does it impact model compression?"
]

async def run_batch_evaluation():
    print("🚀 Starting Batch Evaluation of 10 Queries...")
    print("=============================================\n")
    
    for i, q in enumerate(EVAL_QUERIES, 1):
        print(f"[{i}/10] ❓ Query: {q}")
        start_time = time.time()
        
        # 1. Run the agentic RAG pipeline
        print("  ⏳ Running agentic pipeline (retrieval + generation + grounding + RCA)...")
        result = await query_graph.ainvoke({
            "question": q, "original_question": q,
            "retrieved_docs": [], "relevant_docs": [],
            "answer": "", "grounded": False, "retry_count": 0,
            "sources_footer": "", "image_paths": [],
            "failure_reason": "", "failure_critique": "",
            "rewritten_query": "", "rca_history": [],
        }, config={"run_name": f"batch_eval: {q[:30]}"})
        
        # Extract results — deduplicate and limit contexts to prevent RAGAS timeout
        docs = result.get("relevant_docs") or result.get("retrieved_docs") or []
        seen = set()
        contexts = []
        for d in docs:
            content = d.page_content if hasattr(d, "page_content") else str(d)
            content_key = content.strip()[:500]  # dedup key
            if content_key and content_key not in seen:
                seen.add(content_key)
                contexts.append(content[:1000])   # truncate each chunk
        raw_count = len(docs)
        contexts = contexts[:10]                  # keep top 10 only
        answer = result.get("answer", "")
        
        print(f"  ✅ Pipeline finished in {time.time() - start_time:.1f}s. Answer: {len(answer)} chars. Contexts: {len(contexts)} (from {raw_count} raw docs).")
        
        # 2. Run RAGAS Evaluation
        print("  🧪 Running RAGAS Evaluation (Context Precision, Faithfulness, Answer Relevancy)...")
        try:
            scores = evaluate_response(q, contexts, answer)
            print(format_scores(scores))
        except Exception as e:
            print(f"  ❌ RAGAS Evaluation failed for this query: {e}")
            
        print("\n" + "-"*60 + "\n")
        
    print("🎉 Batch Evaluation Complete! All results have been appended to ragas_results/eval_log.jsonl")
    print("You can view the aggregate metrics by running the Streamlit dashboard: streamlit run dashboard.py")

if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
    asyncio.run(run_batch_evaluation())
