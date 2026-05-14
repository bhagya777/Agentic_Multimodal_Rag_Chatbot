"""
RAGAS Evaluation for Multimodal RAG Pipeline
=============================================
Hooks into the chatbot — evaluates real conversations.

Two ways to use:

  1. LIVE MODE (inside Chainlit chatbot):
       from ragas_eval import evaluate_response
       scores = evaluate_response(question, contexts, answer)

  2. STANDALONE MODE:
       python ragas_eval.py -q "What is hallucination in LLMs?"

All metrics are reference-free — no ground truth needed.
Uses LOCAL Ollama models — no OpenAI key needed.
"""

import os, json, time, math
from dotenv import load_dotenv
load_dotenv()

from rag_langgraph import write_log

# ─── RAGAS imports ────────────────────────────────────────────────────────────
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    ContextUtilization,
)
from ragas.run_config import RunConfig

from langchain_ollama import ChatOllama, OllamaEmbeddings


# =============================================================================
# CONFIG
# =============================================================================
EVAL_MODEL        = "qwen2.5:14b"       # stronger judge — reliable structured JSON for RAGAS
EVAL_EMBED_MODEL  = "nomic-embed-text"  # keep as-is, works fine
EVAL_TIMEOUT      = 600                 # 10 min is enough; 30 min was masking hangs
MAX_EVAL_CONTEXTS = 5                   # 10 × 1000 chars overflows 8192 ctx; 5 × 500 is safe
MAX_CONTEXT_CHARS = 500                 # was 1000 — halve to stay within judge ctx window
RESULTS_DIR       = os.path.join(os.path.dirname(__file__), "ragas_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# LAZY-INIT evaluator models (created once, reused across calls)
# =============================================================================
_eval_llm = None
_eval_embeddings = None
_metrics = None
_run_config = None


def _init_evaluator():
    """Initialize RAGAS evaluator models (lazy, called once)."""
    global _eval_llm, _eval_embeddings, _metrics, _run_config

    if _eval_llm is not None:
        return  # already initialized

    write_log("[RAGAS] Initializing evaluator models...")

    _eval_llm = LangchainLLMWrapper(
        ChatOllama(              # ← HERE
            model=EVAL_MODEL,
            num_ctx=8192,        # ← change to 16384
            temperature=0.2,     # ← change to 0.1
            num_predict=4096,
            timeout=EVAL_TIMEOUT,
        )
    )
    _eval_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=EVAL_EMBED_MODEL, num_ctx=8192)
    )

    _metrics = [
        Faithfulness(llm=_eval_llm),
        ResponseRelevancy(llm=_eval_llm, embeddings=_eval_embeddings),
        ContextUtilization(llm=_eval_llm),
    ]

    _run_config = RunConfig(
        timeout=EVAL_TIMEOUT,
        max_retries=15,
        max_wait=300,
    )


def _run_single_metric(metric, dataset, run_config, metric_name: str, max_attempts: int = 3) -> float:
    """Run a single RAGAS metric with independent retry logic.
    
    Returns the score as a float, or NaN if all attempts fail.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            result = evaluate(
                dataset=dataset,
                metrics=[metric],
                run_config=run_config,
            )
            df = result.to_pandas()
            # Find the metric column (not a data column)
            data_cols = {"user_input", "retrieved_contexts", "response", "reference"}
            for col in df.columns:
                if col not in data_cols:
                    val = float(df[col].iloc[0])
                    if not math.isnan(val) and val > 0.0:
                        write_log(f"[RAGAS]   ✅ {metric_name}: {val:.4f} (attempt {attempt})")
                        return val
                    elif math.isnan(val):
                        write_log(f"[RAGAS]   ⚠️ {metric_name}: NaN on attempt {attempt}/{max_attempts}, retrying...")
                        break  # retry
                    else:
                        # Got 0.0 — could be legitimate or a parsing failure
                        if attempt < max_attempts:
                            write_log(f"[RAGAS]   ⚠️ {metric_name}: 0.0 on attempt {attempt}/{max_attempts}, retrying to confirm...")
                        # FIX — treat 0.0 same as NaN for faithfulness and context metrics
                        else:
                            if name in ("faithfulness", "context_utilization"):
                                write_log(f"[RAGAS]   ❌ {metric_name}: 0.0 likely parse failure, returning NaN")
                                return float('nan')
                            else:
                                write_log(f"[RAGAS]   ℹ️ {metric_name}: 0.0 confirmed (accepting as final)")
                                return val
                        break  # retry
 
        except Exception as e:
            write_log(f"[RAGAS]   ❌ {metric_name}: Error on attempt {attempt}/{max_attempts}: {e}")
        
        time.sleep(2)  # brief cooldown between retries
    
    write_log(f"[RAGAS]   ❌ {metric_name}: All {max_attempts} attempts failed, returning NaN")
    return float('nan')


# =============================================================================
# CORE FUNCTION — call this from anywhere (chatbot, script, notebook)
# =============================================================================
def evaluate_response(
    question: str,
    contexts: list[str],
    answer: str,
) -> dict:
    """
    Evaluate a single RAG response using RAGAS reference-free metrics.
    
    Each metric is run independently with its own retry logic to prevent
    one metric's failure from corrupting the others.

    Args:
        question:  the user's question
        contexts:  list of retrieved context strings
        answer:    the generated answer

    Returns:
        dict with metric scores, e.g.:
        {
            "faithfulness": 0.85,
            "answer_relevancy": 0.92,
            "context_precision": 0.78,
            "eval_time_seconds": 45.2
        }
    """
    _init_evaluator()

    write_log(f"[RAGAS] Evaluating: {question[:60]}...")
    start = time.time()

    # ── Context limiting: prevent timeout from context explosion ──
    original_count = len(contexts)
    # Deduplicate while preserving order
    seen = set()
    unique_contexts = []
    for c in contexts:
        c_stripped = c.strip()
        if c_stripped and c_stripped not in seen:
            seen.add(c_stripped)
            unique_contexts.append(c_stripped[:MAX_CONTEXT_CHARS])
    # Limit to top N
    contexts = unique_contexts[:MAX_EVAL_CONTEXTS]
    if original_count != len(contexts):
        write_log(f"[RAGAS] Reduced contexts from {original_count} to {len(contexts)} (dedup + limit to {MAX_EVAL_CONTEXTS}, truncated to {MAX_CONTEXT_CHARS} chars each)")

    # Build a single-sample dataset
    sample = SingleTurnSample(
        user_input=question,
        retrieved_contexts=contexts,
        response=answer,
    )
    dataset = EvaluationDataset(samples=[sample])

    # ── Run each metric independently with its own retry logic ──
    metric_names = ["faithfulness", "answer_relevancy", "context_utilization"]
    scores = {}
    for metric, name in zip(_metrics, metric_names):
        write_log(f"[RAGAS] Running metric: {name}...")
        scores[name] = _run_single_metric(metric, dataset, _run_config, name, max_attempts=3)

    elapsed = time.time() - start
    scores["eval_time_seconds"] = round(elapsed, 1)

    write_log(f"[RAGAS] Final Scores: {scores}")

    # Auto-save to results dir
    _save_single_result(question, answer, len(contexts), scores)

    return scores


# =============================================================================
# SAVE INDIVIDUAL RESULTS (appends to a JSONL log)
# =============================================================================
def _save_single_result(question, answer, num_contexts, scores):
    """Append each evaluation to a running log file."""
    log_path = os.path.join(RESULTS_DIR, "eval_log.jsonl")
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer_length": len(answer),
        "num_contexts": num_contexts,
        **scores,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# =============================================================================
# FORMAT SCORES for display (used by chatbot UI)
# =============================================================================
def format_scores(scores: dict) -> str:
    """Format RAGAS scores as a readable string for the chatbot."""
    lines = ["📊 **RAGAS Evaluation Scores**"]
    metric_labels = {
        "faithfulness":      "🎯 Faithfulness",
        "answer_relevancy":  "💡 Answer Relevancy",
        "context_utilization": "🔍 Context Precision",
        # Fallback for old log entries
        "context_precision":  "🔍 Context Precision",
        "llm_context_precision_without_reference": "🔍 Context Precision",
    }
    for key, label in metric_labels.items():
        if key in scores:
            val = scores[key]
            if isinstance(val, float) and math.isnan(val):
                bar = "⚠️ NaN Error"
                lines.append(f"  {label}: {bar}")
            else:
                try:
                    bar = "█" * int(val * 10) + "░" * (10 - int(val * 10))
                    lines.append(f"  {label}: {bar} **{val:.2f}**")
                except Exception:
                    lines.append(f"  {label}: ⚠️ Format Error ({val})")

    if "eval_time_seconds" in scores:
        lines.append(f"  ⏱️ Eval time: {scores['eval_time_seconds']}s")

    return "\n".join(lines)


# =============================================================================
# STANDALONE MODE — run from command line
# =============================================================================
if __name__ == "__main__":
    import argparse
    from rag_langgraph import query_graph

    parser = argparse.ArgumentParser(description="RAGAS Evaluation (reference-free)")
    parser.add_argument("-q", "--question", type=str, help="Question to evaluate")
    parser.add_argument("-f", "--file", type=str, help="File with one question per line")
    args = parser.parse_args()

    # Collect questions
    questions = []
    if args.question:
        questions = [args.question]
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            questions = [l.strip() for l in f if l.strip()]
    else:
        print("🧪 RAGAS Interactive Evaluation")
        print("   Type questions, press Enter on empty line to finish.\n")
        while True:
            q = input("❓ Question: ").strip()
            if not q:
                break
            questions.append(q)

    if not questions:
        print("No questions. Exiting.")
        exit()

    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {q}")

        # Run full pipeline
        result = query_graph.invoke({
            "question": q, "original_question": q,
            "retrieved_docs": [], "relevant_docs": [],
            "answer": "", "grounded": False, "retry_count": 0,
            "sources_footer": "", "image_paths": [],
            "failure_reason": "", "failure_critique": "",
            "rewritten_query": "", "rca_history": [],
        })

        docs = result.get("relevant_docs") or result.get("retrieved_docs") or []
        contexts = [d.page_content if hasattr(d, "page_content") else str(d) for d in docs]
        answer = result.get("answer", "")

        # Evaluate with RAGAS
        scores = evaluate_response(q, contexts, answer)
        print(format_scores(scores))
