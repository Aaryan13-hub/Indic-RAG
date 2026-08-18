import os
import sys
import json
import time
import shutil
import numpy as np
import groq
from typing import List

from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

from src.core.embeddings import EmbeddingModel
from src.core.vectorstore import QdrantVectorStore
from src.core.orchestrator import _SYSTEM_PROMPT

def load_eval_queries(num_queries: int = 30) -> List[str]:
    with open("benchmark_artifacts/eval_map.json", "r", encoding="utf-8") as f:
        eval_map = json.load(f)
    with open("benchmark_artifacts/queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
        
    valid_query_ids = [qid for qid, items in eval_map.items() if any(i["is_selected"] == 1 for i in items)]
    return [queries[qid] for qid in valid_query_ids[:num_queries]]

def main():
    temp_db_path = "./qdrant_hindi_benchmark_groq_diag"
    if os.path.exists(temp_db_path):
        shutil.rmtree(temp_db_path, ignore_errors=True)
    shutil.copytree("./qdrant_hindi_benchmark", temp_db_path, ignore=shutil.ignore_patterns("*.lock"))
    
    print("Loading embedding model...")
    embed_model = EmbeddingModel("intfloat/multilingual-e5-small")
    
    print("Initializing Qdrant...")
    db = QdrantVectorStore(
        collection_name="hindi_rag_production",
        persist_directory=temp_db_path,
        vector_dim=384,
        query_prefix="query: ",
        passage_prefix="passage: "
    )
    db.embedding_model = embed_model
    
    queries = load_eval_queries(30)
    print(f"Loaded {len(queries)} queries.")
    
    client = groq.Groq()
    model_name = "openai/gpt-oss-20b"
    reasoning_effort = "low"
    max_completion_tokens = 150
    
    results = []
    
    for i, q in enumerate(queries):
        print(f"\n[{i+1}/{len(queries)}] Query: {q[:40]}...")
        
        # 1. Retrieval
        retrieval_start = time.perf_counter()
        db_results = db.query(q, k=2, with_scores=True)
        retrieval_time = (time.perf_counter() - retrieval_start) * 1000
        
        context_texts = [r["chunk"] for r in db_results]
        context_str = "\n\n".join(context_texts)
        prompt = f"<context>\n{context_str}\n</context>\n\nQuestion: {q}\nAnswer:"
        
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        client_start = time.perf_counter()
        response_stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            reasoning_effort=reasoning_effort,
            max_completion_tokens=max_completion_tokens,
            stream=True
        )
        
        first_token_time = None
        final_usage = None
        
        for chunk in response_stream:
            if not first_token_time:
                first_token_time = time.perf_counter()
            if hasattr(chunk, "x_groq") and chunk.x_groq and chunk.x_groq.usage:
                final_usage = chunk.x_groq.usage
                
        client_end = time.perf_counter()
        
        ttft_ms = (first_token_time - client_start) * 1000 if first_token_time else 0
        total_client_ms = (client_end - client_start) * 1000
        
        if final_usage:
            queue_ms = final_usage.queue_time * 1000 if getattr(final_usage, "queue_time", None) else 0
            prompt_ms = final_usage.prompt_time * 1000 if getattr(final_usage, "prompt_time", None) else 0
            completion_ms = final_usage.completion_time * 1000 if getattr(final_usage, "completion_time", None) else 0
            server_total_ms = final_usage.total_time * 1000 if getattr(final_usage, "total_time", None) else 0
            
            p_tokens = final_usage.prompt_tokens
            c_tokens = final_usage.completion_tokens
            
            r_tokens = 0
            if getattr(final_usage, "completion_tokens_details", None) and getattr(final_usage.completion_tokens_details, "reasoning_tokens", None):
                r_tokens = final_usage.completion_tokens_details.reasoning_tokens
            
            # Groq does not seem to reliably return cached prompt tokens yet, but let's check
            cached_tokens = 0
            if getattr(final_usage, "prompt_tokens_details", None) and getattr(final_usage.prompt_tokens_details, "cached_tokens", None):
                cached_tokens = final_usage.prompt_tokens_details.cached_tokens
        else:
            queue_ms = prompt_ms = completion_ms = server_total_ms = 0
            p_tokens = c_tokens = r_tokens = cached_tokens = 0
            
        overhead_ms = total_client_ms - server_total_ms
        
        print(f"  Client TTFT: {ttft_ms:.1f} ms | Client Total: {total_client_ms:.1f} ms")
        print(f"  Groq Queue: {queue_ms:.1f} ms | Prompt: {prompt_ms:.1f} ms | Completion: {completion_ms:.1f} ms")
        print(f"  Tokens: Prompt {p_tokens} (Cached {cached_tokens}) | Completion {c_tokens} (Reasoning {r_tokens})")
        print(f"  Network/Client Overhead: {overhead_ms:.1f} ms")
        
        results.append({
            "client_total_ms": total_client_ms,
            "client_ttft_ms": ttft_ms,
            "queue_ms": queue_ms,
            "prompt_ms": prompt_ms,
            "completion_ms": completion_ms,
            "server_total_ms": server_total_ms,
            "overhead_ms": overhead_ms,
            "p_tokens": p_tokens,
            "c_tokens": c_tokens,
            "r_tokens": r_tokens,
        })
        
        # Pace the requests to avoid hammering the API
        time.sleep(2.0)
        
    print("\n" + "="*50)
    print("DIAGNOSTICS SUMMARY (30 Requests paced at 2s)")
    print("="*50)
    
    metrics = ["client_total_ms", "client_ttft_ms", "queue_ms", "prompt_ms", "completion_ms", "server_total_ms", "overhead_ms", "p_tokens", "c_tokens", "r_tokens"]
    
    for metric in metrics:
        vals = [r[metric] for r in results]
        p50 = np.percentile(vals, 50)
        p70 = np.percentile(vals, 70)
        p100 = np.percentile(vals, 100)
        mean = np.mean(vals)
        print(f"{metric:>18}: P50={p50:6.1f} | P70={p70:6.1f} | Max={p100:6.1f} | Mean={mean:6.1f}")
        
    if os.path.exists(temp_db_path):
        del db
        time.sleep(1)
        try:
            shutil.rmtree(temp_db_path, ignore_errors=True)
        except:
            pass

if __name__ == "__main__":
    main()
