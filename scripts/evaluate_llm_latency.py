import os
import sys
import json
import time
import shutil
import numpy as np
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.embeddings import EmbeddingModel
from src.core.vectorstore import QdrantVectorStore
from src.core.llm import GroqLLMBackend
from src.core.orchestrator import VoiceRAGOrchestrator

def load_eval_queries(num_queries: int = 35) -> List[str]:
    # Load mapping of queries that have at least one relevant passage
    with open("benchmark_artifacts/eval_map.json", "r", encoding="utf-8") as f:
        eval_map = json.load(f)
        
    with open("benchmark_artifacts/queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
        
    valid_query_ids = []
    for qid, items in eval_map.items():
        if any(i["is_selected"] == 1 for i in items):
            valid_query_ids.append(qid)
            
    selected_ids = valid_query_ids[:num_queries]
    return [queries[qid] for qid in selected_ids]

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM latency")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b")
    parser.add_argument("--reasoning", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()

    # We copy the db to avoid locking the running api.py
    temp_db_path = "./qdrant_hindi_benchmark_llm_test"
    if os.path.exists(temp_db_path):
        shutil.rmtree(temp_db_path)
    shutil.copytree("./qdrant_hindi_benchmark", temp_db_path, ignore=shutil.ignore_patterns("*.lock"))
    
    print(f"Loading embedding model...")
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
    
    print(f"Loading LLM backend: {args.model}")
    llm = GroqLLMBackend(
        model_name=args.model,
        reasoning_effort=args.reasoning,
        max_completion_tokens=args.max_tokens
    )
    
    print("Creating dummy STT...")
    
    # Dummy STT
    class DummySTT:
        def transcribe(self, path, lang=None): return ""
        
    orchestrator = VoiceRAGOrchestrator(
        stt_client=DummySTT(),
        vector_store=db,
        llm_backend=llm,
        embedding_model=embed_model,
        off_topic_threshold=0.826,
        groundedness_threshold=0.75,
    )
    
    queries = load_eval_queries(35)
    print(f"Evaluating {len(queries)} queries...\n")
    
    llm_latencies = []
    e2e_latencies = []
    grounded_count = 0
    refused_off_topic = 0
    
    for i, q in enumerate(queries):
        start_time = time.perf_counter()
        
        # Bypass STT in orchestrator manually since we already have text
        # But wait, orchestrator.process_audio() expects an audio file!
        # We can just manually call the same pipeline steps, OR mock STT to return 'q'.
        orchestrator.stt_client.transcribe = lambda path, lang=None: q
        
        # Use a dummy audio file
        dummy_audio = "dummy.wav"
        with open(dummy_audio, "w") as f: f.write("dummy")
        
        response = orchestrator.process_voice_query(dummy_audio)
        
        e2e_ms = (time.perf_counter() - start_time) * 1000
        
        status = response.get("status")
        if status == "answered":
            llm_latencies.append(response.get("llm_stats", {}).get("total_time_ms", 0))
            e2e_latencies.append(e2e_ms)
            grounded_count += 1
            print(f"[{i+1}/{len(queries)}] {q[:30]}... -> OK (LLM: {llm_latencies[-1]:.0f}ms, E2E: {e2e_ms:.0f}ms)")
        elif status == "refused_off_topic":
            refused_off_topic += 1
            print(f"[{i+1}/{len(queries)}] {q[:30]}... -> OFF-TOPIC")
        elif status == "refused_not_grounded":
            llm_latencies.append(response.get("llm_stats", {}).get("total_time_ms", 0))
            e2e_latencies.append(e2e_ms)
            print(f"[{i+1}/{len(queries)}] {q[:30]}... -> UNGROUNDED (LLM: {llm_latencies[-1]:.0f}ms, E2E: {e2e_ms:.0f}ms)")
        else:
            print(f"[{i+1}/{len(queries)}] {q[:30]}... -> ERROR")
            
        os.remove(dummy_audio)
        
        # Pace the requests to allow the persistent HTTP keep-alive connection to handle traffic organically
        # and to avoid Groq rate-limit queuing.
        time.sleep(2.0)

    # Cleanup DB
    if os.path.exists(temp_db_path):
        # Allow Qdrant client to release locks before rmtree
        del db
        time.sleep(1)
        try:
            shutil.rmtree(temp_db_path)
        except:
            pass

    print("\n" + "="*50)
    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Reasoning: {args.reasoning}")
    print(f"  Max Tokens: {args.max_tokens}")
    print("="*50)
    print(f"Valid Results: {len(llm_latencies)}/{len(queries)}")
    print(f"Off-Topic Skips: {refused_off_topic}")
    print(f"Groundedness Pass Rate: {(grounded_count/len(llm_latencies)*100) if llm_latencies else 0:.1f}%")
    print("-" * 50)
    print(f"LLM Generation Latency (ms):")
    print(f"  P50:  {np.percentile(llm_latencies, 50):.0f}")
    print(f"  P70:  {np.percentile(llm_latencies, 70):.0f}")
    print(f"  P100: {np.percentile(llm_latencies, 100):.0f}")
    print("-" * 50)
    print(f"End-to-End Latency (ms):")
    print(f"  P50:  {np.percentile(e2e_latencies, 50):.0f}")
    print(f"  P70:  {np.percentile(e2e_latencies, 70):.0f}")
    print(f"  P100: {np.percentile(e2e_latencies, 100):.0f}")
    print("="*50)

if __name__ == "__main__":
    main()
