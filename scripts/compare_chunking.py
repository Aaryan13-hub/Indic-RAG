"""
scripts/compare_chunking.py
===========================
A/B test chunking strategies for Hindi retrieval.

Models fixed to: intfloat/multilingual-e5-small
Data fixed to: hintrain.parquet (1,000 sample, seed=42)

Compares:
    1. RecursiveChunker (max 1000, overlap 200)
    2. SentenceChunker  (max 1000, overlap 200)

Usage:
    python -m scripts.compare_chunking
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from src.core.chunking import RecursiveChunker, SentenceChunker
from scripts.ingest_hindi import load_and_sample, extract_and_deduplicate

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QDRANT_BASE_PATH = str(PROJECT_ROOT / "qdrant_hindi_benchmark")
RESULTS_DIR = PROJECT_ROOT / "benchmark_results"

MODEL_HF_ID = "intfloat/multilingual-e5-small"
Q_PREFIX = "query: "
P_PREFIX = "passage: "
BATCH_SIZE = 256
TOP_K_VALUES = [1, 5, 10]

def build_ground_truth(eval_map):
    gt = {}
    for qid, entries in eval_map.items():
        gt[qid] = {e["passage_text"].strip() for e in entries if e["is_selected"] == 1}
    return gt

def check_relevant(chunk_text, relevant_passages):
    chunk_text = chunk_text.strip()
    for p in relevant_passages:
        if chunk_text in p:
            return True
    return False

def run_chunker_eval(
    name: str,
    chunker,
    unique_passages,
    queries,
    ground_truth,
    model,
    dim
):
    print(f"\n=======================================")
    print(f"  Testing Chunker: {name}")
    print(f"=======================================")
    
    # 1. Chunk
    chunks = []
    chunk_counter = 0
    lengths = []
    per_passage = []
    
    for p in unique_passages:
        cs = chunker.chunk(p["text"])
        per_passage.append(len(cs))
        for c in cs:
            chunks.append({
                "chunk_id": chunk_counter, "text": c,
                "passage_id": p["passage_id"], "query_ids": p["query_ids"]
            })
            lengths.append(len(c))
            chunk_counter += 1
            
    stats = {
        "num_chunks": len(chunks),
        "avg_chunks_per_passage": float(np.mean(per_passage)),
        "avg_chunk_length": float(np.mean(lengths)),
    }
    print(f"[CHUNK] {len(chunks)} chunks (avg {stats['avg_chunk_length']:.0f} chars)")
    
    # 2. Embed
    texts = [P_PREFIX + c["text"] for c in chunks]
    print(f"[EMBED] Encoding {len(texts)} chunks ...")
    t0 = time.perf_counter()
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True).tolist()
    total_s = time.perf_counter() - t0
    
    # 3. Index
    collection = f"hindi_chunk_{name}"
    client = QdrantClient(path=QDRANT_BASE_PATH)
    try:
        client.delete_collection(collection)
    except Exception:
        pass
        
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    
    points = [
        PointStruct(
            id=chunks[i]["chunk_id"],
            vector=embeddings[i],
            payload={"chunk": chunks[i]["text"], "passage_id": chunks[i]["passage_id"]}
        )
        for i in range(len(chunks))
    ]
    
    # Upload in batches of 500 to avoid memory issues in Qdrant local client
    UPSERT_BATCH = 500
    for start in range(0, len(points), UPSERT_BATCH):
        end = min(start + UPSERT_BATCH, len(points))
        client.upsert(collection_name=collection, points=points[start:end])
    
    # 4. Evaluate
    print(f"[EVAL] Evaluating ...")
    query_ids = sorted(queries.keys())
    max_k = max(TOP_K_VALUES)
    
    recall_at = {k: [] for k in TOP_K_VALUES}
    mrr_5 = []
    total_lats = []
    skipped = 0
    
    for qid in query_ids:
        relevant = ground_truth.get(qid, set())
        if not relevant:
            skipped += 1
            continue
            
        t_total = time.perf_counter()
        qvec = model.encode(Q_PREFIX + queries[qid]).tolist()
        hits = client.query_points(collection_name=collection, query=qvec, limit=max_k)
        total_ms = (time.perf_counter() - t_total) * 1000
        
        relevance = [check_relevant(h.payload.get("chunk", ""), relevant) for h in hits.points]
        
        for k in TOP_K_VALUES:
            recall_at[k].append(1.0 if any(relevance[:k]) else 0.0)
        mrr_5.append(next((1.0 / (j + 1) for j, r in enumerate(relevance[:5]) if r), 0.0))
        total_lats.append(total_ms)
        
    client.close()
    
    result = {
        "chunker": name,
        "num_chunks": stats["num_chunks"],
        "avg_chunk_length": round(stats["avg_chunk_length"], 0),
        "embedding_time_s": round(total_s, 1),
    }
    for k in TOP_K_VALUES:
        result[f"Recall@{k}"] = round(float(np.mean(recall_at[k])) * 100, 2)
    result["MRR@5"] = round(float(np.mean(mrr_5)) * 100, 2)
    result["Latency_P50"] = round(float(np.percentile(total_lats, 50)), 1)
    
    print(f"  Recall@5: {result['Recall@5']:.2f}%  |  MRR@5: {result['MRR@5']:.2f}%  |  P50: {result['Latency_P50']} ms")
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default=str(PROJECT_ROOT / ".cache/hintrain.parquet"))
    args = parser.parse_args()
    
    # We rely on the hardcoded default path in ingest_hindi if --parquet not set correctly here
    from scripts.ingest_hindi import DEFAULT_PARQUET_PATH
    parquet_path = DEFAULT_PARQUET_PATH
    
    df = load_and_sample(parquet_path, 1000, 42)
    unique_passages, eval_map = extract_and_deduplicate(df)
    queries = dict(zip(df["query_id"].astype(int).tolist(), df["query"].tolist()))
    ground_truth = build_ground_truth(eval_map)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_HF_ID} on {device} ...")
    model = SentenceTransformer(MODEL_HF_ID, device=device)
    dim = model.get_sentence_embedding_dimension()
    
    results = []
    
    # Run Recursive
    res_rec = run_chunker_eval("recursive", RecursiveChunker(), unique_passages, queries, ground_truth, model, dim)
    results.append(res_rec)
    
    # Run Sentence
    res_sen = run_chunker_eval("sentence", SentenceChunker(), unique_passages, queries, ground_truth, model, dim)
    results.append(res_sen)
    
    print("\n==========================================================================")
    print(f"  Chunker         Recall@5    MRR@5      P50 Latency    Chunks    Avg Len")
    print("--------------------------------------------------------------------------")
    for r in results:
        print(f"  {r['chunker']:<15s} {r['Recall@5']:>8.2f}% {r['MRR@5']:>8.2f}% {r['Latency_P50']:>11.1f} ms "
              f"{r['num_chunks']:>9d} {r['avg_chunk_length']:>10.0f}")
    print("==========================================================================\n")
    
    # Save append to MD
    md_path = RESULTS_DIR / "HINDI_EMBEDDING_BENCHMARK.md"
    if md_path.exists():
        with open(md_path, "a", encoding="utf-8") as f:
            f.write("\n## Chunking Strategy Comparison\n\n")
            f.write("Model fixed to `intfloat/multilingual-e5-small`. 1,000 sampled rows.\n\n")
            f.write("| Chunker | Recall@5 | MRR@5 | P50 Latency | Total Chunks | Avg Length |\n")
            f.write("|---------|----------|-------|-------------|--------------|------------|\n")
            for r in results:
                f.write(f"| {r['chunker']} | {r['Recall@5']:.2f}% | {r['MRR@5']:.2f}% | {r['Latency_P50']:.1f} ms | {r['num_chunks']} | {r['avg_chunk_length']:.0f} |\n")
            f.write("\n")
            
        print(f"Appended results to {md_path}")

if __name__ == "__main__":
    main()
