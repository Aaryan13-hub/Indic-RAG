"""
scripts/evaluate_hindi_retrieval.py
====================================
Retrieval evaluation for Hindi embeddings benchmark.

Usage:
    python -m scripts.evaluate_hindi_retrieval
    python -m scripts.evaluate_hindi_retrieval --model intfloat/multilingual-e5-small --query-prefix "query: "

Evaluates:
    Recall@1, Recall@5, Recall@10, MRR@5
    Latency: embedding P50/P70/P100, Qdrant P50/P70/P100, total P50/P70/P100

Requires artifacts from ingest_hindi.py:
    benchmark_artifacts/eval_map.json
    benchmark_artifacts/queries.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_QDRANT_PATH = str(PROJECT_ROOT / "qdrant_hindi_benchmark")
DEFAULT_COLLECTION = "hindi_minilm_baseline"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_ARTIFACTS_DIR = str(PROJECT_ROOT / "benchmark_artifacts")
TOP_K_VALUES = [1, 5, 10]
WARMUP_QUERIES = 5  # warm up model + DB before timing


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Hindi retrieval quality and latency.")
    p.add_argument("--qdrant-path", default=DEFAULT_QDRANT_PATH)
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--model", default=DEFAULT_MODEL_NAME)
    p.add_argument("--query-prefix", default="", help="Prefix for query texts (e.g. 'query: ')")
    p.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS_DIR)
    p.add_argument("--output-json", default=None, help="Path to write JSON results")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------
def load_artifacts(
    artifacts_dir: str,
) -> Tuple[Dict[int, str], Dict[int, List[Dict[str, Any]]]]:
    """Load queries and eval_map from the ingestion artifacts."""
    queries_path = Path(artifacts_dir) / "queries.json"
    eval_map_path = Path(artifacts_dir) / "eval_map.json"

    with open(queries_path, "r", encoding="utf-8") as f:
        queries = {int(k): v for k, v in json.load(f).items()}

    with open(eval_map_path, "r", encoding="utf-8") as f:
        eval_map = {int(k): v for k, v in json.load(f).items()}

    return queries, eval_map


# ---------------------------------------------------------------------------
# Build ground truth: query_id → set of relevant passage texts
# ---------------------------------------------------------------------------
def build_ground_truth(
    eval_map: Dict[int, List[Dict[str, Any]]],
) -> Dict[int, set]:
    """For each query, build the set of passage texts that are marked as relevant."""
    gt: Dict[int, set] = {}
    for qid, entries in eval_map.items():
        relevant = set()
        for e in entries:
            if e["is_selected"] == 1:
                relevant.add(e["passage_text"].strip())
        gt[qid] = relevant
    return gt


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve_single(
    query_text: str,
    model: SentenceTransformer,
    client: QdrantClient,
    collection: str,
    top_k: int,
    query_prefix: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Run a single retrieval query and measure latency components.

    Returns:
        results: list of Qdrant hits (payload dicts)
        latency: {"embed_ms", "search_ms", "total_ms"}
    """
    t_total = time.perf_counter()

    # Embedding
    t_embed = time.perf_counter()
    query_vec = model.encode(query_prefix + query_text).tolist()
    embed_ms = (time.perf_counter() - t_embed) * 1000

    # Qdrant search
    t_search = time.perf_counter()
    search_result = client.query_points(
        collection_name=collection,
        query=query_vec,
        limit=top_k,
    )
    search_ms = (time.perf_counter() - t_search) * 1000

    total_ms = (time.perf_counter() - t_total) * 1000

    results = []
    for hit in search_result.points:
        results.append({
            "chunk_text": hit.payload.get("chunk", ""),
            "passage_id": hit.payload.get("passage_id"),
            "score": float(hit.score),
        })

    return results, {"embed_ms": embed_ms, "search_ms": search_ms, "total_ms": total_ms}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def check_relevant(
    retrieved_chunks: List[Dict[str, Any]],
    relevant_passages: set,
    eval_map_entries: List[Dict[str, Any]],
) -> List[bool]:
    """For each retrieved chunk, check if it belongs to a relevant passage.

    A chunk is relevant if its text is a substring of any relevant passage,
    since RecursiveChunker splits passages into smaller pieces.
    """
    relevance = []
    for chunk in retrieved_chunks:
        chunk_text = chunk["chunk_text"].strip()
        is_rel = False
        for passage_text in relevant_passages:
            if chunk_text in passage_text:
                is_rel = True
                break
        relevance.append(is_rel)
    return relevance


def compute_recall_at_k(relevance_list: List[bool], k: int) -> float:
    """Recall@K: did we find at least one relevant result in top-K?"""
    return 1.0 if any(relevance_list[:k]) else 0.0


def compute_mrr_at_k(relevance_list: List[bool], k: int) -> float:
    """MRR@K: reciprocal rank of first relevant result in top-K."""
    for i, is_rel in enumerate(relevance_list[:k]):
        if is_rel:
            return 1.0 / (i + 1)
    return 0.0


def percentiles(data: List[float]) -> Dict[str, float]:
    """Compute P50, P70, P100."""
    arr = np.array(data)
    return {
        "P50": float(np.percentile(arr, 50)),
        "P70": float(np.percentile(arr, 70)),
        "P100": float(np.percentile(arr, 100)),
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def run_evaluation(
    queries: Dict[int, str],
    eval_map: Dict[int, List[Dict[str, Any]]],
    ground_truth: Dict[int, set],
    model: SentenceTransformer,
    client: QdrantClient,
    collection: str,
    query_prefix: str = "",
) -> Dict[str, Any]:
    """Run full evaluation and return metrics dict."""

    max_k = max(TOP_K_VALUES)
    query_ids = sorted(queries.keys())

    # --- Warmup ---
    print(f"[EVAL]  Warming up ({WARMUP_QUERIES} queries) ...")
    for qid in query_ids[:WARMUP_QUERIES]:
        retrieve_single(queries[qid], model, client, collection, max_k, query_prefix)

    # --- Timed evaluation ---
    print(f"[EVAL]  Evaluating {len(query_ids)} queries at top-K={TOP_K_VALUES} ...")

    recall_at = {k: [] for k in TOP_K_VALUES}
    mrr_at_5: List[float] = []
    embed_latencies: List[float] = []
    search_latencies: List[float] = []
    total_latencies: List[float] = []
    queries_with_no_relevant = 0

    for i, qid in enumerate(query_ids):
        query_text = queries[qid]
        relevant = ground_truth.get(qid, set())

        if not relevant:
            queries_with_no_relevant += 1
            continue  # skip queries with no ground truth

        results, latency = retrieve_single(
            query_text, model, client, collection, max_k, query_prefix,
        )

        # Check relevance
        eval_entries = eval_map.get(qid, [])
        relevance = check_relevant(results, relevant, eval_entries)

        # Metrics
        for k in TOP_K_VALUES:
            recall_at[k].append(compute_recall_at_k(relevance, k))
        mrr_at_5.append(compute_mrr_at_k(relevance, 5))

        # Latency
        embed_latencies.append(latency["embed_ms"])
        search_latencies.append(latency["search_ms"])
        total_latencies.append(latency["total_ms"])

        if (i + 1) % 100 == 0:
            print(f"[EVAL]  {i+1}/{len(query_ids)} queries processed ...", end="\r")

    print()

    # --- Aggregate ---
    results_dict: Dict[str, Any] = {
        "total_queries": len(query_ids),
        "evaluated_queries": len(query_ids) - queries_with_no_relevant,
        "queries_with_no_relevant": queries_with_no_relevant,
    }

    for k in TOP_K_VALUES:
        vals = recall_at[k]
        results_dict[f"Recall@{k}"] = round(float(np.mean(vals)) * 100, 2) if vals else 0.0

    results_dict["MRR@5"] = round(float(np.mean(mrr_at_5)) * 100, 2) if mrr_at_5 else 0.0

    results_dict["latency_embed_ms"] = percentiles(embed_latencies) if embed_latencies else {}
    results_dict["latency_search_ms"] = percentiles(search_latencies) if search_latencies else {}
    results_dict["latency_total_ms"] = percentiles(total_latencies) if total_latencies else {}

    return results_dict


# ---------------------------------------------------------------------------
# Cold start measurement
# ---------------------------------------------------------------------------
def measure_cold_start(model_name: str, query_prefix: str = "") -> Dict[str, float]:
    """Measure cold-start latency: loading model + first encode."""
    print("[COLD]  Measuring cold-start latency ...")

    t0 = time.perf_counter()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = SentenceTransformer(model_name, device=device)
    model_load_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    _ = m.encode(query_prefix + "test query for warmup")
    first_encode_ms = (time.perf_counter() - t0) * 1000

    result = {
        "model_load_ms": round(model_load_ms, 1),
        "first_encode_ms": round(first_encode_ms, 1),
        "cold_start_total_ms": round(model_load_ms + first_encode_ms, 1),
    }
    print(f"[COLD]  Model load: {result['model_load_ms']:.0f} ms, "
          f"first encode: {result['first_encode_ms']:.0f} ms")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    print("=" * 60)
    print("  Hindi Retrieval Evaluation")
    print("=" * 60)

    # Load artifacts
    queries, eval_map = load_artifacts(args.artifacts_dir)
    ground_truth = build_ground_truth(eval_map)
    print(f"[LOAD]  {len(queries)} queries, {len(eval_map)} eval entries")

    # Cold start
    cold_start = measure_cold_start(args.model, args.query_prefix)

    # Load model + Qdrant for warm evaluation
    print(f"[MODEL] Loading {args.model} ...")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model, device=device)
    client = QdrantClient(path=args.qdrant_path)

    # Run evaluation
    metrics = run_evaluation(
        queries, eval_map, ground_truth,
        model, client, args.collection, args.query_prefix,
    )

    client.close()

    # Combine results
    final = {
        "model": args.model,
        "collection": args.collection,
        "cold_start": cold_start,
        **metrics,
    }

    # Print summary
    print()
    print("=" * 60)
    print("  Evaluation Results")
    print("=" * 60)
    print(f"  Model:             {args.model}")
    print(f"  Evaluated queries: {metrics['evaluated_queries']}")
    print()
    for k in TOP_K_VALUES:
        print(f"  Recall@{k:<2d}:         {metrics[f'Recall@{k}']:.2f}%")
    print(f"  MRR@5:             {metrics['MRR@5']:.2f}%")
    print()
    print("  Warm Retrieval Latency:")
    for stage in ["embed", "search", "total"]:
        key = f"latency_{stage}_ms"
        if metrics.get(key):
            p = metrics[key]
            print(f"    {stage:8s}  P50={p['P50']:6.1f} ms  P70={p['P70']:6.1f} ms  P100={p['P100']:7.1f} ms")
    print()
    print("  Cold Start:")
    print(f"    Model load:    {cold_start['model_load_ms']:.0f} ms")
    print(f"    First encode:  {cold_start['first_encode_ms']:.0f} ms")
    print("=" * 60)

    # Save
    if args.output_json:
        output_path = Path(args.output_json)
    else:
        output_path = Path(args.artifacts_dir) / f"eval_{args.model.replace('/', '_')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    print(f"[SAVE]  Results → {output_path}")


if __name__ == "__main__":
    main()
