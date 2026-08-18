"""
scripts/benchmark_embeddings.py
================================
Multi-model A/B benchmark for Hindi retrieval.

Runs the full pipeline (ingest → evaluate) for each embedding model
using EXACTLY the same sampled rows, deduplicated passages, chunks,
and evaluation queries. Only the embedding model changes.

Usage:
    python -m scripts.benchmark_embeddings
    python -m scripts.benchmark_embeddings --sample-size 500

Models benchmarked:
    1. sentence-transformers/all-MiniLM-L6-v2  (baseline, 384d, no prefix)
    2. intfloat/multilingual-e5-small          (384d, requires "query: " / "passage: " prefixes)
    3. BAAI/bge-m3                             (1024d, no prefix)

Produces:
    benchmark_results.json
    benchmark_results.csv
    HINDI_EMBEDDING_BENCHMARK.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from src.core.chunking import RecursiveChunker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_PARQUET_PATH = (
    r"C:\Users\Aryan Shaikh\.cache\huggingface\hub"
    r"\datasets--ai4bharat--MSMARCO-XI\snapshots"
    r"\bf5cdc1f26e581e519018e434db14edd1b77602b\train\hintrain.parquet"
)
SAMPLE_SIZE = 1000
RANDOM_SEED = 42
TOP_K_VALUES = [1, 5, 10]
WARMUP_QUERIES = 5
BATCH_SIZE = 256
QDRANT_BASE_PATH = str(PROJECT_ROOT / "qdrant_hindi_benchmark")
ARTIFACTS_DIR = PROJECT_ROOT / "benchmark_artifacts"
RESULTS_DIR = PROJECT_ROOT / "benchmark_results"

# Models to benchmark — each is a dict with the model config
MODELS: List[Dict[str, Any]] = [
    {
        "name": "all-MiniLM-L6-v2",
        "hf_id": "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix": "",
        "passage_prefix": "",
        "collection": "hindi_minilm_baseline",
    },
    {
        "name": "multilingual-e5-small",
        "hf_id": "intfloat/multilingual-e5-small",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "collection": "hindi_e5_small",
    },
    {
        "name": "bge-m3",
        "hf_id": "BAAI/bge-m3",
        "query_prefix": "",
        "passage_prefix": "",
        "collection": "hindi_bge_m3",
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-model Hindi embedding benchmark.")
    p.add_argument("--parquet-path", default=DEFAULT_PARQUET_PATH)
    p.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--skip-model", action="append", default=[], help="Model names to skip")
    return p.parse_args()


# ===========================================================================
# Data pipeline (shared across all models)
# ===========================================================================

def load_and_sample(parquet_path: str, n: int, seed: int) -> pd.DataFrame:
    print(f"[DATA]  Loading {parquet_path} ...")
    # pq.read_table with column selection fails in pyarrow 25 for nested structs.
    # ParquetFile.iter_batches() is the reliable workaround (no column pre-filter).
    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    # Read all rows in a single batch, convert to Python dicts, keep needed keys
    records = []
    KEEP = {"query_id", "query", "passages"}
    for batch in pf.iter_batches():
        for rec in batch.to_pylist():
            records.append({k: rec[k] for k in KEEP})
    df = pd.DataFrame(records)
    actual_n = min(n, len(df))
    sampled = df.sample(n=actual_n, random_state=seed)
    print(f"[DATA]  {total_rows:,} total → sampled {actual_n:,} (seed={seed})")
    return sampled


def extract_and_deduplicate(df: pd.DataFrame):
    """Returns (unique_passages, eval_map, queries, stats)."""
    seen = {}
    eval_map = {}
    passage_counter = 0
    total_entries = 0
    dupes = 0

    for _, row in df.iterrows():
        qid = int(row["query_id"])
        passages = row["passages"]
        translated = passages["Translated_passages"]
        is_selected = passages["is_selected"]
        entries = []
        for text, sel in zip(translated, is_selected):
            total_entries += 1
            text = text.strip()
            if not text:
                continue
            if text not in seen:
                seen[text] = {"passage_id": passage_counter, "text": text, "query_ids": set()}
                passage_counter += 1
            else:
                dupes += 1
            seen[text]["query_ids"].add(qid)
            entries.append({"passage_text": text, "is_selected": int(sel)})
        eval_map[qid] = entries

    unique_passages = [
        {"passage_id": v["passage_id"], "text": v["text"], "query_ids": sorted(v["query_ids"])}
        for v in seen.values()
    ]

    queries = dict(zip(df["query_id"].astype(int).tolist(), df["query"].tolist()))

    stats = {
        "sampled_rows": len(df),
        "total_passage_entries": total_entries,
        "unique_passages": len(unique_passages),
        "duplicates_removed": dupes,
    }
    print(f"[DATA]  Passages: {total_entries:,} total, {len(unique_passages):,} unique, {dupes:,} dupes removed")
    return unique_passages, eval_map, queries, stats


def chunk_passages(passages):
    chunker = RecursiveChunker()
    chunks = []
    chunk_counter = 0
    lengths = []
    per_passage = []

    for p in passages:
        cs = chunker.chunk(p["text"])
        per_passage.append(len(cs))
        for c in cs:
            chunks.append({
                "chunk_id": chunk_counter, "text": c,
                "passage_id": p["passage_id"], "query_ids": p["query_ids"], "language": "hi",
            })
            lengths.append(len(c))
            chunk_counter += 1

    arr = np.array(lengths) if lengths else np.array([0])
    stats = {
        "num_chunks": len(chunks),
        "avg_chunks_per_passage": round(float(np.mean(per_passage)), 2) if per_passage else 0,
        "avg_chunk_length": round(float(np.mean(arr)), 0),
        "min_chunk_length": int(np.min(arr)),
        "max_chunk_length": int(np.max(arr)),
    }
    print(f"[CHUNK] {len(passages):,} passages → {len(chunks):,} chunks "
          f"(avg {stats['avg_chunks_per_passage']:.1f}/passage)")
    return chunks, stats


# ===========================================================================
# Per-model pipeline
# ===========================================================================

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


def run_model_benchmark(
    model_config: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    queries: Dict[int, str],
    eval_map: Dict[int, List[Dict[str, Any]]],
    ground_truth: Dict[int, set],
) -> Dict[str, Any]:
    """Run full ingest + evaluate for a single model."""
    name = model_config["name"]
    hf_id = model_config["hf_id"]
    q_prefix = model_config["query_prefix"]
    p_prefix = model_config["passage_prefix"]
    collection = model_config["collection"]

    print()
    print("=" * 60)
    print(f"  Benchmarking: {name} ({hf_id})")
    print("=" * 60)

    # --- Load model ---
    t0 = time.perf_counter()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(hf_id, device=device)
    model_load_ms = (time.perf_counter() - t0) * 1000
    dim = model.get_sentence_embedding_dimension()
    print(f"[MODEL] Loaded in {model_load_ms:.0f} ms, dim={dim}")

    # --- Embed chunks ---
    texts = [p_prefix + c["text"] for c in chunks]
    print(f"[EMBED] Encoding {len(texts):,} chunks ...")
    t0 = time.perf_counter()
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True).tolist()
    embed_total_s = time.perf_counter() - t0
    throughput = len(texts) / embed_total_s if embed_total_s > 0 else 0
    print(f"[EMBED] Done in {embed_total_s:.1f}s ({throughput:.0f} chunks/sec)")

    # --- Index into Qdrant ---
    qdrant_path = QDRANT_BASE_PATH
    client = QdrantClient(path=qdrant_path)

    try:
        client.delete_collection(collection)
    except Exception:
        pass

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    UPSERT_BATCH = 500
    for start in range(0, len(chunks), UPSERT_BATCH):
        end = min(start + UPSERT_BATCH, len(chunks))
        points = [
            PointStruct(
                id=chunks[i]["chunk_id"],
                vector=embeddings[i],
                payload={
                    "chunk": chunks[i]["text"],
                    "passage_id": chunks[i]["passage_id"],
                    "chunk_id": chunks[i]["chunk_id"],
                    "query_ids": chunks[i]["query_ids"],
                    "language": "hi",
                },
            )
            for i in range(start, end)
        ]
        client.upsert(collection_name=collection, points=points)

    # Get approximate index size
    coll_info = client.get_collection(collection)
    index_vectors_count = coll_info.points_count

    print(f"[QDRANT] Indexed {index_vectors_count} vectors into '{collection}'")

    # --- Warmup ---
    query_ids = sorted(queries.keys())
    max_k = max(TOP_K_VALUES)

    for qid in query_ids[:WARMUP_QUERIES]:
        qvec = model.encode(q_prefix + queries[qid]).tolist()
        client.query_points(collection_name=collection, query=qvec, limit=max_k)

    # --- Evaluate ---
    print(f"[EVAL]  Evaluating {len(query_ids)} queries ...")
    recall_at = {k: [] for k in TOP_K_VALUES}
    mrr_5 = []
    embed_lats = []
    search_lats = []
    total_lats = []
    skipped = 0

    for i, qid in enumerate(query_ids):
        relevant = ground_truth.get(qid, set())
        if not relevant:
            skipped += 1
            continue

        t_total = time.perf_counter()

        t_e = time.perf_counter()
        qvec = model.encode(q_prefix + queries[qid]).tolist()
        e_ms = (time.perf_counter() - t_e) * 1000

        t_s = time.perf_counter()
        hits = client.query_points(collection_name=collection, query=qvec, limit=max_k)
        s_ms = (time.perf_counter() - t_s) * 1000

        total_ms = (time.perf_counter() - t_total) * 1000

        relevance = [check_relevant(h.payload.get("chunk", ""), relevant) for h in hits.points]

        for k in TOP_K_VALUES:
            recall_at[k].append(1.0 if any(relevance[:k]) else 0.0)
        mrr_5.append(next(
            (1.0 / (j + 1) for j, r in enumerate(relevance[:5]) if r), 0.0
        ))

        embed_lats.append(e_ms)
        search_lats.append(s_ms)
        total_lats.append(total_ms)

    client.close()

    def pcts(data):
        a = np.array(data)
        return {"P50": round(float(np.percentile(a, 50)), 2),
                "P70": round(float(np.percentile(a, 70)), 2),
                "P100": round(float(np.percentile(a, 100)), 2)}

    result = {
        "model": name,
        "hf_id": hf_id,
        "dimension": dim,
        "query_prefix": q_prefix,
        "passage_prefix": p_prefix,
        "evaluated_queries": len(query_ids) - skipped,
        "skipped_no_relevant": skipped,
        "embedding_throughput_chunks_per_sec": round(throughput, 1),
        "embedding_total_time_s": round(embed_total_s, 2),
        "model_load_ms": round(model_load_ms, 0),
        "index_vectors": index_vectors_count,
    }

    for k in TOP_K_VALUES:
        result[f"Recall@{k}"] = round(float(np.mean(recall_at[k])) * 100, 2) if recall_at[k] else 0
    result["MRR@5"] = round(float(np.mean(mrr_5)) * 100, 2) if mrr_5 else 0

    result["latency_embed_ms"] = pcts(embed_lats) if embed_lats else {}
    result["latency_search_ms"] = pcts(search_lats) if search_lats else {}
    result["latency_total_ms"] = pcts(total_lats) if total_lats else {}

    return result


# ===========================================================================
# Report generation
# ===========================================================================

def generate_report(
    all_results: List[Dict[str, Any]],
    data_stats: Dict[str, Any],
    chunk_stats: Dict[str, Any],
) -> str:
    """Generate the HINDI_EMBEDDING_BENCHMARK.md content."""
    lines = []
    lines.append("# Hindi Embedding Benchmark Report")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- **Source**: ai4bharat/MSMARCO-XI Hindi split (`hintrain.parquet`)")
    lines.append(f"- **Total rows in parquet**: 778,638")
    lines.append(f"- **Sample size**: {data_stats['sampled_rows']}")
    lines.append(f"- **Random seed**: {RANDOM_SEED}")
    lines.append(f"- **Total passage entries**: {data_stats['total_passage_entries']:,}")
    lines.append(f"- **Unique passages**: {data_stats['unique_passages']:,}")
    lines.append(f"- **Duplicates removed**: {data_stats['duplicates_removed']:,}")
    lines.append("")
    lines.append("## Chunking")
    lines.append("")
    lines.append(f"- **Chunker**: RecursiveChunker (max_chars=1000, overlap_chars=200)")
    lines.append(f"- **Total chunks**: {chunk_stats['num_chunks']:,}")
    lines.append(f"- **Avg chunks/passage**: {chunk_stats['avg_chunks_per_passage']}")
    lines.append(f"- **Chunk length**: avg={chunk_stats['avg_chunk_length']:.0f}, "
                 f"min={chunk_stats['min_chunk_length']}, max={chunk_stats['max_chunk_length']}")
    lines.append("")
    lines.append("## Models Tested")
    lines.append("")
    lines.append("| # | Model | HF ID | Dimension | Prefix Required |")
    lines.append("|---|-------|-------|-----------|-----------------|")
    for i, r in enumerate(all_results, 1):
        pref = "Yes" if r.get("query_prefix") else "No"
        lines.append(f"| {i} | {r['model']} | `{r['hf_id']}` | {r['dimension']} | {pref} |")
    lines.append("")

    # Retrieval metrics table
    lines.append("## Retrieval Metrics")
    lines.append("")
    lines.append("| Model | Recall@1 | Recall@5 | Recall@10 | MRR@5 |")
    lines.append("|-------|----------|----------|-----------|-------|")
    for r in all_results:
        lines.append(f"| {r['model']} | {r['Recall@1']:.2f}% | {r['Recall@5']:.2f}% | "
                     f"{r['Recall@10']:.2f}% | {r['MRR@5']:.2f}% |")
    lines.append("")

    # Latency table
    lines.append("## Retrieval Latency (warm, per query)")
    lines.append("")
    lines.append("| Model | Embed P50 | Embed P70 | Embed P100 | Search P50 | Search P70 | Search P100 | Total P50 | Total P70 | Total P100 |")
    lines.append("|-------|-----------|-----------|------------|------------|------------|-------------|-----------|-----------|------------|")
    for r in all_results:
        e = r.get("latency_embed_ms", {})
        s = r.get("latency_search_ms", {})
        t = r.get("latency_total_ms", {})
        lines.append(
            f"| {r['model']} "
            f"| {e.get('P50', 0):.1f} ms | {e.get('P70', 0):.1f} ms | {e.get('P100', 0):.1f} ms "
            f"| {s.get('P50', 0):.1f} ms | {s.get('P70', 0):.1f} ms | {s.get('P100', 0):.1f} ms "
            f"| {t.get('P50', 0):.1f} ms | {t.get('P70', 0):.1f} ms | {t.get('P100', 0):.1f} ms |"
        )
    lines.append("")

    # Throughput table
    lines.append("## Embedding Throughput")
    lines.append("")
    lines.append("| Model | Chunks/sec | Total time (s) | Model load (ms) |")
    lines.append("|-------|------------|----------------|-----------------|")
    for r in all_results:
        lines.append(f"| {r['model']} | {r['embedding_throughput_chunks_per_sec']:.0f} "
                     f"| {r['embedding_total_time_s']:.1f} | {r['model_load_ms']:.0f} |")
    lines.append("")

    # Recommendation
    lines.append("## Recommendation")
    lines.append("")

    # Find best model by Recall@5, and check tradeoffs
    best_r5 = max(all_results, key=lambda x: x["Recall@5"])
    baseline = all_results[0]  # all-MiniLM-L6-v2 is always first

    baseline_r5 = baseline["Recall@5"]
    best_r5_val = best_r5["Recall@5"]
    baseline_p50 = baseline.get("latency_total_ms", {}).get("P50", 0)
    best_p50 = best_r5.get("latency_total_ms", {}).get("P50", 0)

    lines.append(f"### 1. Is all-MiniLM-L6-v2 good enough for Hindi retrieval?")
    lines.append("")
    if baseline_r5 >= 70:
        lines.append(f"Baseline Recall@5 = **{baseline_r5:.2f}%** — "
                     f"this is {'acceptable' if baseline_r5 >= 70 else 'borderline'}.")
    else:
        lines.append(f"Baseline Recall@5 = **{baseline_r5:.2f}%** — this is **low** and suggests "
                     f"all-MiniLM-L6-v2 struggles with Hindi text, as expected for an English-first model.")
    lines.append("")

    lines.append(f"### 2. Which alternative performed better?")
    lines.append("")
    if best_r5["model"] != baseline["model"]:
        lines.append(f"**{best_r5['model']}** achieved the highest Recall@5 at **{best_r5_val:.2f}%**.")
    else:
        lines.append(f"No alternative outperformed the baseline on Recall@5.")
    lines.append("")

    lines.append(f"### 3. Improvement magnitude")
    lines.append("")
    if best_r5["model"] != baseline["model"]:
        delta = best_r5_val - baseline_r5
        lines.append(f"Recall@5 improved by **+{delta:.2f}pp** ({baseline_r5:.2f}% → {best_r5_val:.2f}%).")
        mrr_delta = best_r5["MRR@5"] - baseline["MRR@5"]
        lines.append(f"MRR@5 improved by **+{mrr_delta:.2f}pp** ({baseline['MRR@5']:.2f}% → {best_r5['MRR@5']:.2f}%).")
    else:
        lines.append("N/A — baseline was the best.")
    lines.append("")

    lines.append(f"### 4. Latency cost")
    lines.append("")
    if best_r5["model"] != baseline["model"]:
        lat_delta = best_p50 - baseline_p50
        lines.append(f"Total retrieval P50 changed by **{lat_delta:+.1f} ms** "
                     f"({baseline_p50:.1f} ms → {best_p50:.1f} ms).")
    else:
        lines.append(f"N/A — no model change needed.")
    lines.append("")

    lines.append(f"### 5. Quality/latency tradeoff")
    lines.append("")
    # Pick the best considering tradeoff
    # If best model is >5pp better on Recall@5 and latency delta is <50ms, recommend it.
    # Otherwise, consider the second-best.
    if best_r5["model"] != baseline["model"]:
        delta = best_r5_val - baseline_r5
        lat_delta = best_p50 - baseline_p50
        if delta >= 5 and lat_delta < 50:
            lines.append(f"The improvement (+{delta:.1f}pp Recall@5) is significant and the latency "
                         f"cost (+{lat_delta:.1f} ms P50) is acceptable. **Recommend switching.**")
        elif delta >= 5 and lat_delta >= 50:
            lines.append(f"The improvement (+{delta:.1f}pp Recall@5) is significant but the latency "
                         f"cost (+{lat_delta:.1f} ms P50) is substantial. Consider your latency budget.")
            # Check if there's a middle-ground model
            mid_candidates = [r for r in all_results if r["model"] not in [baseline["model"], best_r5["model"]]]
            for mc in mid_candidates:
                mc_delta = mc["Recall@5"] - baseline_r5
                mc_lat = mc.get("latency_total_ms", {}).get("P50", 0) - baseline_p50
                if mc_delta > 0 and mc_lat < lat_delta:
                    lines.append(f"**{mc['model']}** offers a middle ground: +{mc_delta:.1f}pp Recall@5 "
                                 f"at only +{mc_lat:.1f} ms P50.")
        else:
            lines.append(f"The improvement (+{delta:.1f}pp Recall@5) is marginal. "
                         f"**The baseline may be sufficient** depending on quality requirements.")
    else:
        lines.append("Baseline is the best option tested.")
    lines.append("")

    lines.append(f"### 6. Final recommendation")
    lines.append("")
    # Determine recommendation
    if best_r5["model"] != baseline["model"]:
        delta = best_r5_val - baseline_r5
        lat_delta = best_p50 - baseline_p50
        if delta >= 3:
            lines.append(f"> **REPLACE** `all-MiniLM-L6-v2` with **`{best_r5['hf_id']}`**")
            lines.append(f">")
            lines.append(f"> Recall@5: {baseline_r5:.2f}% → {best_r5_val:.2f}% (+{delta:.2f}pp)")
            lines.append(f"> MRR@5: {baseline['MRR@5']:.2f}% → {best_r5['MRR@5']:.2f}%")
            lines.append(f"> Latency P50: {baseline_p50:.1f} ms → {best_p50:.1f} ms")
        else:
            lines.append(f"> **KEEP** `all-MiniLM-L6-v2`")
            lines.append(f"> The improvement from {best_r5['model']} (+{delta:.2f}pp) is too small to justify the switch.")
    else:
        lines.append(f"> **KEEP** `all-MiniLM-L6-v2`")
        lines.append(f"> No tested alternative improved Recall@5.")
    lines.append("")

    return "\n".join(lines)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  Hindi Embedding Benchmark")
    print("  Models:", ", ".join(m["name"] for m in MODELS if m["name"] not in args.skip_model))
    print("=" * 60)

    # --- Shared data pipeline (Steps 2-5) ---
    df = load_and_sample(args.parquet_path, args.sample_size, args.seed)
    unique_passages, eval_map, queries, data_stats = extract_and_deduplicate(df)
    chunks, chunk_stats = chunk_passages(unique_passages)
    ground_truth = build_ground_truth(eval_map)

    # Save shared artifacts
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_DIR / "eval_map.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in eval_map.items()}, f, ensure_ascii=False, indent=2)
    with open(ARTIFACTS_DIR / "queries.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in queries.items()}, f, ensure_ascii=False, indent=2)

    # --- Per-model benchmarks (Steps 6-8) ---
    all_results: List[Dict[str, Any]] = []

    for model_config in MODELS:
        if model_config["name"] in args.skip_model:
            print(f"\n[SKIP]  Skipping {model_config['name']}")
            continue

        result = run_model_benchmark(model_config, chunks, queries, eval_map, ground_truth)
        all_results.append(result)

        # Print interim result
        print(f"\n  {result['model']:30s} Recall@5={result['Recall@5']:.2f}%  "
              f"MRR@5={result['MRR@5']:.2f}%  "
              f"P50={result.get('latency_total_ms', {}).get('P50', 0):.1f} ms")

    # --- Generate outputs (Steps 11-12) ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = RESULTS_DIR / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"data_stats": data_stats, "chunk_stats": chunk_stats, "models": all_results},
                  f, indent=2, ensure_ascii=False)
    print(f"\n[SAVE]  {json_path}")

    # CSV
    csv_path = RESULTS_DIR / "benchmark_results.csv"
    if all_results:
        fieldnames = ["model", "dimension", "Recall@1", "Recall@5", "Recall@10", "MRR@5",
                       "embedding_throughput_chunks_per_sec", "embedding_total_time_s",
                       "model_load_ms"]
        # Add flattened latency columns
        for stage in ["embed", "search", "total"]:
            for p in ["P50", "P70", "P100"]:
                fieldnames.append(f"latency_{stage}_{p}_ms")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in all_results:
                row = dict(r)
                for stage in ["embed", "search", "total"]:
                    lats = r.get(f"latency_{stage}_ms", {})
                    for p in ["P50", "P70", "P100"]:
                        row[f"latency_{stage}_{p}_ms"] = lats.get(p, "")
                writer.writerow(row)
    print(f"[SAVE]  {csv_path}")

    # Markdown report
    report_md = generate_report(all_results, data_stats, chunk_stats)
    md_path = RESULTS_DIR / "HINDI_EMBEDDING_BENCHMARK.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[SAVE]  {md_path}")

    # --- Final summary table ---
    print()
    print("=" * 90)
    print(f"  {'Model':<30s} {'Recall@5':>10s} {'MRR@5':>10s} {'P50 total':>12s} {'Throughput':>12s}")
    print("-" * 90)
    for r in all_results:
        p50 = r.get("latency_total_ms", {}).get("P50", 0)
        print(f"  {r['model']:<30s} {r['Recall@5']:>9.2f}% {r['MRR@5']:>9.2f}% "
              f"{p50:>10.1f} ms {r['embedding_throughput_chunks_per_sec']:>8.0f} c/s")
    print("=" * 90)
    print()
    print(f"Full report: {md_path}")


if __name__ == "__main__":
    main()
