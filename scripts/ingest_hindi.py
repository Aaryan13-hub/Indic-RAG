"""
scripts/ingest_hindi.py
=======================
Offline Hindi ingestion pipeline:
    hintrain.parquet → sample → deduplicate → chunk → embed → Qdrant

Usage:
    python -m scripts.ingest_hindi
    python -m scripts.ingest_hindi --parquet-path /path/to/hintrain.parquet
    python -m scripts.ingest_hindi --sample-size 500 --qdrant-path ./my_qdrant

This script is OFFLINE infrastructure — it builds the vector index that
the live VoiceRAGOrchestrator reads at query time. It never touches the
orchestrator, STT, LLM, or guardrails code.
"""

from __future__ import annotations

import argparse
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
# Make project root importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.chunking import RecursiveChunker
from src.core.embeddings import EmbeddingModel

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# ---------------------------------------------------------------------------
# Configuration — all tuneable from CLI or here
# ---------------------------------------------------------------------------
DEFAULT_PARQUET_PATH = (
    r"C:\Users\Aryan Shaikh\.cache\huggingface\hub"
    r"\datasets--ai4bharat--MSMARCO-XI\snapshots"
    r"\bf5cdc1f26e581e519018e434db14edd1b77602b\train\hintrain.parquet"
)
DEFAULT_QDRANT_PATH = str(PROJECT_ROOT / "qdrant_hindi_benchmark")
DEFAULT_COLLECTION = "hindi_minilm_baseline"
DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_RANDOM_SEED = 42
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 256


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest Hindi MS-MARCO-XI passages into Qdrant for retrieval benchmarks."
    )
    p.add_argument("--parquet-path", default=DEFAULT_PARQUET_PATH, help="Path to hintrain.parquet")
    p.add_argument("--qdrant-path", default=DEFAULT_QDRANT_PATH, help="Qdrant on-disk storage path")
    p.add_argument("--collection", default=DEFAULT_COLLECTION, help="Qdrant collection name")
    p.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Rows to sample")
    p.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed")
    p.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Sentence-transformer model name")
    p.add_argument("--query-prefix", default="", help="Prefix for query texts (e.g. 'query: ')")
    p.add_argument("--passage-prefix", default="", help="Prefix for passage texts (e.g. 'passage: ')")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Embedding batch size")
    p.add_argument("--output-stats", default=None, help="Path to write JSON stats (optional)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 2: Sample
# ---------------------------------------------------------------------------
def load_and_sample(parquet_path: str, n: int, seed: int) -> pd.DataFrame:
    """Load parquet and return a random sample of n rows."""
    print(f"[LOAD]  Reading {parquet_path} ...")
    t0 = time.perf_counter()
    # pq.read_table with column selection fails in pyarrow 25 for nested structs.
    # ParquetFile.iter_batches() is the reliable workaround (no column pre-filter).
    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    records = []
    KEEP = {"query_id", "query", "passages"}
    for batch in pf.iter_batches():
        for rec in batch.to_pylist():
            records.append({k: rec[k] for k in KEEP})
    df = pd.DataFrame(records)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"[LOAD]  {total_rows:,} rows loaded in {load_ms:.0f} ms")

    actual_n = min(n, total_rows)
    sampled = df.sample(n=actual_n, random_state=seed)
    print(f"[SAMPLE] {actual_n:,} rows sampled (seed={seed})")
    return sampled


# ---------------------------------------------------------------------------
# Step 3 + 4: Extract Hindi passages + Deduplicate
# ---------------------------------------------------------------------------
def extract_and_deduplicate(
    df: pd.DataFrame,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """Extract Translated_passages, deduplicate by text, and build the
    evaluation mapping (passage_text → list of {query_id, is_selected}).

    Returns:
        unique_passages: list of {"passage_id": int, "text": str, "query_ids": [...]}
        eval_map:        dict  query_id → [{"passage_text": str, "is_selected": int}]
    """
    # passage_text → {"passage_id": int, "text": str, "query_ids": set}
    seen: Dict[str, Dict[str, Any]] = {}
    eval_map: Dict[int, List[Dict[str, Any]]] = {}
    passage_counter = 0
    total_passage_entries = 0
    duplicates_removed = 0

    for _, row in df.iterrows():
        qid = int(row["query_id"])
        passages = row["passages"]
        translated = passages["Translated_passages"]
        is_selected = passages["is_selected"]

        eval_entries: List[Dict[str, Any]] = []

        for idx, (text, sel) in enumerate(zip(translated, is_selected)):
            total_passage_entries += 1
            text = text.strip()
            if not text:
                continue

            if text not in seen:
                seen[text] = {
                    "passage_id": passage_counter,
                    "text": text,
                    "query_ids": set(),
                }
                passage_counter += 1
            else:
                duplicates_removed += 1

            seen[text]["query_ids"].add(qid)
            eval_entries.append({"passage_text": text, "is_selected": int(sel)})

        eval_map[qid] = eval_entries

    unique_passages = []
    for v in seen.values():
        unique_passages.append({
            "passage_id": v["passage_id"],
            "text": v["text"],
            "query_ids": sorted(v["query_ids"]),
        })

    print(f"[DEDUP] Total passage entries:   {total_passage_entries:,}")
    print(f"[DEDUP] Unique passages:         {len(unique_passages):,}")
    print(f"[DEDUP] Duplicates removed:      {duplicates_removed:,}")

    return unique_passages, eval_map


# ---------------------------------------------------------------------------
# Step 5: Chunk
# ---------------------------------------------------------------------------
def chunk_passages(
    passages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Chunk each unique passage using RecursiveChunker.

    Returns:
        chunks: list of {"chunk_id": int, "text": str, "passage_id": int,
                          "query_ids": [...], "language": "hi"}
        stats:  chunking stats dict
    """
    chunker = RecursiveChunker()  # defaults: max_chars=1000, overlap_chars=200
    chunks: List[Dict[str, Any]] = []
    chunk_counter = 0
    chunk_lengths: List[int] = []
    chunks_per_passage: List[int] = []

    for p in passages:
        passage_chunks = chunker.chunk(p["text"])
        chunks_per_passage.append(len(passage_chunks))
        for c in passage_chunks:
            chunks.append({
                "chunk_id": chunk_counter,
                "text": c,
                "passage_id": p["passage_id"],
                "query_ids": p["query_ids"],
                "language": "hi",
            })
            chunk_lengths.append(len(c))
            chunk_counter += 1

    arr = np.array(chunk_lengths) if chunk_lengths else np.array([0])
    stats = {
        "num_unique_passages": len(passages),
        "num_chunks": len(chunks),
        "avg_chunks_per_passage": float(np.mean(chunks_per_passage)) if chunks_per_passage else 0,
        "avg_chunk_length": float(np.mean(arr)),
        "min_chunk_length": int(np.min(arr)),
        "max_chunk_length": int(np.max(arr)),
        "chunker_max_chars": chunker.max_chars,
        "chunker_overlap_chars": chunker.overlap_chars,
    }

    print(f"[CHUNK] {stats['num_unique_passages']:,} passages → {stats['num_chunks']:,} chunks")
    print(f"[CHUNK] Avg chunks/passage: {stats['avg_chunks_per_passage']:.2f}")
    print(f"[CHUNK] Chunk length: avg={stats['avg_chunk_length']:.0f}, "
          f"min={stats['min_chunk_length']}, max={stats['max_chunk_length']}")

    return chunks, stats


# ---------------------------------------------------------------------------
# Step 6: Embed
# ---------------------------------------------------------------------------
def embed_chunks(
    chunks: List[Dict[str, Any]],
    model_name: str,
    passage_prefix: str = "",
    batch_size: int = 256,
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """Embed all chunk texts and return vectors + timing stats."""
    print(f"[EMBED] Loading model: {model_name} ...")
    model = EmbeddingModel(model_name=model_name)
    dim = model.model.get_sentence_embedding_dimension()

    texts = [passage_prefix + c["text"] for c in chunks]
    print(f"[EMBED] Embedding {len(texts):,} chunks (dim={dim}, batch={batch_size}) ...")

    t0 = time.perf_counter()
    # Encode in one call — SentenceTransformer handles batching internally
    embeddings = model.model.encode(texts, batch_size=batch_size, show_progress_bar=True).tolist()
    total_s = time.perf_counter() - t0

    stats = {
        "model_name": model_name,
        "embedding_dimension": dim,
        "total_embedding_time_s": round(total_s, 2),
        "avg_embedding_time_ms": round((total_s / len(texts)) * 1000, 3) if texts else 0,
        "throughput_chunks_per_sec": round(len(texts) / total_s, 1) if total_s > 0 else 0,
    }

    print(f"[EMBED] Done in {total_s:.1f}s — {stats['throughput_chunks_per_sec']} chunks/sec")
    return embeddings, stats


# ---------------------------------------------------------------------------
# Step 7: Qdrant index
# ---------------------------------------------------------------------------
def index_into_qdrant(
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
    qdrant_path: str,
    collection_name: str,
    dim: int,
) -> None:
    """Persist chunks + embeddings into a Qdrant on-disk collection.

    Recreates the collection from scratch to avoid mixing old/new vectors.
    """
    print(f"[QDRANT] Connecting to {qdrant_path} ...")
    client = QdrantClient(path=qdrant_path)

    # Recreate collection for reproducibility
    try:
        client.delete_collection(collection_name)
        print(f"[QDRANT] Deleted existing collection '{collection_name}'")
    except Exception:
        pass

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    print(f"[QDRANT] Created collection '{collection_name}' (dim={dim}, cosine)")

    # Upload in batches of 500 to avoid memory issues
    BATCH = 500
    total = len(chunks)
    for start in range(0, total, BATCH):
        end = min(start + BATCH, total)
        points = []
        for i in range(start, end):
            c = chunks[i]
            payload = {
                "chunk": c["text"],
                "passage_id": c["passage_id"],
                "chunk_id": c["chunk_id"],
                "query_ids": c["query_ids"],
                "language": c["language"],
            }
            points.append(PointStruct(id=c["chunk_id"], vector=embeddings[i], payload=payload))

        client.upsert(collection_name=collection_name, points=points)
        print(f"[QDRANT] Indexed {end}/{total} chunks", end="\r")

    print(f"\n[QDRANT] Indexing complete — {total:,} vectors in '{collection_name}'")
    client.close()


# ---------------------------------------------------------------------------
# Save evaluation mapping + stats
# ---------------------------------------------------------------------------
def save_artifacts(
    eval_map: Dict[int, List[Dict[str, Any]]],
    queries: Dict[int, str],
    stats: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Save eval_map, queries, and stats to disk for the evaluation script."""
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_path = output_dir / "eval_map.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        # Convert int keys to str for JSON serialization
        json.dump({str(k): v for k, v in eval_map.items()}, f, ensure_ascii=False, indent=2)
    print(f"[SAVE]  Evaluation map → {eval_path}")

    queries_path = output_dir / "queries.json"
    with open(queries_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in queries.items()}, f, ensure_ascii=False, indent=2)
    print(f"[SAVE]  Queries → {queries_path}")

    stats_path = output_dir / "ingestion_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[SAVE]  Stats → {stats_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    print("=" * 60)
    print("  Hindi Ingestion Pipeline")
    print("=" * 60)

    overall_start = time.perf_counter()

    # Step 2: Load + sample
    df = load_and_sample(args.parquet_path, args.sample_size, args.seed)

    # Build query lookup
    queries: Dict[int, str] = dict(zip(
        df["query_id"].astype(int).tolist(),
        df["query"].tolist(),
    ))

    # Step 3+4: Extract + deduplicate
    unique_passages, eval_map = extract_and_deduplicate(df)

    # Step 5: Chunk
    chunks, chunk_stats = chunk_passages(unique_passages)

    # Step 6: Embed
    embeddings, embed_stats = embed_chunks(
        chunks, args.model, passage_prefix=args.passage_prefix, batch_size=args.batch_size,
    )

    # Step 7: Qdrant
    index_into_qdrant(
        chunks, embeddings, args.qdrant_path, args.collection, embed_stats["embedding_dimension"],
    )

    overall_s = time.perf_counter() - overall_start

    # Collect all stats
    all_stats = {
        "sample_size": args.sample_size,
        "random_seed": args.seed,
        "total_rows_in_parquet": 778638,
        "sampled_rows": len(df),
        **chunk_stats,
        **embed_stats,
        "qdrant_path": args.qdrant_path,
        "qdrant_collection": args.collection,
        "total_ingestion_time_s": round(overall_s, 2),
    }

    # Save artifacts
    output_dir = Path(args.qdrant_path).parent / "benchmark_artifacts"
    save_artifacts(eval_map, queries, all_stats, output_dir)

    if args.output_stats:
        with open(args.output_stats, "w") as f:
            json.dump(all_stats, f, indent=2)

    print()
    print("=" * 60)
    print("  Ingestion Summary")
    print("=" * 60)
    for k, v in all_stats.items():
        print(f"  {k:35s} : {v}")
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
