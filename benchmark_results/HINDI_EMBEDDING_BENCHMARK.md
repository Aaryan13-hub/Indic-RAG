# Hindi Embedding Benchmark Report

## Dataset

- **Source**: ai4bharat/MSMARCO-XI Hindi split (`hintrain.parquet`)
- **Total rows in parquet**: 778,638
- **Sample size**: 1000
- **Random seed**: 42
- **Total passage entries**: 9,981
- **Unique passages**: 9,943
- **Duplicates removed**: 38

## Chunking

- **Chunker**: RecursiveChunker (max_chars=1000, overlap_chars=200)
- **Total chunks**: 10,096
- **Avg chunks/passage**: 1.02
- **Chunk length**: avg=334, min=34, max=1000

## Models Tested

| # | Model | HF ID | Dimension | Prefix Required |
|---|-------|-------|-----------|-----------------|
| 1 | all-MiniLM-L6-v2 | `sentence-transformers/all-MiniLM-L6-v2` | 384 | No |
| 2 | multilingual-e5-small | `intfloat/multilingual-e5-small` | 384 | Yes |

## Retrieval Metrics

| Model | Recall@1 | Recall@5 | Recall@10 | MRR@5 |
|-------|----------|----------|-----------|-------|
| all-MiniLM-L6-v2 | 0.80% | 2.71% | 4.15% | 1.54% |
| multilingual-e5-small | 29.82% | 73.05% | 85.81% | 45.53% |

## Retrieval Latency (warm, per query)

| Model | Embed P50 | Embed P70 | Embed P100 | Search P50 | Search P70 | Search P100 | Total P50 | Total P70 | Total P100 |
|-------|-----------|-----------|------------|------------|------------|-------------|-----------|-----------|------------|
| all-MiniLM-L6-v2 | 13.5 ms | 13.8 ms | 50.0 ms | 17.1 ms | 18.0 ms | 26.8 ms | 30.8 ms | 32.2 ms | 72.1 ms |
| multilingual-e5-small | 21.0 ms | 21.9 ms | 125.4 ms | 24.2 ms | 25.3 ms | 35.9 ms | 45.6 ms | 46.9 ms | 149.8 ms |

## Embedding Throughput

| Model | Chunks/sec | Total time (s) | Model load (ms) |
|-------|------------|----------------|-----------------|
| all-MiniLM-L6-v2 | 472 | 21.4 | 8778 |
| multilingual-e5-small | 368 | 27.4 | 10833 |

## Recommendation

### 1. Is all-MiniLM-L6-v2 good enough for Hindi retrieval?

Baseline Recall@5 = **2.71%** — this is **low** and suggests all-MiniLM-L6-v2 struggles with Hindi text, as expected for an English-first model.

### 2. Which alternative performed better?

**multilingual-e5-small** achieved the highest Recall@5 at **73.05%**.

### 3. Improvement magnitude

Recall@5 improved by **+70.34pp** (2.71% → 73.05%).
MRR@5 improved by **+43.99pp** (1.54% → 45.53%).

### 4. Latency cost

Total retrieval P50 changed by **+14.8 ms** (30.8 ms → 45.6 ms).

### 5. Quality/latency tradeoff

The improvement (+70.3pp Recall@5) is significant and the latency cost (+14.8 ms P50) is acceptable. **Recommend switching.**

### 6. Final recommendation

> **REPLACE** `all-MiniLM-L6-v2` with **`intfloat/multilingual-e5-small`**
>
> Recall@5: 2.71% → 73.05% (+70.34pp)
> MRR@5: 1.54% → 45.53%
> Latency P50: 30.8 ms → 45.6 ms

## Chunking Strategy Comparison

Model fixed to `intfloat/multilingual-e5-small`. 1,000 sampled rows.

| Chunker | Recall@5 | MRR@5 | P50 Latency | Total Chunks | Avg Length |
|---------|----------|-------|-------------|--------------|------------|
| recursive | 73.05% | 45.53% | 31.6 ms | 10096 | 334 |
| sentence | 72.89% | 45.41% | 31.4 ms | 10060 | 334 |

