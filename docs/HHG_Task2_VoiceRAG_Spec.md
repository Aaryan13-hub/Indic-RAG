# Voice-Enabled RAG System — Project Spec & Build Plan
### HH Goa 2026, Shortlisting Task 2

---

## 1. What the task actually wants

Build a pipeline that does, end to end:

**Voice input → Speech-to-text → Chunking/Retrieval (vector DB) → Answer generation**

A user speaks a question, your system transcribes it, retrieves relevant context from a provided dataset, and generates a grounded answer. It's judged not just on "does it work" but on how deliberately each stage was engineered — chunking depth, latency discipline, orchestration, and guardrails.

### Core objectives (from the brief)
| # | Requirement | Detail |
|---|---|---|
| 1 | Speech-to-text | Must use **Sarvam** or **ElevenLabs** — pick one, don't mix |
| 2 | Chunking | Must show *multiple* strategies (fixed-size, semantic, overlap handling, metadata-aware) — not one naive splitter |
| 3 | Latency target | Full pipeline (chunk + retrieve + generate) — **under 200ms** end to end |
| 4 | Latency analytics | Report P50 / P70 / P100 across many test queries, not a cherry-picked run |
| 5 | Harness | Structured orchestration — tool calls, retries, structured I/O, error recovery — not a single raw prompt call |
| 6 | Guardrails | Off-topic detection, unsafe-input handling, hallucination/groundedness checks, "I don't know" behavior |
| 7 | Submission | Form + GitHub repo + live working link + 2 videos |
| 8 | Promotion | Both videos on Instagram, X, LinkedIn — by *every* team member individually, with a public IG account, tagged appropriately (I've left this out of the technical plan below since it's a logistics task, not an engineering one — don't forget it) |

**Deadline:** August 22, 2026, 11:59 PM.

### ⚠️ Latency target: use 200ms, not the brief's original number
Working target for the full pipeline (chunking + retrieval + generation) is **under 200ms end to end**. This is achievable for retrieval (chunking + vector search comfortably fits in well under 100ms at this dataset's scale) but genuinely tight once LLM generation is included — even the fastest hosted inference (e.g. Groq) has 150ms+ time-to-first-token, so hitting 200ms *total* means leaning hard on a fast provider and keeping retrieval overhead minimal so most of the 200ms budget is left for generation.

**Recommendation:** measure and report retrieval latency and full end-to-end latency (STT + retrieval + generation) separately in your P50/P70/P100 table, with 200ms as the end-to-end target. Be transparent in your submission about where the time actually goes — this reads as engineering maturity, not failure, even if you occasionally miss the 200ms mark on longer queries.

---

## 2. The dataset

The brief (and the link you confirmed, `https://huggingface.co/datasets/ai4bharat/MSMARCO-XI`) 

- **`ai4bharat/MSMARCO-XI`**  Schema: `query`, `answers`, `passages` (list), `source_lang`, `target_lang`, plus translation metadata. Available per-language splits (e.g. `hi`, `te`, etc.), loadable via `load_dataset("ai4bharat/MSMARCO-XI", "hi", split="train")`.
- **`ai4bharat/IndicMSMARCO`** — sibling dataset, same project/paper, 13 Indian-language subsets, ~1,000 rows each, MIT license, simpler passage/query schema.

**Confirmed schema** (verified directly from the dataset page): each record is a query in a target Indian language paired with translated + original English passages:

```json
{
  "source_lang": "eng_Latn",
  "target_lang": "asm_Beng",
  "meta": { "model_name": "...", "temperature": 0.0, "max_tokens": 4096, "top_p": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0 },
  "query": "...",              // translated query, target language
  "Answer": "...",             // translated answer, target language — capital A, singular
  "query_id": 1185869,
  "query_type": "DESCRIPTION",
  "passages": {
    "is_selected": [1, 0, 0, ...],          // which passage(s) are the ground-truth relevant one
    "English_passages": ["...", "..."],      // original English passage pool
    "Translated_passages": ["...", "..."]    // same passages, translated — position-aligned with English_passages and is_selected
  },
  "Eng_Query": "...",           // original English query
  "Eng_Answer": "..."           // original English answer
}
```

**What this means for the build:**
- Your document corpus (what gets chunked + indexed) should be built from **`passages.English_passages`** or **`passages.Translated_passages`** depending on which language you're demoing in — don't mix the two in one index unless you tag each chunk with its language.
- **`passages.is_selected`** is your ground truth for retrieval evaluation: for a given query, the passage(s) marked `1` are what a correct retriever *should* surface at top-k. Use this to score retrieval precision/recall per chunking strategy — this is exactly the comparison mechanism your experimentation setup needs.
- Use **`query`** (translated) as what you feed through STT/demo in your target Indian language, and **`Eng_Query`** as the English equivalent if you want an English-only demo path instead.
- Use **`Answer`** / **`Eng_Answer`** to sanity-check your generated answers for groundedness, not as something you show the user — they're your eval reference, not part of the live pipeline.
- **`query_type`** (e.g. `DESCRIPTION`) is worth logging — different query types may need different retrieval/generation handling, and it's a free signal already in the data.

**What to do first:** try loading `ai4bharat/MSMARCO-XI` for one language split first (e.g. `"hi"` or `"as"`), print one record, and confirm it matches the shape above before writing ingestion code around it.

**What the dataset is, structurally:** it's a retrieval benchmark — each row has a **query** (in a target Indian language, plus its English original), a set of candidate **passages** (English + translated, position-aligned), a flag marking which passage is the **ground-truth relevant one**, and a reference **answer**. This is exactly the shape a RAG system needs.

**What you need to do with it:**
1. Pick 1–3 target-language splits to scope your demo to (e.g. `hi`, or English-only via `Eng_Query`/`English_passages` — safest, most judge-legible combo).
2. Flatten `passages.English_passages` (or `Translated_passages`, matched by language) into your document corpus — this is what gets chunked, embedded, and indexed.
3. Use `query` / `Eng_Query` as your test/eval set — this is what you speak into your STT during the demo, and what you use to generate your latency P50/P70/P100 numbers.
4. Use `passages.is_selected` (ground truth for which passage is actually relevant) to score retrieval quality per chunking strategy, and use `Answer`/`Eng_Answer` to sanity-check that your generated answers are actually grounded — not shown to the user, just your eval reference.

---

## 3. Your constraints, restated

- **Cost: free.** No paid APIs as a hard dependency for the working submission (a low-cost cloud LLM is acceptable as a *documented alternative*, not the only path).
- **Local hardware ceiling:** RTX 3050 (4GB VRAM), Ryzen 7 7000-series, 16GB system RAM.
- **Deployment latency target:** <200ms end-to-end (see the flag in Section 1 — treat this as retrieval+generation, not counting raw network/STT round trip).
- **You want to experiment** with multiple chunking strategies and swap vector DBs before finalizing — the architecture needs a clean interface boundary, not a hardcoded pipeline.

---

## 4. Recommended stack

### 4a. Speech-to-text — **Sarvam** (recommended over ElevenLabs)
| | Sarvam (Saaras v3) | ElevenLabs |
|---|---|---|
| Free tier | ₹1,000 free credits on signup | Free tier exists but is minutes-limited and much stingier for STT specifically; ElevenLabs' core free tier is optimized for TTS, not STT |
| Cost after free tier | ~₹1.5/min (~$0.018/min) | Materially higher per-minute for transcription |
| Language fit | Built specifically for Indian languages + code-mixed Hinglish — matches the IndicMSMARCO dataset's languages | English-first; Indian-language accuracy is weaker |
| Latency | <150ms time-to-first-token in fast/streaming mode | Also low-latency, but you're paying a language-accuracy tax for this dataset |

**Verdict:** Sarvam is the better fit for this specific dataset and is genuinely free to prototype with (₹1,000 credit covers a full hackathon cycle of testing). Use the REST API for demo clips under 30s; note in your README that ElevenLabs was evaluated and Sarvam was chosen for Indic-language accuracy and cost.

### 4b. LLM — offer both, lead with cloud-free
You asked me to check both. Here's the honest picture:

**Cloud (free) — recommended primary: Groq**
- Free tier: 30 RPM, ~6,000 TPM, ~1,000 requests/day per model, **no credit card required**.
- Runs Llama 3.3 70B / Llama 3.1 8B on custom LPU hardware — genuinely the fastest hosted inference available (300–1,000 tokens/sec), which directly helps your latency target.
- $0 cost as long as you stay inside hackathon-scale usage (a few hundred test queries easily fits).
- Runner-up: **Google Gemini API free tier** — 1,500 requests/day, 1M TPM on Flash/Flash-Lite, no card, no expiry. Slightly higher latency than Groq but more generous daily quota — good as a fallback/secondary provider in your harness.

**Local (offline, $0 forever) — feasible on your hardware, with caveats**
- Your 3050 has only 4GB VRAM — that rules out anything above a ~3–4B parameter model in 4-bit quant if you want it to fit fully on GPU. Realistic options via **Ollama**:
  - `llama3.2:3b` (Q4) — ~2GB VRAM, fits comfortably, decent instruction following.
  - `qwen2.5:3b-instruct` (Q4) — similarly sized, strong multilingual/Indic performance, a good match for this dataset's languages.
  - `phi3:mini` (3.8B, Q4) — fits, strong reasoning-per-parameter.
- These will **not** hit sub-200ms generation latency on a 3050 for anything beyond short answers — expect 1–3+ seconds for a full grounded answer locally. If local-only latency compliance matters for the demo, cloud (Groq) is the only realistic way to stay under 200ms; be upfront about this trade-off in your latency report rather than hiding it.
- CPU offload to the Ryzen 7 + 16GB RAM is a fallback if VRAM is tight, but it's slower — treat GPU-resident 3B models as your local ceiling.

**Verdict:** Build with **Groq as primary** (free, fast, hits your latency target) and keep a **local Ollama model as a swappable fallback** in your harness (for offline demo resilience and to show you engineered for the constraint, not just used a hosted API). Document both, and don't hardcode either — see the harness design below.

### 4c. Vector DB — keep it swappable, don't commit early
Since you explicitly want to experiment:
- **Chroma** — embedded, zero-config, pure Python, runs happily in 16GB RAM, great for fast local iteration.
- **Qdrant** — free, runs locally via Docker or as an embedded library (`qdrant-client` in local mode), better filtering/metadata support if your chunking gets metadata-aware, still $0.
- **FAISS** — no server at all, in-memory, the fastest for pure vector search at this dataset's scale (thousands of passages), good baseline to benchmark others against.

Build a thin `VectorStore` interface (`add(chunks)`, `query(embedding, k)`, `delete()`) with Chroma/Qdrant/FAISS as interchangeable backends behind it. This is a small amount of extra scaffolding that buys you the experimentation room you asked for, and it's exactly the kind of "harness" design the brief is scoring you on anyway.

### 4d. Chunking — plan for at least 3–4 strategies, swappable the same way
- **Fixed-size with overlap** — baseline (e.g. 256 tokens, 20% overlap). Cheap, fast, your control group.
- **Recursive/structure-aware splitting** — split on paragraph/sentence boundaries first, fall back to fixed-size only when a unit is too long.
- **Semantic chunking** — embed sentences, merge adjacent sentences while cosine similarity stays high, split when topic shifts. Slower to build but usually the retrieval-quality winner.
- **Metadata-aware chunking** — tag each chunk with source language, passage ID, position — lets your guardrail layer reason about provenance and lets you filter retrieval by metadata later.
- **Sentence-window / small-to-big** — index small chunks for precise matching, retrieve their surrounding larger window for generation context. Good middle ground on quality vs. speed.

Same pattern as the vector DB: a `Chunker` interface so you can A/B these against your retrieval eval set before picking a final one for submission.

### 4e. Embeddings (needed regardless of vector DB choice)
Use a small, free, local embedding model so embedding itself doesn't burn your cloud budget or blow your latency: `sentence-transformers/all-MiniLM-L6-v2` (fast, tiny, runs on CPU fine) or, given the dataset's Indic-language focus, `ai4bharat/indic-sentence-bert` / a multilingual MiniLM variant if you scope beyond English+Hindi.

### 4f. Guardrails — cheap and effective options
- **Off-topic detection:** similarity threshold between query embedding and top retrieved chunk — below threshold, refuse/redirect instead of hallucinating.
- **Groundedness check:** after generation, verify the answer's key claims appear in the retrieved context (simple overlap check, or a second cheap LLM call asking "is this answer supported by this context, yes/no").
- **Unsafe-input filter:** a lightweight keyword/classifier pass before the query ever reaches retrieval.
- All of this belongs inside the harness (next section), not bolted onto the prompt.

---

## 5. Cost summary (everything above)

| Component | Choice | Cost |
|---|---|---|
| STT | Sarvam Saaras v3 | Free (₹1,000 credit), then ~$0.018/min |
| LLM (primary) | Groq (Llama 3.3/3.1) | Free (30 RPM / ~1,000 req/day) |
| LLM (fallback) | Gemini Flash free tier | Free (1,500 req/day) |
| LLM (offline) | Ollama, Llama-3.2-3B / Qwen2.5-3B (Q4) | Free, runs on your 3050 |
| Embeddings | MiniLM / Indic-SBERT (local) | Free |
| Vector DB | Chroma / Qdrant / FAISS (local) | Free |
| Hosting (live link) | Free-tier Streamlit Cloud / HF Spaces / Render | Free |

Total required spend: **$0**, with Groq's speed doing the heavy lifting on your latency target.

---

## 6. Prompt for the AI coding agent

```
You are helping me build a voice-enabled RAG system for a hackathon submission. 

DATASET: ai4bharat/MSMARCO-XI (the brief referenced "ai4bharat/MSMARCO-IIX", which 
does not exist on Hugging Face — MSMARCO-XI is the confirmed real dataset, verified 
directly). Load it with:
  from datasets import load_dataset
  ds = load_dataset("ai4bharat/MSMARCO-XI", "hi", split="train")

CONFIRMED SCHEMA per record (verified from the dataset page — use these exact field 
names, do not guess or rename them):
  - query (str): translated query, target language
  - Answer (str): translated answer — note capital A, singular, not "answers"
  - query_id (int), query_type (str, e.g. "DESCRIPTION")
  - Eng_Query (str), Eng_Answer (str): original English versions
  - passages (dict) with three position-aligned lists:
      - is_selected (list[int]): 1 = ground-truth relevant passage, 0 = not
      - English_passages (list[str])
      - Translated_passages (list[str])
  - source_lang, target_lang (str), meta (dict, translation-model settings)

Build the document corpus from passages.English_passages or 
passages.Translated_passages (position-aligned with is_selected). Use 
passages.is_selected as ground truth to score retrieval quality per chunking 
strategy — don't just eyeball retrieval results, actually measure precision/recall 
against this field. Use Answer/Eng_Answer only as an internal groundedness check, 
never surfaced to the end user.

Confirm this schema against one real loaded record before building ingestion code 
around it — if what you see differs from the above, stop and tell me rather than 
silently adapting.

STARTING STACK — use these as the concrete v1 choices, not open-ended options:
- Vector DB: start with Chroma (embedded, zero-config, pure Python). Build it 
  behind a VectorStore interface (add / query / delete) from the very first 
  version so I can swap in FAISS or Qdrant later without touching the rest of 
  the pipeline.
- Chunking: start with recursive/structure-aware splitting (split on paragraph 
  then sentence boundaries, fall back to fixed-size ~256 tokens w/ 20% overlap 
  only when a unit is too long). Build this behind a Chunker interface from the 
  start so I can swap in semantic chunking, metadata-aware chunking, and a pure 
  fixed-size baseline later for comparison — don't hardcode the strategy inline 
  anywhere in the retrieval code.
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (local, free, fast on CPU).
- LLM: Groq API (Llama 3.3 70B or Llama 3.1 8B) as the primary generation backend 
  — free tier, no card required, fastest hosted inference available, directly 
  helps hit the latency target. Put this behind an LLM interface too, and add 
  an Ollama-based local model (llama3.2:3b or qwen2.5:3b, Q4 quant) as a 
  swappable offline fallback — my GPU is a 4GB RTX 3050, so any local model 
  must fit comfortably in 4GB VRAM at Q4.
- STT: Sarvam (Saaras v3) via REST API for clips under 30s.

Ground rules for how you work with me on this build:

1. Do NOT attempt to implement the whole pipeline in one shot. Break the work into 
   small, independently reviewable modules yourself — you decide the breakdown, 
   I don't want to see a giant dump of code across every layer at once.

2. After you implement one module, STOP. Show me exactly what you built, how to run 
   it, and how to verify it works, in isolation from the rest of the system. Wait for 
   my explicit approval before starting the next module. Do not proceed on your own 
   judgment that a module is "good enough" — I decide when we move on.

3. Keep the vector database, chunking strategy, and LLM backend all swappable behind 
   clean interfaces, per the starting stack above. I want to run controlled 
   experiments across multiple chunking strategies and multiple vector DB backends 
   before I lock in a final configuration for submission — the starting choices 
   above are a v1 default, not a final decision.

4. For every module, tell me explicitly: what you built, what you deliberately left 
   out or deferred, and what decision (if any) you need from me before continuing.

5. Keep a running latency log as we build — every component that touches the 
   request path (STT call, embedding, retrieval, generation) should be individually 
   timeable, since I need P50/P70/P100 latency numbers broken down by stage, not 
   just an end-to-end number.

6. Default to free-tier / local tooling as specified in the stack above. Flag clearly 
   if you think a paid dependency is genuinely necessary — don't add one silently.

7. Build in guardrail checkpoints (off-topic detection, groundedness/hallucination 
   check, unsafe-input handling) as part of the core pipeline, not as an afterthought 
   bolted on at the end.

Start by proposing your module breakdown and build order to me before writing any 
code, so I can confirm the plan before we begin.
```

---

## 7. Before you submit — checklist

- [ ] Confirm `ai4bharat/MSMARCO-XI` loads with the schema documented above (query, Answer, passages.is_selected/English_passages/Translated_passages) — note the substitution from the PDF's `MSMARCO-IIX` in your README
- [ ] Pick Sarvam or ElevenLabs and use only that one, consistently
- [ ] At least 3 chunking strategies implemented and compared, not just described
- [ ] Retrieval latency measured separately from generation latency; P50/P70/P100 reported for both, across a real query sample (not one run)
- [ ] Harness shows retries/error handling/structured I/O — not a single prompt-in/text-out call
- [ ] Guardrails demonstrably reject at least one off-topic or ungrounded case in your demo video
- [ ] GitHub repo, live link, submission form all filled before the "no resubmissions" deadline: Aug 22, 11:59 PM
- [ ] Promotion requirement handled separately by every team member (IG/X/LinkedIn, one public IG, correct hashtag) — this is a logistics task, track it independently of the build
```
