"""
api.py — Flask server that bridges the static frontend with the Python RAG pipeline.

Endpoints
---------
GET  /                      → Serves frontend/index.html
GET  /static/<path>         → Serves frontend CSS / JS

POST /api/query/audio       → Accepts audio blob (multipart), saves to temp file,
                              runs full pipeline (STT → embed → Qdrant → LLM)
POST /api/query/text        → Accepts {"query": "..."}, skips STT, runs retrieval + LLM

Both POST endpoints return:
{
  "status":      "answered" | "refused_unsafe" | "refused_off_topic" |
                 "refused_not_grounded" | "error",
  "transcript":  str | null,
  "answer":      str | null,
  "guardrail_scores": {
    "top_similarity": float | null,
    "groundedness":   float | null
  },
  "latency": {          ← flattened from logger.metrics (P50 for each stage in ms)
    "stt":         float,
    "embedding":   float,
    "retrieval":   float,
    "generation":  float,
    "end_to_end":  float,
    "guardrail_input":        float,
    "guardrail_off_topic":    float,
    "guardrail_groundedness": float
  },
  "error_detail": str | null
}
"""

import os
import sys
import tempfile
import logging

from dotenv import load_dotenv

# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS

from src.core.stt import GroqSTT
from src.core.embeddings import EmbeddingModel
from src.core.vectorstore import QdrantVectorStore
from src.core.llm import GroqLLMBackend
from src.core.orchestrator import VoiceRAGOrchestrator
from src.core.latency import logger, LatencyLogger

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Build the pipeline once at startup (expensive models loaded in memory)
# ---------------------------------------------------------------------------

log.info("Initializing RAG pipeline…")

stt = GroqSTT()
embed_model = EmbeddingModel("intfloat/multilingual-e5-small")

# Remove stale Qdrant lock file if it exists (left behind after unclean shutdown)
_qdrant_lock = os.path.join(".", "qdrant_hindi_benchmark", ".lock")
if os.path.exists(_qdrant_lock):
    os.remove(_qdrant_lock)
    log.info("Removed stale Qdrant lock file.")

db = QdrantVectorStore(
    collection_name="hindi_rag_production",
    persist_directory="./qdrant_hindi_benchmark",
    vector_dim=384,
    query_prefix="query: ",
    passage_prefix="passage: ",
)
db.embedding_model = embed_model  # share the same loaded model instance

llm = GroqLLMBackend(
    model_name="openai/gpt-oss-20b",
    reasoning_effort="low",
    max_completion_tokens=150
)

orchestrator = VoiceRAGOrchestrator(
    stt_client=stt,
    vector_store=db,
    llm_backend=llm,
    embedding_model=embed_model,
    off_topic_threshold=0.826,    # Calibrated against multilingual-e5-small Hindi benchmark (irrelevant P99 ≈ 0.826, relevant 0.849-0.906)
    groundedness_threshold=0.75,
)

log.info("Pipeline ready.")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="/static")
CORS(app)  # Allow cross-origin requests (needed when opening index.html as file://)


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory(os.path.join(os.path.dirname(__file__), "assets"), filename)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_latency() -> dict:
    """Pull P50 from the shared global logger for every known stage."""
    stages = [
        "stt", "embedding", "retrieval", "generation", "end_to_end",
        "guardrail_input", "guardrail_off_topic", "guardrail_groundedness",
    ]
    result = {}
    for stage in stages:
        stats = logger.get_percentiles(stage)
        result[stage] = round(stats.get("p50", 0.0), 1)
    # Also capture any dynamic retry stages
    for stage, values in logger.metrics.items():
        if stage not in result and values:
            import numpy as np
            result[stage] = round(float(np.percentile(values, 50)), 1)
    return result


def _reset_logger():
    """Reset the global logger between requests so latencies stay per-query."""
    known_stages = [
        "stt", "embedding", "retrieval", "generation", "end_to_end",
        "guardrail_input", "guardrail_off_topic", "guardrail_groundedness",
    ]
    logger.metrics = {stage: [] for stage in known_stages}


def _build_json_response(pipeline_result: dict) -> dict:
    latency = _extract_latency()
    return {
        "status":           pipeline_result.get("status"),
        "transcript":       pipeline_result.get("transcript"),
        "answer":           pipeline_result.get("answer"),
        "guardrail_scores": pipeline_result.get("guardrail_scores", {}),
        "latency":          latency,
        "error_detail":     pipeline_result.get("error_detail"),
    }


# ---------------------------------------------------------------------------
# Audio endpoint
# ---------------------------------------------------------------------------

@app.route("/api/query/audio", methods=["POST"])
def query_audio():
    """
    Accepts an audio file uploaded as multipart/form-data with field name 'audio'.
    Runs the full pipeline (STT → retrieval → LLM) and returns a JSON response.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file in request (expected field 'audio')"}), 400

    audio_file = request.files["audio"]
    if not audio_file or audio_file.filename == "":
        return jsonify({"error": "Empty audio file"}), 400

    # Determine extension: browsers send WebM by default; Groq Whisper accepts it
    filename = audio_file.filename or "recording.webm"
    ext = os.path.splitext(filename)[1] or ".webm"

    _reset_logger()

    # Save to a temp file and run the pipeline
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp_path = tmp.name
        audio_file.save(tmp_path)

    try:
        result = orchestrator.process_voice_query(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return jsonify(_build_json_response(result))


# ---------------------------------------------------------------------------
# Text endpoint
# ---------------------------------------------------------------------------

@app.route("/api/query/text", methods=["POST"])
def query_text():
    """
    Accepts {"query": "..."} as JSON. Skips STT; runs retrieval + LLM.
    """
    body = request.get_json(silent=True) or {}
    query = (body.get("query") or "").strip()

    if not query:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    _reset_logger()

    # Build a minimal "audio" path trick: write the query as a fake transcript
    # by bypassing the STT step directly through the orchestrator internal logic.
    # We do this cleanly by routing through a tiny shim that pre-fills the transcript.
    result = _run_text_pipeline(query)

    return jsonify(_build_json_response(result))


def _run_text_pipeline(query: str) -> dict:
    """
    Run stages 2-7 of the orchestrator pipeline with a known text transcript,
    bypassing the STT stage. This mirrors the orchestrator's internal flow.
    """
    import time
    from src.core.guardrails import check_input_safety, check_off_topic, check_groundedness
    from src.core.orchestrator import _build_response, _SYSTEM_PROMPT, _with_retry

    overall_start = time.perf_counter()

    # Stage 2 – Input safety
    t0 = time.perf_counter()
    is_safe, unsafe_reason = check_input_safety(query)
    logger.log("guardrail_input", (time.perf_counter() - t0) * 1000)
    if not is_safe:
        return _build_response("refused_unsafe", transcript=query, error_detail=unsafe_reason)

    # Stage 3 – Retrieval
    results = db.query(query, k=orchestrator.max_retrieval_results, with_scores=True)

    # Stage 4 – Off-topic
    t0 = time.perf_counter()
    is_on_topic, top_score = check_off_topic(results, threshold=orchestrator.off_topic_threshold)
    logger.log("guardrail_off_topic", (time.perf_counter() - t0) * 1000)
    if not is_on_topic:
        return _build_response(
            "refused_off_topic",
            transcript=query, retrieved_context=results, top_similarity=top_score,
            error_detail=f"Query similarity ({top_score:.3f}) is below the off-topic threshold.",
        )

    # Stage 5 – LLM generation
    context_texts = [r["chunk"] for r in results]
    context_str = "\n\n".join(context_texts)
    prompt = f"<context>\n{context_str}\n</context>\n\nQuestion: {query}\nAnswer:"

    try:
        llm_response = _with_retry(
            fn=lambda: llm.generate(prompt=prompt, system_prompt=_SYSTEM_PROMPT),
            max_attempts=orchestrator.llm_max_attempts,
            base_delay=orchestrator.retry_base_delay,
            stage_label="llm",
        )
    except Exception as exc:
        return _build_response(
            "error", transcript=query, retrieved_context=results,
            top_similarity=top_score, error_detail=f"LLM generation failed: {exc}",
        )

    answer_text = llm_response["text"]

    # Stage 6 – Groundedness
    t0 = time.perf_counter()
    is_grounded, ground_score = check_groundedness(
        answer=answer_text,
        context_chunks=context_texts,
        embedding_model=embed_model,
        threshold=orchestrator.groundedness_threshold,
    )
    logger.log("guardrail_groundedness", (time.perf_counter() - t0) * 1000)
    if not is_grounded:
        return _build_response(
            "refused_not_grounded",
            transcript=query, retrieved_context=results,
            top_similarity=top_score, groundedness=ground_score,
            error_detail=f"Answer groundedness ({ground_score:.3f}) is below threshold.",
        )

    # Stage 7 – End-to-end
    logger.log("end_to_end", (time.perf_counter() - overall_start) * 1000)

    return _build_response(
        "answered",
        transcript=query, answer=answer_text, retrieved_context=results,
        top_similarity=top_score, groundedness=ground_score, llm_stats=llm_response,
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    log.info(f"Starting Flask server on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
