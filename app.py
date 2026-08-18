"""
app.py — Gradio interface for Indic-RAG.
Designed for Hugging Face Spaces (Gradio SDK) — free tier compatible.

Replaces the Flask + HTML/JS frontend while reusing all src.core modules directly.
Entry point: HF Spaces auto-detects `demo` in app.py.
"""

import os
import time
import logging

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bootstrap pipeline (loaded once at startup)
# ---------------------------------------------------------------------------
log.info("Initializing RAG pipeline…")

from src.core.stt import GroqSTT
from src.core.embeddings import EmbeddingModel
from src.core.vectorstore import QdrantVectorStore
from src.core.llm import GroqLLMBackend
from src.core.orchestrator import VoiceRAGOrchestrator, _build_response, _SYSTEM_PROMPT, _with_retry
from src.core.guardrails import check_input_safety, check_off_topic, check_groundedness
from src.core.latency import logger as lat_logger

stt = GroqSTT()
embed_model = EmbeddingModel("intfloat/multilingual-e5-small")

_lock = os.path.join(".", "qdrant_hindi_benchmark", ".lock")
if os.path.exists(_lock):
    os.remove(_lock)
    log.info("Removed stale Qdrant lock file.")

db = QdrantVectorStore(
    collection_name="hindi_rag_production",
    persist_directory="./qdrant_hindi_benchmark",
    vector_dim=384,
    query_prefix="query: ",
    passage_prefix="passage: ",
)
db.embedding_model = embed_model

llm = GroqLLMBackend(
    model_name="openai/gpt-oss-20b",
    reasoning_effort="low",
    max_completion_tokens=150,
)

orchestrator = VoiceRAGOrchestrator(
    stt_client=stt,
    vector_store=db,
    llm_backend=llm,
    embedding_model=embed_model,
    off_topic_threshold=0.826,
    groundedness_threshold=0.75,
)

log.info("Pipeline ready.")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STAGES = [
    "stt", "embedding", "retrieval", "generation", "end_to_end",
    "guardrail_input", "guardrail_off_topic", "guardrail_groundedness",
]


def _reset_latency():
    lat_logger.metrics = {s: [] for s in STAGES}


def _get_latency() -> dict:
    result = {}
    for stage in STAGES:
        stats = lat_logger.get_percentiles(stage)
        result[stage] = round(stats.get("p50", 0.0), 1)
    return result


def _run_text_pipeline(query: str) -> dict:
    """Run the full RAG pipeline on a text query (skips STT)."""
    overall_start = time.perf_counter()

    t0 = time.perf_counter()
    is_safe, unsafe_reason = check_input_safety(query)
    lat_logger.log("guardrail_input", (time.perf_counter() - t0) * 1000)
    if not is_safe:
        return _build_response("refused_unsafe", transcript=query, error_detail=unsafe_reason)

    t0 = time.perf_counter()
    results = db.query(query, k=orchestrator.max_retrieval_results, with_scores=True)
    lat_logger.log("retrieval", (time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    is_on_topic, top_score = check_off_topic(results, threshold=orchestrator.off_topic_threshold)
    lat_logger.log("guardrail_off_topic", (time.perf_counter() - t0) * 1000)
    if not is_on_topic:
        return _build_response(
            "refused_off_topic",
            transcript=query, retrieved_context=results, top_similarity=top_score,
            error_detail=f"Query similarity ({top_score:.3f}) is below the off-topic threshold.",
        )

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

    t0 = time.perf_counter()
    is_grounded, ground_score = check_groundedness(
        answer=answer_text,
        context_chunks=context_texts,
        embedding_model=embed_model,
        threshold=orchestrator.groundedness_threshold,
    )
    lat_logger.log("guardrail_groundedness", (time.perf_counter() - t0) * 1000)
    if not is_grounded:
        return _build_response(
            "refused_not_grounded",
            transcript=query, retrieved_context=results,
            top_similarity=top_score, groundedness=ground_score,
            error_detail=f"Answer groundedness ({ground_score:.3f}) is below threshold.",
        )

    lat_logger.log("end_to_end", (time.perf_counter() - overall_start) * 1000)

    return _build_response(
        "answered",
        transcript=query, answer=answer_text, retrieved_context=results,
        top_similarity=top_score, groundedness=ground_score, llm_stats=llm_response,
    )


STATUS_EMOJI = {
    "answered": "✅",
    "refused_unsafe": "🚫",
    "refused_off_topic": "📵",
    "refused_not_grounded": "🔍",
    "error": "❌",
}

STATUS_LABEL = {
    "answered": "Answered",
    "refused_unsafe": "Refused — Unsafe Input",
    "refused_off_topic": "Refused — Off-topic",
    "refused_not_grounded": "Refused — Not Grounded",
    "error": "Error",
}


def _format_output(result: dict):
    status = result.get("status", "error")
    answer = result.get("answer") or ""
    error_detail = result.get("error_detail") or ""
    guardrail = result.get("guardrail_scores", {})
    latency = _get_latency()

    emoji = STATUS_EMOJI.get(status, "❓")
    label = STATUS_LABEL.get(status, status)

    if status == "answered":
        answer_md = f"### {emoji} {label}\n\n{answer}"
    else:
        answer_md = f"### {emoji} {label}\n\n_{error_detail}_"

    def ms(v):
        return f"`{v} ms`" if v else "—"

    top_sim = guardrail.get("top_similarity")
    ground = guardrail.get("groundedness")

    metrics_md = f"""
### ⏱ Latency

| Stage | P50 |
|---|---|
| 🔤 STT | {ms(latency.get('stt'))} |
| 🧠 Embedding | {ms(latency.get('embedding'))} |
| 🗃 Retrieval | {ms(latency.get('retrieval'))} |
| 🤖 LLM Generation | {ms(latency.get('generation'))} |
| 🏁 End-to-End | {ms(latency.get('end_to_end'))} |

### 🛡 Guardrail Scores

| Check | Score |
|---|---|
| Top Similarity | {f"`{top_sim:.3f}`" if top_sim is not None else "—"} |
| Groundedness | {f"`{ground:.3f}`" if ground is not None else "—"} |
"""
    return answer_md, metrics_md


# ---------------------------------------------------------------------------
# Gradio query handlers
# ---------------------------------------------------------------------------

def handle_text(query: str):
    if not query.strip():
        return "⚠️ Please enter a query in Hindi.", ""
    _reset_latency()
    result = _run_text_pipeline(query.strip())
    return _format_output(result)


def handle_audio(audio_path):
    if audio_path is None:
        return "⚠️ Please record audio first.", ""
    _reset_latency()
    result = orchestrator.process_voice_query(audio_path)
    transcript = result.get("transcript") or ""
    answer_md, metrics_md = _format_output(result)
    if transcript:
        answer_md = f"**📝 Transcript:** _{transcript}_\n\n{answer_md}"
    return answer_md, metrics_md


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

import gradio as gr

CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;500;600;700&display=swap');

body, .gradio-container {
    background: #0a0a0f !important;
    font-family: 'Inter', sans-serif !important;
}

.indic-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid rgba(99,102,241,.25);
    margin-bottom: 1.5rem;
}
.indic-header h1 {
    font-family: 'JetBrains Mono', monospace;
    font-size: clamp(1.8rem, 5vw, 3rem);
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 .5rem;
    letter-spacing: -1px;
}
.indic-header .subtitle {
    font-size: .9rem;
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: .08em;
}
.indic-header .badges {
    display: flex;
    justify-content: center;
    gap: .5rem;
    margin-top: .75rem;
    flex-wrap: wrap;
}
.badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: .7rem;
    padding: .25rem .6rem;
    border-radius: 999px;
    border: 1px solid;
    letter-spacing: .05em;
}
.badge-purple { color: #a78bfa; border-color: rgba(167,139,250,.35); background: rgba(167,139,250,.08); }
.badge-pink   { color: #f472b6; border-color: rgba(244,114,182,.35); background: rgba(244,114,182,.08); }
.badge-green  { color: #34d399; border-color: rgba(52,211,153,.35); background: rgba(52,211,153,.08); }

.pipeline-status {
    display: flex;
    gap: 1rem;
    justify-content: center;
    flex-wrap: wrap;
    padding: 1.25rem 1rem;
    border-top: 1px solid rgba(99,102,241,.15);
    margin-top: 1.5rem;
}
.status-item {
    font-family: 'JetBrains Mono', monospace;
    font-size: .72rem;
    color: #64748b;
    letter-spacing: .05em;
}
.status-item span { color: #34d399; margin-right: .35rem; }
"""

HEADER_HTML = """
<div class="indic-header">
  <h1>INDIC — RAG &nbsp;🇮🇳</h1>
  <p class="subtitle">HINDI VOICE RAG · HACKER HOUSE GOA 2026</p>
  <div class="badges">
    <span class="badge badge-purple">multilingual-e5-small</span>
    <span class="badge badge-pink">Groq Whisper v3</span>
    <span class="badge badge-green">GPT-OSS-20B</span>
    <span class="badge badge-purple">Qdrant · Local</span>
  </div>
</div>
"""

PIPELINE_STATUS_HTML = """
<div class="pipeline-status">
  <span class="status-item"><span>✓</span> Qdrant: hindi_rag_production</span>
  <span class="status-item"><span>✓</span> Embedding: multilingual-e5-small</span>
  <span class="status-item"><span>✓</span> LLM: GPT-OSS-20B (Groq)</span>
  <span class="status-item"><span>✓</span> STT: Whisper Large v3 (Groq)</span>
  <span class="status-item"><span>✓</span> Guardrails: Input · Off-topic · Groundedness</span>
</div>
"""

with gr.Blocks(css=CSS, title="Indic-RAG | Hindi Voice RAG") as demo:

    gr.HTML(HEADER_HTML)

    with gr.Tabs():

        # ── Text Query Tab
        with gr.Tab("⌨️  Text Query"):
            with gr.Row():
                with gr.Column(scale=3):
                    text_input = gr.Textbox(
                        label="Hindi Query",
                        placeholder="यहाँ अपना प्रश्न लिखें... (Type your Hindi question here)",
                        lines=3,
                    )
                    text_btn = gr.Button("▶  SEND QUERY", variant="primary", size="lg")

                with gr.Column(scale=2):
                    gr.Markdown("""
### 📋 Instructions
- Type your query **in Hindi** (Devanagari script)
- The pipeline retrieves context from the **local Qdrant** vector database
- Guardrails check for **safety**, **topic relevance**, and **groundedness**
- Average end-to-end latency: **~600ms**
                    """)

            text_answer = gr.Markdown(value="_Waiting for query…_")
            text_metrics = gr.Markdown()

            text_btn.click(fn=handle_text, inputs=[text_input], outputs=[text_answer, text_metrics])
            text_input.submit(fn=handle_text, inputs=[text_input], outputs=[text_answer, text_metrics])

        # ── Voice Query Tab
        with gr.Tab("🎙️  Voice Query"):
            with gr.Row():
                with gr.Column(scale=3):
                    audio_input = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="Record your Hindi question",
                    )
                    audio_btn = gr.Button("▶  TRANSCRIBE & QUERY", variant="primary", size="lg")

                with gr.Column(scale=2):
                    gr.Markdown("""
### 🎤 Voice Instructions
- Click the **microphone** button to start recording
- Speak your question **clearly in Hindi**
- Click **stop** when done, then hit **Transcribe & Query**
- Audio is transcribed via **Groq Whisper Large v3**
- Script is enforced to **Devanagari** only
                    """)

            audio_answer = gr.Markdown(value="_Waiting for audio…_")
            audio_metrics = gr.Markdown()

            audio_btn.click(fn=handle_audio, inputs=[audio_input], outputs=[audio_answer, audio_metrics])

    gr.HTML(PIPELINE_STATUS_HTML)

# HF Spaces auto-launches via `demo`; this is for local dev only
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
