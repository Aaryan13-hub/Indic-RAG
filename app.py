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

db = QdrantVectorStore(
    collection_name="hindi_rag_production",
    persist_directory="./qdrant_hindi_benchmark",  # ignored when QDRANT_URL is set
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
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,400;0,700;1,400&family=Bodoni+Moda:wght@700&display=swap');

:root {
    --bg-dark:   #0d1a0d;
    --bg-panel:  rgba(10, 24, 10, 0.92);
    --neon-yellow: #eaea00;
    --neon-green:  #83d99c;
    --border:    rgba(131, 217, 156, 0.3);
    --text-dim:  #6b8f6b;
}

body, .gradio-container {
    background-color: var(--bg-dark) !important;
    background-image:
        repeating-linear-gradient(0deg, rgba(131,217,156,0.03) 0px, transparent 1px, transparent 3px),
        radial-gradient(ellipse at 10% 50%, rgba(20,60,20,0.6) 0%, transparent 60%),
        radial-gradient(ellipse at 90% 50%, rgba(20,60,20,0.6) 0%, transparent 60%);
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--neon-green) !important;
}

/* Scanline overlay */
body::after {
    content: '';
    position: fixed; inset: 0;
    background: repeating-linear-gradient(transparent, transparent 2px, rgba(0,0,0,0.07) 2px, rgba(0,0,0,0.07) 4px);
    pointer-events: none; z-index: 9999;
}

.gradio-container { max-width: 1200px !important; }

/* Panels */
.gr-box, .gr-panel, .gr-form, .gr-block {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
}

/* Labels */
label, .gr-form label, .block .label-wrap span {
    color: var(--neon-green) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

/* Textboxes and inputs */
textarea, input[type=text] {
    background: rgba(0,20,0,0.8) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
    color: var(--neon-yellow) !important;
    font-family: 'JetBrains Mono', monospace !important;
    caret-color: var(--neon-yellow);
}
textarea:focus, input[type=text]:focus {
    border-color: var(--neon-yellow) !important;
    box-shadow: 0 0 8px rgba(234,234,0,0.3) !important;
}

/* Primary button */
.gr-button-primary, button.primary {
    background: var(--neon-yellow) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    transition: all 0.15s ease !important;
}
.gr-button-primary:hover, button.primary:hover {
    background: #000 !important;
    color: var(--neon-yellow) !important;
    box-shadow: 0 0 12px rgba(234,234,0,0.4), inset 0 0 0 1px var(--neon-yellow) !important;
}

/* Secondary button */
button.secondary {
    background: transparent !important;
    color: var(--neon-green) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Tabs */
.tab-nav button {
    color: var(--text-dim) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    background: transparent !important;
}
.tab-nav button.selected {
    color: var(--neon-yellow) !important;
    border-bottom-color: var(--neon-yellow) !important;
}

/* Markdown output */
.gr-prose, .gr-markdown {
    color: var(--neon-green) !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.gr-prose h3 { color: var(--neon-yellow) !important; }
.gr-prose code {
    background: rgba(234,234,0,0.08) !important;
    color: var(--neon-yellow) !important;
    border: 1px solid rgba(234,234,0,0.2) !important;
    border-radius: 0 !important;
}

/* Header */
.hh-header {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
    position: relative;
}
.hh-header .studio-tag {
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    color: var(--text-dim);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.hh-header h1 {
    font-family: 'Bodoni Moda', serif;
    font-size: clamp(2.5rem, 8vw, 5rem);
    font-weight: 700;
    color: var(--neon-yellow);
    line-height: 1;
    letter-spacing: -1px;
    margin: 0;
    text-shadow: 0 0 40px rgba(234,234,0,0.25);
}
.hh-header .goa {
    font-family: 'Bodoni Moda', serif;
    font-size: clamp(1.8rem, 5vw, 3.2rem);
    color: #f472b6;
    display: block;
    margin-top: -0.25rem;
}
.hh-header .meta {
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    color: var(--text-dim);
    margin-top: 0.75rem;
    text-transform: uppercase;
}

/* Pipeline status bar */
.pipeline-bar {
    display: flex;
    gap: 1.5rem;
    justify-content: center;
    flex-wrap: wrap;
    padding: 1rem;
    border-top: 1px solid var(--border);
    margin-top: 1.5rem;
    font-size: 0.7rem;
    letter-spacing: 0.06em;
    color: var(--text-dim);
}
.pipeline-bar .ok { color: var(--neon-green); margin-right: 0.3rem; }

/* Audio recorder */
.audio-recorder { border: 1px solid var(--border) !important; border-radius: 0 !important; }
"""

HEADER_HTML = """
<div class="hh-header">
  <div class="studio-tag">2:47 PM STUDIO</div>
  <h1>HACKER<br>HOUSE<span class="goa">गोवा</span></h1>
  <div class="meta">GOA, INDIA &nbsp;·&nbsp; 28–31 OCT 2026 &nbsp;·&nbsp; INDIC RAG 🇮🇳</div>
</div>
"""

PIPELINE_STATUS_HTML = """
<div class="pipeline-bar">
  <span><span class="ok">✓</span> Qdrant: hindi_rag_production (Cloud)</span>
  <span><span class="ok">✓</span> Embedding: multilingual-e5-small</span>
  <span><span class="ok">✓</span> LLM: GPT-OSS-20B (Groq)</span>
  <span><span class="ok">✓</span> STT: Whisper Large v3 (Groq)</span>
  <span><span class="ok">✓</span> Guardrails: Input · Off-topic · Groundedness</span>
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
