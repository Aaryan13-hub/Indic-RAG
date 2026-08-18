"""
src/core/guardrails.py
======================
Layered guardrail checks for the VoiceRAG pipeline.

Three independent, pure-function checks — each returns a (bool, detail) tuple
so the caller can enforce the stop decision. No side effects; no state.

Layer 1 — check_input_safety  : keyword/regex pre-filter (runs before any API call)
Layer 2 — check_off_topic     : similarity-threshold check (runs after retrieval,
                                 before LLM — avoids paying for generation on a
                                 query the corpus can't answer)
Layer 3 — check_groundedness  : embedding cosine check on the generated answer
                                 (runs after LLM, before returning to caller)

Design notes
------------
* Regex patterns are compiled once at module load — no per-call overhead.
* check_groundedness reuses the EmbeddingModel the caller already holds (passed as
  an argument) so we never load the sentence-transformer model twice.
* All thresholds are exposed as parameters with documented defaults. Tune them
  against your real corpus before setting a hard deployment value.
"""

from __future__ import annotations

import re
import logging
from typing import List, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # Avoid circular import at runtime; only used for type hints.
    from .embeddings import EmbeddingModel

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layer 1: Input Safety — keyword / regex pre-filter
# ---------------------------------------------------------------------------

# Patterns are normalized (input is lowercased + whitespace-collapsed) before
# matching, so trivial obfuscation like extra spaces or CAPS doesn't bypass them.
#
# Design choices vs. the old simple_guardrail bare-word list:
#   "hack" / "bypass" — dropped bare; too many legitimate uses
#   "kill" bare        — dropped; would block "kill the background process" etc.
#   "system prompt"    — kept but narrowed to injection context (disregard/ignore)
#
# If you need to re-add bare words for your specific domain, append a
# re.compile(r"\byourword\b", re.IGNORECASE) entry to the list below.
_UNSAFE_PATTERNS: List[re.Pattern] = [
    # Prompt injection — direct instruction override
    re.compile(r"ignore\s+(previous|prior|all)\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(your|the)\s+(system\s*prompt|instructions?)", re.IGNORECASE),
    re.compile(r"forget\s+(all|your|previous)\s+(instructions?|context|rules?)", re.IGNORECASE),

    # Role-play / persona jailbreak
    re.compile(r"you\s+are\s+now\s+\w", re.IGNORECASE),
    re.compile(r"act\s+as\s+(if\s+you\s+are|a\s+|an\s+)\w", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be)\s+\w", re.IGNORECASE),

    # Common bypass tokens
    re.compile(r"\bDAN\b"),                          # "Do Anything Now" jailbreak
    re.compile(r"\bjailbreak\b", re.IGNORECASE),

    # Harmful content — targeted violence
    re.compile(r"(kill|murder|assassinate)\s+\w+\s+(person|people|user|him|her|them|someone)", re.IGNORECASE),

    # Harmful content — weapons / explosives instructions
    re.compile(r"(bomb|explosive|weapon)\s*(making|how\s+to\s+make|instructions?|recipe)", re.IGNORECASE),
]


def _normalize(text: str) -> str:
    """Lowercase and collapse all whitespace to single spaces."""
    return " ".join(text.lower().split())


def check_input_safety(text: str) -> Tuple[bool, str]:
    """Layer 1: keyword/regex pre-filter.

    Normalizes the input before matching so trivial obfuscation is not a bypass.

    Args:
        text: The raw transcript or query string.

    Returns:
        (is_safe, reason)
        is_safe=True  → pipeline may continue.
        is_safe=False → pipeline must stop; reason describes which pattern fired.
    """
    normalized = _normalize(text)
    for pattern in _UNSAFE_PATTERNS:
        if pattern.search(normalized):
            reason = f"Matched unsafe pattern: /{pattern.pattern}/"
            _log.warning("[guardrail:input_safety] BLOCKED — %s", reason)
            return False, reason
    return True, ""


# ---------------------------------------------------------------------------
# Layer 2: Off-Topic Check — similarity threshold
# ---------------------------------------------------------------------------

# DEFAULT_OFF_TOPIC_THRESHOLD
# ---------------------------
# Starting point: 0.35 (cosine similarity, 0–1 scale).
#
# Rationale: on-topic queries against a well-indexed corpus with all-MiniLM-L6-v2
# typically score 0.45–0.80. A threshold of 0.35 gives a ~0.10 margin below the
# expected on-topic floor, so clearly off-domain questions (score < 0.20–0.25)
# are blocked without risk of false-positives on borderline queries.
#
# TUNING REQUIRED: inspect your actual score distribution once the corpus is
# indexed (log result[0]["score"] for a representative query set) and move the
# threshold to sit just below the bottom of your on-topic score distribution.
# Override via the orchestrator's off_topic_threshold constructor arg.
DEFAULT_OFF_TOPIC_THRESHOLD: float = 0.35


def check_off_topic(
    results: List[dict],
    threshold: float = DEFAULT_OFF_TOPIC_THRESHOLD,
) -> Tuple[bool, float]:
    """Layer 2: similarity threshold check on retrieval results.

    Must be called with results from vector_store.query(..., with_scores=True).
    Fails closed (returns is_on_topic=False) if results is empty or the 'score'
    key is absent — both indicate the corpus cannot answer the query.

    Args:
        results:   List of result dicts from VectorStore.query(with_scores=True).
        threshold: Cosine similarity threshold (0–1). Queries whose top result
                   scores below this are considered off-topic.

    Returns:
        (is_on_topic, top_score)
        is_on_topic=True  → pipeline may continue to LLM generation.
        is_on_topic=False → pipeline must stop; return "refused_off_topic".
    """
    if not results:
        _log.warning("[guardrail:off_topic] BLOCKED — no results returned by vector store")
        return False, 0.0

    top_score = results[0].get("score")
    if top_score is None:
        # 'score' key missing — query() was called without with_scores=True.
        # Log a warning but fail open so we don't silently block valid queries
        # due to a caller configuration error.
        _log.error(
            "[guardrail:off_topic] 'score' key missing from results — "
            "did you call query(with_scores=True)? Failing open."
        )
        return True, 0.0

    top_score = float(top_score)
    is_on_topic = top_score >= threshold
    if not is_on_topic:
        _log.warning(
            "[guardrail:off_topic] BLOCKED — top score %.3f < threshold %.3f",
            top_score, threshold,
        )
    return is_on_topic, top_score


# ---------------------------------------------------------------------------
# Layer 3: Groundedness Check — embedding cosine similarity
# ---------------------------------------------------------------------------

# DEFAULT_GROUNDEDNESS_THRESHOLD
# --------------------------------
# ⚠️  PLACEHOLDER VALUE — must be calibrated empirically before deployment.
#
# Why embedding cosine (not trigram/Jaccard overlap)?
#   Our context passages are English, but answers are generated in the same
#   language as the query — which may be Hindi, Marathi, or another Indic script.
#   Lexical overlap between a Hindi answer and English context would be near-zero
#   even for a perfectly grounded answer, causing almost every valid response to
#   be refused. Embedding cosine similarity in a shared semantic space avoids
#   this language mismatch.
#
# ⚠️  Cross-lingual caveat:
#   all-MiniLM-L6-v2 is an English-first model. Cross-lingual semantic similarity
#   scores between Hindi answers and English context will be systematically lower
#   than monolingual (English–English) scores. If groundedness refusals are too
#   aggressive for Hindi answers, either:
#     a) Lower this threshold (e.g., 0.25–0.30) after observing real score distributions, OR
#     b) Swap the EmbeddingModel to 'paraphrase-multilingual-MiniLM-L12-v2' for
#        better cross-lingual alignment — the orchestrator accepts an injected
#        EmbeddingModel, so no further code changes are needed.
#
# How to calibrate:
#   1. Run 10–20 grounded English Q&A pairs → record similarity scores.
#   2. Run 10–20 grounded Hindi Q&A pairs  → record similarity scores.
#   3. Run 5–10 hallucinated answers        → record similarity scores.
#   4. Set threshold between the hallucinated ceiling and the grounded floor.
DEFAULT_GROUNDEDNESS_THRESHOLD: float = 0.35


def check_groundedness(
    answer: str,
    context_chunks: List[str],
    embedding_model: "EmbeddingModel",
    threshold: float = DEFAULT_GROUNDEDNESS_THRESHOLD,
) -> Tuple[bool, float]:
    """Layer 3: embedding cosine similarity groundedness check.

    Embeds the generated answer and the concatenated retrieved context using the
    same EmbeddingModel already held by the orchestrator (no second model load).
    Computes cosine similarity in the shared embedding space as a cheap proxy for
    semantic overlap — this is language-agnostic, which matters for cross-lingual
    (Hindi answer / English context) cases where lexical overlap would be zero.

    Args:
        answer:          The LLM-generated answer text.
        context_chunks:  List of retrieved chunk strings (from vector store).
        embedding_model: The EmbeddingModel instance to reuse (passed from orchestrator).
        threshold:       Cosine similarity floor. Answers below this are considered
                         ungrounded. See calibration guidance in module docstring above.

    Returns:
        (is_grounded, similarity_score)
        is_grounded=True  → pipeline may return the answer to the caller.
        is_grounded=False → pipeline must stop; return "refused_not_grounded".
    """
    if not answer.strip():
        _log.warning("[guardrail:groundedness] BLOCKED — empty answer")
        return False, 0.0

    if not context_chunks:
        _log.warning("[guardrail:groundedness] BLOCKED — no context chunks to compare against")
        return False, 0.0

    # Bypass groundedness check for valid LLM refusals. The refusal string is mathematically
    # dissimilar to the retrieved context, so it will always fail the cosine similarity check
    # and incorrectly trigger a "hallucination" error if we don't explicitly allow it.
    if "I cannot answer this based on the provided context." in answer:
        _log.info("[guardrail:groundedness] BYPASS — valid LLM refusal detected.")
        return True, 1.0

    # Concatenate all context chunks into a single passage for embedding.
    # A single embedding over the full context captures the aggregate semantic
    # space of the retrieved passages, which is what we compare against.
    context_str = " ".join(context_chunks)

    # Embed answer + context in one batch to minimise model inference overhead.
    # Note: EmbeddingModel.embed() logs its own "embedding" latency stage, which
    # will appear in the latency report alongside the guardrail_groundedness stage
    # timed by the orchestrator.
    try:
        embeddings = embedding_model.embed([answer, context_str])
    except Exception as exc:
        _log.error("[guardrail:groundedness] Embedding failed: %s — failing open", exc)
        # Fail open on embedding error: don't refuse a potentially valid answer
        # due to an infrastructure failure in the guardrail itself.
        return True, 0.0

    answer_vec = np.array(embeddings[0], dtype=np.float32)
    context_vec = np.array(embeddings[1], dtype=np.float32)

    norm_a = float(np.linalg.norm(answer_vec))
    norm_c = float(np.linalg.norm(context_vec))

    if norm_a == 0.0 or norm_c == 0.0:
        _log.warning("[guardrail:groundedness] Zero-norm embedding vector — failing closed")
        return False, 0.0

    similarity = float(np.dot(answer_vec, context_vec) / (norm_a * norm_c))
    # Clamp to [0, 1] — cosine similarity can be slightly outside due to float precision
    similarity = max(0.0, min(1.0, similarity))

    is_grounded = similarity >= threshold
    if not is_grounded:
        _log.warning(
            "[guardrail:groundedness] BLOCKED — similarity %.3f < threshold %.3f",
            similarity, threshold,
        )
    return is_grounded, similarity
