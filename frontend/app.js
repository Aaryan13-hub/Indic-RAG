/**
 * app.js — Frontend logic for the Indic-RAG Voice Interface.
 *
 * Features:
 *  - MediaRecorder Web API for real mic recording
 *  - POST audio blob to /api/query/audio
 *  - POST text query to /api/query/text
 *  - Render transcript + answer in terminal output
 *  - Update all 6 metric cards with real latency data
 */

const API_BASE = window.location.origin.startsWith('file:')
    ? 'http://localhost:5000'
    : window.location.origin;

// ─── DOM refs ────────────────────────────────────────────────────────────────
const terminalOutput = document.getElementById('terminal-output');
const inputField     = document.getElementById('query-input');
const sendButton     = document.getElementById('send-btn');
const micButton      = document.getElementById('mic-btn');

// Metric card value elements
const metricEls = {
    stt:         document.getElementById('metric-stt'),
    total:       document.getElementById('metric-total'),
    generation:  document.getElementById('metric-generation'),
    retrieval:   document.getElementById('metric-retrieval'),
    embedding:   document.getElementById('metric-embedding'),
    groundedness:document.getElementById('metric-groundedness'),
};

// ─── State ───────────────────────────────────────────────────────────────────
let mediaRecorder  = null;
let audioChunks    = [];
let isRecording    = false;

// ─── Terminal helpers ─────────────────────────────────────────────────────────
function removeCursor() {
    const cursor = terminalOutput.querySelector('.cursor-blink-wrap');
    if (cursor) cursor.remove();
}

function appendLine(html, type = 'system') {
    removeCursor();
    const p = document.createElement('p');
    p.style.marginBottom = '6px';
    p.style.lineHeight   = '1.5';
    if (type === 'user') {
        p.innerHTML = `<span style="color:var(--color-secondary-fixed);">&gt; YOU: ${escapeHtml(html)}</span>`;
    } else if (type === 'answer') {
        p.innerHTML = `<span style="color:var(--color-primary); font-weight: bold;">&gt; SYSTEM: ${escapeHtml(html)}</span>`;
    } else if (type === 'transcript') {
        p.innerHTML = `<span style="color:var(--color-tertiary);">&gt; STT: ${escapeHtml(html)}</span>`;
    } else if (type === 'error') {
        p.innerHTML = `<span style="color:var(--color-error);">&gt; ERROR: ${escapeHtml(html)}</span>`;
    } else if (type === 'warn') {
        p.innerHTML = `<span style="color:var(--color-tertiary);">&gt; ${escapeHtml(html)}</span>`;
    } else {
        p.innerHTML = `<span style="color:var(--color-outline);">&gt; ${html}</span>`;
    }
    terminalOutput.appendChild(p);
    addCursor();
    terminalOutput.scrollTop = terminalOutput.scrollHeight;
}

function addCursor() {
    removeCursor();
    const wrap = document.createElement('p');
    wrap.className = 'cursor-blink-wrap';
    wrap.style.marginTop = '12px';
    wrap.style.opacity   = '0.5';
    wrap.innerHTML = '<span class="cursor-blink"></span>';
    terminalOutput.appendChild(wrap);
}

function escapeHtml(text) {
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function setLoading(on) {
    sendButton.disabled = on;
    sendButton.style.opacity = on ? '0.5' : '1';
}

// ─── Metric updater ───────────────────────────────────────────────────────────
function updateMetrics(latency) {
    if (metricEls.stt)          metricEls.stt.textContent         = latency.stt          ? `${latency.stt}ms`         : '—';
    if (metricEls.total)        metricEls.total.textContent       = latency.end_to_end   ? `${latency.end_to_end}ms`  : '—';
    if (metricEls.generation)   metricEls.generation.textContent  = latency.generation   ? `${latency.generation}ms`  : '—';
    if (metricEls.retrieval)    metricEls.retrieval.textContent   = latency.retrieval    ? `${latency.retrieval}ms`   : '—';
    if (metricEls.embedding)    metricEls.embedding.textContent   = latency.embedding    ? `${latency.embedding}ms`   : '—';
    if (metricEls.groundedness) metricEls.groundedness.textContent= latency.guardrail_groundedness
        ? `${latency.guardrail_groundedness}ms` : '—';
}

// ─── Pipeline response renderer ───────────────────────────────────────────────
function renderResponse(data) {
    const { status, transcript, answer, guardrail_scores, latency, error_detail } = data;

    if (transcript) {
        appendLine(transcript, 'transcript');
    }

    if (status === 'answered' && answer) {
        appendLine(answer, 'answer');
        if (guardrail_scores) {
            const sim   = guardrail_scores.top_similarity   != null ? guardrail_scores.top_similarity.toFixed(3)  : '—';
            const grnd  = guardrail_scores.groundedness     != null ? guardrail_scores.groundedness.toFixed(3)   : '—';
            appendLine(`Relevance: ${sim} | Groundedness: ${grnd}`, 'system');
        }
    } else if (status === 'refused_unsafe') {
        appendLine('BLOCKED: Input failed safety check.', 'error');
    } else if (status === 'refused_off_topic') {
        appendLine('BLOCKED: Query is off-topic for this knowledge base.', 'warn');
    } else if (status === 'refused_not_grounded') {
        appendLine('BLOCKED: Answer could not be verified against retrieved context.', 'warn');
    } else if (status === 'error') {
        appendLine(error_detail || 'An unknown pipeline error occurred.', 'error');
    }

    if (latency) updateMetrics(latency);
    appendLine('PIPELINE COMPLETE.', 'system');
}

// ─── Text query ───────────────────────────────────────────────────────────────
async function handleTextSend() {
    const query = inputField.value.trim();
    if (!query) return;

    appendLine(query, 'user');
    inputField.value = '';
    setLoading(true);
    appendLine('PROCESSING TEXT QUERY...', 'system');

    try {
        const res = await fetch(`${API_BASE}/api/query/text`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        const data = await res.json();
        renderResponse(data);
    } catch (err) {
        appendLine(`Failed to reach server: ${err.message}`, 'error');
    } finally {
        setLoading(false);
    }
}

sendButton.addEventListener('click', handleTextSend);
inputField.addEventListener('keypress', e => { if (e.key === 'Enter') handleTextSend(); });

// ─── Voice / microphone ───────────────────────────────────────────────────────
async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioChunks  = [];
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = e => {
            if (e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.onstop = async () => {
            // Stop all mic tracks to release the hardware indicator
            stream.getTracks().forEach(t => t.stop());

            const mimeType = mediaRecorder.mimeType || 'audio/webm';
            const ext      = mimeType.includes('ogg') ? '.ogg' : '.webm';
            const blob     = new Blob(audioChunks, { type: mimeType });

            appendLine('AUDIO CAPTURED — SENDING TO PIPELINE...', 'system');
            setLoading(true);

            const formData = new FormData();
            formData.append('audio', blob, `recording${ext}`);

            try {
                const res = await fetch(`${API_BASE}/api/query/audio`, {
                    method: 'POST',
                    body: formData,
                });
                if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
                const data = await res.json();
                renderResponse(data);
            } catch (err) {
                appendLine(`Failed to reach server: ${err.message}`, 'error');
            } finally {
                setLoading(false);
            }
        };

        mediaRecorder.start();
        isRecording = true;
        micButton.classList.add('recording');
        micButton.style.backgroundColor = 'var(--color-secondary-fixed)';
        micButton.style.color           = 'var(--color-black)';
        appendLine('LISTENING... (click mic again to stop)', 'system');
    } catch (err) {
        if (err.name === 'NotAllowedError') {
            appendLine('Microphone access denied. Please allow mic access in your browser.', 'error');
        } else {
            appendLine(`Mic error: ${err.message}`, 'error');
        }
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    isRecording = false;
    micButton.classList.remove('recording');
    micButton.style.backgroundColor = '';
    micButton.style.color           = '';
}

micButton.addEventListener('click', () => {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
});

// ─── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    addCursor();
});
