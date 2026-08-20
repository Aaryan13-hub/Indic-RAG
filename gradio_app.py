import gradio as gr
import time
import os

# ==============================================================================
# 1. Custom CSS to match your Hacker House Goa Theme
# ==============================================================================
custom_css = """
body, .gradio-container {
    background-image: url('file/assets/bg.png');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    font-family: 'JetBrains Mono', monospace;
}

/* Make blocks transparent to show background */
.gr-box, .gr-panel {
    background-color: rgba(18, 20, 20, 0.85) !important;
    border: 1px solid #83d99c !important;
    border-radius: 0px !important;
}

/* Terminal Text Style */
.terminal-text {
    color: #eaea00 !important;
    font-family: 'JetBrains Mono', monospace;
}

/* Primary buttons */
.gr-button-primary {
    background-color: #eaea00 !important;
    color: #000 !important;
    border: none !important;
    font-weight: bold !important;
    border-radius: 0px !important;
}
.gr-button-primary:hover {
    background-color: #000 !important;
    color: #eaea00 !important;
    border: 1px solid #eaea00 !important;
}

/* Headings */
h1, h2, h3 {
    color: #83d99c !important;
    font-family: 'Bodoni Moda', serif !important;
    text-transform: uppercase;
}
"""

# ==============================================================================
# 2. Dummy Backend Functions (Where you will connect your actual logic)
# ==============================================================================
def process_query(audio_path, text_query):
    """
    HOW TO CONNECT YOUR BACKEND:
    ----------------------------
    1. Import your orchestrator at the top of this file:
       `from src.core.orchestrator import VoiceRAGOrchestrator` (and initialize it)
       
    2. Replace this dummy logic with your actual pipeline call.
       If audio_path is provided, pass it to orchestrator.process_voice_query(audio_path)
       If text_query is provided, pass it to your text pipeline.
    """
    
    # Dummy processing delay to simulate backend
    time.sleep(1.5)
    
    if audio_path:
        input_used = "Voice Input Received."
        dummy_answer = "यह आपके ऑडियो प्रश्न का उत्तर है। (This is the answer to your audio query.)"
    elif text_query:
        input_used = f"Text Input: {text_query}"
        dummy_answer = "यह आपके टेक्स्ट प्रश्न का उत्तर है। (This is the answer to your text query.)"
    else:
        return "Please provide audio or text.", "Error", "Error"

    # In reality, extract these from your orchestrator's 'result' dictionary
    dummy_terminal_logs = f"> INIT GOA_HOUSE_NODE...\n> {input_used}\n> PIPELINE COMPLETE.\n> GENERATING RESPONSE..."
    
    metrics = (
        "**STT Latency:** 350ms\n"
        "**Retrieval:** 45ms\n"
        "**LLM Gen:** 450ms\n"
        "**Total:** 845ms"
    )
    
    return dummy_terminal_logs, dummy_answer, metrics

# ==============================================================================
# 3. Gradio Interface Layout
# ==============================================================================
with gr.Blocks(css=custom_css, title="Indic-RAG Hacker House Goa") as demo:
    
    gr.Markdown("# 2:47 PM STUDIO | HACKER HOUSE GOA")
    
    with gr.Row():
        # LEFT COLUMN (Terminal & Input)
        with gr.Column(scale=2):
            gr.Markdown("## TERMINAL")
            
            terminal_output = gr.Textbox(
                label="System Logs", 
                value="> STATUS: ONLINE. WAITING FOR INPUT...", 
                interactive=False, 
                lines=6,
                elem_classes=["terminal-text"]
            )
            
            llm_response = gr.Textbox(
                label="LLM Output (Hindi)", 
                interactive=False, 
                lines=4
            )
            
            with gr.Row():
                # Audio Input
                audio_in = gr.Audio(sources=["microphone"], type="filepath", label="Mic Input")
                
            with gr.Row():
                # Text Input
                text_in = gr.Textbox(label="Manual Text Input", placeholder="Type query here...")
                send_btn = gr.Button("SEND QUERY", variant="primary")
                
            metrics_display = gr.Markdown("### Metrics\n(Waiting for query...)")
            
        # RIGHT COLUMN (Status & Instructions)
        with gr.Column(scale=1):
            gr.Markdown("### PIPELINE STATUS")
            gr.Markdown(
                "✓ **Qdrant:** `hindi_rag_production`\n\n"
                "✓ **Embedding:** `multilingual-e5-small`\n\n"
                "✓ **LLM:** `GPT OSS 20B (Groq)`\n\n"
                "✓ **STT:** `Whisper Large v3 (Groq)`"
            )
            
            gr.Markdown("### OPERATOR INSTRUCTIONS")
            gr.Markdown(
                "> Click the Mic to record your voice.\n\n"
                "> Or type a manual query.\n\n"
                "*Ensure local context is synced.*"
            )

    # Wire up the button to the backend function
    send_btn.click(
        fn=process_query,
        inputs=[audio_in, text_in],
        outputs=[terminal_output, llm_response, metrics_display]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
