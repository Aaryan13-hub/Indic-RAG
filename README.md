# Indic-RAG 🇮🇳

An ultra-fast, fully offline-vectorized Voice RAG pipeline for Hindi, built at Hacker House Goa 2026.

This project enables voice-to-voice RAG in Hindi. It transcribes audio using Groq Whisper, embeds the text using `multilingual-e5-small` via PyTorch, retrieves context from a local Qdrant Vector database, applies multiple safety & hallucination guardrails, and generates an answer using Groq's GPT-OSS-20b model.

**End-to-End Latency:** ~600ms  
**LLM Generation:** ~450ms

## 🚀 Features
- **Voice STT:** Groq `whisper-large-v3` (Enforced Devanagari Script)
- **Local Embeddings:** `intfloat/multilingual-e5-small` (Running via PyTorch `sentence-transformers`)
- **Vector DB:** Qdrant (Local pre-built `.sqlite` storage)
- **LLM Engine:** Groq `openai/gpt-oss-20b`
- **Dynamic Guardrails:** Input Safety, Off-topic prevention, and strict Groundedness checks.

---

## 🛠️ Quick Start Setup (For Collaborators)

This repository includes the **fully ingested Qdrant database** (`qdrant_hindi_benchmark/`). You do not need to re-ingest the data! Just clone and run.

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Indic-RAG.git
cd Indic-RAG
```

### 3. Install Dependencies
It is highly recommended to use a virtual environment.
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 4. Setup Environment Variables
Create a `.env` file in the root directory by copying the example:
```bash
cp .env.example .env
```
Open the `.env` file and add your actual Groq API key:
```env
GROQ_API_KEY=gsk_your_real_key_here
```

### 5. Run the Server
Start the Flask API and the Frontend Server:
```bash
python api.py
```

### 6. Use the App
Open your browser and navigate to:  
👉 **`http://localhost:5000`**

Click the Microphone button to record a query in Hindi, or use the terminal input!

---

## 📂 Project Structure

- `api.py` - Flask server, routing, and HTTP connection pooling.
- `main.py` - CLI version of the RAG pipeline.
- `frontend/` - HTML/CSS/JS for the beautiful Glassmorphism UI.
- `src/core/` - The brains of the operation (Orchestrator, STT, Embeddings, LLM, Guardrails).
- `qdrant_hindi_benchmark/` - **The pre-built Vector Database.** Do not delete this!
- `scripts/` - Diagnostics, embedding benchmarks, and chunking A/B tests.
