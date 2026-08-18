import os
import sys
import glob
from dotenv import load_dotenv

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

from src.core.stt import GroqSTT
from src.core.embeddings import EmbeddingModel
from src.core.vectorstore import QdrantVectorStore
from src.core.llm import GroqLLMBackend
from src.core.orchestrator import VoiceRAGOrchestrator
from src.core.latency import logger

def main():
    print("Initializing Voice RAG Pipeline...")
    
    # Initialize components
    # 1. STT Provider (Swappable to ElevenLabsSTT later)
    stt = GroqSTT()
    print(f"[*] Active STT Provider: {stt.__class__.__name__}")
    
    # 2. Embedding Model (E5 requires prefixes)
    embed_model = EmbeddingModel("intfloat/multilingual-e5-small")
    
    # 3. Vector DB (Connected to the offline ingested Hindi index)
    db = QdrantVectorStore(
        collection_name="hindi_rag_production",
        persist_directory="./qdrant_hindi_benchmark",
        vector_dim=384,
        query_prefix="query: ",
        passage_prefix="passage: ",  # Used during add(), ignored in query()
    )
    # Inject the shared embedding model so it's not loaded twice
    db.embedding_model = embed_model
    
    # 4. LLM Backend
    llm = GroqLLMBackend()
    
    # 5. Orchestrator
    orchestrator = VoiceRAGOrchestrator(
        stt_client=stt,
        vector_store=db,
        llm_backend=llm,
        embedding_model=embed_model,
        off_topic_threshold=0.75,     # Tuned for E5 cosine similarity
        groundedness_threshold=0.75,  # Tuned for E5 cosine similarity
    )
    
    # Find audio file
    audio_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    if not audio_file:
        audio_files = glob.glob("*.wav") + glob.glob("*.mp3") + glob.glob("*.m4a")
        if audio_files:
            audio_file = audio_files[0]
            
    if not audio_file or not os.path.exists(audio_file):
        print(f"\n[!] Error: No audio file found.")
        print("Please place a .wav, .mp3, or .m4a file in this directory.")
        return
        
    try:
        response = orchestrator.process_voice_query(audio_file)
        
        print("\n==================================================")
        print(" FINAL ANSWER")
        print("==================================================")
        print(response['answer'])
        print("==================================================\n")
        
        logger.report()
        
    except Exception as e:
        print(f"Pipeline Error: {e}")

if __name__ == "__main__":
    main()
