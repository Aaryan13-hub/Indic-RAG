import os
import sys
import glob
from dotenv import load_dotenv

load_dotenv()

from src.core.stt import GroqSTT
from src.core.vectorstore import QdrantVectorStore
from src.core.llm import GroqLLMBackend
from src.core.orchestrator import VoiceRAGOrchestrator
from src.core.latency import logger

def main():
    print("Initializing Voice RAG Pipeline...")
    
    # Initialize components
    stt = GroqSTT()
    db = QdrantVectorStore(collection_name="test_collection")
    llm = GroqLLMBackend()
    
    orchestrator = VoiceRAGOrchestrator(stt_client=stt, vector_store=db, llm_backend=llm)
    
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
