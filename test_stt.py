import os
from dotenv import load_dotenv
load_dotenv()

from src.core.stt import GroqSTT
from src.core.latency import logger

def test_stt():
    print("Testing Groq Whisper Speech-to-Text API...")
    
    # Look for any audio file in the current directory if not explicitly passed
    import sys
    import glob
    
    audio_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    if not audio_file:
        audio_files = glob.glob("*.wav") + glob.glob("*.mp3") + glob.glob("*.m4a")
        if audio_files:
            audio_file = audio_files[0]
            
    if not audio_file or not os.path.exists(audio_file):
        print(f"\n[!] Error: No audio file found.")
        print("Please record a short audio file (e.g. test.m4a or test.wav)")
        print("Save it in the 'Indic-RAG' folder and run this script again.")
        print("Or pass it directly: uv run test_stt.py my_audio.m4a")
        return
        
    try:
        stt = GroqSTT()
        transcript = stt.transcribe(audio_file)
        
        print("\nSuccess! Transcribed Text:")
        print(f"\"{transcript}\"")
        
        print("\nLatency Report:")
        logger.report()
        
    except Exception as e:
        print(f"\nSTT failed: {e}")

if __name__ == "__main__":
    test_stt()
