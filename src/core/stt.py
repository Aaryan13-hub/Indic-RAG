import os
import time
import requests
from typing import Dict, Any
from .latency import logger

import groq

class SarvamSTT:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("SARVAM_API_KEY")
        if not self.api_key:
            raise ValueError("SARVAM_API_KEY environment variable is not set.")
        
        self.url = "https://api.sarvam.ai/speech-to-text"
        
    def transcribe(self, audio_file_path: str, language_code: str = "hi-IN") -> str:
        """
        Transcribes the given audio file using Sarvam's saaras:v3 model.
        Logs the network latency.
        """
        start_time = time.perf_counter()
        
        headers = {
            "api-subscription-key": self.api_key
        }
        
        data = {
            "model": "saaras:v3",
            "mode": "transcribe" # Could also be 'translate' to translate directly to English
        }
        
        # Determine content type based on extension
        ext = os.path.splitext(audio_file_path)[1].lower()
        content_type = "audio/wav"
        if ext == ".mp3":
            content_type = "audio/mpeg"
        elif ext == ".ogg":
            content_type = "audio/ogg"
        elif ext == ".m4a":
            content_type = "audio/mp4"
            
        with open(audio_file_path, "rb") as f:
            files = {
                "file": (os.path.basename(audio_file_path), f, content_type)
            }
            
            response = requests.post(self.url, headers=headers, data=data, files=files)
            
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.log("stt", duration_ms)
        
        if response.status_code != 200:
            raise Exception(f"Sarvam API Error {response.status_code}: {response.text}")
            
        result = response.json()
        return result.get("transcript", "")

class ElevenLabsSTT:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable is not set.")
            
        from elevenlabs.client import ElevenLabs
        self.client = ElevenLabs(api_key=self.api_key)
        
    def transcribe(self, audio_file_path: str, language_code: str = None) -> str:
        """
        Transcribes the given audio file using ElevenLabs Scribe model.
        Logs the network latency.
        """
        start_time = time.perf_counter()
        
        with open(audio_file_path, "rb") as f:
            # Scribe automatically detects language, so language_code is not strictly required here
            response = self.client.speech_to_text.convert(
                file=f,
                model_id="scribe_v1",  # Use scribe_v1 for ElevenLabs STT
                tag_audio_events=False
            )
            
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.log("stt", duration_ms)
        
        return response.text

class GroqSTT:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
            
        import httpx
        
        custom_http_client = httpx.Client(
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100, keepalive_expiry=60.0)
        )
        self.client = groq.Groq(
            api_key=self.api_key,
            http_client=custom_http_client
        )
        
    def transcribe(self, audio_file_path: str, language_code: str = None) -> str:
        """
        Transcribes the given audio file using Groq's whisper-large-v3 model.
        Logs the network latency.
        """
        start_time = time.perf_counter()
        
        with open(audio_file_path, "rb") as f:
            kwargs = {
                "file": (os.path.basename(audio_file_path), f.read()),
                "model": "whisper-large-v3",
                "prompt": "The audio may contain Hindi or English.",
            }
            if language_code:
                kwargs["language"] = language_code
            else:
                kwargs["language"] = "hi"  # Default to Hindi to prevent Urdu script
                
            transcription = self.client.audio.transcriptions.create(**kwargs)
            
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.log("stt", duration_ms)
        
        return transcription.text
