import os
import time
from typing import Dict, Any
from .interfaces import LLMBackend
from .latency import logger
import groq

class GroqLLMBackend(LLMBackend):
    def __init__(self, model_name: str = "openai/gpt-oss-120b", reasoning_effort: str = None, max_completion_tokens: int = None):
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = max_completion_tokens
        import httpx
        
        custom_http_client = httpx.Client(
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100, keepalive_expiry=60.0)
        )
        self.client = groq.Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
            http_client=custom_http_client
        )
        
    def generate(self, prompt: str, system_prompt: str = None) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        kwargs = {
            "messages": messages,
            "model": self.model_name,
            "stream": True
        }
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = self.max_completion_tokens

        # We stream the response to accurately measure Time-To-First-Token (TTFT)
        response = self.client.chat.completions.create(**kwargs)
        
        first_token_time = None
        full_text = ""
        
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                full_text += chunk.choices[0].delta.content
                
        end_time = time.perf_counter()
        
        ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
        total_time_ms = (end_time - start_time) * 1000
        
        logger.log("generation", total_time_ms)
        
        return {
            "text": full_text,
            "ttft_ms": ttft_ms,
            "total_time_ms": total_time_ms
        }

class OllamaLLMBackend(LLMBackend):
    def __init__(self, model_name: str = "llama3.2:3b"):
        self.model_name = model_name
        # Lazy import — ollama is only available in local dev, not on HF Spaces
        import ollama as _ollama
        # Assumes local ollama is running on default port
        self.client = _ollama.Client()
        
    def generate(self, prompt: str, system_prompt: str = None) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat(
            model=self.model_name,
            messages=messages,
            stream=True
        )
        
        first_token_time = None
        full_text = ""
        
        for chunk in response:
            if chunk.get('message', {}).get('content'):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                full_text += chunk['message']['content']
                
        end_time = time.perf_counter()
        
        ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
        total_time_ms = (end_time - start_time) * 1000
        
        logger.log("generation", total_time_ms)
        
        return {
            "text": full_text,
            "ttft_ms": ttft_ms,
            "total_time_ms": total_time_ms
        }
