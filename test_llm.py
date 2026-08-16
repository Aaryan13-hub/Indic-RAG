import os
from dotenv import load_dotenv
load_dotenv()

from src.core.llm import GroqLLMBackend, OllamaLLMBackend
from src.core.latency import logger

def test_llms():
    print("Testing Groq (Cloud)...")
    try:
        groq_llm = GroqLLMBackend()
        resp = groq_llm.generate("Explain the theory of relativity in one simple sentence.", system_prompt="You are a helpful physics tutor.")
        print(f"Groq Response: {resp['text']}")
        print(f"Groq TTFT: {resp['ttft_ms']:.1f}ms | Total: {resp['total_time_ms']:.1f}ms\n")
    except Exception as e:
        print(f"Groq failed: {e}\n")

    print("Testing Ollama (Local)...")
    try:
        ollama_llm = OllamaLLMBackend()
        resp = ollama_llm.generate("Explain the theory of relativity in one simple sentence.", system_prompt="You are a helpful physics tutor.")
        print(f"Ollama Response: {resp['text']}")
        print(f"Ollama TTFT: {resp['ttft_ms']:.1f}ms | Total: {resp['total_time_ms']:.1f}ms\n")
    except Exception as e:
        print(f"Ollama failed: {e}\n(Make sure Ollama is running locally and the model is pulled)")

    print("Latency Report:")
    logger.report()

if __name__ == "__main__":
    test_llms()
