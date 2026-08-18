import time
import os
from dotenv import load_dotenv
import groq

load_dotenv()
client = groq.Groq()

def benchmark_request(req_num):
    start = time.perf_counter()
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": "Say hi."}],
        reasoning_effort="low",
        max_completion_tokens=50,
        stream=True
    )
    
    ttft = None
    final_chunk = None
    for chunk in response:
        if ttft is None:
            ttft = time.perf_counter()
        if hasattr(chunk, 'x_groq') and chunk.x_groq and chunk.x_groq.usage:
            final_chunk = chunk
            
    end = time.perf_counter()
    
    ttft_ms = (ttft - start) * 1000
    usage = final_chunk.x_groq.usage
    queue_ms = usage.queue_time * 1000
    prompt_ms = usage.prompt_time * 1000
    
    network_ttft_overhead = ttft_ms - queue_ms - prompt_ms
    
    print(f"Req {req_num} | TTFT: {ttft_ms:6.1f}ms | Queue: {queue_ms:6.1f}ms | Prompt: {prompt_ms:6.1f}ms | Network Overhead (TTFT): {network_ttft_overhead:6.1f}ms")

print("Running 5 sequential requests with 1s pacing using the SAME Groq client to test connection reuse...")
for i in range(1, 6):
    benchmark_request(i)
    time.sleep(1)
