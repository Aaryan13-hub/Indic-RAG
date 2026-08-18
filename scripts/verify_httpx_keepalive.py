import os
import sys
import time
import httpx
import groq
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    print("GROQ_API_KEY not found.")
    sys.exit(1)

def run_test(client_name, client_instance, num_requests=5, pacing_seconds=6.0):
    print(f"\n{'='*50}\nTesting {client_name} (Pacing: {pacing_seconds}s)\n{'='*50}")
    
    overheads = []
    
    for i in range(1, num_requests + 1):
        start = time.perf_counter()
        response = client_instance.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="low",
            max_completion_tokens=10,
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
        
        # Pure network overhead to reach TTFT
        network_ttft_overhead = ttft_ms - queue_ms - prompt_ms
        overheads.append(network_ttft_overhead)
        
        print(f"Req {i} | TTFT: {ttft_ms:6.1f}ms | Queue: {queue_ms:6.1f}ms | Network Overhead (TTFT): {network_ttft_overhead:6.1f}ms")
        
        if i < num_requests:
            time.sleep(pacing_seconds)
            
    avg_subsequent = sum(overheads[1:]) / (len(overheads) - 1) if len(overheads) > 1 else 0
    print(f"-> First Request Overhead: {overheads[0]:.1f} ms (DNS + TCP + TLS)")
    print(f"-> Subsequent Average Overhead: {avg_subsequent:.1f} ms")
    return overheads

# 1. Default Groq Client
default_client = groq.Groq(api_key=api_key)

# 2. Optimized Persistent Client
custom_http_client = httpx.Client(
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=100, keepalive_expiry=60.0)
)
optimized_client = groq.Groq(api_key=api_key, http_client=custom_http_client)

run_test("Default Groq Client", default_client, num_requests=5, pacing_seconds=6.0)
run_test("Optimized Persistent Groq Client", optimized_client, num_requests=5, pacing_seconds=6.0)

