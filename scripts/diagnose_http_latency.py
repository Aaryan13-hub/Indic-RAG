import os
import sys
import json
import subprocess
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    print("GROQ_API_KEY not found.")
    sys.exit(1)

# A minimal payload to simulate the request
payload = {
    "model": "openai/gpt-oss-20b",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ],
    "reasoning_effort": "low",
    "max_completion_tokens": 150,
    "stream": False
}

with open("temp_payload.json", "w") as f:
    json.dump(payload, f)

# curl format string for detailed latency profiling
curl_format = """
{
  "time_namelookup": %{time_namelookup},
  "time_connect": %{time_connect},
  "time_appconnect": %{time_appconnect},
  "time_pretransfer": %{time_pretransfer},
  "time_starttransfer": %{time_starttransfer},
  "time_total": %{time_total}
}
"""

with open("curl_format.txt", "w") as f:
    f.write(curl_format)

cmd = [
    "curl", "-s", "-w", "@curl_format.txt", "-o", "curl_response.json",
    "-X", "POST", "https://api.groq.com/openai/v1/chat/completions",
    "-H", f"Authorization: Bearer {api_key}",
    "-H", "Content-Type: application/json",
    "-d", "@temp_payload.json"
]

print("Running cURL to measure HTTP layer latencies...")
result = subprocess.run(cmd, capture_output=True, text=True)

try:
    timings = json.loads(result.stdout)
    with open("curl_response.json", "r", encoding="utf-8") as f:
        response_data = json.load(f)
        
    usage = response_data.get("usage", {})
    x_groq = response_data.get("x_groq", {})
    
    # Calculate detailed breakdown
    dns_time = timings["time_namelookup"]
    tcp_time = timings["time_connect"] - timings["time_namelookup"]
    tls_time = timings["time_appconnect"] - timings["time_connect"]
    req_sent = timings["time_pretransfer"] - timings["time_appconnect"]
    ttfb = timings["time_starttransfer"] - timings["time_pretransfer"]
    transfer = timings["time_total"] - timings["time_starttransfer"]
    
    print("\n--- HTTP LAYER TIMINGS (cURL) ---")
    print(f"DNS Lookup:      {dns_time * 1000:6.2f} ms")
    print(f"TCP Connect:     {tcp_time * 1000:6.2f} ms")
    print(f"TLS Handshake:   {tls_time * 1000:6.2f} ms")
    print(f"Pre-transfer:    {req_sent * 1000:6.2f} ms")
    print(f"TTFB (Wait):     {ttfb * 1000:6.2f} ms  <-- This includes Server Time + Queue Time")
    print(f"Transfer Time:   {transfer * 1000:6.2f} ms")
    print(f"TOTAL cURL Time: {timings['time_total'] * 1000:6.2f} ms")
    
    print("\n--- GROQ SERVER TIMINGS (from response) ---")
    q_time = usage.get('queue_time', 0)
    p_time = usage.get('prompt_time', 0)
    c_time = usage.get('completion_time', 0)
    t_time = usage.get('total_time', 0)
    
    print(f"Queue Time:      {q_time * 1000:6.2f} ms")
    print(f"Prompt Time:     {p_time * 1000:6.2f} ms")
    print(f"Completion Time: {c_time * 1000:6.2f} ms")
    print(f"Server Total:    {t_time * 1000:6.2f} ms (Reported)")
    
    print("\n--- LATENCY ACCOUNTING OVERLAP ANALYSIS ---")
    calculated_server_sum = (p_time + c_time) * 1000
    print(f"Prompt + Completion = {calculated_server_sum:.2f} ms")
    print(f"Is Queue included in Server Total? {'Yes' if t_time >= (p_time + c_time + q_time) else 'No'}")
    
    # Calculate true network overhead
    # TTFB includes the time the request spent in queue and processing.
    true_network_latency = (timings['time_total'] - q_time - t_time) * 1000
    print(f"True Network/Protocol Overhead: {true_network_latency:.2f} ms")
    
finally:
    if os.path.exists("temp_payload.json"): os.remove("temp_payload.json")
    if os.path.exists("curl_format.txt"): os.remove("curl_format.txt")
    if os.path.exists("curl_response.json"): os.remove("curl_response.json")
