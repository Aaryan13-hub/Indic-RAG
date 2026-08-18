import os
import groq
from dotenv import load_dotenv

load_dotenv()

client = groq.Groq()
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    reasoning_effort="low",
    max_completion_tokens=150,
    stream=False
)

print("--- STREAM=FALSE ---")
print("Usage:", response.usage)
if hasattr(response, "x_groq"):
    print("x_groq:", response.x_groq)

print("\n--- STREAM=TRUE ---")
response_stream = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "Tell me a joke"}],
    reasoning_effort="low",
    max_completion_tokens=150,
    stream=True
)

for chunk in response_stream:
    if hasattr(chunk, "x_groq") and chunk.x_groq:
        print("Final chunk x_groq:", chunk.x_groq)
