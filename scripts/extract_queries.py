import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import os
from qdrant_client import QdrantClient

# Set up client pointing to the local DB
client = QdrantClient(path="./qdrant_hindi_benchmark")
collection_name = "hindi_rag_production"

# Scroll for 5 points
try:
    records, next_page_offset = client.scroll(
        collection_name=collection_name,
        limit=5,
        with_payload=True
    )
    
    print("\n--- EXTRACTED CHUNKS ---")
    for r in records:
        chunk = r.payload.get("chunk", "NO CHUNK")
        print(f"ID: {r.id}\nText: {chunk}\n")
        
except Exception as e:
    print(f"Error reading Qdrant: {e}")
