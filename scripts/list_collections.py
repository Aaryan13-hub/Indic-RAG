import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from qdrant_client import QdrantClient

client = QdrantClient(path="./qdrant_hindi_benchmark")
collections = client.get_collections()
print("Collections found in local DB:")
for c in collections.collections:
    info = client.get_collection(c.name)
    print(f"  - {c.name}  ({info.points_count} points)")
