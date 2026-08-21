"""
scripts/migrate_to_qdrant_cloud.py
===================================
One-time migration: copies all vectors from the local Qdrant SQLite DB
(qdrant_hindi_benchmark/) to your Qdrant Cloud cluster.

Usage:
    python -m scripts.migrate_to_qdrant_cloud

Requirements:
    QDRANT_URL and QDRANT_API must be set in your .env file.
"""

import os
import sys
import time
from pathlib import Path

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOCAL_QDRANT_PATH = str(PROJECT_ROOT / "qdrant_hindi_benchmark")
COLLECTION_NAME   = "hindi_rag_production"
VECTOR_DIM        = 384
BATCH_SIZE        = 100   # How many points to read/write per iteration

CLOUD_URL = os.environ.get("QDRANT_URL")
CLOUD_KEY = os.environ.get("QDRANT_API")

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
if not CLOUD_URL or not CLOUD_KEY:
    print("[ERROR] QDRANT_URL and QDRANT_API must be set in your .env file.")
    sys.exit(1)

print(f"[CONFIG] Local DB    : {LOCAL_QDRANT_PATH}")
print(f"[CONFIG] Cloud URL   : {CLOUD_URL}")
print(f"[CONFIG] Collection  : {COLLECTION_NAME}")
print()

# ---------------------------------------------------------------------------
# Connect to local DB
# ---------------------------------------------------------------------------
print("[1/5] Connecting to LOCAL Qdrant...")
local_client = QdrantClient(path=LOCAL_QDRANT_PATH)

try:
    local_info = local_client.get_collection(COLLECTION_NAME)
    total_points = local_info.points_count
    print(f"[1/5] Local collection has {total_points:,} points.")
except Exception as e:
    print(f"[ERROR] Could not read local collection: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Connect to cloud DB
# ---------------------------------------------------------------------------
print("\n[2/5] Connecting to CLOUD Qdrant...")
cloud_client = QdrantClient(url=CLOUD_URL, api_key=CLOUD_KEY)

# Create collection in cloud if it doesn't exist
try:
    cloud_client.get_collection(COLLECTION_NAME)
    print(f"[2/5] Collection '{COLLECTION_NAME}' already exists in cloud.")
    existing = cloud_client.get_collection(COLLECTION_NAME).points_count
    print(f"[2/5] Cloud already has {existing:,} points. Will upsert (no duplicates).")
except Exception:
    print(f"[2/5] Creating collection '{COLLECTION_NAME}' in cloud...")
    cloud_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    print("[2/5] Collection created.")

# ---------------------------------------------------------------------------
# Migrate in batches
# ---------------------------------------------------------------------------
print(f"\n[3/5] Migrating {total_points:,} points in batches of {BATCH_SIZE}...")
offset = None
migrated = 0
t_start = time.perf_counter()

while True:
    # Scroll a batch of points from local
    records, next_offset = local_client.scroll(
        collection_name=COLLECTION_NAME,
        limit=BATCH_SIZE,
        offset=offset,
        with_payload=True,
        with_vectors=True,
    )

    if not records:
        break

    # Build PointStruct list
    points = [
        PointStruct(
            id=r.id,
            vector=r.vector,
            payload=r.payload,
        )
        for r in records
    ]

    # Upsert into cloud
    cloud_client.upsert(collection_name=COLLECTION_NAME, points=points)
    migrated += len(records)

    elapsed = time.perf_counter() - t_start
    rate = migrated / elapsed if elapsed > 0 else 0
    print(f"  Migrated {migrated:,}/{total_points:,} points  ({rate:.0f} pts/sec)", end="\r")

    if next_offset is None:
        break
    offset = next_offset

print()

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
print(f"\n[4/5] Verifying migration...")
time.sleep(2)  # Give cloud a moment to index
cloud_count = cloud_client.get_collection(COLLECTION_NAME).points_count
print(f"  Local points  : {total_points:,}")
print(f"  Cloud points  : {cloud_count:,}")

if cloud_count >= total_points:
    print("\n[5/5] ✅ Migration SUCCESSFUL! All points are in Qdrant Cloud.")
else:
    diff = total_points - cloud_count
    print(f"\n[5/5] ⚠️  Migration INCOMPLETE: {diff} points missing. Re-run the script.")

local_client.close()
cloud_client.close()
