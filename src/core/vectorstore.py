import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any
import time
from .interfaces import VectorStore
from .embeddings import EmbeddingModel
from .latency import logger

class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        collection_name: str = "rag_collection",
        persist_directory: str = "./qdrant_db",
        vector_dim: int = 384,
        query_prefix: str = "",
        passage_prefix: str = "",
        qdrant_url: str = None,
        qdrant_api_key: str = None,
    ):
        # ---------------------------------------------------------------------------
        # Connection toggle: Use Qdrant Cloud if QDRANT_URL env var is set,
        # otherwise fall back to the local SQLite DB on disk.
        # ---------------------------------------------------------------------------
        cloud_url = qdrant_url or os.environ.get("QDRANT_URL")
        cloud_key = qdrant_api_key or os.environ.get("QDRANT_API")

        if cloud_url:
            self.client = QdrantClient(url=cloud_url, api_key=cloud_key)
            print(f"[Qdrant] Connected to CLOUD: {cloud_url}")
        else:
            self.client = QdrantClient(path=persist_directory)
            print(f"[Qdrant] Connected to LOCAL: {persist_directory}")

        self.embedding_model = EmbeddingModel()
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

        # Check if collection exists; create it if not
        try:
            self.client.get_collection(collection_name=self.collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
            )

            
    def add(self, chunks: List[str], metadata: List[Dict[str, Any]] = None):
        if not chunks:
            return
            
        embeddings = self.embedding_model.embed(chunks, prefix=self.passage_prefix)
        
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Create a unique integer or UUID ID
            # Qdrant accepts UUID or positive integers. We'll use a positive integer derived from time and index
            point_id = int(time.time() * 1000) + i
            
            payload = metadata[i] if metadata and i < len(metadata) else {}
            payload["chunk"] = chunk  # Store the text chunk in the payload
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
            )
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Indexed {len(chunks)} chunks into Qdrant.")

    def query(self, query_text: str, k: int = 5, with_scores: bool = False) -> List[Dict[str, Any]]:
        """Query the vector store.

        When with_scores=True, each result dict includes a 'score' key
        (cosine similarity in [0, 1]). Qdrant returns cosine similarity
        directly via hit.score when the collection uses Distance.COSINE.
        """
        start_time = time.perf_counter()

        query_embedding = self.embedding_model.embed([query_text], prefix=self.query_prefix)[0]

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=k,
        )

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.log("retrieval", duration_ms)

        formatted_results = []
        for hit in search_result.points:
            entry: Dict[str, Any] = {
                "chunk": hit.payload.get("chunk", ""),
                "metadata": hit.payload,
            }
            if with_scores:
                # Qdrant returns cosine similarity directly (Distance.COSINE collection)
                entry["score"] = float(max(0.0, min(1.0, hit.score)))
            formatted_results.append(entry)

        return formatted_results

    def delete(self):
        """Clear the database collection."""
        self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
        )
