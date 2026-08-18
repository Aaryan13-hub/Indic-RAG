import os
import chromadb
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any
import time
from .interfaces import VectorStore
from .embeddings import EmbeddingModel
from .latency import logger

class ChromaVectorStore(VectorStore):
    def __init__(self, collection_name: str = "rag_collection", persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_model = EmbeddingModel()
        
        # We get or create the collection
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
    def add(self, chunks: List[str], metadata: List[Dict[str, Any]] = None):
        if not chunks:
            return
            
        # 1. Generate embeddings (Latency tracked inside embed())
        embeddings = self.embedding_model.embed(chunks)
        
        # 2. Add to Chroma
        # Generate unique IDs based on timestamp
        ids = [f"chunk_{int(time.time() * 1000)}_{i}" for i in range(len(chunks))]
        
        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadata if metadata else [{} for _ in chunks],
            ids=ids
        )
        print(f"Indexed {len(chunks)} chunks into ChromaDB.")

    def query(self, query_text: str, k: int = 5, with_scores: bool = False) -> List[Dict[str, Any]]:
        """Query the vector store.

        When with_scores=True, each result dict includes a 'score' key
        (cosine similarity in [0, 1]). Chroma returns L2 distances;
        for unit-normalized vectors the conversion is:
            cosine_sim = 1 - (L2_dist ** 2) / 2
        which is exact for all-MiniLM-L6-v2 (embeddings are unit-normalised).
        """
        # Track total retrieval latency (embedding query + searching DB)
        start_time = time.perf_counter()

        # 1. Embed the query
        query_embedding = self.embedding_model.embed([query_text])[0]

        # 2. Search Chroma — always include distances so they're available if needed
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.log("retrieval", duration_ms)

        # Format the results
        formatted_results = []
        if results["documents"] and len(results["documents"]) > 0:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results["metadatas"] else [{} for _ in docs]
            distances = results["distances"][0] if results.get("distances") else [None] * len(docs)
            for doc, meta, dist in zip(docs, metas, distances):
                entry: Dict[str, Any] = {"chunk": doc, "metadata": meta}
                if with_scores:
                    # Convert L2 distance → cosine similarity (unit-normalised vectors)
                    # cosine_sim = 1 - dist^2 / 2, clamped to [0, 1]
                    if dist is not None:
                        entry["score"] = float(max(0.0, min(1.0, 1.0 - (dist ** 2) / 2.0)))
                    else:
                        entry["score"] = 0.0
                formatted_results.append(entry)

        return formatted_results

    def delete(self):
        """Clear the database collection."""
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.create_collection(name)

class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        collection_name: str = "rag_collection",
        persist_directory: str = "./qdrant_db",
        vector_dim: int = 384,
        query_prefix: str = "",
        passage_prefix: str = "",
    ):
        self.client = QdrantClient(path=persist_directory)
        self.embedding_model = EmbeddingModel()
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        
        # Check if collection exists, if not create it
        # all-MiniLM-L6-v2 outputs 384-dimensional vectors
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
