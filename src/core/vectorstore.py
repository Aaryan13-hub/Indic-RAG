import os
import chromadb
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

    def query(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        # Track total retrieval latency (embedding query + searching DB)
        start_time = time.perf_counter()
        
        # 1. Embed the query
        query_embedding = self.embedding_model.embed([query_text])[0]
        
        # 2. Search Chroma
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.log("retrieval", duration_ms)
        
        # Format the results
        formatted_results = []
        if results['documents'] and len(results['documents']) > 0:
            docs = results['documents'][0]
            metas = results['metadatas'][0] if results['metadatas'] else [{} for _ in docs]
            for doc, meta in zip(docs, metas):
                formatted_results.append({
                    "chunk": doc,
                    "metadata": meta
                })
                
        return formatted_results

    def delete(self):
        """Clear the database collection."""
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.create_collection(name)
