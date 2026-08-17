from sentence_transformers import SentenceTransformer
from typing import List
import time
from .latency import logger

class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Load the embedding model locally."""
        print(f"Loading embedding model: {model_name}...")
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings and log the latency."""
        start_time = time.perf_counter()
        
        embeddings = self.model.encode(texts).tolist()
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.log("embedding", duration_ms)
        
        return embeddings
