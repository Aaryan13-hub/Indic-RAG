from sentence_transformers import SentenceTransformer
from typing import List
import time
import torch
from .latency import logger


class EmbeddingModel:
    """Wrapper around a SentenceTransformer model with latency logging.

    Benchmark result (Hindi MS-MARCO-XI, 1K sample, seed=42):
        all-MiniLM-L6-v2         Recall@5 =  2.71%  (English-first, fails on Hindi)
        intfloat/multilingual-e5-small  Recall@5 = 73.05%  ← selected

    E5 prefix convention (IMPORTANT):
        Passages at ingest time → embed with prefix="passage: "
        Queries  at query  time → embed with prefix="query: "
    The caller is responsible for passing the correct prefix; this class does
    not assume a role. This lets the same EmbeddingModel instance serve both
    ingest (vectorstore.add) and retrieval/groundedness (orchestrator).

    Swapping the model:
        Just pass a different model_name. If you switch to a model that does
        NOT require prefixes (e.g. BAAI/bge-m3), pass prefix="" at all call
        sites — no other changes required.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
    ) -> None:
        """Load the embedding model onto the best available device.

        Args:
            model_name: HuggingFace model ID or local path.
                        Default: intfloat/multilingual-e5-small
                        (selected by Hindi benchmark — Recall@5 73.05%).
        """
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        print(f"Embedding model ready on {device}.")

    def embed(self, texts: List[str], prefix: str = "") -> List[List[float]]:
        """Generate embeddings for a list of texts and log latency.

        Args:
            texts:  List of text strings to embed.
            prefix: Optional prefix prepended to every text before encoding.
                    For intfloat/multilingual-e5-small:
                      - Pass "passage: " when embedding corpus chunks.
                      - Pass "query: "   when embedding user queries.
                    Leave empty ("") for models that do not require prefixes.

        Returns:
            List of embedding vectors (list of floats per text).
        """
        start_time = time.perf_counter()

        if prefix:
            prefixed = [prefix + t for t in texts]
        else:
            prefixed = texts

        embeddings = self.model.encode(prefixed).tolist()

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.log("embedding", duration_ms)

        return embeddings
