from abc import ABC, abstractmethod
from typing import List, Dict, Any

class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: List[str], metadata: List[Dict[str, Any]] = None):
        """Add chunks to the vector store."""
        pass

    @abstractmethod
    def query(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """Query the vector store for top k most similar chunks.
        Should return a list of dicts with at least 'chunk' and 'metadata' keys.
        """
        pass

    @abstractmethod
    def delete(self):
        """Clear the vector store."""
        pass

class Chunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """Split text into chunks."""
        pass

class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = None) -> Dict[str, Any]:
        """Generate response from LLM. 
        Must return a dict containing at least:
        - 'text': The generated text
        - 'ttft_ms': Time to first token in milliseconds (if streaming/measurable)
        - 'total_time_ms': Total generation time in milliseconds
        """
        pass
