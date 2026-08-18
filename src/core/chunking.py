from typing import List
from .interfaces import Chunker
import re

class RecursiveChunker(Chunker):
    def __init__(self, max_chars: int = 1000, overlap_chars: int = 200):
        # 1000 chars is roughly 200-250 tokens
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk(self, text: str) -> List[str]:
        if not text:
            return []
            
        chunks = []
        
        # 1. Split by paragraphs
        paragraphs = text.split("\n\n")
        
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
                
            if len(p) <= self.max_chars:
                chunks.append(p)
            else:
                # 2. Split by sentences if paragraph is too long
                # Simple regex for sentence splitting (handles '.', '!', '?')
                sentences = re.split(r'(?<=[.!?]) +', p)
                
                current_chunk = ""
                for s in sentences:
                    s = s.strip()
                    if not s:
                        continue
                        
                    # If adding the sentence exceeds max length, push current chunk
                    if len(current_chunk) + len(s) + 1 > self.max_chars and current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                        
                    # If a single sentence is STILL longer than max_chars, we do fixed-size fallback
                    if len(s) > self.max_chars:
                        fixed_chunks = self._fixed_size_split(s)
                        chunks.extend(fixed_chunks)
                    else:
                        if current_chunk:
                            current_chunk += " " + s
                        else:
                            current_chunk = s
                            
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
        return chunks
        
    def _fixed_size_split(self, text: str) -> List[str]:
        """Fallback: split string by fixed char length with overlap."""
        chunks = []
        step = self.max_chars - self.overlap_chars
        if step <= 0:
            step = self.max_chars # Safety fallback
            
        for i in range(0, len(text), step):
            chunks.append(text[i:i + self.max_chars])
        return chunks


class SentenceChunker(Chunker):
    """Splits text exclusively by sentences, ignoring paragraph boundaries.
    
    Uses both English (.!?) and Devanagari (।) punctuation.
    """
    def __init__(self, max_chars: int = 1000, overlap_chars: int = 200):
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk(self, text: str) -> List[str]:
        if not text:
            return []
            
        chunks = []
        # Split on standard punctuation OR Devanagari danda (।)
        # The lookbehind (?<=[.!?।]) ensures we keep the punctuation on the sentence.
        sentences = re.split(r'(?<=[.!?।]) +', text.strip())
        
        current_chunk = ""
        for s in sentences:
            s = s.strip()
            if not s:
                continue
                
            if len(current_chunk) + len(s) + 1 > self.max_chars and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                
            if len(s) > self.max_chars:
                fixed_chunks = self._fixed_size_split(s)
                chunks.extend(fixed_chunks)
            else:
                if current_chunk:
                    current_chunk += " " + s
                else:
                    current_chunk = s
                    
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def _fixed_size_split(self, text: str) -> List[str]:
        chunks = []
        step = self.max_chars - self.overlap_chars
        if step <= 0:
            step = self.max_chars
        for i in range(0, len(text), step):
            chunks.append(text[i:i + self.max_chars])
        return chunks
