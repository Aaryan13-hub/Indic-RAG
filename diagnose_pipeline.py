import os
import sys
from dotenv import load_dotenv

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

from src.core.embeddings import EmbeddingModel
from src.core.vectorstore import QdrantVectorStore
from src.core.llm import GroqLLMBackend
from src.core.orchestrator import VoiceRAGOrchestrator, _SYSTEM_PROMPT
from src.core.guardrails import check_off_topic, check_groundedness

def run_diagnostics():
    print("Loading models...")
    embed_model = EmbeddingModel("intfloat/multilingual-e5-small")
    
    db = QdrantVectorStore(
        collection_name="hindi_rag_production",
        persist_directory="./qdrant_temp",
        vector_dim=384,
        query_prefix="query: ",
        passage_prefix="passage: ",
    )
    db.embedding_model = embed_model
    
    llm = GroqLLMBackend()
    
    # Defaults in orchestrator:
    # off_topic_threshold = 0.75
    # groundedness_threshold = 0.75
    off_topic_threshold = 0.75
    groundedness_threshold = 0.75
    max_retrieval_results = 2 # based on orchestrator.py defaults? Wait, api.py sets it to default. Let me check api.py.
    # api.py uses default orchestrator:
    # orchestrator = VoiceRAGOrchestrator(
    #     ..., off_topic_threshold=0.75, groundedness_threshold=0.75
    # )
    # Let me use max_retrieval_results = 2 (default in orchestrator)
    max_retrieval_results = 2
    
    queries = [
        "क्या उल्कापिंडों में प्लूटोनियम होता है?"
    ]
    
    for query in queries:
        print("="*80)
        print(f"QUERY: {query}")
        print("="*80)
        
        # 1. Retrieval
        print("\n--- 1. RETRIEVAL ---")
        results = db.query(query, k=5, with_scores=True)
        for i, r in enumerate(results):
            print(f"Rank {i+1}: Score: {r['score']:.4f} | Text: {r['chunk'][:150]}...")
            
        # 2. Off-topic check
        print("\n--- 2. OFF-TOPIC CHECK ---")
        results_for_pipeline = results[:max_retrieval_results]
        is_on_topic, top_score = check_off_topic(results_for_pipeline, threshold=off_topic_threshold)
        print(f"Top Score: {top_score:.4f} | Threshold: {off_topic_threshold}")
        print(f"Result: {'PASS' if is_on_topic else 'FAIL'}")
        
        if not is_on_topic:
            print(f"Final Reason: BLOCKED - off-topic (score {top_score:.4f} < {off_topic_threshold})")
            continue
            
        # 3. Context & LLM
        print("\n--- 3. LLM GENERATION ---")
        context_texts = [r["chunk"] for r in results_for_pipeline]
        context_str = "\n\n".join(context_texts)
        prompt = (
            f"<context>\n{context_str}\n</context>\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )
        print("EXACT PROMPT:")
        print(prompt)
        
        llm_response = llm.generate(prompt=prompt, system_prompt=_SYSTEM_PROMPT)
        answer_text = llm_response["text"]
        print("\nEXACT LLM RESPONSE:")
        print(answer_text)
        
        # 4. Groundedness
        print("\n--- 4. GROUNDEDNESS CHECK ---")
        is_grounded, ground_score = check_groundedness(
            answer=answer_text,
            context_chunks=context_texts,
            embedding_model=embed_model,
            threshold=groundedness_threshold,
        )
        print(f"Groundedness Score: {ground_score:.4f} | Threshold: {groundedness_threshold}")
        print(f"Result: {'PASS' if is_grounded else 'FAIL'}")
        
        if not is_grounded:
            print(f"Final Reason: BLOCKED - answer not grounded (score {ground_score:.4f} < {groundedness_threshold})")
        else:
            print("Final Reason: ACCEPTED - pipeline succeeded")

if __name__ == "__main__":
    run_diagnostics()
