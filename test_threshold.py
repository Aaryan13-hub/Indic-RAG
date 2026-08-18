import os
import json
from src.core.embeddings import EmbeddingModel
from src.core.vectorstore import QdrantVectorStore
from src.core.guardrails import check_off_topic

def main():
    print("Loading model and DB...")
    embed_model = EmbeddingModel("intfloat/multilingual-e5-small")
    
    db = QdrantVectorStore(
        collection_name="hindi_rag_production",
        persist_directory="./qdrant_temp",
        vector_dim=384,
        query_prefix="query: ",
        passage_prefix="passage: ",
    )
    db.embedding_model = embed_model
    
    threshold = 0.826
    
    with open("benchmark_artifacts/queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
    
    # Grab first 5 known answerable queries
    known_queries = list(queries.values())[:5]
    
    test_queries = [
        "Who is Virat Kohli",
        "विराट कोहली कौन है",
        "what is capital of India",
        "अल्बर्ट आइंस्टीन कोण है"
    ] + known_queries
    
    output = []
    
    for q in test_queries:
        output.append("="*80)
        output.append(f"QUERY: {q}")
        
        results = db.query(q, k=5, with_scores=True)
        is_on_topic, top_score = check_off_topic(results, threshold=threshold)
        
        output.append(f"Top Score: {top_score:.4f} | Threshold: {threshold}")
        output.append(f"0.826 Guardrail Result: {'PASS (Relevant)' if is_on_topic else 'BLOCKED (Off-Topic)'}")
        output.append("\nTop 5 Chunks:")
        
        for i, r in enumerate(results):
            text_snippet = r['chunk'][:100].replace('\n', ' ')
            output.append(f"  {i+1}. Score: {r['score']:.4f} | {text_snippet}...")
            
        output.append("")
        
    with open("threshold_test_output.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output))
        
    print("Test complete. Results saved to threshold_test_output.txt")

if __name__ == "__main__":
    main()
