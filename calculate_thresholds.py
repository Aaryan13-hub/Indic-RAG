import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

def load_data():
    with open("benchmark_artifacts/queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open("benchmark_artifacts/eval_map.json", "r", encoding="utf-8") as f:
        eval_map = json.load(f)
    return queries, eval_map

def main():
    queries, eval_map = load_data()
    
    pos_pairs = [] # (query_text, passage_text)
    neg_pairs = [] # (query_text, passage_text)
    
    # Extract positive pairs
    for qid, qtext in queries.items():
        if qid in eval_map:
            for entry in eval_map[qid]:
                if entry["is_selected"] == 1:
                    pos_pairs.append((qtext, entry["passage_text"]))
                    
    # Extract negative pairs by pairing queries with random passages from other queries
    import random
    random.seed(42)
    all_passages = []
    for entries in eval_map.values():
        for entry in entries:
            all_passages.append(entry["passage_text"])
            
    for qtext, _ in pos_pairs:
        # Pick 3 random passages
        for _ in range(3):
            neg_pairs.append((qtext, random.choice(all_passages)))
            
    print(f"Total positive pairs: {len(pos_pairs)}")
    print(f"Total negative pairs: {len(neg_pairs)}")
    
    # We don't need all if it's too large, but 1000 is fine
    pos_pairs = pos_pairs[:500]
    neg_pairs = neg_pairs[:1500]
    
    print("Loading model...")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("intfloat/multilingual-e5-small", device=device)
    
    def compute_sims(pairs):
        q_texts = ["query: " + p[0] for p in pairs]
        p_texts = ["passage: " + p[1] for p in pairs]
        
        q_emb = model.encode(q_texts, convert_to_tensor=True, show_progress_bar=True, batch_size=32)
        p_emb = model.encode(p_texts, convert_to_tensor=True, show_progress_bar=True, batch_size=32)
        
        # Diagonal elements of cos_sim are the pair scores
        sims = torch.nn.functional.cosine_similarity(q_emb, p_emb)
        return sims.cpu().numpy()

    print("Computing positive similarities...")
    pos_sims = compute_sims(pos_pairs)
    
    print("Computing negative similarities...")
    neg_sims = compute_sims(neg_pairs)
    
    print("\n--- DISTRIBUTION RESULTS ---")
    print("Relevant Pairs (Positive Class):")
    print(f"  Min: {np.min(pos_sims):.4f}")
    print(f"  P10: {np.percentile(pos_sims, 10):.4f}")
    print(f"  P50: {np.percentile(pos_sims, 50):.4f}")
    print(f"  P90: {np.percentile(pos_sims, 90):.4f}")
    print(f"  Max: {np.max(pos_sims):.4f}")

    print("\nIrrelevant Pairs (Negative Class):")
    print(f"  Min: {np.min(neg_sims):.4f}")
    print(f"  P10: {np.percentile(neg_sims, 10):.4f}")
    print(f"  P50: {np.percentile(neg_sims, 50):.4f}")
    print(f"  P90: {np.percentile(neg_sims, 90):.4f}")
    print(f"  P95: {np.percentile(neg_sims, 95):.4f}")
    print(f"  P99: {np.percentile(neg_sims, 99):.4f}")
    print(f"  Max: {np.max(neg_sims):.4f}")
    
    # Recommendation
    # We want threshold above P95 of negative class, but below P10 of positive class if possible
    threshold = np.percentile(neg_sims, 99)
    print(f"\nRECOMMENDATION: Set threshold at P99 of negative class to eliminate almost all off-topic matches.")
    print(f"Suggested Threshold: {threshold:.3f}")
    
    # How much positive data would we lose at this threshold?
    pos_retained = np.mean(pos_sims >= threshold) * 100
    print(f"At threshold {threshold:.3f}, {pos_retained:.1f}% of RELEVANT chunks would still pass.")

if __name__ == "__main__":
    main()
