import os
from dotenv import load_dotenv
load_dotenv()

from src.core.vectorstore import QdrantVectorStore
from src.core.latency import logger

def test_db():
    print("Initializing VectorStore...")
    store = QdrantVectorStore(collection_name="test_collection")
    
    # Clean up before test
    store.delete()
    
    chunks = [
        "The Manhattan Project was a research and development undertaking during World War II that produced the first nuclear weapons.",
        "Albert Einstein was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time.",
        "New York City comprises 5 boroughs sitting where the Hudson River meets the Atlantic Ocean."
    ]
    metadata = [{"source": "wiki_manhattan"}, {"source": "wiki_einstein"}, {"source": "wiki_nyc"}]
    
    print("\nAdding chunks...")
    store.add(chunks, metadata)
    
    print("\nQuerying 'Who was Albert Einstein?'...")
    results = store.query("Who was Albert Einstein?", k=1)
    
    print("\nResults:")
    for r in results:
        print(f"- {r['chunk']} (Metadata: {r['metadata']})")
        
    print("\nLatency Report:")
    logger.report()

if __name__ == "__main__":
    test_db()
