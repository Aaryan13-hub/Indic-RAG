from src.core.chunking import RecursiveChunker

def test_chunking():
    chunker = RecursiveChunker(max_chars=200, overlap_chars=50)
    
    long_text = """The Manhattan Project was a research and development undertaking during World War II that produced the first nuclear weapons. It was led by the United States with the support of the United Kingdom and Canada. From 1942 to 1946, the project was under the direction of Major General Leslie Groves of the U.S. Army Corps of Engineers. Nuclear physicist J. Robert Oppenheimer was the director of the Los Alamos Laboratory that designed the actual bombs.

The first nuclear device ever detonated was an implosion-type bomb at the Trinity test, conducted at New Mexico's Alamogordo Bombing and Gunnery Range on 16 July 1945. Little Boy and Fat Man bombs were used a month later in the atomic bombings of Hiroshima and Nagasaki, respectively, with Japan capitulating shortly thereafter.

This is a very long single sentence that deliberately exceeds the two hundred character limit without any punctuation breaks in order to trigger the fixed size chunking fallback mechanism that we implemented so we can be absolutely sure that it works perfectly even when a paragraph has no sentence boundaries at all and just keeps going on and on and on and on."""

    chunks = chunker.chunk(long_text)
    
    print(f"Original text length: {len(long_text)} chars")
    print(f"Generated {len(chunks)} chunks:\n")
    
    for i, c in enumerate(chunks):
        print(f"--- Chunk {i+1} (Length: {len(c)}) ---")
        print(c)
        print()

if __name__ == "__main__":
    test_chunking()
