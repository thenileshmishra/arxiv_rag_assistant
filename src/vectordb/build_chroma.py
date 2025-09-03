import argparse
import json
from pathlib import Path
from typing import List

import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm   

CHUNK_DIR = Path("data/chunks")

def load_chunks() -> List[dict]:
    all_chunks = []
    for f in CHUNK_DIR.glob("*.jsonl"):
        with open(f, encoding="utf-8") as fin:
            for line in fin:
                rec = json.loads(line)
                all_chunks.append(rec)
    return all_chunks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist-dir", type=str, default="data/chroma")
    parser.add_argument("--collection-name", type=str, default="arxiv-chunks")
    args = parser.parse_args()

    #Sentence-BERT embedding function
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L6-v2")

    #Create persistent ChromaDB client
    client = chromadb.PersistentClient(path=args.persist_dir)

    #Drop collection if already exists
    try:
        client.delete_collection(name=args.collection_name)
    except Exception:
        pass

    collection = client.create_collection(name=args.collection_name, embedding_function=embed_fn)

    # Load Chunks
    chunks = load_chunks()
    print(f"[INFO] Loaded {len(chunks)} chunks from {CHUNK_DIR}/")

    # Add in batches
    batch_size = 100
    for i in tqdm(range(0, len(chunks), batch_size), desc="Indexing Chunks"):
        batch = chunks[i:i + batch_size]
        ids = [f"{c['arxiv_id']}_chunk{c['chunk_id']}" for c in batch]
        texts = [c["text"] for c in batch]
        metadatas = [{"arxiv_id": c["arxiv_id"], "title": c["title"]} for c in batch]
        collection.add(ids=ids, documents=texts, metadatas=metadatas,)
    
    print(f"[OK] Indexed {len(chunks)} chunks into ChromaDB at {args.collection_name}/")


if __name__ == "__main__":
    main()
    