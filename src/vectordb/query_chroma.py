#!/usr/bin/env python3
"""
Query Chroma vector DB to check retrieval.

Usage:
  python src/vectordb/query_chroma.py --query "What is reinforcement learning?"
"""
import argparse
import chromadb
from chromadb.utils import embedding_functions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist-dir", type=str, default="data/chroma")
    parser.add_argument("--collection-name", type=str, default="arxiv_chunks")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--k", type=int, default=3, help="Top-k results")
    args = parser.parse_args()

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=args.persist_dir)
    collection = client.get_collection(name=args.collection_name, embedding_function=embed_fn)

    results = collection.query(
        query_texts=[args.query],
        n_results=args.k,
    )

    print("\n=== Query ===")
    print(args.query)
    print("\n=== Top Results ===")
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"\n[{i+1}] Title: {meta['title']}")
        print(doc[:500], "...")
        print("—" * 40)

if __name__ == "__main__":
    main()
