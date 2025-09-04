#!/usr/bin/env python3
"""
Retriever: query Chroma and optionally rerank with a Cross-Encoder.
Usage (example):
  python src/pipeline/retriever.py --query "What is reinforcement learning?" --k 8 --rerank 5
"""
import argparse
from typing import List, Dict

import chromadb
from sentence_transformers import CrossEncoder

class Retriever:
    def __init__(self, persist_dir: str = "data/chroma", collection_name: str = "arxiv-chunks",
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_collection(name=collection_name)
        # load cross-encoder for reranking (fast & small)
        self.reranker = CrossEncoder(reranker_model) if reranker_model else None
    
    def query(self, query: str, k: int = 10, rerank_top_n: int = 0) -> List[Dict]:
    # raw dense retrieval
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas"]  # ✅ removed "ids"
        )
        ids = results["ids"][0]   # still works, ids are always returned
        docs = results["documents"][0]
        metas = results["metadatas"][0]

        candidates = [{"id": ids[i], "text": docs[i], "metadata": metas[i]} for i in range(len(docs))]

        # optional rerank top-N using cross-encoder (higher score -> more relevant)
        if rerank_top_n and self.reranker:
            top_slice = candidates[:rerank_top_n]
            inputs = [[query, c["text"]] for c in top_slice]
            scores = self.reranker.predict(inputs)  # shape (rerank_top_n,)
            for i, s in enumerate(scores):
                top_slice[i]["score"] = float(s)
            top_sorted = sorted(top_slice, key=lambda x: x["score"], reverse=True)
            return top_sorted + candidates[rerank_top_n:]
        return candidates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--rerank", type=int, default=5)
    args = parser.parse_args()

    r = Retriever()
    res = r.query(args.query, k=args.k, rerank_top_n=args.rerank)
    for i, c in enumerate(res[:args.k]):
        print(f"\n[{i+1}] id: {c['id']}")
        print(c['text'][:500].replace("\n", " "))
        if "score" in c:
            print(f"score: {c['score']}")
