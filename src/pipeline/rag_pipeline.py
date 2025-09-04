"""
Simple RAG pipeline:
 - retrieve top-k docs from Chroma (and rerank)
 - concatenate until token budget reached
 - call LLM wrapper and return answer + used passages

Usage (example):
  python src/pipeline/rag_pipeline.py --query "What is reinforcement learning?" --model openai
"""
import argparse
import tiktoken

from src.pipeline.retriever import Retriever
from src.generator.llm_wrapper import LLM


SYSTEM_PROMPT = (
    "You are an expert assistant. Answer using ONLY the provided context passages. "
    "If the answer is not contained in the context, respond: 'Not found in provided documents.' "
    "Keep answers concise and cite passages by their id when relevant."
)

def build_final_prompt(contexts, question):
    joined = "\n\n---\n\n".join([f"[{c['id']}]\n{c['text']}" for c in contexts])
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{joined}\n\nQuestion: {question}\nAnswer:"
    return prompt

def select_contexts_by_token_budget(candidates, token_budget=3000, enc=None):
    enc = enc or tiktoken.get_encoding("cl100k_base")
    selected = []
    used = 0
    for c in candidates:
        tok_len = len(enc.encode(c["text"]))
        if used + tok_len > token_budget:
            # if no contexts selected and single chunk > budget, still include it (avoid empty)
            if not selected:
                selected.append(c)
            break
        selected.append(c)
        used += tok_len
    return selected

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--rerank", type=int, default=4)
    parser.add_argument("--model", choices=["openai", "hf", "gemini"], default="openai")
    parser.add_argument("--model-name", type=str, default="gpt-4o-mini")
    parser.add_argument("--token-budget", type=int, default=3000)
    args = parser.parse_args()

    # 1) retrieve + rerank
    retr = Retriever()
    candidates = retr.query(args.query, k=args.top_k, rerank_top_n=args.rerank)

    # 2) select contexts under token budget
    enc = tiktoken.get_encoding("cl100k_base")
    contexts = select_contexts_by_token_budget(candidates, token_budget=args.token_budget, enc=enc)

    # 3) build prompt and call LLM
    prompt = build_final_prompt(contexts, args.query)
    llm = LLM(mode=args.model, model_name=args.model_name, temperature=0.0, max_tokens=512)
    answer = llm.generate(prompt)

    # 4) print outputs (answer + which contexts used)
    print("\n=== Answer ===\n")
    print(answer)
    print("\n=== Source passages used ===\n")
    for i, c in enumerate(contexts):
        print(f"[{i+1}] id: {c['id']}  meta: {c.get('metadata')}")
        print(c['text'][:400].replace("\n", " "), "...\n" + "-"*60)

if __name__ == "__main__":
    main()
