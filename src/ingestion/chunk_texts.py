
import argparse
import csv
import json
from pathlib import Path
from typing import Iterator

import tiktoken
from tqdm import tqdm

PROC_INDEX = Path("data/processed/index.csv")
CHUNK_DIR = Path("data/chunks")
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

def iter_rows():
    with open(PROC_INDEX, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            yield row

def chunk_by_tokens(text: str, chunk_size: int, overlap: int, enc) -> Iterator[tuple[int, int, str]]:
    tokens = enc.encode(text)
    step = chunk_size - overlap
    i = 0
    while i < len(tokens):
        window = tokens[i:i + chunk_size]
        yield i, i + len(window), enc.decode(window)
        i += step

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=150)
    args = parser.parse_args()

    enc = tiktoken.get_encoding("cl100k_base")
    total_chunks = 0

    for row in tqdm(list(iter_rows()), desc="Chunking"):
        text_path = Path(row["text_path"])
        if not text_path.exists():
            continue
        text = text_path.read_text(encoding="utf-8")
        out_path = CHUNK_DIR / f"{row['arxiv_id']}.jsonl"

        with out_path.open("w", encoding="utf-8") as f:
            cid = 0
            for start, end, chunk_text in chunk_by_tokens(text, args.chunk_size, args.overlap, enc):
                rec = {
                    "chunk_id": cid,
                    "arxiv_id": row["arxiv_id"],
                    "title": row["title"],
                    "start_token": start,
                    "end_token": end,
                    "text": chunk_text
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cid += 1
                total_chunks += 1

    print(f"[OK] Wrote {total_chunks} chunks to {CHUNK_DIR}/")

if __name__ == "__main__":
    main()
