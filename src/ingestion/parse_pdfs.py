#!/usr/bin/env python3
"""
Extract readable text from PDFs and lightly clean it.

Usage:
  python src/ingestion/parse_pdfs.py
"""
import csv
import re
from pathlib import Path
from typing import Dict, List

import fitz  # PyMuPDF
from tqdm import tqdm

RAW_META = Path("data/raw/metadata.csv")
RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)
PROC_INDEX = PROC_DIR / "index.csv"

def read_metadata() -> List[Dict]:
    rows = []
    with open(RAW_META, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows

def extract_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text("text")
        pages.append(text)
    return "\n".join(pages)

def clean_text(text: str, trim_references: bool = True) -> str:
    # 1) Merge hyphenated words across line-breaks: e.g., "gener-\native" -> "genernative"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # 2) Replace newlines inside paragraphs with spaces, but keep paragraph breaks
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # 3) Remove lines that are too short or look like headers/footers noise
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if len(ln) > 2]
    text = "\n".join(lines)
    # 4) Optionally trim everything after a "References" section (heuristic)
    if trim_references:
        m = re.search(r"\nreferences\b.*", text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            text = text[:m.start()]
    return text.strip()

def main():
    rows = read_metadata()
    out_rows = []
    for r in tqdm(rows, desc="Parsing PDFs"):
        pdf_path = Path(r["pdf_path"])
        if not pdf_path.exists():
            continue
        try:
            raw = extract_text(pdf_path)
            cleaned = clean_text(raw, trim_references=True)
            out_file = PROC_DIR / f"{r['arxiv_id']}.txt"
            out_file.write_text(cleaned, encoding="utf-8")
            out_rows.append({
                "arxiv_id": r["arxiv_id"],
                "title": r["title"],
                "pdf_path": r["pdf_path"],
                "text_path": str(out_file),
                "num_chars": len(cleaned)
            })
        except Exception as e:
            print(f"[WARN] Failed to parse {pdf_path.name}: {e}")

    # index
    with open(PROC_INDEX, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["arxiv_id", "title", "pdf_path", "text_path", "num_chars"])
        w.writeheader()
        w.writerows(out_rows)

    print(f"[OK] Parsed {len(out_rows)} PDFs. Index -> {PROC_INDEX}")

if __name__ == "__main__":
    main()
