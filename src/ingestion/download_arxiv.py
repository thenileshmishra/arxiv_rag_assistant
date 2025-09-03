#!/usr/bin/env python3
"""
Download recent arXiv PDFs with metadata.

Usage:
  python src/ingestion/download_arxiv.py --query "cat:cs.LG OR cat:cs.AI" --max-results 100
"""
import argparse
import csv
import os
from pathlib import Path
from typing import List

import arxiv
from slugify import slugify
from tqdm import tqdm

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
META_CSV = RAW_DIR / "metadata.csv"

def sanitize_filename(title: str, arxiv_id: str) -> str:
    # Keep names readable and unique
    base = slugify(title)[:80]  # limit length
    return f"{base}-{arxiv_id}.pdf"

def download_papers(query: str, max_results: int, categories_hint: str = "") -> List[dict]:
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    client = arxiv.Client(page_size=50, delay_seconds=3, num_retries=3)
    records = []

    for result in tqdm(client.results(search), total=max_results, desc="Downloading PDFs"):
        arxiv_id = result.get_short_id()
        title = result.title
        authors = ", ".join(a.name for a in result.authors)
        categories = ", ".join(result.categories)
        filename = sanitize_filename(title, arxiv_id)
        pdf_path = RAW_DIR / filename

        if not pdf_path.exists():
            try:
                result.download_pdf(dirpath=str(RAW_DIR), filename=filename)
            except Exception as e:
                print(f"[WARN] Failed to download {arxiv_id}: {e}")
                continue

        records.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "categories": categories or categories_hint,
            "pdf_path": str(pdf_path),
            "primary_category": result.primary_category,
            "published": result.published.strftime("%Y-%m-%d"),
            "updated": result.updated.strftime("%Y-%m-%d"),
            "entry_id": result.entry_id,
        })

    return records

def write_metadata(rows: List[dict]):
    fieldnames = [
        "arxiv_id", "title", "authors", "categories", "primary_category",
        "published", "updated", "entry_id", "pdf_path"
    ]
    write_header = not META_CSV.exists()
    with open(META_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerows(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="cat:cs.LG OR cat:cs.AI",
                        help="arXiv query. Examples: 'cat:cs.AI', 'cat:cs.LG OR cat:cs.AI'")
    parser.add_argument("--max-results", type=int, default=100,
                        help="Max number of PDFs to download")
    args = parser.parse_args()

    rows = download_papers(args.query, args.max_results)
    write_metadata(rows)
    print(f"[OK] Downloaded {len(rows)} PDFs. Metadata -> {META_CSV}")

if __name__ == "__main__":
    main()
