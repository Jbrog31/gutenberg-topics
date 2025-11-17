import re
from pathlib import Path

import requests

from .config import BOOKS, RAW_DIR

URL_PATTERNS = [
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt"
]

def strip_gutenberg_header_footer(text):
    start_pattern = r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*"
    end_pattern = r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*"
    start_match = re.search(start_pattern, text, flags=re.IGNORECASE | re.DOTALL)
    end_match = re.search(end_pattern, text, flags=re.IGNORECASE | re.DOTALL)
    start = start_match.end() if start_match else 0
    end = end_match.start() if end_match else len(text)
    return text[start:end].strip()

def fetch_book_text(book_id):
    last_error = None
    for pattern in URL_PATTERNS:
        url = pattern.format(id=book_id)
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and "Project Gutenberg" in r.text:
                return r.text
        except Exception as e:
            last_error = e
    if last_error is not None:
        raise RuntimeError(f"Failed to download {book_id}: {last_error}")
    raise RuntimeError(f"Failed to download {book_id}: no valid URL pattern")

def download_book(book_id, overwrite=False):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    outfile = RAW_DIR / f"{book_id}.txt"
    if outfile.exists() and not overwrite:
        return outfile
    raw_text = fetch_book_text(book_id)
    cleaned = strip_gutenberg_header_footer(raw_text)
    outfile.write_text(cleaned, encoding="utf-8")
    return outfile

def download_all_books(overwrite=False):
    paths = []
    for book_id in BOOKS.keys():
        path = download_book(book_id, overwrite=overwrite)
        paths.append(path)
    return paths

def main():
    paths = download_all_books(overwrite=False)
    for p in paths:
        print(f"Saved {p}")

if __name__ == "__main__":
    main()

