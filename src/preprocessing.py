import re
from pathlib import Path

import pandas as pd

from .config import BOOKS, RAW_DIR, PROCESSED_DIR


def read_book_text(book_id):
    path = RAW_DIR / f"{book_id}.txt"
    return path.read_text(encoding="utf-8")


def split_into_chapters(text):
    pattern = re.compile(
        r"(^\s*(chapter|letter|section)\s+[xivlcdm0-9]+\.?\s*$)",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return [text.strip()]
    chapters = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            chapters.append(chunk)
    return chapters


def build_chapter_dataframe():
    rows = []
    for book_id, title in BOOKS.items():
        text = read_book_text(book_id)
        chapters = split_into_chapters(text)
        for i, ch in enumerate(chapters, start=1):
            rows.append(
                {
                    "book_id": book_id,
                    "book_title": title,
                    "chapter_idx": i,
                    "chapter_id": f"{title}__{i}",
                    "text": ch,
                }
            )
    df = pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    outpath = PROCESSED_DIR / "chapters.csv"
    df.to_csv(outpath, index=False)
    return df, outpath


def main():
    df, outpath = build_chapter_dataframe()
    print(f"Saved {len(df)} chapters to {outpath}")


if __name__ == "__main__":
    main()

