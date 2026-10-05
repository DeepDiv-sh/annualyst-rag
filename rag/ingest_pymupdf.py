import json
import re
from pathlib import Path

import pymupdf

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/chunks/pymupdf_chunks.jsonl")

CHUNK_SIZE = 1000
OVERLAP = 150
MIN_PAGE_CHARS = 50

def clean(text):
    return re.sub("\s+", " ", text).strip()

def chunk_text(text, size, overlap):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            cut = text.rfind(" ", start, end)
            if cut > start + size//2:
                end = cut
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
        if end >= n:
            break
        
        start = end - overlap
        space = text.find(" ", start, end)
        if space != -1:
            start = space + 1
    return chunks

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    total_chunks = 0
    
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for pdf_path in pdfs:
            company = pdf_path.stem.rsplit("_", 1)[0]
            doc = pymupdf.open(pdf_path)
            used, skipped, n_chunks = 0, 0, 0
            for page_index, page in enumerate(doc):
                page_num = page_index + 1
                text = clean(page.get_text())
                if len(text) < MIN_PAGE_CHARS:
                    skipped += 1
                    continue
                used += 1
                
                for i, chunk in enumerate(chunk_text(text, CHUNK_SIZE, OVERLAP)):
                    record = {
                        "chunk_id": f"{company}_p{page_num}_c{i}",
                        "company": company,
                        "page": page_num,
                        "type": "text",
                        "section": None,
                        "source": "pymupdf",
                        "text": chunk,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_chunks += 1
            doc.close()
            total_chunks += n_chunks
            print(f"{company:12s} pages used={used:4d} skipped={skipped:3d} chunks={n_chunks:5d}")
    print(f"\nTOTAL chunks: {total_chunks} -> {OUT_PATH}")
    
if __name__ == "__main__":
    main()
    