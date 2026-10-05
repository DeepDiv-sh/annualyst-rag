import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "BAAI/bge-small-en-v1.5"

def resolve(p):
    path = Path(p)
    return path if path.is_absolute() else ROOT/path

def load_chunks(path):
    with path.open(encoding = "utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/chunks/pymupdf_chunks.jsonl")
    ap.add_argument("--out", default="data/index/pymupdf")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    
    chunks_path = resolve(args.chunks)
    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    chunks = load_chunks(chunks_path)
    texts = [c["text"] for c in chunks]
    print(f"Loaded {len(chunks)} chunks from {chunks_path}", flush=True)
    
    model = SentenceTransformer(MODEL_NAME, device='cpu')
    
    t0 = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    secs = time.perf_counter() -t0
    print(f"Embedded {embeddings.shape[0]} chunks, dim={embeddings.shape[1]}, in {secs:.1f}s", flush=True)
    
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    assert index.ntotal == len(chunks), "vector count and chunk count must match"
    
    scores, ids = index.search(embeddings[:1], 1)
    assert ids[0][0] == 0, "self-retrieval failed: index is broken"
    
    faiss.write_index(index, str(out_dir / "index.faiss"))
    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    meta = {
        "model": MODEL_NAME,
        "dim" : int(embeddings.shape[1]),
        "num_chunks": len(chunks),
        "source_chunks": str(chunks_path.relative_to(ROOT)),
        "normalized": True,
        "embed_seconds": round(secs, 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    
    print(f"Saved index + chunks + meta to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
    