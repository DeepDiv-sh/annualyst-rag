import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent

QUERY_INSTRUCTION = "Retrieve this sentence for searching relevant passages: "

def resolve(p):
    path = Path(p)
    return path if path.is_absolute() else ROOT / path

class Retriver:
    def __init__(self, index_dir: str = "data/index/pymupdf"):
        index_dir = resolve(index_dir)
        
        self.meta = json.loads((index_dir / "meta.json").read_text())
        self.index = faiss.read_index(str(index_dir/"index.faiss"))
        with (index_dir / "chunks.jsonl").open(encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f if line.strip()]
            
        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                f"index has {self.index.ntotal} vectors but {len(self.chunks)} chunks; rebuild the index"
            )
        
        self.model = SentenceTransformer(self.meta["model"], device = "cpu")
        
    def embed_query(self, query):
        vec = self.model.encode(
            [QUERY_INSTRUCTION + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.astype(np.float32)
    
    def search(self, query, k: int = 5):
        scores, ids = self.index.search(self.embed_query(query), k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            c = self.chunks[idx]
            results.append(
                {
                    "score": float(score),
                    "chunk_id": c["chunk_id"],
                    "company": c["company"],
                    "page": c["page"],
                    "text": c["text"],
                }
            )
        return results
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="the question to search for")
    ap.add_argument("--index", default="data/index/pymupdf")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    
    retriver = Retriver(args.index)
    hits = retriver.search(args.query, args.k)
    
    print(f"\nQ: {args.query}\n")
    for rank, h in enumerate(hits, start=1):
        preview = h["text"][:300].replace("\n", " ")
        print(f"#{rank}  score={h['score']:.3f}  {h['company']}  p.{h['page']}  [{h['chunk_id']}]")
        print(f"    {preview}...\n")


if __name__ == "__main__":
    main()