"""Cross-encoder reranking on top of any retriever from rag/hybrid.py.

Stage 1 (retriever): fast, rough — fetch the top `depth` candidates (default 30).
Stage 2 (reranker):  slow, careful — score every (question, chunk) pair together, keep the best k.

Usage:
    uv run python -m rag.rerank "What was HDFC Bank's net profit in FY26?" --index data/index/docling_fine5
    uv run python -m rag.rerank "..." --index data/index/docling_fine5 --model bge -k 5
"""
import argparse
import time

import numpy as np
from sentence_transformers import CrossEncoder

RERANKERS = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # ~1.9 s / 30 candidates on laptop CPU
    "bge": "BAAI/bge-reranker-base",                    # ~8.1 s / 30 candidates, stronger
}


class Reranker:
    def __init__(self, name="minilm", max_length=512, batch_size=16):
        self.name = name
        self.model_id = RERANKERS.get(name, name)  # short name or a full Hugging Face id
        self.model = CrossEncoder(self.model_id, max_length=max_length)
        self.batch_size = batch_size
        print(f"[rerank] {self.model_id}")

    def rerank(self, query, candidates, top_k):
        """candidates: list of (chunk, retriever_score). Returns top_k (chunk, rerank_score), best first.
        Scores are only comparable within one model (MiniLM gives raw logits, bge gives 0-1)."""
        if not candidates:
            return []
        pairs = [(query, chunk["text"]) for chunk, _ in candidates]
        scores = np.asarray(self.model.predict(pairs, batch_size=self.batch_size,
                                               show_progress_bar=False), dtype=np.float32)
        order = np.argsort(-scores)[:top_k]
        return [(candidates[i][0], float(scores[i])) for i in order]


class RerankedRetriever:
    """Wraps a retriever: fetch `depth` candidates, rerank, return top k.
    Exposes the same search / search_batch interface as the retrievers in rag/hybrid.py."""

    def __init__(self, base, reranker, depth=30):
        self.base, self.reranker, self.depth = base, reranker, depth
        self.last_candidates = []   # stage-1 lists from the last call (for the eval's ceiling check)
        self.timing = {}

    def search_batch(self, queries, k):
        n = max(len(queries), 1)
        t0 = time.perf_counter()
        candidates = self.base.search_batch(queries, self.depth)
        t1 = time.perf_counter()
        out = [self.reranker.rerank(q, c, k) for q, c in zip(queries, candidates)]
        t2 = time.perf_counter()
        self.last_candidates = candidates
        self.timing = {"retrieve_s_per_q": round((t1 - t0) / n, 3),
                       "rerank_s_per_q": round((t2 - t1) / n, 3)}
        return out

    def search(self, query, k=5):
        return self.search_batch([query], k)[0]


if __name__ == "__main__":
    from rag.hybrid import make_retriever

    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--index", default="data/index/docling_fine5")
    ap.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
    ap.add_argument("--model", default="minilm", help="minilm | bge | any cross-encoder HF id")
    ap.add_argument("--depth", type=int, default=30)
    ap.add_argument("-k", type=int, default=5)
    a = ap.parse_args()

    r = RerankedRetriever(make_retriever(a.mode, a.index), Reranker(a.model), a.depth)
    results = r.search(a.query, a.k)
    print(f"\nretrieve {r.timing['retrieve_s_per_q']}s | rerank {r.timing['rerank_s_per_q']}s")
    for rank, (c, s) in enumerate(results, 1):
        print(f"\n#{rank}  {c['company']} p{c['page']}  [{c.get('type')}]  rerank={s:.4f}")
        print(c["text"][:300])