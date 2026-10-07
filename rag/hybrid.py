import argparse
import json
import os

import bm25s
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import Stemmer
    STEMMER = Stemmer.Stemmer("english")
except ImportError:
    STEMMER = None

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_chunks(index_dir):
    return load_jsonl(os.path.join(index_dir, "chunks.jsonl"))


class _Base:
    chunks: list

    def search_ids(self, queries, k):
        raise NotImplementedError

    def search_batch(self, queries, k):
        return [[(self.chunks[i], s) for i, s in row] for row in self.search_ids(queries, k)]

    def search(self, query, k=5):
        return self.search_batch([query], k)[0]


class DenseRetriever(_Base):
    def __init__(self, index_dir, chunks=None):
        with open(os.path.join(index_dir, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        name = (meta.get("model") or meta.get("model_name")
                or meta.get("embedding_model") or "BAAI/bge-small-en-v1.5")
        self.model = SentenceTransformer(name)
        self.prefix = BGE_QUERY_PREFIX if "bge" in name.lower() else ""
        self.index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
        self.chunks = chunks if chunks is not None else load_chunks(index_dir)
        assert self.index.ntotal == len(self.chunks), "index and chunks.jsonl out of sync"
        print(f"[dense] {name}, {len(self.chunks)} chunks")

    def search_ids(self, queries, k):
        q = self.model.encode([self.prefix + x for x in queries], normalize_embeddings=True,
                              batch_size=32, show_progress_bar=False)
        scores, ids = self.index.search(np.asarray(q, dtype="float32"), k)
        return [[(int(i), float(s)) for i, s in zip(ri, rs) if i != -1]
                for ri, rs in zip(ids, scores)]


class BM25Retriever(_Base):
    def __init__(self, index_dir, chunks=None):
        self.chunks = chunks if chunks is not None else load_chunks(index_dir)
        tokens = bm25s.tokenize([c["text"] for c in self.chunks], stopwords="en",
                                stemmer=STEMMER, show_progress=False)
        self.bm25 = bm25s.BM25()
        self.bm25.index(tokens, show_progress=False)
        print(f"[bm25] indexed {len(self.chunks)} chunks (stemmer={'on' if STEMMER else 'off'})")

    def search_ids(self, queries, k):
        q = bm25s.tokenize(queries, stopwords="en", stemmer=STEMMER,
                           return_ids=False, show_progress=False)
        ids, scores = self.bm25.retrieve(q, k=min(k, len(self.chunks)), show_progress=False)
        return [[(int(i), float(s)) for i, s in zip(ri, rs)] for ri, rs in zip(ids, scores)]


class HybridRetriever(_Base):
    def __init__(self, index_dir, rrf_k=60, fetch_k=50):
        self.chunks = load_chunks(index_dir)
        self.dense = DenseRetriever(index_dir, self.chunks)
        self.bm25 = BM25Retriever(index_dir, self.chunks)
        self.rrf_k, self.fetch_k = rrf_k, fetch_k

    def search_ids(self, queries, k):
        dense = self.dense.search_ids(queries, self.fetch_k)
        sparse = self.bm25.search_ids(queries, self.fetch_k)
        out = []
        for d, b in zip(dense, sparse):
            fused = {}
            for ranked in (d, b):
                for rank, (i, _) in enumerate(ranked, start=1):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (self.rrf_k + rank)
            out.append(sorted(fused.items(), key=lambda x: -x[1])[:k])
        return out


def make_retriever(mode, index_dir):
    return {"dense": DenseRetriever, "bm25": BM25Retriever,
            "hybrid": HybridRetriever}[mode](index_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--index", default="data/index/docling")
    ap.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
    ap.add_argument("-k", type=int, default=5)
    a = ap.parse_args()
    for rank, (c, s) in enumerate(make_retriever(a.mode, a.index).search(a.query, a.k), 1):
        print(f"\n#{rank}  {c['company']} p{c['page']}  [{c.get('type')}]  score={s:.4f}")
        print(c["text"][:300])