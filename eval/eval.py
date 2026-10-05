import argparse
import json
import os
from collections import defaultdict
 
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
 
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
KS = (1, 5, 10)
MAX_K = max(KS)
 
 
def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
 
 
class DenseRetriever:
    """Mirrors rag/retrieve.py: same model (from meta.json), same bge prefix, FAISS IP."""
 
    def __init__(self, index_dir):
        with open(os.path.join(index_dir, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        model_name = (meta.get("model") or meta.get("model_name")
                      or meta.get("embedding_model") or "BAAI/bge-small-en-v1.5")
        print(f"[eval] model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
        self.chunks = load_jsonl(os.path.join(index_dir, "chunks.jsonl"))
        assert self.index.ntotal == len(self.chunks), "index and chunks.jsonl out of sync"
        self.prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""
 
    def search_batch(self, queries, k):
        q = self.model.encode([self.prefix + x for x in queries],
                              normalize_embeddings=True, batch_size=32,
                              show_progress_bar=False)
        scores, ids = self.index.search(np.asarray(q, dtype="float32"), k)
        return [[(self.chunks[i], float(s)) for i, s in zip(row_i, row_s) if i != -1]
                for row_i, row_s in zip(ids, scores)]
 
 
def first_hit_rank(results, company, gold):
    for rank, (chunk, _) in enumerate(results, start=1):
        if chunk["company"] == company and int(chunk["page"]) in gold:
            return rank
    return None
 
 
def summarize(ranks):
    n = len(ranks)
    out = {"n": n}
    for k in KS:
        out[f"recall@{k}"] = sum(r is not None and r <= k for r in ranks) / n if n else 0.0
    out["mrr@10"] = sum(1.0 / r for r in ranks if r is not None) / n if n else 0.0
    return out
 
 
def fmt_row(name, s):
    return (f"{name:<16}{s['n']:>4}" + "".join(f"{s[f'recall@{k}']:>10.2f}" for k in KS)
            + f"{s['mrr@10']:>10.3f}")
 
 
def print_table(title, groups):
    print(f"\n{title}")
    print(f"{'':<16}{'n':>4}" + "".join(f"{'R@' + str(k):>10}" for k in KS) + f"{'MRR':>10}")
    for name, s in groups.items():
        print(fmt_row(name, s))
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="e.g. data/index/pymupdf")
    ap.add_argument("--questions", default="eval/questions.jsonl")
    ap.add_argument("--name", default=None, help="results name (default: index folder name)")
    args = ap.parse_args()
    name = args.name or os.path.basename(os.path.normpath(args.index))
 
    questions = load_jsonl(args.questions)
    answerable = [q for q in questions if q.get("gold_pages")]
    print(f"[eval] {len(questions)} questions, {len(answerable)} answerable "
          f"({len(questions) - len(answerable)} unanswerable skipped)")
 
    retriever = DenseRetriever(args.index)
    all_results = retriever.search_batch([q["question"] for q in answerable], MAX_K)
 
    rows, ranks = [], []
    by_type, by_company = defaultdict(list), defaultdict(list)
    for q, results in zip(answerable, all_results):
        gold = {int(p) for p in q["gold_pages"]}
        rank = first_hit_rank(results, q["company"], gold)
        ranks.append(rank)
        by_type[q.get("type", "?")].append(rank)
        by_company[q["company"]].append(rank)
        rows.append({
            "id": q["id"], "company": q["company"], "type": q.get("type"),
            "question": q["question"], "gold_pages": sorted(gold), "hit_rank": rank,
            "retrieved": [{"company": c["company"], "page": c["page"], "score": round(s, 4)}
                          for c, s in results],
        })
 
    overall = summarize(ranks)
    print_table("OVERALL", {"all": overall})
    print_table("BY TYPE", {t: summarize(r) for t, r in sorted(by_type.items())})
    print_table("BY COMPANY", {c: summarize(r) for c, r in sorted(by_company.items())})
 
    misses = [r for r in rows if r["hit_rank"] is None or r["hit_rank"] > 5]
    print(f"\nMISSES @5 ({len(misses)}):")
    for r in misses:
        top3 = ", ".join(f"{x['company']} p{x['page']}" for x in r["retrieved"][:3])
        print(f"  {r['id']:<16} gold={r['gold_pages']}  rank={r['hit_rank']}  top3: {top3}")
 
    os.makedirs("eval/results", exist_ok=True)
    with open(f"eval/results/{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "config": name, "index": args.index, "questions": args.questions, "overall": overall,
        "by_type": {t: summarize(r) for t, r in by_type.items()},
        "by_company": {c: summarize(r) for c, r in by_company.items()},
    }
    with open(f"eval/results/{name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[eval] saved eval/results/{name}.jsonl and {name}_summary.json")
 
 
if __name__ == "__main__":
    main()
