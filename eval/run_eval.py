import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.hybrid import load_jsonl, make_retriever  # noqa: E402

KS = (1, 5, 10)
MAX_K = max(KS)


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


def print_table(title, groups):
    print(f"\n{title}")
    print(f"{'':<16}{'n':>4}" + "".join(f"{'R@' + str(k):>10}" for k in KS) + f"{'MRR':>10}")
    for name, s in groups.items():
        print(f"{name:<16}{s['n']:>4}" + "".join(f"{s[f'recall@{k}']:>10.2f}" for k in KS)
              + f"{s['mrr@10']:>10.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="dense")
    ap.add_argument("--store", choices=["faiss", "qdrant"], default="faiss",
                    help="where the dense vectors are searched (Fin-E5 indexes only for qdrant)")
    ap.add_argument("--company-filter", action="store_true",
                    help="detect the company in the question and search only its chunks (needs qdrant)")
    ap.add_argument("--rerank", default=None,
                    help="cross-encoder to rerank with: minilm | bge (default: no reranking)")
    ap.add_argument("--rerank-depth", type=int, default=30,
                    help="how many retriever candidates the reranker sees")
    ap.add_argument("--questions", default="eval/questions.jsonl")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    base = os.path.basename(os.path.normpath(args.index))
    name = args.name or (base if args.mode == "dense" else f"{base}_{args.mode}")
    if not args.name:
        if args.store == "qdrant":
            name += "_qdrant"
        if args.company_filter:
            name += "_cf"
        if args.rerank:
            name += f"_rerank_{args.rerank}"

    questions = load_jsonl(args.questions)
    answerable = [q for q in questions if q.get("gold_pages")]
    print(f"[eval] config={name}  {len(answerable)} answerable / {len(questions)} questions")

    retriever = make_retriever(args.mode, args.index, store=args.store,
                               company_filter=args.company_filter)
    base_retriever = retriever
    if args.rerank:
        from rag.rerank import RerankedRetriever, Reranker
        retriever = RerankedRetriever(retriever, Reranker(args.rerank), depth=args.rerank_depth)

    t0 = time.perf_counter()
    all_results = retriever.search_batch([q["question"] for q in answerable], MAX_K)
    total_s = time.perf_counter() - t0

    rows, ranks = [], []
    by_type, by_company = defaultdict(list), defaultdict(list)
    for q, results in zip(answerable, all_results):
        gold = {int(p) for p in q["gold_pages"]}
        rank = first_hit_rank(results, q["company"], gold)
        ranks.append(rank)
        by_type[q.get("type", "?")].append(rank)
        by_company[q["company"]].append(rank)
        rows.append({"id": q["id"], "company": q["company"], "type": q.get("type"),
                     "question": q["question"], "gold_pages": sorted(gold), "hit_rank": rank,
                     "retrieved": [{"company": c["company"], "page": c["page"],
                                    "type": c.get("type"), "score": round(s, 4)}
                                   for c, s in results]})

    overall = summarize(ranks)
    print_table("OVERALL", {"all": overall})
    print_table("BY TYPE", {t: summarize(r) for t, r in sorted(by_type.items())})
    print_table("BY COMPANY", {c: summarize(r) for c, r in sorted(by_company.items())})

    # Reranker ceiling: a reranker can only reorder what stage 1 found.
    extra = {"seconds_total": round(total_s, 1),
             "seconds_per_question": round(total_s / max(len(answerable), 1), 3)}
    if args.rerank:
        in_cands = [first_hit_rank(c, q["company"], {int(p) for p in q["gold_pages"]}) is not None
                    for q, c in zip(answerable, retriever.last_candidates)]
        extra["rerank_model"] = args.rerank
        extra["rerank_depth"] = args.rerank_depth
        extra[f"candidate_recall@{args.rerank_depth}"] = round(sum(in_cands) / len(in_cands), 3)
        extra.update(retriever.timing)
        print(f"\nRERANK CEILING: gold page in top-{args.rerank_depth} candidates for "
              f"{sum(in_cands)}/{len(in_cands)} questions "
              f"({extra[f'candidate_recall@{args.rerank_depth}']:.2f})")
        print(f"TIMING (avg per question): retrieve {retriever.timing['retrieve_s_per_q']}s | "
              f"rerank {retriever.timing['rerank_s_per_q']}s")
    else:
        print(f"\nTIMING: {extra['seconds_per_question']}s per question (batched)")

    # Company filter diagnostics: detection uses only the question text; the eval's
    # "company" field is used here only to check whether the detection was right.
    if args.company_filter:
        filters = base_retriever.last_filters
        detected = sum(bool(f) for f in filters)
        correct = sum(f == {q["company"]} for q, f in zip(answerable, filters))
        wrong = [(q["id"], sorted(f)) for q, f in zip(answerable, filters)
                 if f and q["company"] not in f]
        extra["company_filter"] = {"detected": detected, "exact_match": correct,
                                   "wrong": len(wrong), "n": len(answerable)}
        print(f"COMPANY FILTER: detected in {detected}/{len(answerable)} questions | "
              f"exactly right {correct} | wrong {len(wrong)}"
              + (f" -> {wrong}" if wrong else ""))

    misses = [r for r in rows if r["hit_rank"] is None or r["hit_rank"] > 5]
    print(f"\nMISSES @5 ({len(misses)}):")
    for r in misses:
        top3 = ", ".join(f"{x['company']} p{x['page']}" for x in r["retrieved"][:3])
        print(f"  {r['id']:<16} gold={r['gold_pages']}  rank={r['hit_rank']}  top3: {top3}")

    os.makedirs("eval/results", exist_ok=True)
    with open(f"eval/results/{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"eval/results/{name}_summary.json", "w", encoding="utf-8") as f:
        json.dump({"config": name, "index": args.index, "mode": args.mode,
                   "store": args.store, "company_filter": args.company_filter,
                   "rerank": args.rerank, "questions": args.questions, "overall": overall,
                   "by_type": {t: summarize(r) for t, r in by_type.items()},
                   "by_company": {c: summarize(r) for c, r in by_company.items()},
                   **extra}, f, indent=2)
    print(f"\n[eval] saved eval/results/{name}.jsonl and {name}_summary.json")


if __name__ == "__main__":
    main()