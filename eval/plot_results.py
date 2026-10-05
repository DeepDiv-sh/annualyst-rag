"""
Plot + tabulate every eval/results/*_summary.json side by side.

Usage (from repo root):
  python eval/plot_results.py
Outputs:
  eval/results/plots/overall.png    Recall@1/5/10 + MRR per config
  eval/results/plots/by_type.png    Recall@5 per question type per config
  eval/results/plots/by_company.png Recall@5 per company per config
  eval/results/plots/recall_curve.png  Recall vs k per config
  Prints a markdown table (paste into README).
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = "eval/results"
OUT = os.path.join(RESULTS, "plots")
KS = (1, 5, 10)
# preferred display order; unknown configs go at the end
ORDER = ["pymupdf", "docling", "hybrid", "rerank"]


def load_summaries():
    summaries = []
    for path in glob.glob(os.path.join(RESULTS, "*_summary.json")):
        with open(path, encoding="utf-8") as f:
            summaries.append(json.load(f))
    summaries.sort(key=lambda s: (ORDER.index(s["config"]) if s["config"] in ORDER else 99,
                                  s["config"]))
    return summaries


def grouped_bars(ax, groups, configs, values, ylabel, title):
    """values[c][g] -> bar height for config c, group g."""
    x = np.arange(len(groups))
    width = 0.8 / max(len(configs), 1)
    for i, c in enumerate(configs):
        heights = [values[c].get(g, 0.0) for g in groups]
        bars = ax.bar(x + (i - (len(configs) - 1) / 2) * width, heights, width, label=c)
        for b, h in zip(bars, heights):
            ax.text(b.get_x() + b.get_width() / 2, h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)


def main():
    summaries = load_summaries()
    if not summaries:
        print("No *_summary.json in eval/results. Run eval/run_eval.py first.")
        return
    os.makedirs(OUT, exist_ok=True)
    configs = [s["config"] for s in summaries]
    by_cfg = {s["config"]: s for s in summaries}

    # 1. overall metrics
    metrics = [f"recall@{k}" for k in KS] + ["mrr@10"]
    vals = {c: {m: by_cfg[c]["overall"][m] for m in metrics} for c in configs}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    n = summaries[0]["overall"]["n"]
    grouped_bars(ax, metrics, configs, vals, "score", f"Retrieval quality (n={n})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "overall.png"), dpi=150)

    # 2. recall@5 by type, 3. by company
    for key, fname, title in [("by_type", "by_type.png", "Recall@5 by question type"),
                              ("by_company", "by_company.png", "Recall@5 by company")]:
        groups = sorted({g for s in summaries for g in s[key]})
        vals = {c: {g: by_cfg[c][key][g]["recall@5"] for g in by_cfg[c][key]} for c in configs}
        labels = [f"{g}\n(n={summaries[0][key].get(g, {}).get('n', '?')})" for g in groups]
        fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(groups)), 4.5))
        grouped_bars(ax, groups, configs, vals, "recall@5", title)
        ax.set_xticklabels(labels)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname), dpi=150)

    # 4. recall curve
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for c in configs:
        ax.plot(KS, [by_cfg[c]["overall"][f"recall@{k}"] for k in KS], marker="o", label=c)
    ax.set_xticks(KS)
    ax.set_xlabel("k (chunks retrieved)")
    ax.set_ylabel("recall")
    ax.set_ylim(0, 1.05)
    ax.set_title("Recall vs k  (gap @5 to @10 = reranker headroom)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "recall_curve.png"), dpi=150)

    # markdown table for README
    types = sorted({t for s in summaries for t in s["by_type"]})
    header = "| config | R@1 | R@5 | R@10 | MRR | " + " | ".join(f"R@5 {t}" for t in types) + " |"
    print(header)
    print("|" + "---|" * (5 + len(types)))
    for s in summaries:
        o = s["overall"]
        cells = [f"{o['recall@1']:.2f}", f"{o['recall@5']:.2f}", f"{o['recall@10']:.2f}",
                 f"{o['mrr@10']:.3f}"]
        cells += [f"{s['by_type'][t]['recall@5']:.2f}" if t in s["by_type"] else "-" for t in types]
        print(f"| {s['config']} | " + " | ".join(cells) + " |")
    print(f"\nSaved 4 plots to {OUT}/")


if __name__ == "__main__":
    main()