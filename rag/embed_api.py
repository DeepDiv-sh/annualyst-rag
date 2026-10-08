"""Embed chunks with the AbaciNLP (Fin-E5) API and save a FAISS-compatible index folder.

Output (same layout as rag/index.py, plus embeddings.npy):
    <out>/embeddings.npy   float32, L2-normalised, row i = chunk i
    <out>/chunks.jsonl     the embedded chunks, same order
    <out>/index.faiss      IndexFlatIP over the embeddings (= cosine similarity)
    <out>/meta.json        model, dim, query prefix, token usage
    <out>/parts/           per-batch checkpoints (safe to delete after a successful run)

Resumable: batches already saved in <out>/parts/ are skipped on a rerun,
so you never pay twice for the same chunks.

Usage:
    uv run python -m rag.embed_api --chunks data/chunks/docling_chunks.jsonl --out data/index/docling_fine5_test --limit 200
    uv run python -m rag.embed_api --chunks data/chunks/docling_chunks.jsonl --out data/index/docling_fine5
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import APIStatusError, BadRequestError, OpenAI
from tqdm import tqdm

BASE_URL = "https://abacinlp.com/v1"
MODEL = "abacinlp-text-v1"  # Fin-E5 (finance-tuned e5-Mistral-7B), served via the AbaciNLP API
# Queries get this prefix at search time. Documents (chunks) are embedded as plain text.
QUERY_TASK = "Given a financial question, retrieve relevant passages that answer the query."
QUERY_PREFIX = f"Instruct: {QUERY_TASK}\nQuery: "
MAX_CHARS = 8000  # safety cap for one text, used only if a batch keeps failing


def load_chunks(path, limit=None):
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
                if limit and len(chunks) >= limit:
                    break
    return chunks


def create_with_retry(client, model, texts, tries=5):
    """One API call. Retries server-side errors the SDK doesn't retry itself
    (e.g. AbaciNLP's 404 'upstream_error'), waiting 3, 6, 12, 24 s."""
    for attempt in range(tries):
        try:
            return client.embeddings.create(model=model, input=texts, encoding_format="float")
        except BadRequestError:
            raise  # a bad input won't fix itself by waiting
        except APIStatusError as e:
            if e.status_code in (401, 403) or attempt == tries - 1:
                raise
            wait = 3 * 2 ** attempt
            print(f"\n  API error {e.status_code}: retry {attempt + 1}/{tries - 1} in {wait}s")
            time.sleep(wait)


def embed_batch(client, model, texts):
    """Embed a list of texts. Returns (vectors, tokens_used).
    If the batch keeps failing, retry one text at a time (truncated) to isolate a bad chunk."""
    texts = [t if t.strip() else "[empty]" for t in texts]
    try:
        resp = create_with_retry(client, model, texts)
        data = sorted(resp.data, key=lambda d: d.index)  # keep the same order as `texts`
        tokens = resp.usage.total_tokens if resp.usage else 0
        return [d.embedding for d in data], tokens
    except APIStatusError as e:
        if e.status_code in (401, 403):
            raise
        print(f"\n  batch failed ({e.status_code}): retrying one by one, truncated to {MAX_CHARS} chars")
        vecs, tokens = [], 0
        for i, t in enumerate(texts):
            try:
                r = create_with_retry(client, model, [t[:MAX_CHARS]])
            except APIStatusError:
                print(f"  text {i} in this batch keeps failing ({len(t)} chars): {t[:120]!r}")
                raise
            vecs.append(r.data[0].embedding)
            tokens += r.usage.total_tokens if r.usage else 0
        return vecs, tokens


def save_npy_atomic(path, arr):
    """Write to a temp file then rename, so a crash never leaves a half-written checkpoint."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", required=True, help="input chunks .jsonl")
    p.add_argument("--out", required=True, help="output index folder")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--batch", type=int, default=32, help="texts per API request")
    p.add_argument("--limit", type=int, default=None, help="embed only the first N chunks (dry run)")
    a = p.parse_args()

    load_dotenv()
    # The OpenAI SDK retries 429 / 5xx / timeouts itself, with exponential backoff.
    client = OpenAI(api_key=os.environ["ABACI_API_KEY"], base_url=BASE_URL,
                    max_retries=8, timeout=120)

    out = Path(a.out)
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)

    # Checkpoints are named by start index, so the batch size must not change between runs.
    cfg_path = parts / "config.json"
    cfg = {"chunks": a.chunks, "batch": a.batch, "model": a.model}
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        if old != cfg:
            raise SystemExit(f"{parts} was created with {old}; you passed {cfg}. "
                             f"Use the same settings or a new --out folder.")
    else:
        cfg_path.write_text(json.dumps(cfg))

    chunks = load_chunks(a.chunks, a.limit)
    n = len(chunks)
    starts = list(range(0, n, a.batch))
    done = sum((parts / f"{s:07d}.npy").exists() for s in starts)
    print(f"{n:,} chunks | {len(starts)} batches of {a.batch} | {done} already done | -> {out}")

    t0 = time.time()
    run_tokens = 0
    for s in tqdm(starts, desc="embedding"):
        vec_path = parts / f"{s:07d}.npy"
        if vec_path.exists():
            continue  # already paid for
        texts = [c["text"] for c in chunks[s:s + a.batch]]
        vecs, tokens = embed_batch(client, a.model, texts)
        arr = np.asarray(vecs, dtype=np.float32)
        if arr.shape[0] != len(texts):
            raise RuntimeError(f"batch {s}: sent {len(texts)} texts, got {arr.shape[0]} vectors")
        save_npy_atomic(vec_path, arr)
        with open(parts / "usage.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"start": s, "n": len(texts), "tokens": tokens}) + "\n")
        run_tokens += tokens
    elapsed = time.time() - t0

    # ---------- assemble the final index ----------
    emb = np.concatenate([np.load(parts / f"{s:07d}.npy") for s in starts])
    if emb.shape[0] != n:
        raise RuntimeError(f"expected {n} vectors, got {emb.shape[0]}")
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    np.save(out / "embeddings.npy", emb)

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(out / "index.faiss"))

    with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    total_tokens = 0
    usage_path = parts / "usage.jsonl"
    if usage_path.exists():
        with open(usage_path, encoding="utf-8") as f:
            total_tokens = sum(json.loads(line)["tokens"] for line in f if line.strip())

    meta = {
        "model": a.model,
        "provider": "abacinlp",
        "base_url": BASE_URL,
        "embedder": "api",
        "dim": int(emb.shape[1]),
        "normalized": True,
        "n_chunks": n,
        "chunks_source": a.chunks,
        "query_prefix": QUERY_PREFIX,
        "total_tokens": total_tokens,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\ndone in {elapsed:.0f}s | dim {emb.shape[1]} | {n:,} vectors -> {out}")
    print(f"tokens this run: {run_tokens:,} | total tokens for this index: {total_tokens:,} "
          f"| avg {total_tokens / max(n, 1):.0f} tokens/chunk")


if __name__ == "__main__":
    main()