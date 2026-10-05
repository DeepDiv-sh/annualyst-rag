import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from rag.retrieve import Retriver

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT/".env")

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """You answer questions about Indian company annual reports (FY2025-26).

Rules:
1. Use ONLY the context chunks provided. Never use outside knowledge.
2. Every factual claim must be supported by at least one chunk; list the IDs of the chunks you used in "citations".
3. If the context does not contain the answer, set "found" to false, set "answer" to a one-line statement that the reports provided do not contain this information, and return an empty "citations" list. Do not guess.
4. Be precise with numbers: keep units (crore, lakh, %, '000) and say which year a figure belongs to.
5. Note: due to PDF font extraction, the rupee symbol may appear as ` or I or H or J directly before a number (e.g. `74,671.3 crore or I3,110 crore). Treat these as Rs./₹.
6. Distinguish the company from its subsidiaries (e.g. HDFC Bank vs HDFC Securities). Only answer about the entity asked for."""

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "rag_answer",
        "strict": True,
        "schema": {
            "type":"object",
            "properties":{
                "answer":{"type":"string"},
                "citations":{"type":"array", "items":{"type":"string"}},
                "found":{"type":"boolean"},
            },
            "required":["answer", "citations", "found"],
            "additionalProperties":False,
        }
    }
}

def build_context(hits):
    blocks = []
    for h in hits:
        header = f"[{h['chunk_id']}] ({h['company']}, page {h['page']})"
        blocks.append(f"{header}\n{h['text']}")
    return "\n\n---\n\n".join(blocks)

class Answerer:
    def __init__(self, index_dir: str = "data/index/pymupdf", model: str = DEFAULT_MODEL, k: int = 5):
        self.retriver = Retriver(index_dir)
        self.client = OpenAI()
        self.model = model
        self.k = k
        
    def answer(self, question):
        hits = self.retriver.search(question, k=self.k)
        context = build_context(hits)
        resp = self.client.chat.completions.create(
            model = self.model,
            temperature=0,
            response_format=RESPONSE_SCHEMA,
            messages=[
                {"role":"system", "content":SYSTEM_PROMPT},
                {"role":"user", "content":f"Context:\n\n{context}\n\nQuestion:{question}"}
            ]
        )
        
        out = json.loads(resp.choices[0].message.content)
        
        by_id = {h["chunk_id"]: h for h in hits}
        valid = [cid for cid in out["citations"] if cid in by_id]
        dropped = [cid for cid in out["citations"] if cid not in by_id]
        
        return {
            "question": question,
            "answer": out["answer"],
            "found": out["found"],
            "citations":[
                {"chunk_id": cid, "company": by_id[cid]["company"], "page": by_id[cid]["page"]} for cid in valid
            ],
            "dropped_citations": dropped,
            "hits":hits,
            "usage":{
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            },
        }
        
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--index", default="data/index/pymupdf")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--show-context", action="store_true")
    args = ap.parse_args()
    
    result = Answerer(args.index, args.model, args.k).answer(args.question)

    if args.show_context:
        print("\n=== retrieved ===")
        for i, h in enumerate(result["hits"], 1):
            print(f"#{i} {h['score']:.3f} {h['chunk_id']}  {h['text'][:200]}...")

    print(f"\nQ: {result['question']}")
    print(f"\nA: {result['answer']}")
    print(f"\nfound: {result['found']}")
    if result["citations"]:
        print("\nSources:")
        for c in result["citations"]:
            print(f"  - {c['company']} p.{c['page']}  [{c['chunk_id']}]")
    if result["dropped_citations"]:
        print(f"\n(warning: model cited unknown chunks, dropped: {result['dropped_citations']})")
    u = result["usage"]
    print(f"\n[tokens: {u['prompt_tokens']} in / {u['completion_tokens']} out | model: {args.model}]")


if __name__ == "__main__":
    main()