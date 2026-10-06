"""
Docling chunker: structure-aware chunks from Docling JSON.

Rules:
  1. Headers/footers dropped (body only; page_header/page_footer labels skipped).
  2. Section headings attached: `section` field + "Section: ..." line at top of text.
  3. Tables kept whole as markdown; long tables split by rows with the header row repeated.
Text chunks are ~max_chars and never cross a page or a heading (same size as the
PyMuPDF baseline, so the ablation measures parsing quality, not chunk size).

Usage (from repo root):
  python -m rag.chunk_docling                      # all companies
  python -m rag.chunk_docling --companies itc      # quick test (overwrites output file)
Output: data/chunks/docling_chunks.jsonl  (same schema as pymupdf_chunks.jsonl)
"""
import argparse
import json
import numbers
import re
from pathlib import Path

from docling_core.types.doc import DoclingDocument

SKIP_LABELS = {"page_header", "page_footer", "picture", "chart"}
HEADING_LABELS = {"title", "section_header"}
TABLE_LABELS = {"table", "document_index"}


# ---------- small helpers ----------

def label_of(item):
    lab = getattr(item, "label", "")
    return str(getattr(lab, "value", lab)).lower()


def page_of(item, default):
    prov = getattr(item, "prov", None)
    return prov[0].page_no if prov else default


def clean(s):
    s = re.sub(r"\s+", " ", str(s if s is not None else "")).strip()
    return "" if s.lower() in ("nan", "none") else s


def cell(s):
    return clean(s).replace("|", "/")


def md_line(cells):
    return "| " + " | ".join(cells) + " |"


def split_long(text, max_chars, overlap):
    """Split one long text at whitespace into ~max_chars windows with overlap."""
    if len(text) <= max_chars:
        return [text]
    out, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + max_chars // 2, end)
            if cut != -1:
                end = cut
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [o for o in out if o]


def load_doc(path):
    try:
        return DoclingDocument.load_from_json(path)
    except AttributeError:  # older docling-core
        with open(path, encoding="utf-8") as f:
            return DoclingDocument.model_validate(json.load(f))


# ---------- tables ----------

def table_rows(table, doc):
    """Return (header, rows) as lists of cleaned strings, or (None, []) if empty."""
    try:
        df = table.export_to_dataframe(doc=doc)
    except TypeError:  # older docling-core has no doc arg
        df = table.export_to_dataframe()
    if df is None or df.empty:
        return None, []
    rows = [[cell(v) for v in r] for r in df.astype(str).values.tolist()]
    cols = list(df.columns)
    if all(isinstance(c, numbers.Integral) for c in cols):
        # Docling found no header row -> treat first row as header
        header, rows = rows[0], rows[1:]
    else:
        header = [cell(c) for c in cols]
    return header, rows


def table_chunks(header, rows, prefix, max_chars):
    """Markdown table split by rows; every part repeats prefix + header."""
    head = md_line(header) + "\n" + md_line(["---"] * len(header))
    budget = max(200, max_chars - len(prefix) - len(head) - 1)
    parts, cur, size = [], [], 0
    for r in rows:
        line = md_line(r)
        if cur and size + len(line) + 1 > budget:
            parts.append(prefix + head + "\n" + "\n".join(cur))
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur or not parts:
        parts.append(prefix + head + ("\n" + "\n".join(cur) if cur else ""))
    return parts


# ---------- the walker ----------

class DocChunker:
    def __init__(self, doc, company, max_chars, overlap, table_max):
        self.doc, self.company = doc, company
        self.max_chars, self.overlap, self.table_max = max_chars, overlap, table_max
        self.chunks = []
        self.stack = []                      # [(level, heading text)]
        self.buf, self.buf_len = [], 0
        self.buf_page, self.buf_section = None, None
        self.n_tables = self.n_split = self.n_fake_tables = 0

    def section(self):
        return " > ".join(t for _, t in self.stack)

    @staticmethod
    def prefix(section):
        return f"Section: {section}\n" if section else ""

    def emit(self, text, page, section, typ):
        self.chunks.append({"company": self.company, "page": int(page), "type": typ,
                            "section": section, "source": "docling", "text": text})

    def flush_text(self):
        if not self.buf:
            return
        pre = self.prefix(self.buf_section)
        body = "\n".join(self.buf)
        for piece in split_long(body, self.max_chars - len(pre), self.overlap):
            self.emit(pre + piece, self.buf_page, self.buf_section, "text")
        self.buf, self.buf_len = [], 0

    def add_text(self, text, page):
        sec = self.section()
        limit = self.max_chars - len(self.prefix(sec))
        if self.buf and (page != self.buf_page or sec != self.buf_section
                         or self.buf_len + len(text) + 1 > limit):
            self.flush_text()
        if not self.buf:
            self.buf_page, self.buf_section = page, sec
        self.buf.append(text)
        self.buf_len += len(text) + 1

    def set_heading(self, text, level):
        self.flush_text()
        level = max(1, int(level or 1))
        self.stack = [(l, t) for l, t in self.stack if l < level] + [(level, text)]

    def add_table(self, table, page):
        self.flush_text()
        header, rows = table_rows(table, self.doc)
        if header is None:
            return
        if len(header) <= 1:                 # 1-column "table" = layout box, treat as text
            self.n_fake_tables += 1
            text = " ".join(c for r in [header] + rows for c in r if c)
            if text:
                self.add_text(text, page)
                self.flush_text()
            return
        sec = self.section()
        cap = ""
        if hasattr(table, "caption_text"):
            try:
                cap = clean(table.caption_text(self.doc))
            except Exception:
                cap = ""
        pre = self.prefix(sec) + (f"Table: {cap}\n" if cap else "")
        parts = table_chunks(header, rows, pre, self.table_max)
        self.n_tables += 1
        self.n_split += len(parts) > 1
        for part in parts:
            self.emit(part, page, sec, "table")

    def run(self):
        last_page = 1
        for item, _ in self.doc.iterate_items():   # body only: furniture excluded
            lab = label_of(item)
            page = page_of(item, last_page)
            last_page = page
            if lab in SKIP_LABELS:
                continue
            if lab in TABLE_LABELS:
                self.add_table(item, page)
                continue
            text = clean(getattr(item, "text", ""))
            if not text:
                continue
            if lab in HEADING_LABELS:
                self.set_heading(text, getattr(item, "level", 1))
            else:
                self.add_text(text, page)
        self.flush_text()
        return self.chunks


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/parsed/docling")
    ap.add_argument("--out", default="data/chunks/docling_chunks.jsonl")
    ap.add_argument("--max-chars", type=int, default=1000)
    ap.add_argument("--overlap", type=int, default=150)
    ap.add_argument("--table-max-chars", type=int, default=1500)
    ap.add_argument("--companies", nargs="*", help="subset, e.g. itc hdfcbank")
    args = ap.parse_args()

    paths = sorted(Path(args.in_dir).glob("*.json"))
    if args.companies:
        paths = [p for p in paths if p.stem.split("_")[0] in set(args.companies)]
    if not paths:
        raise SystemExit(f"No JSON files found in {args.in_dir}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    print(f"{'company':<12}{'chunks':>8}{'text':>7}{'table':>7}{'tables':>8}"
          f"{'split':>7}{'1-col':>7}{'avg':>6}{'max':>6}")
    with open(out, "w", encoding="utf-8") as f:
        for p in paths:
            company = p.stem.split("_")[0]
            chunker = DocChunker(load_doc(p), company,
                                 args.max_chars, args.overlap, args.table_max_chars)
            chunks = chunker.run()
            for i, c in enumerate(chunks):
                f.write(json.dumps({"chunk_id": f"{company}_dl_{i:05d}", **c},
                                   ensure_ascii=False) + "\n")
            lens = [len(c["text"]) for c in chunks] or [0]
            n_tab = sum(c["type"] == "table" for c in chunks)
            print(f"{company:<12}{len(chunks):>8}{len(chunks) - n_tab:>7}{n_tab:>7}"
                  f"{chunker.n_tables:>8}{chunker.n_split:>7}{chunker.n_fake_tables:>7}"
                  f"{sum(lens) // len(lens):>6}{max(lens):>6}")
            total += len(chunks)
    print(f"\nTOTAL chunks: {total}  ->  {out}")


if __name__ == "__main__":
    main()