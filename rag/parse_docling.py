"""
Parse annual-report PDFs with Docling (layout + table structure, CPU, no OCR).

Outputs per PDF:
  <output>/<name>.json  -> lossless DoclingDocument (source of truth for chunking)
  <output>/<name>.md    -> human-readable export, for eyeballing tables only

Usage:
  # timing test on 20 pages of one report
  uv run python rag/parse_docling.py --input data/raw/hdfcbank_fy26.pdf \
      --output data/parsed/docling_test --pages 100-120

  # full run (resumable: finished PDFs are skipped)
  uv run python rag/parse_docling.py --input data/raw --output data/parsed/docling
"""

import argparse
import os
import time
from pathlib import Path

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption

# AcceleratorOptions moved modules across Docling versions
try:
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
except ImportError:
    from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions


def build_converter(num_threads: int) -> DocumentConverter:
    opts = PdfPipelineOptions()
    opts.do_ocr = False                      # corpus is digital; OCR is the slowest CPU step
    opts.do_table_structure = True           # rebuild real table grids
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    opts.accelerator_options = AcceleratorOptions(
        num_threads=num_threads, device=AcceleratorDevice.CPU
    )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def parse_page_range(s: str | None):
    if not s:
        return None
    start, end = s.split("-")
    return (int(start), int(end))  # 1-indexed, inclusive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="a PDF file or a folder of PDFs")
    ap.add_argument("--output", required=True, help="output folder")
    ap.add_argument("--pages", default=None, help="page range for testing, e.g. 100-120")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    args = ap.parse_args()

    inp = Path(args.input)
    pdfs = [inp] if inp.is_file() else sorted(inp.glob("*.pdf"))
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    page_range = parse_page_range(args.pages)

    print(f"{len(pdfs)} PDF(s) | threads={args.threads} | pages={args.pages or 'all'}", flush=True)
    converter = build_converter(args.threads)

    summary = []
    for pdf in pdfs:
        json_path = out_dir / f"{pdf.stem}.json"
        md_path = out_dir / f"{pdf.stem}.md"

        if json_path.exists():
            print(f"[skip] {pdf.name} (already parsed)", flush=True)
            continue

        print(f"[start] {pdf.name}", flush=True)
        t0 = time.perf_counter()
        try:
            kwargs = {"raises_on_error": False}
            if page_range:
                kwargs["page_range"] = page_range
            result = converter.convert(pdf, **kwargs)
        except Exception as e:  # unexpected crash: log and move on
            print(f"[FAIL] {pdf.name}: {type(e).__name__}: {e}", flush=True)
            summary.append((pdf.name, "CRASH", 0, 0, 0.0))
            continue
        secs = time.perf_counter() - t0

        status = result.status
        if status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
            print(f"[FAIL] {pdf.name}: status={status.name}", flush=True)
            for err in result.errors:
                print(f"    {err}", flush=True)
            summary.append((pdf.name, status.name, 0, 0, secs))
            continue
        if status == ConversionStatus.PARTIAL_SUCCESS:
            print(f"[warn] {pdf.name}: partial success", flush=True)
            for err in result.errors:
                print(f"    {err}", flush=True)

        doc = result.document
        doc.save_as_json(json_path)
        md_path.write_text(doc.export_to_markdown(), encoding="utf-8")

        n_pages = len(doc.pages)
        n_tables = len(doc.tables)
        spp = secs / max(n_pages, 1)
        print(
            f"[done] {pdf.name}: {n_pages} pages, {n_tables} tables, "
            f"{secs:.1f}s ({spp:.2f} s/page)",
            flush=True,
        )
        summary.append((pdf.name, status.name, n_pages, n_tables, secs))

    print("\n=== summary ===", flush=True)
    for name, st, p, t, s in summary:
        print(f"{name:28s} {st:16s} pages={p:4d} tables={t:4d} time={s:7.1f}s", flush=True)


if __name__ == "__main__":
    main()