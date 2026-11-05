#!/usr/bin/env -S uv run python
"""PDF text-size auditor: flags text spans below the journal minimum.

Scans every text span in a PDF and reports those under a size threshold
(default 7pt — the safe line for print figures). Mirrors the behaviour of
nature-figure's audit_pdf_text.py: report-only by default, hard-fail with
--fail-below for CI pipelines.

Usage:
  uv run python audit_pdf.py figure.pdf
  uv run python audit_pdf.py figure.pdf --min-size 6 --fail-below
"""
import argparse
import sys

try:
    import fitz
except ImportError:
    print("audit_pdf.py requires PyMuPDF — install with: uv pip install pymupdf", file=sys.stderr)
    sys.exit(2)

MIN_SIZE = 7.0


def audit(pdf_path, min_size=MIN_SIZE):
    """Return [(page_no, size, text, bbox)] for spans below min_size."""
    offenders = []
    doc = fitz.open(pdf_path)
    try:
        for pno, page in enumerate(doc):
            d = page.get_text("rawdict")
            for block in d["blocks"]:
                if block["type"] != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        txt = "".join(ch["c"] for ch in span.get("chars", [])).strip()
                        if not txt:
                            continue
                        if span["size"] < min_size:
                            offenders.append((pno + 1, round(span["size"], 2), txt[:40],
                                              tuple(round(v, 1) for v in span["bbox"])))
    finally:
        doc.close()
    return offenders


def main():
    ap = argparse.ArgumentParser(description="Audit PDF text sizes against the journal minimum")
    ap.add_argument("pdf", help="Input PDF")
    ap.add_argument("--min-size", type=float, default=MIN_SIZE,
                    help=f"Minimum font size in pt (default: {MIN_SIZE})")
    ap.add_argument("--fail-below", action="store_true",
                    help="Exit code 1 if any span is below --min-size (CI mode)")
    ap.add_argument("--max-reports", type=int, default=20,
                    help="Max offenders to print (default: 20)")
    args = ap.parse_args()

    offenders = audit(args.pdf, args.min_size)
    if not offenders:
        print(f"OK: no text below {args.min_size}pt in {args.pdf}")
        sys.exit(0)

    print(f"FOUND {len(offenders)} text span(s) below {args.min_size}pt in {args.pdf}:")
    for pno, size, txt, bbox in offenders[: args.max_reports]:
        print(f"  p{pno} {size:5.2f}pt '{txt}' bbox={bbox}")
    if len(offenders) > args.max_reports:
        print(f"  ... and {len(offenders) - args.max_reports} more")
    sys.exit(1 if args.fail_below else 0)


if __name__ == "__main__":
    main()
