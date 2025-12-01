import argparse
import os
import re
from pathlib import Path

try:
    from pypdf import PdfReader
except Exception as e:
    raise SystemExit("pypdf is required. Install with: python -m pip install --user pypdf")


def sanitize_filename(name: str) -> str:
    # Keep letters, numbers, dash, underscore; replace others with underscore
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    # Trim long names
    return base[:128]


def extract_pdf(pdf_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(pdf_path))
    texts = []
    for i, page in enumerate(reader.pages):
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    text = "\n\n".join(texts)

    out_name = sanitize_filename(pdf_path.stem) + ".txt"
    out_path = out_dir / out_name
    out_path.write_text(text, encoding="utf-8", errors="ignore")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Extract text from PDFs to .txt files")
    parser.add_argument("pdfs", nargs="+", help="Paths to PDF files")
    parser.add_argument("--out", default="pdf_extracted", help="Output directory for text files")
    args = parser.parse_args()

    out_dir = Path(args.out)
    for p in args.pdfs:
        pdf_path = Path(p)
        if not pdf_path.exists():
            print(f"[skip] Not found: {pdf_path}")
            continue
        try:
            out_path = extract_pdf(pdf_path, out_dir)
            print(f"[ok] {pdf_path} -> {out_path}")
        except Exception as e:
            print(f"[error] {pdf_path}: {e}")


if __name__ == "__main__":
    main()
