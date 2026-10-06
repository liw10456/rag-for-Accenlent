"""Document loaders: turn files in a folder into (text, source, metadata) records.

- PDFs are loaded page by page so every chunk can cite the page it came from.
- The first subfolder becomes a `collection` (e.g. library/papers/x.pdf -> "papers"),
  so queries can be restricted to one collection.
- Reference / bibliography sections are dropped by default: they are dense with
  other papers' titles and keywords and pollute retrieval.
- Front matter in book-style PDFs (copyright pages, tables of contents) is dropped
  by default for the same reason. Found on real proceedings volumes, where the
  copyright page won every "who are the authors" query.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

TEXT_EXTS = {".md", ".txt"}
PDF_EXTS = {".pdf"}
SUPPORTED = TEXT_EXTS | PDF_EXTS

REF_HEADING = re.compile(r"^\s*(?:\d+\.?\s*)?(references|bibliography|works cited|literature cited)\s*$", re.I | re.M)


COPYRIGHT_MARKERS = re.compile(
    r"all rights reserved|copyright|\u00a9|\bisbn\b|\bissn\b|library of congress|printed in|"
    r"no warranty|without warranty|give a warranty|trademarks?\b|the publisher",
    re.I,
)
CONTENTS_HEADING = re.compile(r"^\s*(table of )?contents\s*$", re.I | re.M)
PAGE_NUMBER_LINE = re.compile(r"^\s*\d{1,4}\s+\S|\S[\s.]{2,}\d{1,4}\s*$")


def is_front_matter(text: str, page_no: int) -> bool:
    """Heuristic: copyright pages and tables of contents near the start of a PDF.

    Counts marker *lines*, not marker words, so a paper's first page with a one-line
    "(c) 2024 IEEE. All rights reserved." footer is kept, while a book's copyright
    page (ISBN, ISSN, (c), warranty, publisher ... on separate lines) is dropped.
    """
    if page_no > 40:
        return False
    lines = [l for l in text.splitlines() if l.strip()]
    if sum(bool(COPYRIGHT_MARKERS.search(l)) for l in lines) >= 3:
        return True
    if len(lines) < 5:
        return False
    # table-of-contents entries: a page number plus a title-like run of words
    entries = sum(
        bool(PAGE_NUMBER_LINE.search(l)) and len(re.findall(r"[A-Za-z]{2,}", l)) >= 4 for l in lines
    ) / len(lines)
    return (bool(CONTENTS_HEADING.search(text)) and entries >= 0.3) or entries >= 0.6


@dataclass
class Document:
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


def strip_references(text: str) -> tuple[str, bool]:
    """Cut text at a standalone 'References' heading. Returns (text, found)."""
    m = REF_HEADING.search(text)
    return (text[: m.start()], True) if m else (text, False)


def _load_pdf(path: Path, meta: dict, keep_references: bool, keep_front_matter: bool) -> list[Document]:
    try:
        from pypdf import PdfReader  # optional dependency
    except ImportError as e:
        raise ImportError("Reading PDFs needs pypdf:  python -m pip install pypdf") from e
    docs = []
    for page_no, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = (page.extract_text() or "").strip()
        if not keep_front_matter and is_front_matter(text, page_no):
            continue
        in_refs = False
        if not keep_references:
            text, in_refs = strip_references(text)
            text = text.strip()
        if text:  # scanned pages with no text layer come back empty
            docs.append(Document(text, path.name, {**meta, "page": page_no}))
        if in_refs:
            break  # everything after the references heading is skipped
    return docs


def load_folder(folder: str | Path, keep_references: bool = False, keep_front_matter: bool = False) -> list[Document]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"No such folder: {folder}")
    docs: list[Document] = []
    for path in sorted(folder.glob("**/*")):
        rel = path.relative_to(folder)
        if any(part.startswith(".") for part in rel.parts) or not path.is_file():
            continue
        meta = {"collection": rel.parts[0]} if len(rel.parts) > 1 else {}
        ext = path.suffix.lower()
        if ext in TEXT_EXTS:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if not keep_references:
                text = strip_references(text)[0]
            docs.append(Document(text, path.name, meta))
        elif ext in PDF_EXTS:
            docs += _load_pdf(path, meta, keep_references, keep_front_matter)
    return docs
