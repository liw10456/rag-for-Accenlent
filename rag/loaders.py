"""Document loaders: turn files in a folder into (text, source, metadata) records.

- PDFs are loaded page by page so every chunk can cite the page it came from.
- The first subfolder becomes a `collection` (e.g. library/papers/x.pdf -> "papers"),
  so queries can be restricted to one collection.
- Reference / bibliography sections are dropped by default: they are dense with
  other papers' titles and keywords and pollute retrieval.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

TEXT_EXTS = {".md", ".txt"}
PDF_EXTS = {".pdf"}
SUPPORTED = TEXT_EXTS | PDF_EXTS

REF_HEADING = re.compile(r"^\s*(?:\d+\.?\s*)?(references|bibliography|works cited|literature cited)\s*$", re.I | re.M)


@dataclass
class Document:
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


def strip_references(text: str) -> tuple[str, bool]:
    """Cut text at a standalone 'References' heading. Returns (text, found)."""
    m = REF_HEADING.search(text)
    return (text[: m.start()], True) if m else (text, False)


def _load_pdf(path: Path, meta: dict, keep_references: bool) -> list[Document]:
    try:
        from pypdf import PdfReader  # optional dependency
    except ImportError as e:
        raise ImportError("Reading PDFs needs pypdf:  python -m pip install pypdf") from e
    docs = []
    for page_no, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = (page.extract_text() or "").strip()
        in_refs = False
        if not keep_references:
            text, in_refs = strip_references(text)
            text = text.strip()
        if text:  # scanned pages with no text layer come back empty
            docs.append(Document(text, path.name, {**meta, "page": page_no}))
        if in_refs:
            break  # everything after the references heading is skipped
    return docs


def load_folder(folder: str | Path, keep_references: bool = False) -> list[Document]:
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
            docs += _load_pdf(path, meta, keep_references)
    return docs
