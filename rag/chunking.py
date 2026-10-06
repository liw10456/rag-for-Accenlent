"""Chunking strategies.

Interview talking point: chunk size is a trade-off.
- Too large  -> the embedding averages many topics, retrieval gets fuzzy, prompts get expensive.
- Too small  -> each chunk lacks context, answers lose surrounding facts.
Overlap reduces the chance that a fact is split across a chunk boundary.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


def fixed_size_chunks(text: str, source: str, size: int = 400, overlap: int = 80) -> list[Chunk]:
    """Split by word count with a sliding window."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = text.split()
    chunks, start, i = [], 0, 0
    while start < len(words):
        piece = " ".join(words[start : start + size])
        chunks.append(Chunk(id=f"{source}#{i}", text=piece, source=source))
        if start + size >= len(words):
            break
        start += size - overlap
        i += 1
    return chunks


def paragraph_chunks(text: str, source: str, max_words: int = 250) -> list[Chunk]:
    """Respect document structure: split on headings / blank lines, then merge small
    paragraphs until max_words. Each chunk keeps the most recent markdown heading
    as metadata so the LLM knows which section the text came from."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    chunks: list[Chunk] = []
    buf: list[str] = []
    heading = ""
    buf_heading = ""

    def flush():
        if buf:
            idx = len(chunks)
            chunks.append(
                Chunk(
                    id=f"{source}#{idx}",
                    text="\n\n".join(buf),
                    source=source,
                    metadata={"section": buf_heading},
                )
            )
            buf.clear()

    for block in blocks:
        if block.startswith("#"):
            flush()
            heading = block.lstrip("#").strip()
            buf_heading = heading
            continue
        if not buf:
            buf_heading = heading
        if sum(len(b.split()) for b in buf) + len(block.split()) > max_words:
            flush()
            buf_heading = heading
        buf.append(block)
    flush()
    return chunks
