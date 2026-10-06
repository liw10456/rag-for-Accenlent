"""Metadata filters shared by the vector store and BM25.

where = {"collection": "papers"}     exact match on chunk metadata
where = {"source": "a.pdf"}          exact file name
where = {"file": "glos"}             case-insensitive substring of the file name
"""
from __future__ import annotations

from .chunking import Chunk


def matches(chunk: Chunk, where: dict | None) -> bool:
    if not where:
        return True
    for key, val in where.items():
        if key == "file":
            if str(val).lower() not in chunk.source.lower():
                return False
        elif key == "source":
            if chunk.source != val:
                return False
        elif chunk.metadata.get(key) != val:
            return False
    return True
