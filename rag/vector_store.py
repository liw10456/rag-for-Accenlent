"""A minimal vector store: brute-force cosine similarity with numpy.

Brute force is exact and fine up to ~100k vectors. Beyond that you'd use an ANN
index (HNSW via FAISS / pgvector / Qdrant), trading a little recall for big
latency wins. That trade-off is a common interview question.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .chunking import Chunk


class VectorStore:
    def __init__(self):
        self.vectors: np.ndarray | None = None
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        assert len(chunks) == len(vectors)
        self.chunks.extend(chunks)
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])

    def search(self, query_vec: np.ndarray, k: int = 5, where: dict | None = None) -> list[tuple[Chunk, float]]:
        if self.vectors is None:
            return []
        # vectors are L2-normalized -> dot = cosine. errstate silences spurious FP warnings
        # that numpy + Apple Accelerate BLAS can raise on macOS; the results are unaffected.
        with np.errstate(all="ignore"):
            scores = self.vectors @ query_vec
        order = np.argsort(-scores)
        results = []
        for i in order:
            c = self.chunks[i]
            if where and any(c.metadata.get(key, c.source if key == "source" else None) != val for key, val in where.items()):
                continue  # metadata filtering (e.g. per-tenant / per-permission)
            results.append((c, float(scores[i])))
            if len(results) == k:
                break
        return results

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        (path / "chunks.json").write_text(
            json.dumps([c.__dict__ for c in self.chunks], ensure_ascii=False, indent=1)
        )

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        path = Path(path)
        store = cls()
        store.vectors = np.load(path / "vectors.npy")
        store.chunks = [Chunk(**d) for d in json.loads((path / "chunks.json").read_text())]
        return store
