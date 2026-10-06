"""End-to-end RAG pipeline: ingest -> retrieve (dense / bm25 / hybrid [+ rerank]) -> generate."""
from __future__ import annotations

from pathlib import Path

from .bm25 import BM25
from .chunking import Chunk, fixed_size_chunks, paragraph_chunks
from .embeddings import get_embedder
from .generate import generate
from .retrieval import CrossEncoderReranker, reciprocal_rank_fusion
from .vector_store import VectorStore


class RAG:
    def __init__(self, embedder="auto", chunking: str = "paragraph", rerank: bool = False):
        self.embedder = get_embedder(embedder) if isinstance(embedder, str) else embedder
        self.chunking = chunking
        self.store = VectorStore()
        self.bm25: BM25 | None = None
        self.reranker = CrossEncoderReranker() if rerank else None

    # ---------- ingestion ----------
    def chunk(self, text: str, source: str) -> list[Chunk]:
        if self.chunking == "fixed":
            return fixed_size_chunks(text, source, size=120, overlap=30)
        return paragraph_chunks(text, source)

    def ingest_dir(self, folder: str | Path) -> int:
        chunks: list[Chunk] = []
        for path in sorted(Path(folder).glob("**/*")):
            if path.suffix.lower() in {".md", ".txt"}:
                chunks += self.chunk(path.read_text(encoding="utf-8"), path.name)
        self.store.add(chunks, self.embedder.embed([c.text for c in chunks]))
        self.bm25 = BM25(self.store.chunks)
        return len(chunks)

    # ---------- retrieval ----------
    def retrieve(self, query: str, k: int = 5, mode: str = "hybrid") -> list[tuple[Chunk, float]]:
        fetch = k * 4 if self.reranker else k
        qvec = self.embedder.embed([query])[0]
        if mode == "dense":
            results = self.store.search(qvec, fetch)
        elif mode == "bm25":
            results = self.bm25.search(query, fetch)
        elif mode == "hybrid":
            results = reciprocal_rank_fusion(
                [self.store.search(qvec, fetch * 2), self.bm25.search(query, fetch * 2)]
            )[:fetch]
        else:
            raise ValueError(f"unknown mode {mode}")
        if self.reranker:
            results = self.reranker.rerank(query, results, k)
        return results[:k]

    # ---------- generation ----------
    def ask(self, question: str, k: int = 4, mode: str = "hybrid") -> dict:
        hits = self.retrieve(question, k, mode)
        answer = generate(question, [c for c, _ in hits])
        return {"answer": answer, "sources": [(c.id, round(s, 4)) for c, s in hits]}
