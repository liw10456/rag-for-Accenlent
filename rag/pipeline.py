"""End-to-end RAG pipeline: ingest -> retrieve (dense / bm25 / hybrid [+ rerank]) -> generate.

The index can be saved to disk, so a private library is embedded once and then
queried instantly.
"""
from __future__ import annotations

import json
from pathlib import Path

from .bm25 import BM25
from .chunking import Chunk, fixed_size_chunks, paragraph_chunks
from .embeddings import HashingEmbedder, SentenceTransformerEmbedder, get_embedder
from .generate import generate
from .loaders import load_folder
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
    def chunk(self, text: str, source: str, metadata: dict | None = None) -> list[Chunk]:
        metadata = metadata or {}
        if self.chunking == "fixed":
            chunks = fixed_size_chunks(text, source, size=120, overlap=30)
        else:
            chunks = paragraph_chunks(text, source)
        page = metadata.get("page")
        for i, c in enumerate(chunks):
            c.metadata = {**metadata, **{k: v for k, v in c.metadata.items() if v}}
            c.id = f"{source}:p{page}#{i}" if page else f"{source}#{i}"
        return chunks

    def ingest_dir(self, folder: str | Path, keep_references: bool = False) -> int:
        chunks: list[Chunk] = []
        for doc in load_folder(folder, keep_references=keep_references):
            chunks += self.chunk(doc.text, doc.source, doc.metadata)
        if not chunks:
            raise ValueError(f"No readable .md / .txt / .pdf content found in {folder}")
        self.store.add(chunks, self.embedder.embed([c.text for c in chunks]))
        self.bm25 = BM25(self.store.chunks)
        return len(chunks)

    # ---------- persistence ----------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        self.store.save(path)
        (path / "config.json").write_text(json.dumps({"embedder": self.embedder.name, "chunking": self.chunking}))

    @classmethod
    def load(cls, path: str | Path, rerank: bool = False) -> "RAG":
        path = Path(path)
        cfg = json.loads((path / "config.json").read_text())
        name = cfg["embedder"]
        # queries must be embedded with the same model the index was built with
        embedder = HashingEmbedder() if name == "hashing" else SentenceTransformerEmbedder(name)
        rag = cls(embedder=embedder, chunking=cfg["chunking"], rerank=rerank)
        rag.store = VectorStore.load(path)
        rag.bm25 = BM25(rag.store.chunks)
        return rag

    # ---------- retrieval ----------
    def retrieve(
        self,
        query: str,
        k: int = 5,
        mode: str = "hybrid",
        where: dict | None = None,
        max_per_source: int | None = None,
    ) -> list[tuple[Chunk, float]]:
        """where: metadata filter, e.g. {"collection": "papers"}.
        max_per_source: cap chunks per file, so cross-paper questions see several
        papers instead of five chunks of the same one."""
        fetch = k * 4 if (self.reranker or max_per_source) else k
        qvec = self.embedder.embed([query])[0]
        if mode == "dense":
            results = self.store.search(qvec, fetch, where)
        elif mode == "bm25":
            results = self.bm25.search(query, fetch, where)
        elif mode == "hybrid":
            results = reciprocal_rank_fusion(
                [self.store.search(qvec, fetch * 2, where), self.bm25.search(query, fetch * 2, where)]
            )[:fetch]
        else:
            raise ValueError(f"unknown mode {mode}")
        if self.reranker:
            results = self.reranker.rerank(query, results, len(results))
        if max_per_source:
            seen: dict[str, int] = {}
            diverse = []
            for c, s in results:
                if seen.get(c.source, 0) < max_per_source:
                    seen[c.source] = seen.get(c.source, 0) + 1
                    diverse.append((c, s))
            results = diverse
        return results[:k]

    def collections(self) -> list[str]:
        return sorted({c.metadata["collection"] for c in self.store.chunks if c.metadata.get("collection")})

    # ---------- generation ----------
    def ask(self, question: str, k: int = 4, mode: str = "hybrid", where: dict | None = None,
            max_per_source: int | None = None) -> dict:
        hits = self.retrieve(question, k, mode, where, max_per_source)
        answer = generate(question, [c for c, _ in hits])
        return {"answer": answer, "sources": [(c.id, round(s, 4)) for c, s in hits]}
