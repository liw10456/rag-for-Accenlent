"""Hybrid retrieval with Reciprocal Rank Fusion (RRF), plus optional reranking.

RRF merges ranked lists without needing to calibrate scores from different systems
(BM25 scores and cosine similarities live on different scales):

    rrf(d) = sum over lists  1 / (k + rank_in_list(d))
"""
from __future__ import annotations

from .chunking import Chunk


def reciprocal_rank_fusion(result_lists: list[list[tuple[Chunk, float]]], k: int = 60) -> list[tuple[Chunk, float]]:
    scores: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for results in result_lists:
        for rank, (chunk, _) in enumerate(results, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
            by_id[chunk.id] = chunk
    return sorted(((by_id[i], s) for i, s in scores.items()), key=lambda x: -x[1])


class CrossEncoderReranker:
    """Second-stage reranker. A bi-encoder (embeddings) scores query and doc
    separately; a cross-encoder reads them together, which is slower but much more
    accurate. Typical pattern: retrieve top-50 cheaply, rerank to top-5."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder  # optional dependency

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[tuple[Chunk, float]], k: int = 5) -> list[tuple[Chunk, float]]:
        if not results:
            return []
        scores = self.model.predict([(query, c.text) for c, _ in results])
        ranked = sorted(zip((c for c, _ in results), scores), key=lambda x: -x[1])
        return [(c, float(s)) for c, s in ranked[:k]]
