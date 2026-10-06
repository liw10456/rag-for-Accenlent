"""BM25 keyword retrieval, implemented from scratch.

Why keep keyword search when we have embeddings? Dense retrieval is weak on exact
tokens: error codes, product SKUs, names, acronyms. BM25 nails those.

score(q, d) = sum_t IDF(t) * tf(t,d) * (k1+1) / (tf(t,d) + k1 * (1 - b + b*|d|/avgdl))
"""
from __future__ import annotations

import math
from collections import Counter

from .chunking import Chunk
from .embeddings import tokenize

STOPWORDS = set(
    "a an the of to in on for and or is are was were be by with as at it this that from "
    "how what which who when where why do does can i my you your we our".split()
)


def _terms(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(_terms(c.text)) for c in chunks]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avgdl = sum(self.lens) / max(len(self.lens), 1)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        q = _terms(query)
        scores = []
        for i, doc in enumerate(self.docs):
            s = 0.0
            for t in q:
                tf = doc.get(t, 0)
                if tf:
                    denom = tf + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl)
                    s += self.idf[t] * tf * (self.k1 + 1) / denom
            scores.append(s)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.chunks[i], scores[i]) for i in order if scores[i] > 0]
