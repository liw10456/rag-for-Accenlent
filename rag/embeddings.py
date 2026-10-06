"""Embedding models.

Two implementations behind one interface:
- HashingEmbedder: pure numpy, no downloads. Hashes word unigrams/bigrams and
  character trigrams into a fixed-size vector. It is NOT semantic (it can't tell
  that "car" ~ "automobile"), but it lets the whole pipeline run offline.
- SentenceTransformerEmbedder: real semantic embeddings (e.g. all-MiniLM-L6-v2,
  bge-small-en). Install `sentence-transformers` to use it.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class HashingEmbedder:
    name = "hashing"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        toks = tokenize(text)
        feats = list(toks)
        feats += [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for t in toks:
            padded = f"#{t}#"
            feats += [f"c:{padded[i:i+3]}" for i in range(len(padded) - 2)]
        return feats

    def _hash(self, feat: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.md5(feat.encode()).digest()[:8], "little")
        return h % self.dim, 1.0 if (h >> 63) & 1 else -1.0

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for f in self._features(text):
                idx, sign = self._hash(f)
                out[row, idx] += sign
        out = np.sign(out) * np.log1p(np.abs(out))  # dampen frequent features
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.model = SentenceTransformer(model_name)
        self.name = model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )


def get_embedder(kind: str = "auto"):
    """'auto' uses sentence-transformers if installed, otherwise hashing."""
    if kind in ("auto", "st"):
        try:
            return SentenceTransformerEmbedder()
        except Exception:
            if kind == "st":
                raise
    return HashingEmbedder()
