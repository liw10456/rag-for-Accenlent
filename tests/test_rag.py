from pathlib import Path

import numpy as np

from rag import RAG
from rag.bm25 import BM25
from rag.chunking import Chunk, fixed_size_chunks, paragraph_chunks
from rag.embeddings import HashingEmbedder
from rag.retrieval import reciprocal_rank_fusion
from rag.vector_store import VectorStore

DATA = Path(__file__).resolve().parent.parent / "data" / "sample"


def test_fixed_chunks_overlap():
    text = " ".join(str(i) for i in range(100))
    chunks = fixed_size_chunks(text, "t", size=40, overlap=10)
    assert chunks[0].text.split()[-10:] == chunks[1].text.split()[:10]
    assert chunks[-1].text.split()[-1] == "99"


def test_paragraph_chunks_keep_section():
    chunks = paragraph_chunks("# A\n\nhello world\n\n# B\n\nsecond part", "t")
    assert [c.metadata["section"] for c in chunks] == ["A", "B"]


def test_embeddings_normalized_and_similar():
    e = HashingEmbedder()
    v = e.embed(["rate limit exceeded", "rate limits exceeded", "invoice pdf download"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-5)
    assert v[0] @ v[1] > v[0] @ v[2]


def test_bm25_exact_token():
    chunks = [Chunk("a", "error NB-429 rate limit", "s"), Chunk("b", "billing invoices", "s")]
    assert BM25(chunks).search("NB-429")[0][0].id == "a"


def test_rrf_rewards_agreement():
    a, b, c = (Chunk(i, i, "s") for i in "abc")
    fused = reciprocal_rank_fusion([[(a, 1), (b, 1)], [(b, 1), (c, 1)]])
    assert fused[0][0].id == "b"


def test_vector_store_roundtrip(tmp_path):
    e = HashingEmbedder()
    chunks = [Chunk("x", "hello", "s"), Chunk("y", "world", "s")]
    store = VectorStore()
    store.add(chunks, e.embed([c.text for c in chunks]))
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert loaded.search(e.embed(["hello"])[0], k=1)[0][0].id == "x"


def test_end_to_end():
    rag = RAG(embedder="hashing")
    rag.ingest_dir(DATA)
    top = rag.retrieve("What does NB-429 mean?", k=1)[0][0]
    assert "NB-429" in top.text
    assert "[1]" in rag.ask("What does NB-429 mean?")["answer"]
