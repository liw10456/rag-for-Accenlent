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


def test_save_load_roundtrip(tmp_path):
    rag = RAG(embedder="hashing")
    rag.ingest_dir(DATA)
    rag.save(tmp_path / "idx")
    loaded = RAG.load(tmp_path / "idx")
    q = "What does NB-429 mean?"
    assert loaded.retrieve(q, k=1)[0][0].id == rag.retrieve(q, k=1)[0][0].id


def test_pdf_pages_are_cited(tmp_path):
    import pytest

    pytest.importorskip("pypdf")
    canvas = pytest.importorskip("reportlab.pdfgen.canvas")
    pdf = tmp_path / "lib" / "paper.pdf"
    pdf.parent.mkdir()
    c = canvas.Canvas(str(pdf))
    c.drawString(72, 720, "Introduction to the study design.")
    c.showPage()
    c.drawString(72, 720, "The sensor sampling rate was 200 Hz.")
    c.save()
    rag = RAG(embedder="hashing")
    rag.ingest_dir(pdf.parent)
    top = rag.retrieve("sensor sampling rate", k=1, mode="bm25")[0][0]
    assert top.metadata["page"] == 2 and top.id == "paper.pdf:p2#0"


def test_long_block_is_split():
    text = " ".join(f"w{i}" for i in range(600))  # one block, no blank lines
    assert len(paragraph_chunks(text, "t", max_words=250)) >= 3


def test_references_are_stripped_and_collections_filter(tmp_path):
    lib = tmp_path / "lib"
    (lib / "papers").mkdir(parents=True)
    (lib / "architecture").mkdir()
    (lib / "papers" / "a.md").write_text("# Methods\n\nWe used a 200 Hz pressure sensor.\n\nReferences\n\nSmith J. Pressure sensor arrays. 2019.")
    (lib / "papers" / "b.md").write_text("# Methods\n\nA 100 Hz pressure sensor was placed on the palate.")
    (lib / "architecture" / "fw.md").write_text("# Firmware\n\nThe pressure sensor is read over I2C by the MCU.")
    rag = RAG(embedder="hashing")
    rag.ingest_dir(lib)
    assert rag.collections() == ["architecture", "papers"]
    assert not any("Smith" in c.text for c in rag.store.chunks)
    hits = rag.retrieve("pressure sensor", k=5, where={"collection": "papers"})
    assert hits and all(c.metadata["collection"] == "papers" for c, _ in hits)
    arch = rag.retrieve("pressure sensor", k=5, mode="bm25", where={"collection": "architecture"})
    assert [c.source for c, _ in arch] == ["fw.md"]


def test_max_per_source_diversifies():
    rag = RAG(embedder="hashing")
    rag.ingest_dir(DATA)
    hits = rag.retrieve("error codes plans billing", k=4, max_per_source=1)
    assert len({c.source for c, _ in hits}) == len(hits)


def test_front_matter_detection():
    from rag.loaders import is_front_matter

    copyright_page = "\n".join([
        "ISSN 0302-9743 ISSN 1611-3349 (electronic)",
        "ISBN 978-3-030-58795-6 ISBN 978-3-030-58796-3 (eBook)",
        "(c) Springer Nature Switzerland AG 2020",
        "This work is subject to copyright. All rights are reserved by the Publisher.",
        "Neither the publisher nor the authors or the editors give a warranty.",
    ]).replace("(c)", "©")
    toc_page = "\n".join(["Contents"] + [
        f"{100 + 9 * i} Jane Doe and John Roe Tongue Interfaces for Assistive Devices Study {i}" for i in range(8)
    ])
    first_page = "\n".join([
        "GLOS: GLOve for Speech Recognition",
        "A. Author, Department of Electrical Engineering, Example University",
        "Abstract - We present a wearable glove that recognizes speech gestures.",
        "I. Introduction",
        "Speech impairments affect millions of people worldwide.",
        "© Copyright 2024 IEEE - All rights reserved.",
    ])
    table_page = "\n".join(["Table 2. Results"] + [f"{i} 200 Hz 0.{i}3 ms" for i in range(1, 9)])
    assert is_front_matter(copyright_page, 4)
    assert is_front_matter(toc_page, 12)
    assert not is_front_matter(first_page, 1)
    assert not is_front_matter(table_page, 5)
    assert not is_front_matter(copyright_page, 120)  # deep in a book: not front matter


def test_file_filter_substring():
    rag = RAG(embedder="hashing")
    rag.ingest_dir(DATA)
    hits = rag.retrieve("what does it cost", k=3, where={"file": "BILLING"})
    assert hits and all(c.source == "nimbus_billing.md" for c, _ in hits)
