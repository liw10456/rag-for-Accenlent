# RAG from Scratch

A retrieval-augmented generation (RAG) pipeline built from first principles — chunking, embeddings, vector search, BM25, hybrid fusion, reranking, grounded generation with citations, and retrieval evaluation — with no RAG framework (no LangChain / LlamaIndex) hiding the mechanics.

The core runs on **numpy only**, offline, with no API keys. Plug in real embeddings, a cross-encoder reranker, or an LLM by installing optional packages.

## Architecture

```
            ┌──────────── ingestion ────────────┐
 docs ──►  chunking  ──►  embeddings  ──►  vector store
 (.md/.txt) (paragraph |    (hashing |       (numpy cosine,
            fixed+overlap)   MiniLM)          metadata filter)
                   └──────────────────►  BM25 index

            ┌──────────── query time ───────────┐
 question ─►  dense top-k ─┐
                           ├─► RRF fusion ─► [cross-encoder rerank] ─► top-k chunks
          ─►  BM25 top-k  ─┘                                              │
                                                                          ▼
                         prompt with numbered sources ─► LLM ─► answer + [1][2] citations
```

| Module | What it does |
|---|---|
| `rag/chunking.py` | Fixed-size sliding window with overlap; structure-aware paragraph chunking that keeps the section heading as metadata |
| `rag/embeddings.py` | `HashingEmbedder` (offline, non-semantic baseline) and `SentenceTransformerEmbedder` (semantic) behind one interface |
| `rag/vector_store.py` | Exact cosine search, metadata filtering, save/load |
| `rag/bm25.py` | BM25 keyword retrieval implemented from the formula |
| `rag/retrieval.py` | Reciprocal Rank Fusion and optional cross-encoder reranking |
| `rag/generate.py` | Grounded prompt with citations, "I don't know" fallback, prompt-injection guard; Anthropic / OpenAI / extractive fallback |
| `eval/run_eval.py` | Hit@1, Hit@3, MRR over a golden question set, comparing chunking × retrieval strategies |

## Quick start

```bash
pip install -r requirements.txt
python cli.py ask "What does NB-429 mean?"
python cli.py chat                       # interactive
python -m eval.run_eval --verbose        # retrieval benchmark
pytest -q
```

Optional upgrades:

```bash
pip install sentence-transformers        # semantic embeddings + --rerank
export ANTHROPIC_API_KEY=...             # or OPENAI_API_KEY, for LLM-written answers
python cli.py ask "Is SSO supported?" --rerank
python -m eval.run_eval --embedder st
```

Point it at your own documents with `--data path/to/folder`.

## Results

16-question golden set over the sample docs (`data/sample/`, a fictional cloud provider). A retrieved chunk counts as relevant only if it contains the fact needed to answer. Embedder: offline hashing baseline.

| Chunking  | Retrieval | Hit@1 | Hit@3 | MRR  |
|-----------|-----------|-------|-------|------|
| fixed     | dense     | 0.81  | 1.00  | 0.89 |
| fixed     | bm25      | 0.69  | 0.94  | 0.81 |
| fixed     | hybrid    | 0.81  | 1.00  | 0.90 |
| paragraph | dense     | 0.75  | 0.94  | 0.82 |
| paragraph | bm25      | 0.81  | 0.94  | 0.86 |
| paragraph | **hybrid**| **0.88** | 0.94 | **0.90** |

### What I learned from the failures

- **Hybrid beats either retriever alone.** BM25 and dense make different mistakes, so fusing them with RRF gives the best top-1 accuracy — without having to calibrate their very different score scales.
- **Vocabulary mismatch is the main failure.** "Is SSO supported?" fails in every configuration because the doc says "single sign-on through SAML". Keyword search can't bridge an acronym to its expansion, and the hashing embedder isn't semantic. Fixes to try: real semantic embeddings, query rewriting / expansion with an LLM, or indexing a synonyms field.
- **Chunking interacts with retrieval.** Paragraph chunks help BM25 (a section's terms stay together) but slightly hurt the hashing embedder on short sections. There is no universally best chunk size; it has to be measured on your own data.
- **Measure before tuning.** Without the golden set, every one of these changes would have been a guess.

## Design decisions and trade-offs

- **Exact search vs. ANN.** Brute-force cosine is exact and fast enough below ~100k vectors. At scale I'd switch to an HNSW index (FAISS, pgvector, Qdrant), accepting a small recall loss for large latency gains.
- **RRF over score blending.** Weighted sums of BM25 and cosine scores need per-corpus normalization; RRF uses ranks only, so it's robust out of the box.
- **Retrieve wide, rerank narrow.** With `--rerank`, the retriever fetches 4×k candidates and a cross-encoder picks the final k. Cross-encoders read query and passage together, so they're more accurate but too slow to run over the whole corpus.
- **Grounding and safety.** The prompt requires citations, tells the model to say "I don't know" when sources lack the answer, and treats retrieved text as data so instructions planted in documents are ignored.
- **Metadata filtering.** `VectorStore.search(where=...)` restricts results by source or section — the hook you'd use for per-user permissions in a multi-tenant system.

## Next steps

- [ ] Semantic embeddings + cross-encoder results in the table above
- [ ] LLM query rewriting (multi-query / HyDE) to fix vocabulary mismatch
- [ ] Generation metrics: faithfulness and answer relevance (RAGAS or LLM-as-judge)
- [ ] Incremental re-indexing when documents change
- [ ] Persistent ANN index (pgvector or FAISS HNSW) and a latency benchmark

## License

MIT
