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

## Use it on your own documents: a private library

Built for reading research papers, technical reports and architecture docs. Drop `.md`, `.txt` or `.pdf` files into a folder, organized by subfolder:

```
library/                 # gitignored: never pushed to GitHub
├── papers/              # each subfolder becomes a "collection"
├── reports/
└── architecture/
```

```bash
python -m pip install pypdf sentence-transformers
python cli.py index --data library --index .index/library --embedder st      # embed once
python cli.py chat  --index .index/library                                   # ask anything

# cross-paper question: only papers, 8 chunks, at most 2 per paper
python cli.py ask "Which sensors and sampling rates were used?" \
    --index .index/library --only papers -k 8 --max-per-source 2

python cli.py ask "How does the firmware talk to the sensor?" --index .index/library --only architecture
```

What's tuned for papers and technical docs:

- **Page-level citations.** PDFs are read page by page, so every source reads like `paper.pdf:p7#0`.
- **References are dropped.** Bibliography sections are full of other papers' titles and keywords and would otherwise hijack retrieval. Use `--keep-references` to index them anyway.
- **Collections.** The first subfolder is stored as metadata; `--only` restricts search to it.
- **Source diversity.** `--max-per-source` caps chunks per file, so a "compare the studies" question sees several papers instead of five chunks of one.
- **Comparison-aware prompt.** With an LLM key set, the model is told to attribute findings to each paper and point out agreements, disagreements and method differences.
- **Saved index.** The index stores the embedder name, so queries always use the same model as the documents. Re-run `index` after adding files. Scanned PDFs without a text layer need OCR first.

**Privacy:** loading, embedding and retrieval run locally. Only if you set `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` are the question and the top-k retrieved chunks sent to that provider to write the answer; without a key, answers are extracted locally.

## Results

16-question golden set over the sample docs (`data/sample/`, a fictional cloud provider). A retrieved chunk counts as relevant only if it contains the fact needed to answer.

| Chunking  | Retrieval | Hashing embedder<br>Hit@1 / Hit@3 / MRR | MiniLM (semantic)<br>Hit@1 / Hit@3 / MRR |
|-----------|-----------|-----------------------|-----------------------|
| fixed     | dense     | 0.81 / 1.00 / 0.89    | 0.88 / 0.94 / 0.92    |
| fixed     | bm25      | 0.69 / 0.94 / 0.81    | 0.69 / 0.94 / 0.81    |
| fixed     | hybrid    | 0.81 / 1.00 / 0.90    | 0.88 / 1.00 / 0.94    |
| paragraph | dense     | 0.75 / 0.94 / 0.82    | **1.00 / 1.00 / 1.00** |
| paragraph | bm25      | 0.81 / 0.94 / 0.86    | 0.81 / 0.94 / 0.86    |
| paragraph | hybrid    | 0.88 / 0.94 / 0.90    | 0.94 / 1.00 / 0.97    |

BM25 rows are identical across embedders, as they should be: BM25 doesn't use embeddings. That doubles as a sanity check on the experiment.

### What I learned from the failures

- **Vocabulary mismatch was the main failure, and semantic embeddings fixed it.** "Is SSO supported?" failed in every hashing configuration because the doc says "single sign-on through SAML". BM25 still misses it with any embedder — keyword search can't bridge an acronym to its expansion — but MiniLM retrieves the right chunk at rank 1.
- **Hybrid is not automatically better.** With the weak hashing embedder, hybrid gave the best top-1 accuracy because the two retrievers made different mistakes. With a strong semantic embedder, paragraph-chunked dense retrieval alone was perfect, and fusing in BM25 *lowered* Hit@1 (1.00 → 0.94): BM25 ranked a different chunk first for "Who is allowed to delete a project?" and equal-weight RRF let that noise through. Fixes to try: weighted RRF, or a cross-encoder reranker as the final judge.
- **Chunking interacts with the embedder.** Structure-aware paragraph chunks gave the semantic model clean, single-topic passages (dense 0.88 → 1.00 Hit@1 vs. fixed-size), while fixed windows mixed sections — "How long are backups kept?" fell to rank 7. With the hashing embedder the effect went the other way. There is no universally best chunk size; it has to be measured.
- **Small eval sets mislead in both directions.** One question is 6 points of Hit@1 here, and a perfect 1.00 says more about the set being small and easy than about the system being done. Next: a larger, harder golden set (paraphrased questions, multi-hop, unanswerable questions).
- **Measure before tuning.** Without the golden set, every one of these conclusions — including "hybrid always helps" — would have been a guess, and one of them would have been wrong.

## Design decisions and trade-offs

- **Exact search vs. ANN.** Brute-force cosine is exact and fast enough below ~100k vectors. At scale I'd switch to an HNSW index (FAISS, pgvector, Qdrant), accepting a small recall loss for large latency gains.
- **RRF over score blending.** Weighted sums of BM25 and cosine scores need per-corpus normalization; RRF uses ranks only, so it's robust out of the box.
- **Retrieve wide, rerank narrow.** With `--rerank`, the retriever fetches 4×k candidates and a cross-encoder picks the final k. Cross-encoders read query and passage together, so they're more accurate but too slow to run over the whole corpus.
- **Grounding and safety.** The prompt requires citations, tells the model to say "I don't know" when sources lack the answer, and treats retrieved text as data so instructions planted in documents are ignored.
- **Metadata filtering.** `VectorStore.search(where=...)` restricts results by source or section — the hook you'd use for per-user permissions in a multi-tenant system.

## Next steps

- [x] Semantic embeddings (MiniLM) — fixed the vocabulary-mismatch failure
- [ ] Weighted RRF and cross-encoder reranking, so BM25 can't override a strong dense ranking
- [ ] Larger, harder golden set: paraphrases, multi-hop and unanswerable questions
- [ ] LLM query rewriting (multi-query / HyDE)
- [ ] Generation metrics: faithfulness and answer relevance (RAGAS or LLM-as-judge)
- [ ] Incremental re-indexing when documents change
- [ ] Persistent ANN index (pgvector or FAISS HNSW) and a latency benchmark

## License

MIT
