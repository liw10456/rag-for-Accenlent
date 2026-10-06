# rag-from-local

[![tests](https://github.com/liw10456/rag-from-local/actions/workflows/tests.yml/badge.svg)](https://github.com/liw10456/rag-from-local/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**A retrieval-augmented generation (RAG) system built from first principles, and used as a private, local research library for papers, technical reports and architecture docs.**

No LangChain, no LlamaIndex: chunking, embeddings, vector search, BM25, hybrid fusion, reranking, grounded generation with citations, and evaluation are each implemented and measured directly, so every design choice is visible and testable. The core runs offline on numpy alone.

## Highlights

- **Hybrid retrieval** — dense embeddings + BM25 keyword search, fused with Reciprocal Rank Fusion, with optional cross-encoder reranking.
- **Measured, not guessed** — a golden-set benchmark (Hit@1, Hit@3, MRR) compares 2 chunking strategies × 3 retrievers × 2 embedders. Switching to semantic embeddings took the best configuration from **0.88 → 1.00 Hit@1**.
- **Built for real documents** — PDFs are read page by page with page-level citations; reference sections and book front matter (copyright and contents pages) are filtered out because they were measurably hijacking retrieval on real research PDFs.
- **Private by default** — embedding and retrieval run locally; personal libraries and indexes are gitignored. An LLM is optional and only ever sees the top-k retrieved passages.
- **Production habits** — 14 unit tests, GitHub Actions CI on Python 3.10–3.13, persisted indexes that record which embedding model built them.

## Demo

```
$ python cli.py ask "What happens if my credit card is declined?"

If a card payment fails, Nimbus retries the charge three times over seven days. [1]
...
Sources:
    0.0328  nimbus_billing.md#4
```

Point it at your own documents:

```
$ python cli.py index --data ~/library --index ~/.library-index --embedder st
Indexed 169 chunks ... Collections: architecture, papers, reports

$ python cli.py ask "Which sensors and sampling rates were used?" \
    --index ~/.library-index --only papers -k 8 --max-per-source 2
```

## Results

16-question golden set over the sample docs (`data/sample/`, a fictional cloud provider). A retrieved chunk counts as relevant only if it contains the fact needed to answer.

| Chunking  | Retrieval | Hashing embedder (offline)<br>Hit@1 / Hit@3 / MRR | MiniLM (semantic)<br>Hit@1 / Hit@3 / MRR |
|-----------|-----------|-----------------------|-----------------------|
| fixed     | dense     | 0.81 / 1.00 / 0.89    | 0.88 / 0.94 / 0.92    |
| fixed     | bm25      | 0.69 / 0.94 / 0.81    | 0.69 / 0.94 / 0.81    |
| fixed     | hybrid    | 0.81 / 1.00 / 0.90    | 0.88 / 1.00 / 0.94    |
| paragraph | dense     | 0.75 / 0.94 / 0.82    | **1.00 / 1.00 / 1.00** |
| paragraph | bm25      | 0.81 / 0.94 / 0.86    | 0.81 / 0.94 / 0.86    |
| paragraph | hybrid    | 0.88 / 0.94 / 0.90    | 0.94 / 1.00 / 0.97    |

BM25 rows are identical across embedders, as they should be — BM25 doesn't use embeddings — which doubles as a sanity check on the experiment.

### What the experiments showed

- **Vocabulary mismatch was the main failure, and semantic embeddings fixed it.** "Is SSO supported?" failed in every offline configuration because the document says "single sign-on through SAML". BM25 still misses it — keyword search can't bridge an acronym to its expansion — but MiniLM retrieves it at rank 1.
- **Hybrid is not automatically better.** With a weak embedder, fusing in BM25 helped because the two retrievers made different mistakes. With a strong embedder, dense alone was perfect and equal-weight fusion *lowered* Hit@1 (1.00 → 0.94): BM25's top pick for "Who is allowed to delete a project?" leaked through RRF. Next step: weighted fusion or a reranker as the final judge.
- **Chunking interacts with the embedder.** Structure-aware paragraph chunks gave the semantic model clean single-topic passages; fixed windows mixed sections and pushed one answer down to rank 7.
- **Small eval sets mislead in both directions.** One question is 6 points of Hit@1 here. A perfect score means the set is too small and easy, not that the system is finished — a larger set with paraphrased, multi-hop and unanswerable questions is on the roadmap.

## Real-world use: a private research library

I use the same pipeline on a local library of assistive-technology research papers and technical reports. Real PDFs surfaced problems the clean sample docs never did:

| Problem found on real PDFs | Fix |
|---|---|
| A proceedings volume's **copyright page** ranked #1 for "Which institution are the authors affiliated with?" — it is full of the words *authors*, *editors*, *publisher* | Front-matter filter: drops copyright and table-of-contents pages near the start of a PDF. It counts marker *lines*, so a paper's one-line "© IEEE. All rights reserved." footer doesn't get its first page (the one with the affiliations) thrown away |
| The **table of contents** matched "which school" through paper titles containing *Middle School* | Same filter: TOC entries are detected as page numbers plus title-like word runs, while numeric data tables are kept |
| **Reference lists** match almost any topical query | Bibliography sections are cut at the "References" heading |
| Cross-paper questions returned five chunks of the **same paper** | `--max-per-source` caps chunks per file so several studies get compared |
| "The authors" is ambiguous across a library | `--file` restricts a question to one document; `--only` to one collection (subfolder) |

## Architecture

```
            ┌──────────────────── ingestion ────────────────────┐
 files  ──►  loaders  ──►  chunking  ──►  embeddings  ──►  vector store
 .md .txt    (PDF pages,   (paragraph |    (hashing |       (cosine, metadata
 .pdf         refs + front  fixed+overlap)  MiniLM)          filters, save/load)
              matter cut)        └────────────────────────►  BM25 index

            ┌──────────────────── query time ───────────────────┐
 question ─►  dense top-k ──┐
                            ├─► RRF fusion ─► [cross-encoder rerank] ─► per-source cap ─► top-k
          ─►  BM25 top-k  ──┘                                                              │
                                                                                           ▼
                        prompt with numbered sources ─► LLM (optional) ─► answer with [1][2] citations
```

| Module | Responsibility |
|---|---|
| `rag/loaders.py` | `.md` / `.txt` / `.pdf` loading, page metadata, collections from subfolders, reference and front-matter filtering |
| `rag/chunking.py` | Fixed-size sliding window with overlap; structure-aware paragraph chunking that keeps section headings |
| `rag/embeddings.py` | Offline hashing embedder and sentence-transformers embedder behind one interface |
| `rag/vector_store.py` | Exact cosine search, metadata filtering, persistence |
| `rag/bm25.py` | BM25 implemented from the formula |
| `rag/retrieval.py` | Reciprocal Rank Fusion, cross-encoder reranking |
| `rag/generate.py` | Grounded prompt with citations, "I don't know" behavior, prompt-injection guard; Anthropic / OpenAI / local extractive fallback |
| `rag/pipeline.py` | Ingest → retrieve → generate, index save/load |
| `eval/run_eval.py` | Retrieval benchmark over the golden set |

## Quick start

```bash
git clone https://github.com/liw10456/rag-from-local.git
cd rag-from-local
python -m pip install -r requirements.txt

python cli.py ask "What does NB-429 mean?"     # sample docs, fully offline
python -m eval.run_eval --verbose              # retrieval benchmark
python -m pytest -q                            # tests
```

Optional upgrades:

```bash
python -m pip install pypdf sentence-transformers   # PDFs, semantic embeddings, --rerank
export ANTHROPIC_API_KEY=...                        # or OPENAI_API_KEY, for LLM-written answers
```

### Your own library

```
library/                 # anywhere on disk; gitignored if kept inside the repo
├── papers/              # each subfolder becomes a collection
├── reports/
└── architecture/
```

```bash
python cli.py index --data ~/library --index ~/.library-index --embedder st   # embed once
python cli.py chat  --index ~/.library-index                                  # then ask anything
```

| Flag | Effect |
|---|---|
| `--only papers` | search one collection |
| `--file glos` | search files whose name contains "glos" |
| `-k 8 --max-per-source 2` | wider retrieval, at most 2 chunks per file — good for "compare the studies" |
| `--mode dense \| bm25 \| hybrid` | choose the retriever |
| `--rerank` | cross-encoder reranking |
| `--keep-references`, `--keep-front-matter` | turn the filters off |

**Privacy:** loading, embedding and retrieval run locally. Only with an API key set are the question and the top-k retrieved passages sent to that provider; without one, answers are extracted locally.

## Design decisions

- **Exact search vs. ANN.** Brute-force cosine is exact and fast enough below ~100k vectors. At scale I'd move to an HNSW index (FAISS, pgvector, Qdrant), trading a little recall for large latency wins.
- **RRF over score blending.** BM25 scores and cosine similarities live on different scales; RRF uses ranks only, so it needs no per-corpus calibration.
- **Retrieve wide, rerank narrow.** The retriever fetches 4×k candidates and a cross-encoder picks the final k: more accurate than a bi-encoder, too slow to run over a whole corpus.
- **Grounding and safety.** The prompt requires citations, says "I don't know" when sources lack the answer, and treats retrieved text as data so instructions planted inside documents are ignored.
- **Index records its embedder.** Queries are always embedded with the model that built the index; mixing models silently breaks retrieval.
- **Metadata filters as a security hook.** The same `where` filter behind `--only` and `--file` is how per-user permissions would be enforced in a multi-tenant deployment.

## Roadmap

- [x] Semantic embeddings — fixed the vocabulary-mismatch failure
- [x] PDF ingestion with page citations, reference and front-matter filtering
- [x] Persisted indexes, collections, per-file filtering, source diversity
- [ ] Weighted RRF and reranker evaluation, so BM25 can't override a strong dense ranking
- [ ] Larger golden set: paraphrased, multi-hop and unanswerable questions; a golden set on real research PDFs
- [ ] Generation metrics: faithfulness and answer relevance (RAGAS or LLM-as-judge)
- [ ] LLM query rewriting (multi-query / HyDE)
- [ ] Incremental re-indexing and an ANN index with a latency benchmark

## License

MIT © WenChing Li
