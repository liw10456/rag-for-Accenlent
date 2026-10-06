"""Command-line entry point.

Sample docs (rebuilt in memory every run):
  python cli.py ask "What does NB-429 mean?"

Your own private library (embed once, query many times):
  python cli.py index --data ~/my-library --index .index/my-library
  python cli.py ask "What did the reviewers criticize?" --index .index/my-library
  python cli.py chat --index .index/my-library

Organize the library in subfolders (papers/, reports/, architecture/) and filter:
  python cli.py ask "Which sensors were used?" --index .index/my-library --only papers -k 8 --max-per-source 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rag import RAG


def show(result: dict) -> None:
    print("\n" + result["answer"] + "\n")
    print("Sources:")
    for cid, score in result["sources"]:
        print(f"  {score:>8}  {cid}")


def main():
    ap = argparse.ArgumentParser(description="RAG from scratch")
    ap.add_argument("command", choices=["index", "ask", "chat"])
    ap.add_argument("question", nargs="?")
    ap.add_argument("--data", default="data/sample", help="folder of .md / .txt / .pdf files")
    ap.add_argument("--index", help="saved index folder (written by `index`, read by `ask`/`chat`)")
    ap.add_argument("--mode", default="hybrid", choices=["dense", "bm25", "hybrid"])
    ap.add_argument("--embedder", default="auto", choices=["auto", "hashing", "st"])
    ap.add_argument("--chunking", default="paragraph", choices=["paragraph", "fixed"])
    ap.add_argument("--rerank", action="store_true", help="cross-encoder reranking (needs sentence-transformers)")
    ap.add_argument("-k", type=int, default=4, help="chunks to retrieve (use 8+ for cross-paper questions)")
    ap.add_argument("--only", help="search one collection (first subfolder of the library, e.g. papers)")
    ap.add_argument("--max-per-source", type=int, help="cap chunks per file so several papers are compared")
    ap.add_argument("--keep-references", action="store_true", help="index reference/bibliography sections too")
    args = ap.parse_args()

    if args.command == "index":
        if not args.index:
            ap.error("index needs --index <folder to save to>")
        rag = RAG(embedder=args.embedder, chunking=args.chunking)
        n = rag.ingest_dir(Path(args.data).expanduser(), keep_references=args.keep_references)
        rag.save(Path(args.index).expanduser())
        print(f"Indexed {n} chunks from {args.data} -> {args.index} (embedder: {rag.embedder.name})")
        if rag.collections():
            print("Collections:", ", ".join(rag.collections()))
        return

    if args.index:
        rag = RAG.load(Path(args.index).expanduser(), rerank=args.rerank)
        print(f"Loaded {len(rag.store.chunks)} chunks from {args.index} (embedder: {rag.embedder.name})")
    else:
        rag = RAG(embedder=args.embedder, chunking=args.chunking, rerank=args.rerank)
        n = rag.ingest_dir(Path(args.data).expanduser(), keep_references=args.keep_references)
        print(f"Indexed {n} chunks from {args.data} (embedder: {rag.embedder.name})")

    where = {"collection": args.only} if args.only else None
    opts = dict(k=args.k, mode=args.mode, where=where, max_per_source=args.max_per_source)

    if args.command == "ask":
        if not args.question:
            ap.error("ask needs a question")
        show(rag.ask(args.question, **opts))
    else:
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q:
                show(rag.ask(q, **opts))


if __name__ == "__main__":
    main()
