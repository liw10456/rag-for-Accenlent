"""Command-line entry point.

  python cli.py ask "What does NB-429 mean?"
  python cli.py ask "How do I roll back?" --mode bm25 --data ./my_docs
  python cli.py chat
"""
from __future__ import annotations

import argparse

from rag import RAG


def show(result: dict) -> None:
    print("\n" + result["answer"] + "\n")
    print("Sources:")
    for cid, score in result["sources"]:
        print(f"  {score:>8}  {cid}")


def main():
    ap = argparse.ArgumentParser(description="RAG from scratch")
    ap.add_argument("command", choices=["ask", "chat"])
    ap.add_argument("question", nargs="?")
    ap.add_argument("--data", default="data/sample")
    ap.add_argument("--mode", default="hybrid", choices=["dense", "bm25", "hybrid"])
    ap.add_argument("--embedder", default="auto", choices=["auto", "hashing", "st"])
    ap.add_argument("--chunking", default="paragraph", choices=["paragraph", "fixed"])
    ap.add_argument("--rerank", action="store_true", help="cross-encoder reranking (needs sentence-transformers)")
    ap.add_argument("-k", type=int, default=4)
    args = ap.parse_args()

    rag = RAG(embedder=args.embedder, chunking=args.chunking, rerank=args.rerank)
    n = rag.ingest_dir(args.data)
    print(f"Indexed {n} chunks from {args.data} (embedder: {rag.embedder.name})")

    if args.command == "ask":
        if not args.question:
            ap.error("ask needs a question")
        show(rag.ask(args.question, k=args.k, mode=args.mode))
    else:
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q:
                show(rag.ask(q, k=args.k, mode=args.mode))


if __name__ == "__main__":
    main()
