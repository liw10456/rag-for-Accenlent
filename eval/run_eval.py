"""Retrieval evaluation on a small golden set.

A retrieved chunk counts as relevant if it contains the `answer_contains` string,
i.e. it actually holds the fact needed to answer (chunk-level, stricter than
document-level relevance).

Metrics:
- Hit@k : fraction of questions with at least one relevant chunk in the top k
- MRR   : mean of 1/rank of the first relevant chunk (0 if none in top 10)

Usage:  python -m eval.run_eval [--embedder hashing|st]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag import RAG

ROOT = Path(__file__).resolve().parent.parent


def load_golden():
    return [json.loads(l) for l in (ROOT / "eval" / "golden.jsonl").read_text().splitlines() if l.strip()]


def evaluate(rag: RAG, mode: str, golden: list[dict], max_k: int = 10) -> dict:
    hit1 = hit3 = rr = 0.0
    misses = []
    for g in golden:
        hits = rag.retrieve(g["question"], k=max_k, mode=mode)
        needle = g["answer_contains"].lower()
        rank = next((i for i, (c, _) in enumerate(hits, 1) if needle in c.text.lower()), None)
        if rank:
            rr += 1 / rank
            hit1 += rank == 1
            hit3 += rank <= 3
        if rank != 1:
            misses.append((g["question"], rank))
    n = len(golden)
    return {"hit@1": hit1 / n, "hit@3": hit3 / n, "mrr": rr / n, "misses": misses}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="hashing", choices=["hashing", "st", "auto"])
    ap.add_argument("--verbose", action="store_true", help="print questions missed at rank 1")
    args = ap.parse_args()

    golden = load_golden()
    print(f"Golden set: {len(golden)} questions | embedder: {args.embedder}\n")
    print("| chunking  | retrieval | Hit@1 | Hit@3 | MRR  |")
    print("|-----------|-----------|-------|-------|------|")
    for chunking in ("fixed", "paragraph"):
        rag = RAG(embedder=args.embedder, chunking=chunking)
        rag.ingest_dir(ROOT / "data" / "sample")
        for mode in ("dense", "bm25", "hybrid"):
            r = evaluate(rag, mode, golden)
            print(f"| {chunking:<9} | {mode:<9} | {r['hit@1']:.2f}  | {r['hit@3']:.2f}  | {r['mrr']:.2f} |")
            if args.verbose:
                for q, rank in r["misses"]:
                    print(f"|   miss: {q} -> rank {rank}")


if __name__ == "__main__":
    main()
