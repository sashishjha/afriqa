#!/usr/bin/env python3
"""
Experiment 2 — Retrieval Benchmark
====================================
Compare BM25 · Dense (LaBSE / MPNet) · Hybrid
Evaluate Recall@{1,3,5,10} globally, per-language, and per-subset.
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import set_seed, load_config, save_json, setup_logging, get_device
from src.data_loader import load_data, SUBSETS, SUBSET_TO_LANG
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


# ── recall computation ──────────────────────────────────────────────

def compute_recall_at_k(
    queries: list[str],
    gold_answers: list[str],
    retriever,
    ks: list[int] = (1, 3, 5, 10),
    batch: bool = False,
) -> dict[str, float]:
    """Recall = fraction of queries whose gold answer appears in top-K retrieved answers."""
    max_k = max(ks)

    if batch and hasattr(retriever, "batch_retrieve"):
        all_results = retriever.batch_retrieve(queries, top_k=max_k)
    else:
        all_results = [retriever.retrieve(q, top_k=max_k) for q in queries]

    recall = {f"recall@{k}": 0.0 for k in ks}
    for gold, results in zip(gold_answers, all_results):
        gold_norm = gold.strip().lower()
        for k in ks:
            topk_answers = [r["answer"].strip().lower() for r in results[:k]]
            if gold_norm in topk_answers:
                recall[f"recall@{k}"] += 1.0
    n = len(queries) if queries else 1
    return {key: val / n for key, val in recall.items()}


# ── evaluation harness ──────────────────────────────────────────────

def evaluate_retriever(
    name: str,
    retriever,
    val_df: pd.DataFrame,
    ks: list[int],
    logger,
) -> dict:
    """Run recall evaluation globally, per-subset, and per-language."""

    result = {"retriever": name}

    # global
    logger.info(f"  [{name}] Global evaluation …")
    t0 = time.time()
    g = compute_recall_at_k(
        val_df["input"].tolist(),
        val_df["output"].tolist(),
        retriever,
        ks=ks,
        batch=True,
    )
    elapsed = time.time() - t0
    g["latency_s"] = round(elapsed, 2)
    g["n"] = len(val_df)
    result["global"] = g
    logger.info(f"    global  {g}")

    # per-subset
    result["per_subset"] = {}
    for sub in sorted(val_df["subset"].unique()):
        sdf = val_df[val_df["subset"] == sub]
        m = compute_recall_at_k(
            sdf["input"].tolist(), sdf["output"].tolist(), retriever, ks=ks,
        )
        m["n"] = len(sdf)
        result["per_subset"][sub] = m

    # per-language
    result["per_language"] = {}
    for lang in sorted(set(SUBSET_TO_LANG.values())):
        subs = [s for s, l in SUBSET_TO_LANG.items() if l == lang]
        ldf = val_df[val_df["subset"].isin(subs)]
        if len(ldf) == 0:
            continue
        m = compute_recall_at_k(
            ldf["input"].tolist(), ldf["output"].tolist(), retriever, ks=ks,
        )
        m["n"] = len(ldf)
        result["per_language"][lang] = m

    return result


# ── main ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Experiment 2 – Retrieval Benchmark")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--output_dir", type=str, default="outputs/exp2_retrieval")
    p.add_argument("--dense_models", nargs="+",
                   default=[
                       "sentence-transformers/LaBSE",
                       "sentence-transformers/all-mpnet-base-v2",
                   ])
    p.add_argument("--hybrid_alphas", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    p.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10])
    p.add_argument("--skip_bm25", action="store_true")
    p.add_argument("--skip_dense", action="store_true")
    p.add_argument("--skip_hybrid", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger = setup_logging()
    set_seed(cfg["training"]["seed"])
    os.makedirs(args.output_dir, exist_ok=True)
    device = str(get_device())

    # ── data ────────────────────────────────────────────────────────
    logger.info("Loading data …")
    train_df = load_data(cfg["data"]["train_path"])
    val_df = load_data(cfg["data"]["val_path"])
    logger.info(f"Train : {len(train_df):,}  |  Val : {len(val_df):,}")

    corpus_q = train_df["input"].tolist()
    corpus_a = train_df["output"].tolist()

    all_reports = []

    # ── BM25 ────────────────────────────────────────────────────────
    bm25_ret = None
    if not args.skip_bm25:
        logger.info("Building BM25 index …")
        bm25_ret = BM25Retriever(
            k1=cfg["retrieval"]["bm25_k1"],
            b=cfg["retrieval"]["bm25_b"],
        )
        bm25_ret.build_index(corpus_q, corpus_a)
        report = evaluate_retriever("BM25", bm25_ret, val_df, args.ks, logger)
        all_reports.append(report)
        save_json(report, os.path.join(args.output_dir, "bm25_report.json"))

    # ── Dense ───────────────────────────────────────────────────────
    dense_retrievers = {}
    if not args.skip_dense:
        for dm in args.dense_models:
            short = dm.split("/")[-1]
            logger.info(f"Building Dense index [{short}] …")
            dr = DenseRetriever(model_name=dm, device=device)
            dr.build_index(corpus_q, corpus_a, batch_size=cfg["retrieval"]["dense_batch_size"])
            dense_retrievers[short] = dr
            report = evaluate_retriever(f"Dense-{short}", dr, val_df, args.ks, logger)
            all_reports.append(report)
            save_json(report, os.path.join(args.output_dir, f"dense_{short}_report.json"))

    # ── Hybrid ──────────────────────────────────────────────────────
    if not args.skip_hybrid:
        if bm25_ret is None:
            logger.info("Building BM25 for hybrid (was skipped earlier) …")
            bm25_ret = BM25Retriever(
                k1=cfg["retrieval"]["bm25_k1"],
                b=cfg["retrieval"]["bm25_b"],
            )
            bm25_ret.build_index(corpus_q, corpus_a)

        for dm_name, dr in dense_retrievers.items():
            for alpha in args.hybrid_alphas:
                tag = f"Hybrid-{dm_name}-a{alpha}"
                logger.info(f"Evaluating {tag} …")
                hr = HybridRetriever(bm25_ret, dr, alpha=alpha)
                report = evaluate_retriever(tag, hr, val_df, args.ks, logger)
                all_reports.append(report)
                safe_tag = tag.replace("/", "_")
                save_json(report, os.path.join(args.output_dir, f"{safe_tag}_report.json"))

    # ── summary table ───────────────────────────────────────────────
    logger.info("\n" + "=" * 90)
    logger.info("RETRIEVAL BENCHMARK SUMMARY")
    logger.info("=" * 90)
    header = f"{'Retriever':40s}"
    for k in args.ks:
        header += f"  R@{k:>2}"
    logger.info(header)
    logger.info("-" * 90)

    for r in all_reports:
        row = f"{r['retriever']:40s}"
        for k in args.ks:
            val = r["global"].get(f"recall@{k}", 0.0)
            row += f"  {val:.4f}"
        logger.info(row)

    logger.info("=" * 90)

    # save combined
    save_json(all_reports, os.path.join(args.output_dir, "all_reports.json"))
    logger.info(f"All reports saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
