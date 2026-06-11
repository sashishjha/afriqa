#!/usr/bin/env python3
"""
Experiment 3 — RAG Baseline
============================
Question + Retrieved Examples  →  mT5-base  →  Answer

Steps:
  1. Build best retriever from Exp 2 (default: Hybrid-LaBSE α=0.5)
  2. Load trained mT5 from Exp 1 (or train from scratch)
  3. Inference via RAGPipeline
  4. Evaluate (global + per-subset)
  5. Generate submission
"""

import os
import sys
import argparse
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import set_seed, get_device, load_config, save_json, setup_logging
from src.data_loader import load_data, create_seq2seq_dataset
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.training.trainer import setup_seq2seq_model, load_trained_model, train_seq2seq
from src.rag.pipeline import RAGPipeline
from src.evaluation.evaluator import Evaluator


def parse_args():
    p = argparse.ArgumentParser(description="Experiment 3 – RAG Baseline")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--output_dir", type=str, default="outputs/exp3_rag_baseline")
    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--exp1_model_dir", type=str, default="outputs/exp1_generation/best_model",
                   help="Path to trained Exp 1 model. If not found, trains from scratch.")
    p.add_argument("--retriever_type", type=str, default="hybrid",
                   choices=["bm25", "dense", "hybrid"])
    p.add_argument("--dense_model", type=str, default=None)
    p.add_argument("--hybrid_alpha", type=float, default=0.5)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--skip_eval", action="store_true")
    p.add_argument("--skip_submission", action="store_true")
    return p.parse_args()


def build_retriever(args, cfg, train_df, logger, device="cpu"):
    """Construct the requested retriever type."""
    corpus_q = train_df["input"].tolist()
    corpus_a = train_df["output"].tolist()
    dense_model_name = args.dense_model or cfg["retrieval"]["dense_model"]

    bm25_ret, dense_ret = None, None

    if args.retriever_type in ("bm25", "hybrid"):
        logger.info("Building BM25 index …")
        bm25_ret = BM25Retriever(
            k1=cfg["retrieval"]["bm25_k1"],
            b=cfg["retrieval"]["bm25_b"],
        )
        bm25_ret.build_index(corpus_q, corpus_a)

    if args.retriever_type in ("dense", "hybrid"):
        logger.info(f"Building Dense index ({dense_model_name}) …")
        dense_ret = DenseRetriever(model_name=dense_model_name, device=device)
        dense_ret.build_index(corpus_q, corpus_a,
                              batch_size=cfg["retrieval"]["dense_batch_size"])

    if args.retriever_type == "bm25":
        return bm25_ret
    if args.retriever_type == "dense":
        return dense_ret

    logger.info(f"Creating Hybrid retriever (α={args.hybrid_alpha}) …")
    return HybridRetriever(bm25_ret, dense_ret, alpha=args.hybrid_alpha)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger = setup_logging()
    set_seed(cfg["training"]["seed"])
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    model_name = args.model_name or cfg["model"]["default"]

    # ── data ────────────────────────────────────────────────────────
    logger.info("Loading data …")
    train_df = load_data(cfg["data"]["train_path"])
    val_df = load_data(cfg["data"]["val_path"])
    test_df = load_data(cfg["data"]["test_path"])

    # ── retriever ───────────────────────────────────────────────────
    retriever = build_retriever(args, cfg, train_df, logger, device=str(device))

    # ── generator ───────────────────────────────────────────────────
    if os.path.isdir(args.exp1_model_dir):
        logger.info(f"Loading trained generator from {args.exp1_model_dir}")
        model, tokenizer = load_trained_model(
            model_name, args.exp1_model_dir, model_type="seq2seq",
        )
    else:
        logger.info("No pre-trained generator found — training from scratch …")
        model, tokenizer = setup_seq2seq_model(
            model_name,
            lora_r=cfg["lora"]["r"],
            lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"]["dropout"],
        )
        train_ds = create_seq2seq_dataset(train_df, tokenizer,
                                          cfg["training"]["max_input_length"],
                                          cfg["training"]["max_target_length"])
        val_ds = create_seq2seq_dataset(val_df, tokenizer,
                                        cfg["training"]["max_input_length"],
                                        cfg["training"]["max_target_length"])
        trainer = train_seq2seq(
            model=model, tokenizer=tokenizer,
            train_ds=train_ds, val_ds=val_ds,
            output_dir=os.path.join(args.output_dir, "generator"),
            batch_size=args.batch_size,
            gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
            learning_rate=cfg["training"]["learning_rate"],
            num_epochs=cfg["training"]["num_epochs"],
            warmup_ratio=cfg["training"]["warmup_ratio"],
            weight_decay=cfg["training"]["weight_decay"],
            fp16=cfg["training"]["fp16"],
            logging_steps=cfg["training"]["logging_steps"],
            save_steps=cfg["training"]["save_steps"],
            eval_steps=cfg["training"]["eval_steps"],
            gen_max_length=cfg["generation"]["max_new_tokens"],
            num_beams=cfg["generation"]["num_beams"],
        )
        model = trainer.model

    model = model.to(device)

    # ── RAG pipeline ────────────────────────────────────────────────
    gen_kw = {
        "max_new_tokens": cfg["generation"]["max_new_tokens"],
        "num_beams": cfg["generation"]["num_beams"],
        "length_penalty": cfg["generation"]["length_penalty"],
        "no_repeat_ngram_size": cfg["generation"]["no_repeat_ngram_size"],
        "early_stopping": True,
    }
    rag = RAGPipeline(
        retriever=retriever,
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=args.top_k,
        gen_kwargs=gen_kw,
    )

    # ── evaluate ────────────────────────────────────────────────────
    if not args.skip_eval:
        logger.info("RAG inference on validation set …")
        val_preds = rag.answer_batch(val_df["input"].tolist(), batch_size=args.batch_size)

        evaluator = Evaluator(bertscore_model=cfg["evaluation"]["bertscore_model"])
        per_subset = evaluator.evaluate_per_subset(
            val_preds, val_df["output"].tolist(), val_df["subset"].tolist(),
        )
        evaluator.print_report(per_subset)

        report_path = os.path.join(args.output_dir, "val_results.json")
        evaluator.save_report(per_subset, report_path)
        logger.info(f"Results saved → {report_path}")

        # compare vs exp1 if available
        exp1_report_path = "outputs/exp1_generation/val_results.json"
        if os.path.exists(exp1_report_path):
            import json
            with open(exp1_report_path) as f:
                exp1 = json.load(f)
            logger.info("\n--- RAG vs Pure Generation (Exp1) ---")
            g1 = exp1.get("__global__", {})
            g3 = per_subset.get("__global__", {})
            for k in ("rouge1", "rougeL"):
                v1 = g1.get(k, 0)
                v3 = g3.get(k, 0)
                delta = v3 - v1
                logger.info(f"  {k}: Exp1={v1:.4f}  Exp3={v3:.4f}  Δ={delta:+.4f}")

    # ── submission ──────────────────────────────────────────────────
    if not args.skip_submission:
        logger.info("RAG inference on test set …")
        test_preds = rag.answer_batch(test_df["input"].tolist(), batch_size=args.batch_size)
        sub = pd.DataFrame({"ID": test_df["ID"], "output": test_preds})
        sub_path = os.path.join("submissions", "exp3_submission.csv")
        os.makedirs("submissions", exist_ok=True)
        sub.to_csv(sub_path, index=False)
        logger.info(f"Submission saved → {sub_path}  ({len(sub)} rows)")


if __name__ == "__main__":
    main()
