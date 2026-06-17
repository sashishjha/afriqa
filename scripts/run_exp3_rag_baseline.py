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
from src.retrieval.tfidf_retriever import TFIDFRetriever
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
                   choices=["bm25", "dense", "hybrid", "tfidf"])
    p.add_argument("--per_subset_retrieval", action="store_true",
                   help="Build separate TF-IDF retrievers per language subset "
                        "(critical for Amharic Ge'ez script).")
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

    if args.retriever_type == "tfidf":
        logger.info("Building TF-IDF character n-gram index (competitor approach) …")
        tfidf_ret = TFIDFRetriever(
            analyzer=cfg["retrieval"].get("tfidf_analyzer", "char_wb"),
            ngram_range=(
                cfg["retrieval"].get("tfidf_ngram_min", 2),
                cfg["retrieval"].get("tfidf_ngram_max", 5),
            ),
        )
        tfidf_ret.build_index(corpus_q, corpus_a)
        return tfidf_ret

    if args.retriever_type == "bm25":
        return bm25_ret
    if args.retriever_type == "dense":
        return dense_ret

    logger.info(f"Creating Hybrid retriever (α={args.hybrid_alpha}) …")
    return HybridRetriever(bm25_ret, dense_ret, alpha=args.hybrid_alpha)


from src.retrieval.tfidf_retriever import PerSubsetTFIDFRetriever

def build_per_subset_retriever(args, cfg, train_df, logger):
    """Build language-specific TF-IDF retrievers (one per subset)."""
    logger.info("Building PER-SUBSET TF-IDF retrievers (Amharic fix enabled)...")

    # Ensure 'subset' column exists in train_df
    if "subset" not in train_df.columns:
        logger.warning("No 'subset' column in training data — falling back to global retriever.")
        return None

    retriever = PerSubsetTFIDFRetriever(
        ngram_range=(2, 5),
        max_features=80000,
    )
    retriever.fit(
        questions=train_df["input"].tolist(),
        answers=train_df["output"].tolist(),
        subsets=train_df["subset"].tolist(),
    )
    return retriever


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

    # Preserve subset column for per-subset retrieval in case downstream code drops it
    val_subset_col = val_df["subset"].copy() if "subset" in val_df.columns else None
    test_subset_col = test_df["subset"].copy() if "subset" in test_df.columns else None

    # ── retriever ───────────────────────────────────────────────────
    if args.per_subset_retrieval and args.retriever_type == "tfidf":
        per_subset_ret = build_per_subset_retriever(args, cfg, train_df, logger)
        logger.info("✅ Per-subset TF-IDF retrieval enabled (Amharic fix active)")
        retriever = per_subset_ret
    else:
        # Original single-retriever path (now also benefits from Ge'ez transliteration)
        retriever = build_retriever(args, cfg, train_df, logger, device=str(device))
        logger.info("✅ Global retriever built (Ge'ez transliteration is automatic)")

    # Free GPU memory used by DenseRetriever's SentenceTransformer encoder.
    # The FAISS index is already on CPU; only the encoder model was on GPU.
    # We must free it before loading the mT5 generator onto the same GPU.
    if hasattr(retriever, 'model'):
        # DenseRetriever
        retriever.model = retriever.model.cpu()
    elif hasattr(retriever, 'dense') and retriever.dense is not None:
        # HybridRetriever wrapping a DenseRetriever
        retriever.dense.model = retriever.dense.model.cpu()
    torch.cuda.empty_cache()
    logger.info("Freed GPU memory from retriever encoder.")

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
        if val_subset_col is not None and "subset" not in val_df.columns:
            val_df["subset"] = val_subset_col.values
            logger.info("✅ 'subset' column preserved for per-subset retrieval")
        val_subsets = val_df["subset"].tolist() if args.per_subset_retrieval and args.retriever_type == "tfidf" else None
        val_preds = rag.answer_batch(val_df["input"].tolist(), batch_size=args.batch_size, subsets=val_subsets)

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
        if test_subset_col is not None and "subset" not in test_df.columns:
            test_df["subset"] = test_subset_col.values
            logger.info("✅ 'subset' column preserved for per-subset retrieval")
        test_subsets = test_df["subset"].tolist() if args.per_subset_retrieval and args.retriever_type == "tfidf" else None
        test_preds = rag.answer_batch(test_df["input"].tolist(), batch_size=args.batch_size, subsets=test_subsets)
        sub = pd.DataFrame({
            "ID": test_df["ID"],
            "TargetRLF1": test_preds,
            "TargetR1F1": test_preds,
            "TargetLLM": test_preds,
        })
        sub_name = os.path.basename(args.output_dir.rstrip("/")) + "_submission.csv"
        sub_path = os.path.join("submissions", sub_name)
        os.makedirs("submissions", exist_ok=True)
        sub.to_csv(sub_path, index=False)
        logger.info(f"Submission saved → {sub_path}  ({len(sub)} rows)")


if __name__ == "__main__":
    main()
