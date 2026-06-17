#!/usr/bin/env python3
"""
validate_local.py  —  Local Validation Harness
================================================
Evaluate predictions against ground truth (Val.csv) without burning
Zindi submissions. Computes per-subset ROUGE-1, ROUGE-L, and
(optionally) chrF++ for Amharic.

Usage:
    python validate_local.py \
        --predictions-path output/submission.csv \
        --val-path data/Val.csv

Author : Team Sashish (IISc Bengaluru)
"""

import argparse
import sys
import warnings

import numpy as np
import pandas as pd


def extract_subset(row_id: str) -> str:
    """Extract subset code from ID, e.g. 'ID_VL_Aka_Gha_XXX' -> 'Aka_Gha'."""
    parts = row_id.split("_")
    if len(parts) >= 4:
        return parts[2] + "_" + parts[3]
    return "Unknown"


def compute_rouge_scores(predictions: list, references: list):
    """
    Compute ROUGE-1 and ROUGE-L F1 scores.
    Returns (rouge1_scores, rougeL_scores) as lists of floats.
    """
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)

    rouge1_scores = []
    rougeL_scores = []

    for pred, ref in zip(predictions, references):
        pred_str = str(pred).strip() if pred and str(pred).strip().lower() != "nan" else ""
        ref_str  = str(ref).strip()  if ref  and str(ref).strip().lower() != "nan"  else ""

        if not ref_str:
            # Skip rows with empty references
            rouge1_scores.append(np.nan)
            rougeL_scores.append(np.nan)
            continue

        if not pred_str:
            rouge1_scores.append(0.0)
            rougeL_scores.append(0.0)
            continue

        scores = scorer.score(ref_str, pred_str)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    return rouge1_scores, rougeL_scores


def compute_chrf_scores(predictions: list, references: list):
    """
    Compute chrF++ scores (useful for Amharic where ROUGE fails).
    Returns list of chrF++ scores. Requires sacrebleu.
    """
    try:
        import sacrebleu
    except ImportError:
        return None

    scores = []
    for pred, ref in zip(predictions, references):
        pred_str = str(pred).strip() if pred else ""
        ref_str  = str(ref).strip()  if ref  else ""
        if not ref_str:
            scores.append(np.nan)
            continue
        if not pred_str:
            scores.append(0.0)
            continue
        chrf = sacrebleu.sentence_chrf(pred_str, [ref_str])
        scores.append(chrf.score / 100.0)  # Normalise to 0-1
    return scores


def main():
    parser = argparse.ArgumentParser(description="Local validation for Zindi Health QA")
    parser.add_argument("--predictions-path", type=str, required=True,
                        help="Path to predictions CSV (submission.csv format or debug CSV)")
    parser.add_argument("--val-path", type=str, required=True,
                        help="Path to Val.csv (ground truth)")
    parser.add_argument("--q-col", type=str, default="input",
                        help="Question column name in Val.csv")
    parser.add_argument("--a-col", type=str, default="output",
                        help="Answer column name in Val.csv")
    parser.add_argument("--pred-col", type=str, default="TargetR1F1",
                        help="Prediction column name in predictions CSV")
    parser.add_argument("--compute-chrf", action="store_true",
                        help="Also compute chrF++ (requires sacrebleu)")
    args = parser.parse_args()

    # Load data
    print(f"Loading predictions: {args.predictions_path}")
    pred_df = pd.read_csv(args.predictions_path)

    print(f"Loading ground truth: {args.val_path}")
    val_df = pd.read_csv(args.val_path)

    # Detect prediction column
    if args.pred_col in pred_df.columns:
        pred_col = args.pred_col
    elif "prediction" in pred_df.columns:
        pred_col = "prediction"
    elif "TargetRLF1" in pred_df.columns:
        pred_col = "TargetRLF1"
    else:
        print(f"ERROR: Cannot find prediction column. Available: {list(pred_df.columns)}")
        sys.exit(1)

    # Detect answer column
    a_col = args.a_col
    if a_col not in val_df.columns:
        for c in val_df.columns:
            if c.strip().lower() in ("output", "answer"):
                a_col = c
                break
        else:
            print(f"ERROR: Cannot find answer column. Available: {list(val_df.columns)}")
            sys.exit(1)

    # Merge on ID
    print(f"Prediction column: '{pred_col}' | Answer column: '{a_col}'")
    merged = val_df.merge(pred_df[["ID", pred_col]], on="ID", how="inner")
    print(f"Matched {len(merged)} rows (val={len(val_df)}, pred={len(pred_df)})")

    if len(merged) == 0:
        print("ERROR: No matching IDs found. Check ID formats.")
        sys.exit(1)

    # Extract subsets
    if "subset" not in merged.columns:
        merged["subset"] = merged["ID"].apply(extract_subset)

    # Compute ROUGE
    print("\nComputing ROUGE scores...")
    predictions = merged[pred_col].tolist()
    references  = merged[a_col].tolist()

    rouge1_scores, rougeL_scores = compute_rouge_scores(predictions, references)
    merged["rouge1"] = rouge1_scores
    merged["rougeL"] = rougeL_scores

    # Optionally compute chrF++
    chrf_scores = None
    if args.compute_chrf:
        print("Computing chrF++ scores...")
        chrf_scores = compute_chrf_scores(predictions, references)
        if chrf_scores is not None:
            merged["chrf"] = chrf_scores
        else:
            print("  WARNING: sacrebleu not installed. Skipping chrF++.")

    # ---- Per-Subset Results ----
    print("\n" + "=" * 85)
    print("  PER-SUBSET RESULTS")
    print("=" * 85)

    header = f"{'Subset':<12s} | {'Count':>5s} | {'ROUGE-1':>8s} | {'ROUGE-L':>8s} | {'Weighted':>8s}"
    if chrf_scores is not None:
        header += f" | {'chrF++':>8s}"
    print(header)
    print("-" * len(header))

    subset_results = {}
    for subset in sorted(merged["subset"].unique()):
        mask = merged["subset"] == subset
        sub = merged[mask]
        n = len(sub)
        r1 = sub["rouge1"].dropna().mean()
        rl = sub["rougeL"].dropna().mean()
        w  = 0.5 * r1 + 0.5 * rl

        line = f"{subset:<12s} | {n:5d} | {r1:8.4f} | {rl:8.4f} | {w:8.4f}"

        if chrf_scores is not None and "chrf" in sub.columns:
            chrf_mean = sub["chrf"].dropna().mean()
            line += f" | {chrf_mean:8.4f}"
        elif subset == "Amh_Eth":
            line += " |  ROUGE unreliable for Ge'ez script"

        print(line)
        subset_results[subset] = {"count": n, "rouge1": r1, "rougeL": rl, "weighted": w}

    # ---- Overall Results ----
    overall_r1 = merged["rouge1"].dropna().mean()
    overall_rl = merged["rougeL"].dropna().mean()

    # Competition metric: 0.37 * R1 + 0.37 * RL + 0.26 * LLM_Judge
    # LLM Judge is currently 0 for everyone
    llm_judge = 0.0
    competition_score = 0.37 * overall_r1 + 0.37 * overall_rl + 0.26 * llm_judge

    print("-" * 85)
    print(f"{'OVERALL':<12s} | {len(merged):5d} | {overall_r1:8.4f} | {overall_rl:8.4f} | {0.5*overall_r1+0.5*overall_rl:8.4f}")
    print()
    print("=" * 50)
    print("  COMPETITION METRIC ESTIMATE")
    print("=" * 50)
    print(f"  ROUGE-1 F1    : {overall_r1:.4f}  (weight 0.37)")
    print(f"  ROUGE-L F1    : {overall_rl:.4f}  (weight 0.37)")
    print(f"  LLM Judge     : {llm_judge:.4f}  (weight 0.26, currently deactivated)")
    print(f"  -----------------------------------------")
    print(f"  Estimated LB  : {competition_score:.4f}")
    print(f"  ROUGE-only LB : {0.37*overall_r1 + 0.37*overall_rl:.4f}")
    print()

    # ---- Warnings ----
    amh_mask = merged["subset"] == "Amh_Eth"
    if amh_mask.any():
        amh_r1 = merged.loc[amh_mask, "rouge1"].dropna()
        if len(amh_r1) > 0 and amh_r1.mean() < 0.01:
            print("WARNING: Amharic (Amh_Eth) ROUGE scores are near zero.")
            print("   The rouge-score library cannot tokenise Ge'ez (Ethiopic) script.")
            print("   Use chrF++ as a proxy: rerun with --compute-chrf (requires: pip install sacrebleu)")
            print()

    print("Done!")


if __name__ == "__main__":
    main()
