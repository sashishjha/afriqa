"""ROUGE, BERTScore, and combined metrics.
Modified for AMD ROCm (RBCCPS cluster, IISc - AMD MI210 GPUs).
- compute_bertscore auto-detects device (ROCm/CUDA/CPU).
"""

import numpy as np
from rouge_score import rouge_scorer


def compute_rouge(predictions: list, references: list) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    r1, rL = [], []
    for p, r in zip(predictions, references):
        s = scorer.score(r, p)
        r1.append(s["rouge1"].fmeasure)
        rL.append(s["rougeL"].fmeasure)
    return {"rouge1": float(np.mean(r1)), "rougeL": float(np.mean(rL))}


def compute_bertscore(
    predictions: list,
    references: list,
    model_name: str = "Davlan/afro-xlmr-base",
) -> dict:
    import torch
    from bert_score import score as bs_score

    # Auto-detect device: ROCm maps torch.cuda to HIP
    device = "cuda" if torch.cuda.is_available() else "cpu"

    P, R, F = bs_score(
        predictions, references,
        model_type=model_name, verbose=False,
        device=device,
    )
    return {
        "bertscore_p": float(P.mean()),
        "bertscore_r": float(R.mean()),
        "bertscore_f1": float(F.mean()),
    }


def compute_all_metrics(
    predictions: list,
    references: list,
    bertscore_model: str = "Davlan/afro-xlmr-base",
) -> dict:
    m = compute_rouge(predictions, references)
    try:
        m.update(compute_bertscore(predictions, references, bertscore_model))
    except Exception as e:
        print(f"[WARN] BERTScore failed: {e}")
    return m
