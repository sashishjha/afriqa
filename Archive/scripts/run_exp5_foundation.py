#!/usr/bin/env python3
"""
Experiment 5 - Foundation Model Benchmark
==========================================
Compare mT5-base, mT5-large, ByT5-base, NLLB-600M, Aya-101, Qwen2.5-1.5B

Each model is trained with LoRA, evaluated on validation (global + per-subset),
and scored on ROUGE + cost metrics.  A summary table is printed at the end.

Modified for AMD ROCm (RBCCPS cluster, IISc - AMD MI210 GPUs).
- gpu_memory_mb() works for both ROCm and CUDA.
- Logs ROCm detection info at startup.
"""

import os
import sys
import time
import argparse
import traceback
import torch
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import set_seed, get_device, load_config, save_json, setup_logging
from src.data_loader import load_data, create_seq2seq_dataset, create_causal_dataset
from src.training.trainer import (
    setup_seq2seq_model,
    setup_causal_model,
    train_seq2seq,
    train_causal,
)
from src.evaluation.evaluator import Evaluator

# -- model registry --

MODELS = {
    "mt5-base": {
        "hf_name": "google/mt5-base",
        "type": "seq2seq",
        "lora_r": 16,
    },
    "mt5-large": {
        "hf_name": "google/mt5-large",
        "type": "seq2seq",
        "lora_r": 16,
    },
    "byt5-base": {
        "hf_name": "google/byt5-base",
        "type": "seq2seq",
        "lora_r": 16,
    },
    "nllb-600M": {
        "hf_name": "facebook/nllb-200-distilled-600M",
        "type": "seq2seq",
        "lora_r": 16,
    },
    "aya-101": {
        "hf_name": "CohereForAI/aya-101",
        "type": "seq2seq",
        "lora_r": 16,
    },
    "qwen2.5-1.5B": {
        "hf_name": "Qwen/Qwen2.5-1.5B",
        "type": "causal",
        "lora_r": 16,
    },
}


# -- predictor classes --

class Seq2SeqPredictor:
    def __init__(self, model, tokenizer, device, gen_kwargs):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.gen_kwargs = gen_kwargs

    @torch.no_grad()
    def predict(self, questions, batch_size=8, prefix="answer_question: "):
        preds = []
        for i in range(0, len(questions), batch_size):
            batch = questions[i : i + batch_size]
            texts = [f"{prefix}{q}" for q in batch]
            enc = self.tokenizer(
                texts, return_tensors="pt",
                max_length=256, truncation=True, padding=True,
            ).to(self.device)
            out = self.model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                **self.gen_kwargs,
            )
            preds.extend(self.tokenizer.batch_decode(out, skip_special_tokens=True))
        return [s.strip() for s in preds]


class CausalPredictor:
    def __init__(self, model, tokenizer, device, gen_kwargs):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.gen_kwargs = gen_kwargs

    @torch.no_grad()
    def predict(self, questions, batch_size=4):
        preds = []
        for i in range(0, len(questions), batch_size):
            batch = questions[i : i + batch_size]
            prompts = [f"Question: {q}\nAnswer:" for q in batch]
            enc = self.tokenizer(
                prompts, return_tensors="pt",
                max_length=256, truncation=True, padding=True,
            ).to(self.device)
            input_lengths = enc["attention_mask"].sum(dim=1).tolist()
            out = self.model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                **self.gen_kwargs,
            )
            for seq, prompt_len in zip(out, input_lengths):
                decoded = self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                preds.append(decoded.strip())
        return preds


# -- cost helpers --

def count_params(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


def gpu_memory_mb() -> float:
    """Return peak allocated GPU memory in MB.
    Works for both CUDA and ROCm (torch.cuda.* maps to HIP on ROCm)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


def _is_rocm() -> bool:
    """Return True if PyTorch was built with ROCm (HIP) backend."""
    return hasattr(torch.version, "hip") and torch.version.hip is not None


# -- single model run --

def run_one_model(
    tag: str,
    spec: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: dict,
    args,
    logger,
    device,
) -> dict:
    """Train + evaluate a single model. Returns a results dict."""
    logger.info(f"\n{'='*60}\n  MODEL: {tag}  ({spec['hf_name']})\n{'='*60}")

    out_dir = os.path.join(args.output_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    result = {"model": tag, "hf_name": spec["hf_name"], "type": spec["type"]}

    # -- setup --
    t0 = time.time()
    if spec["type"] == "seq2seq":
        model, tokenizer = setup_seq2seq_model(spec["hf_name"], lora_r=spec["lora_r"])
    else:
        model, tokenizer = setup_causal_model(spec["hf_name"], lora_r=spec["lora_r"])
    result.update(count_params(model))
    logger.info(f"Params total={result['total_params']:,}  trainable={result['trainable_params']:,}")

    # -- datasets --
    max_in = cfg.get("max_input_length", 256)
    max_tgt = cfg.get("max_target_length", 512)
    bs = args.batch_size or cfg.get("batch_size", 8)
    ep = args.epochs or cfg.get("num_epochs", 5)

    if spec["type"] == "seq2seq":
        train_ds = create_seq2seq_dataset(train_df, tokenizer, max_in, max_tgt)
        val_ds = create_seq2seq_dataset(val_df, tokenizer, max_in, max_tgt)
    else:
        train_ds = create_causal_dataset(train_df, tokenizer, max_in + max_tgt)
        val_ds = create_causal_dataset(val_df, tokenizer, max_in + max_tgt)

    # -- train --
    mem_before = gpu_memory_mb()
    if spec["type"] == "seq2seq":
        train_seq2seq(model, tokenizer, train_ds, val_ds, output_dir=out_dir,
                      batch_size=bs, num_epochs=ep)
    else:
        train_causal(model, tokenizer, train_ds, val_ds, output_dir=out_dir,
                     batch_size=bs, num_epochs=ep)
    train_time = time.time() - t0
    mem_peak = gpu_memory_mb()
    result["train_time_s"] = round(train_time, 1)
    result["gpu_peak_mb"] = round(mem_peak, 1)
    logger.info(f"Training done in {train_time:.0f}s | GPU peak {mem_peak:.0f} MB")

    # -- evaluate --
    gen_kwargs = dict(max_new_tokens=max_tgt, num_beams=4)
    if spec["type"] == "seq2seq":
        predictor = Seq2SeqPredictor(model, tokenizer, device, gen_kwargs)
    else:
        predictor = CausalPredictor(model, tokenizer, device, gen_kwargs)

    questions = val_df["input"].tolist()
    references = val_df["output"].tolist()
    subsets = val_df["subset"].tolist()
    preds = predictor.predict(questions, batch_size=bs)

    evaluator = Evaluator()
    per_sub = evaluator.evaluate_per_subset(preds, references, subsets)
    evaluator.print_report(per_sub)
    evaluator.save_report(per_sub, os.path.join(out_dir, "eval_results.json"))
    result["eval"] = per_sub

    # -- cleanup --
    del model, tokenizer, predictor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# -- main --

def parse_args():
    p = argparse.ArgumentParser(description="Experiment 5 - Foundation Model Benchmark")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--output_dir", type=str, default="outputs/exp5_foundation")
    p.add_argument("--models", nargs="+", default=None,
                   help=f"Models to run. Choices: {list(MODELS.keys())}. Default: all.")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="Subsample training data for faster sweeps")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger = setup_logging()
    set_seed(cfg.get("training", {}).get("seed", 42))
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    # -- ROCm detection info --
    if _is_rocm():
        logger.info(f"ROCm backend detected (HIP version: {torch.version.hip})")
        logger.info("Using bfloat16 for training (AMD MI210 native support)")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    elif torch.cuda.is_available():
        logger.info(f"CUDA backend detected (version: {torch.version.cuda})")
        logger.info("Using float16 for training")
    else:
        logger.warning("No GPU detected! Training will be very slow on CPU.")

    # -- data --
    train_path = cfg.get("data", {}).get("train", cfg.get("train_data", "data/train.csv"))
    val_path = cfg.get("data", {}).get("val", cfg.get("val_data", "data/val.csv"))
    train_df = load_data(train_path)
    val_df = load_data(val_path)
    if args.max_train_samples:
        train_df = train_df.head(args.max_train_samples)
    logger.info(f"Train: {len(train_df)} rows | Val: {len(val_df)} rows")

    # -- run models --
    keys = args.models or list(MODELS.keys())
    summary = []
    for key in keys:
        if key not in MODELS:
            logger.warning(f"Unknown model key '{key}', skipping.")
            continue
        try:
            r = run_one_model(key, MODELS[key], train_df, val_df, cfg, args, logger, device)
            summary.append(r)
        except Exception as e:
            logger.error(f"Model {key} FAILED: {e}")
            traceback.print_exc()
            summary.append({"model": key, "error": str(e)})

    # -- summary --
    save_json(summary, os.path.join(args.output_dir, "summary.json"))
    logger.info("Experiment 5 complete.")


if __name__ == "__main__":
    main()
