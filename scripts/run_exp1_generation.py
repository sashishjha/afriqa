#!/usr/bin/env python3
"""
Experiment 1 — Pure Generation Baseline
========================================
mT5-base + LoRA  →  Question → Answer

End-to-end: load data → tokenise → train → evaluate (global + per-subset) → generate submission.
"""

import os
import sys
import argparse
import torch
import pandas as pd

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import set_seed, get_device, load_config, save_json, setup_logging
from src.data_loader import (
    load_data, create_seq2seq_dataset, SUBSETS,
)
from src.training.trainer import setup_seq2seq_model, load_trained_model, train_seq2seq
from src.evaluation.evaluator import Evaluator


# ── helpers ─────────────────────────────────────────────────────────

class Predictor:
    """Lightweight seq2seq predictor (self-contained so file runs standalone)."""

    def __init__(self, model, tokenizer, device, gen_kwargs):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.gen_kwargs = gen_kwargs

    @torch.no_grad()
    def predict_batch(self, questions: list, batch_size: int = 16,
                      input_prefix: str = "answer_question: ") -> list:
        all_preds = []
        for start in range(0, len(questions), batch_size):
            batch = questions[start : start + batch_size]
            texts = [f"{input_prefix}{q}" for q in batch]
            enc = self.tokenizer(
                texts,
                return_tensors="pt",
                max_length=256,
                truncation=True,
                padding=True,
            ).to(self.device)
            out = self.model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                **self.gen_kwargs,
            )
            decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)
            all_preds.extend([s.strip() for s in decoded])
            if (start // batch_size) % 10 == 0:
                done = min(start + batch_size, len(questions))
                print(f"  [predict] {done}/{len(questions)}")
        return all_preds


# ── main ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Experiment 1 – mT5-base + LoRA generation")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--model_name", type=str, default=None,
                   help="Override model name (default: from config)")
    p.add_argument("--output_dir", type=str, default="outputs/exp1_generation")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lora_r", type=int, default=None)
    p.add_argument("--skip_train", action="store_true",
                   help="Skip training; load from output_dir/best_model")
    p.add_argument("--skip_eval", action="store_true")
    p.add_argument("--skip_submission", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger = setup_logging()

    set_seed(cfg["training"]["seed"])
    device = get_device()
    logger.info(f"Device: {device}")

    # ── resolve hyper-params ────────────────────────────────────────
    model_name = args.model_name or cfg["model"]["default"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    num_epochs = args.epochs or cfg["training"]["num_epochs"]
    lr = args.lr or cfg["training"]["learning_rate"]
    lora_r = args.lora_r or cfg["lora"]["r"]
    lora_alpha = cfg["lora"]["alpha"]
    lora_dropout = cfg["lora"]["dropout"]
    grad_acc = cfg["training"]["gradient_accumulation_steps"]
    max_input = cfg["training"]["max_input_length"]
    max_target = cfg["training"]["max_target_length"]
    gen_kwargs = cfg["generation"]

    logger.info(f"Model          : {model_name}")
    logger.info(f"LoRA r         : {lora_r}")
    logger.info(f"Batch size     : {batch_size}")
    logger.info(f"Epochs         : {num_epochs}")
    logger.info(f"Learning rate  : {lr}")

    # ── data ────────────────────────────────────────────────────────
    logger.info("Loading data …")
    train_df = load_data(cfg["data"]["train_path"])
    val_df = load_data(cfg["data"]["val_path"])
    test_df = load_data(cfg["data"]["test_path"])

    logger.info(f"Train : {len(train_df):,}  |  Val : {len(val_df):,}  |  Test : {len(test_df):,}")

    # ── model + tokeniser ───────────────────────────────────────────
    if args.skip_train:
        best_dir = os.path.join(args.output_dir, "best_model")
        logger.info(f"Loading trained model from {best_dir}")
        model, tokenizer = load_trained_model(model_name, best_dir, model_type="seq2seq")
    else:
        logger.info("Setting up model + LoRA …")
        model, tokenizer = setup_seq2seq_model(
            model_name,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

    # ── tokenise ────────────────────────────────────────────────────
    logger.info("Tokenising datasets …")
    train_ds = create_seq2seq_dataset(train_df, tokenizer, max_input, max_target)
    val_ds = create_seq2seq_dataset(val_df, tokenizer, max_input, max_target)

    # ── train ───────────────────────────────────────────────────────
    if not args.skip_train:
        logger.info("Starting training …")
        trainer = train_seq2seq(
            model=model,
            tokenizer=tokenizer,
            train_ds=train_ds,
            val_ds=val_ds,
            output_dir=args.output_dir,
            batch_size=batch_size,
            gradient_accumulation_steps=grad_acc,
            learning_rate=lr,
            num_epochs=num_epochs,
            warmup_ratio=cfg["training"]["warmup_ratio"],
            weight_decay=cfg["training"]["weight_decay"],
            fp16=cfg["training"]["fp16"],
            logging_steps=cfg["training"]["logging_steps"],
            save_steps=cfg["training"]["save_steps"],
            eval_steps=cfg["training"]["eval_steps"],
            gen_max_length=gen_kwargs["max_new_tokens"],
            num_beams=gen_kwargs["num_beams"],
        )
        model = trainer.model
        logger.info("Training complete.")

    # ── evaluation ──────────────────────────────────────────────────
    predictor = None
    if not args.skip_eval:
        logger.info("Evaluating on validation set …")
        predictor = Predictor(
            model=model,
            tokenizer=tokenizer,
            device=device,
            gen_kwargs={
                "max_new_tokens": gen_kwargs["max_new_tokens"],
                "num_beams": gen_kwargs["num_beams"],
                "length_penalty": gen_kwargs["length_penalty"],
                "no_repeat_ngram_size": gen_kwargs["no_repeat_ngram_size"],
                "early_stopping": True,
            },
        )
        val_preds = predictor.predict_batch(val_df["input"].tolist(), batch_size=batch_size)

        evaluator = Evaluator(bertscore_model=cfg["evaluation"]["bertscore_model"])
        per_subset = evaluator.evaluate_per_subset(
            val_preds, val_df["output"].tolist(), val_df["subset"].tolist(),
        )
        evaluator.print_report(per_subset)

        report_path = os.path.join(args.output_dir, "val_results.json")
        evaluator.save_report(per_subset, report_path)
        logger.info(f"Results saved to {report_path}")

    # ── submission ──────────────────────────────────────────────────
    if not args.skip_submission:
        logger.info("Generating test predictions …")
        if predictor is None:
            predictor = Predictor(
                model=model,
                tokenizer=tokenizer,
                device=device,
                gen_kwargs={
                    "max_new_tokens": gen_kwargs["max_new_tokens"],
                    "num_beams": gen_kwargs["num_beams"],
                    "length_penalty": gen_kwargs["length_penalty"],
                    "no_repeat_ngram_size": gen_kwargs["no_repeat_ngram_size"],
                    "early_stopping": True,
                },
            )
        test_preds = predictor.predict_batch(test_df["input"].tolist(), batch_size=batch_size)

        sub = pd.DataFrame({"ID": test_df["ID"], "output": test_preds})
        sub_path = os.path.join("submissions", "exp1_submission.csv")
        os.makedirs("submissions", exist_ok=True)
        sub.to_csv(sub_path, index=False)
        logger.info(f"Submission saved to {sub_path}  ({len(sub)} rows)")


if __name__ == "__main__":
    main()
