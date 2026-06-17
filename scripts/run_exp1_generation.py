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

def preflight_model_check(model_name: str, cfg: dict, logger):
    """
    Apply model-specific configuration overrides and safety checks.
    Call this BEFORE setup_seq2seq_model().
    Returns: updated cfg dict.
    """
    model_lower = model_name.lower()

    # ── Aya-101 (13B, mT5-XXL based) ──────────────────────────
    if "aya-101" in model_lower or "aya_101" in model_lower:
        logger.info("🔍 Detected Aya-101 model — applying 13B-specific settings...")

        # Force gradient checkpointing for memory
        cfg.setdefault("training", {})
        cfg["training"]["gradient_checkpointing"] = True

        # Warn about memory
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            logger.info(f"   GPU VRAM: {vram_gb} GB")
            if vram_gb < 40:
                logger.warning(
                    f"   ⚠️  Aya-101 needs ~26GB per GPU in fp16. "
                    f"Your GPU has {vram_gb}GB — this may OOM. "
                    f"Consider reducing batch_size to 1 or using LoRA r=4."
                )

    # ── AfriTeVa V2 (T5 v1.1 based) ──────────────────────────
    elif "afriteva" in model_lower:
        logger.info("🔍 Detected AfriTeVa V2 model — applying T5-v1.1 fixes...")

        # AfriTeVa was pretrained with dropout OFF — re-enable it
        cfg.setdefault("training", {})
        if "dropout_override" not in cfg["training"]:
            cfg["training"]["dropout_override"] = 0.1
            logger.info("   ✅ Dropout re-enabled (0.1) for finetuning")

        # T5 v1.1 uses gated-gelu; ensure transformers version supports it
        try:
            from transformers import T5Config
            test_cfg = T5Config(feed_forward_proj="gated-gelu")
            logger.info("   ✅ gated-gelu activation supported")
        except Exception as e:
            logger.error(
                f"   ❌ gated-gelu not supported by your transformers version: {e}\n"
                f"   Run: pip install transformers>=4.30.0"
            )
            raise

    return cfg


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


def get_latest_checkpoint(output_dir: str) -> str:
    """Finds the checkpoint with the highest step number in output_dir."""
    if not os.path.isdir(output_dir):
        return None
    checkpoints = [
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not checkpoints:
        return None
    # Sort by the number after "checkpoint-"
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    return os.path.join(output_dir, checkpoints[-1])


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
    p.add_argument("--resume_checkpoint", type=str, default=None,
                   help="Path to a specific checkpoint to resume from (overrides automatic discovery).")
    p.add_argument("--merge_val", action="store_true",
                   help="Merge train+val into one training set (for final submission runs).")
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
    cfg = preflight_model_check(model_name, cfg, logger)
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
    input_prefix = cfg.get("prompt", {}).get("input_prefix", "answer_question: ")
    lora_targets = cfg["lora"].get("target_modules", None)  # None = use auto-detection

    logger.info(f"Model          : {model_name}")
    logger.info(f"LoRA r         : {lora_r}")
    logger.info(f"Batch size     : {batch_size}")
    logger.info(f"Epochs         : {num_epochs}")
    logger.info(f"Learning rate  : {lr}")
    logger.info(f"Prompt prefix  : {repr(input_prefix)}")

    # ── data ────────────────────────────────────────────────────────
    logger.info("Loading data …")
    train_df = load_data(cfg["data"]["train_path"])
    val_df = load_data(cfg["data"]["val_path"])
    test_df = load_data(cfg["data"]["test_path"])

    logger.info(f"Train : {len(train_df):,}  |  Val : {len(val_df):,}  |  Test : {len(test_df):,}")

    # ── merge train+val if requested (final submission mode) ─────────
    if args.merge_val:
        logger.info("Merging train + val into single training set …")
        train_df = pd.concat([train_df, val_df], ignore_index=True)
        logger.info(f"Merged train size: {len(train_df):,}")

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
            target_modules=lora_targets,
        )

        # ═══ NEW: Post-setup model patches (Aya, AfriTeVa, etc.) ══════════
        model_lower = model_name.lower()

        # ── AfriTeVa: re-enable dropout (was OFF during pretraining) ──────
        dropout_override = cfg.get("training", {}).get("dropout_override", None)
        if dropout_override is not None:
            patched = 0
            for module in model.modules():
                if hasattr(module, "dropout") and hasattr(module.dropout, "p"):
                    module.dropout.p = dropout_override
                    patched += 1
                if hasattr(module, "dropout_rate"):
                    module.dropout_rate = dropout_override
                    patched += 1
            logger.info(f"✅ Dropout overridden to {dropout_override} ({patched} modules patched)")

        # ── Large models (Aya-101): enable gradient checkpointing ─────────
        if cfg.get("training", {}).get("gradient_checkpointing", False):
            model.gradient_checkpointing_enable()
            logger.info("✅ Gradient checkpointing enabled")

        # ── Tokenizer: force slow SentencePiece if config says so ─────────
        use_fast = cfg.get("training", {}).get("use_fast_tokenizer", True)
        if not use_fast and tokenizer.is_fast:
            logger.info("⚠️  Config requests use_fast=False but got fast tokenizer. Reloading...")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_fast=False, trust_remote_code=True
            )
            logger.info(f"✅ Slow tokenizer reloaded: vocab_size={tokenizer.vocab_size}")
        # ═══ END post-setup patches ═══════════════════════════════════════

    # ── tokenise ────────────────────────────────────────────────────
    logger.info("Tokenising datasets …")
    train_ds = create_seq2seq_dataset(train_df, tokenizer, max_input, max_target,
                                      input_prefix=input_prefix)

    if args.merge_val:
        # val_df is now part of training — use a small proxy sample for Trainer's
        # periodic eval (just to monitor loss; don't treat as true held-out score)
        proxy_val_df = val_df.sample(n=min(500, len(val_df)), random_state=42)
        val_ds = create_seq2seq_dataset(proxy_val_df, tokenizer, max_input, max_target,
                                        input_prefix=input_prefix)
        logger.info(f"merge_val=True: using {len(proxy_val_df)} sample proxy val (not held-out)")
    else:
        val_ds = create_seq2seq_dataset(val_df, tokenizer, max_input, max_target,
                                        input_prefix=input_prefix)


    # ── train ───────────────────────────────────────────────────────
    if not args.skip_train:
        resume_checkpoint = None
        if args.resume_checkpoint:
            resume_checkpoint = args.resume_checkpoint
            logger.info(f"Using user-specified checkpoint:\n    {resume_checkpoint}\n    Resuming training.")
        else:
            latest = get_latest_checkpoint(args.output_dir)
            if latest:
                resume_checkpoint = latest
                logger.info(f"Found latest checkpoint:\n    {resume_checkpoint}\n\n    Resuming training.")
            else:
                logger.info("No checkpoint found.\nStarting training from scratch.")

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
            length_penalty=gen_kwargs.get("length_penalty", 0.8),
            min_length=gen_kwargs.get("min_length", 15),
            resume_from_checkpoint=resume_checkpoint,
        )
        model = trainer.model
        logger.info("Training complete.")

    # ── DDP safety: unwrap model + gate eval/submission to rank 0 ──
    # After torchrun, model is wrapped in DistributedDataParallel.
    # .generate() on a DDP model crashes on ROCm. Unwrap it.
    if hasattr(model, "module"):
        model = model.module
        logger.info("Unwrapped model from DDP wrapper for inference.")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        logger.info(f"Rank {local_rank}: training done. Exiting (rank 0 handles eval+submission).")
        return

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
                "length_penalty": gen_kwargs.get("length_penalty", 0.8),
                "no_repeat_ngram_size": gen_kwargs["no_repeat_ngram_size"],
                "min_length": gen_kwargs.get("min_length", 15),
                "early_stopping": True,
            },
        )
        val_preds = predictor.predict_batch(val_df["input"].tolist(), batch_size=batch_size,
                                             input_prefix=input_prefix)

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
                    "length_penalty": gen_kwargs.get("length_penalty", 0.8),
                    "no_repeat_ngram_size": gen_kwargs["no_repeat_ngram_size"],
                    "min_length": gen_kwargs.get("min_length", 15),
                    "early_stopping": True,
                },
            )
        test_preds = predictor.predict_batch(test_df["input"].tolist(), batch_size=batch_size,
                                              input_prefix=input_prefix)

        sub = pd.DataFrame({
            "ID": test_df["ID"], 
            "TargetRLF1": test_preds,
            "TargetR1F1": test_preds,
            "TargetLLM": test_preds
        })
        # Derive submission name from output_dir (e.g. "outputs/exp2_optimized" → "exp2_optimized_submission.csv")
        exp_name = os.path.basename(args.output_dir.rstrip("/"))
        sub_path = os.path.join("submissions", f"{exp_name}_submission.csv")
        os.makedirs("submissions", exist_ok=True)
        sub.to_csv(sub_path, index=False)
        logger.info(f"Submission saved to {sub_path}  ({len(sub)} rows)")


if __name__ == "__main__":
    main()
    # Clean DDP shutdown (prevents hangs when non-rank-0 processes exit)
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
