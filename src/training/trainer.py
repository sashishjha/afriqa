"""Model setup (LoRA), training loops for seq2seq and causal LM.
Modified for AMD ROCm (RBCCPS cluster, IISc - AMD MI210 GPUs).
- Uses bf16 (bfloat16) on ROCm instead of fp16 (AMD MI210 has native bf16 support).
- Auto-detects CUDA vs ROCm and selects the correct precision mode.
"""

import os
import torch
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from src.metrics import compute_rouge


def _is_rocm() -> bool:
    """Return True if PyTorch was built with ROCm (HIP) backend."""
    return hasattr(torch.version, "hip") and torch.version.hip is not None


def _get_dtype():
    """Return appropriate torch dtype: bfloat16 for ROCm, float16 for CUDA."""
    if _is_rocm():
        return torch.bfloat16
    return torch.float16


# -- target-module heuristics --

_SEQ2SEQ_TARGETS = {
    "mt5": ["q", "v"],
    "byt5": ["q", "v"],
    "aya": ["q", "v"],
    "afriteva": ["q", "v"],
    "nllb": ["q_proj", "v_proj"],
    "default": ["q", "v"],
}

_CAUSAL_TARGETS = {
    "qwen": ["q_proj", "v_proj"],
    "gemma": ["q_proj", "v_proj"],
    "llama": ["q_proj", "v_proj"],
    "default": ["q_proj", "v_proj"],
}


def _pick_targets(model_name: str, mapping: dict) -> list:
    low = model_name.lower()
    for key, modules in mapping.items():
        if key in low:
            return modules
    return mapping["default"]


# -- model factories --

def setup_seq2seq_model(
    model_name: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    use_lora: bool = True,
    target_modules: list = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)

    if use_lora:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        targets = target_modules or _pick_targets(model_name, _SEQ2SEQ_TARGETS)
        cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=targets,
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()

    return model, tokenizer


def setup_causal_model(
    model_name: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    use_lora: bool = True,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    # ROCm: use bfloat16 instead of float16 for better stability on MI210
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=_get_dtype(), trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    if use_lora:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=_pick_targets(model_name, _CAUSAL_TARGETS),
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()

    return model, tokenizer


def load_trained_model(
    base_model_name: str,
    adapter_path: str,
    model_type: str = "seq2seq",
):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, use_fast=False)
    if model_type == "seq2seq":
        base = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, trust_remote_code=True)
    else:
        # ROCm: use bfloat16 instead of float16
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=_get_dtype(), trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            base.config.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.merge_and_unload()
    return model, tokenizer


# -- compute-metrics callback for Trainer --

def _make_compute_metrics(tokenizer):
    def _normalize_token_ids(values):
        if isinstance(values, tuple):
            values = values[0]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values.argmax(axis=-1)
        if values.dtype.kind == "f":
            values = np.rint(values)
        values = values.astype(np.int64, copy=False)
        values = np.where(values < 0, tokenizer.pad_token_id, values)
        return values

    def compute(eval_pred):
        preds, labels = eval_pred
        preds = _normalize_token_ids(preds)
        labels = _normalize_token_ids(labels)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return compute_rouge(decoded_preds, decoded_labels)

    return compute


# -- training entry-points --

def train_seq2seq(
    model,
    tokenizer,
    train_ds,
    val_ds,
    output_dir: str,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 3e-4,
    num_epochs: int = 5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    fp16: bool = True,
    logging_steps: int = 100,
    save_steps: int = 500,
    eval_steps: int = 500,
    gen_max_length: int = 256,
    num_beams: int = 4,
    length_penalty: float = 0.8,
    min_length: int = 15,
    resume_from_checkpoint: str = None,
):
    # ROCm: use bf16 instead of fp16 (MI300X has native bfloat16 support)
    use_bf16 = _is_rocm() and torch.cuda.is_available()
    use_fp16 = (not _is_rocm()) and fp16 and torch.cuda.is_available()

    # DDP safety: load_best_model_at_end deadlocks when multiple GPU processes
    # all try to load the best checkpoint simultaneously.
    import torch.distributed as dist
    is_distributed = dist.is_available() and dist.is_initialized()
    safe_load_best = not is_distributed

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=safe_load_best,  # False under DDP to avoid deadlock
        metric_for_best_model="rouge1",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=gen_max_length,
        generation_num_beams=num_beams,
        save_total_limit=3,
        report_to="none",
        dataloader_num_workers=0,   # 0 = safe for ROCm DDP (worker crashes otherwise)
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adafactor",
        ddp_find_unused_parameters=False,  # required for LoRA layers under DDP
    )
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=_make_compute_metrics(tokenizer),
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(os.path.join(output_dir, "best_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "best_model"))
    return trainer


def train_causal(
    model,
    tokenizer,
    train_ds,
    val_ds,
    output_dir: str,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    fp16: bool = True,
    logging_steps: int = 100,
    save_steps: int = 500,
    eval_steps: int = 500,
    resume_from_checkpoint: str = None,
):
    # ROCm: use bf16 instead of fp16 (MI210 has native bfloat16 support)
    use_bf16 = _is_rocm() and torch.cuda.is_available()
    use_fp16 = (not _is_rocm()) and fp16 and torch.cuda.is_available()

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        report_to="none",
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(os.path.join(output_dir, "best_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "best_model"))
    return trainer
