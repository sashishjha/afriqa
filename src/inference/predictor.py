"""Inference helpers for seq2seq and causal-LM models.
Modified for AMD ROCm (RBCCPS cluster, IISc - AMD MI210 GPUs).
- Auto-detects GPU (ROCm/CUDA) instead of hardcoding 'cuda'.
"""

import torch
from tqdm import tqdm


def _auto_device() -> str:
    """Return 'cuda' if a GPU is available (works with ROCm HIP), else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


class Predictor:
    def __init__(self, model, tokenizer, device=None, model_type="seq2seq"):
        if device is None:
            device = _auto_device()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_type = model_type

    def predict(self, question: str, gen_kwargs: dict | None = None) -> str:
        gen_kwargs = gen_kwargs or {}
        if self.model_type == "seq2seq":
            prompt = f"answer_question: {question}"
        else:
            prompt = f"Question: {question}\nAnswer:"
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask"),
                **gen_kwargs,
            )
        pred = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if self.model_type == "causal":
            # Remove prompt from output
            if pred.startswith(prompt):
                pred = pred[len(prompt):].strip()
        return pred.strip()

    def predict_batch(self, questions: list, batch_size: int = 8, gen_kwargs: dict | None = None) -> list:
        gen_kwargs = gen_kwargs or {}
        preds = []
        for start in tqdm(range(0, len(questions), batch_size), desc="Predicting"):
            batch = questions[start : start + batch_size]
            prompts = []
            for q in batch:
                if self.model_type == "seq2seq":
                    prompts.append(f"answer_question: {q}")
                else:
                    prompts.append(f"Question: {q}\nAnswer:")
            enc = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.device)
            prompt_lengths = enc["attention_mask"].sum(dim=1).tolist()
            with torch.no_grad():
                out = self.model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc.get("attention_mask"),
                    **gen_kwargs,
                )
            if self.model_type == "causal":
                for seq, prompt_len in zip(out, prompt_lengths):
                    decoded = self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                    preds.append(decoded.strip())
            else:
                decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)
                preds.extend([p.strip() for p in decoded])
        return preds
