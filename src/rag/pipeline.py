"""Retrieval-Augmented Generation pipeline.
Modified for AMD ROCm (RBCCPS cluster, IISc - AMD MI210 GPUs).
- Auto-detects GPU (ROCm/CUDA) instead of hardcoding 'cuda'.
"""

import torch
from tqdm import tqdm


def _auto_device() -> str:
    """Return 'cuda' if a GPU is available (works with ROCm HIP), else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


class RAGPipeline:
    def __init__(self, retriever, model, tokenizer, device=None, top_k: int = 5,
                 gen_kwargs: dict | None = None):
        if device is None:
            device = _auto_device()
        self.retriever = retriever
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.top_k = top_k
        self.gen_kwargs = gen_kwargs or {
            "max_new_tokens": 512,
            "num_beams": 4,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
        }

    @staticmethod
    def build_context(docs: list) -> str:
        parts = []
        for i, d in enumerate(docs):
            parts.append(f"Example {i+1}:\nQ: {d['question']}\nA: {d['answer']}")
        return "\n\n".join(parts)

    def _make_prompt(self, question: str, context: str) -> str:
        return f"context: {context} question: {question}"

    # -- single-question --

    def answer(self, question: str) -> str:
        docs = self.retriever.retrieve(question, self.top_k)
        ctx = self.build_context(docs)
        prompt = self._make_prompt(question, ctx)
        enc = self.tokenizer(
            prompt, return_tensors="pt",
            max_length=1024, truncation=True,
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                **self.gen_kwargs,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    # -- batch --

    def answer_batch(self, questions: list, batch_size: int = 8, subsets: list = None) -> list:
        preds = []
        for start in tqdm(range(0, len(questions), batch_size), desc="RAG gen"):
            batch_q = questions[start : start + batch_size]
            if subsets:
                batch_s = subsets[start : start + batch_size]
            prompts = []
            for idx, q in enumerate(batch_q):
                if subsets:
                    docs = self.retriever.retrieve(q, subset=batch_s[idx], top_k=self.top_k)
                else:
                    docs = self.retriever.retrieve(q, self.top_k)
                ctx = self.build_context(docs)
                prompts.append(self._make_prompt(q, ctx))
            enc = self.tokenizer(
                prompts, return_tensors="pt",
                max_length=1024, truncation=True, padding=True,
            ).to(self.device)
            with torch.no_grad():
                out = self.model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    **self.gen_kwargs,
                )
            preds.extend(self.tokenizer.batch_decode(out, skip_special_tokens=True))
        return preds

    # -- augment a DataFrame for training-time RAG --

    def augment_dataframe(self, df, top_k=3):
        """
        Augment each row in df with retrieved context.
        Supports both global TFIDFRetriever and PerSubsetTFIDFRetriever.
        """
        from src.retrieval.tfidf_retriever import PerSubsetTFIDFRetriever
        import logging
        logger = logging.getLogger("afriqa")

        is_per_subset = isinstance(self.retriever, PerSubsetTFIDFRetriever)

        augmented_inputs = []
        for idx, row in df.iterrows():
            query = row["input"]

            # ── Route to the correct retriever signature ──────────
            if is_per_subset:
                # PerSubsetTFIDFRetriever needs the subset label
                subset = row.get("subset", None)
                if subset is None:
                    logger.warning(
                        f"Row {idx}: no 'subset' column found — "
                        f"falling back to global retriever"
                    )
                    results = self.retriever.fallback_retriever.retrieve(query, top_k=top_k + 1)
                else:
                    results = self.retriever.retrieve(query, subset=subset, top_k=top_k + 1)
            else:
                # Standard TFIDFRetriever / BM25 / Dense / Hybrid
                results = self.retriever.retrieve(query, top_k=top_k + 1)

            docs = [d for d in results if d["question"].strip() != query.strip()][:top_k]
            ctx = self.build_context(docs)
            augmented_input = f"context: {ctx} question: {query}" if ctx else query
            augmented_inputs.append(augmented_input)

        out = df.copy()
        out["input"] = augmented_inputs
        logger.info(
            f"Augmented {len(df)} rows with top-{top_k} retrieval "
            f"(per_subset={is_per_subset})"
        )
        return out
