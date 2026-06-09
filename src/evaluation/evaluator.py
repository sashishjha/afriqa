"""Per-subset and global evaluation harness."""

import json, os
from src.metrics import compute_rouge, compute_bertscore, compute_all_metrics


class Evaluator:
    def __init__(self, bertscore_model: str = "Davlan/afro-xlmr-base"):
        self.bs_model = bertscore_model

    def evaluate(self, preds: list, refs: list) -> dict:
        return compute_all_metrics(preds, refs, self.bs_model)

    def evaluate_per_subset(self, preds: list, refs: list, subsets: list) -> dict:
        results = {}
        for sub in sorted(set(subsets)):
            mask = [s == sub for s in subsets]
            sp = [p for p, m in zip(preds, mask) if m]
            sr = [r for r, m in zip(refs, mask) if m]
            if sp:
                results[sub] = compute_rouge(sp, sr)
                results[sub]["n"] = len(sp)
        results["__global__"] = compute_rouge(preds, refs)
        results["__global__"]["n"] = len(preds)
        return results

    def print_report(self, results: dict):
        print("\n" + "=" * 72)
        print("EVALUATION REPORT")
        print("=" * 72)
        for sub in sorted(results):
            m = results[sub]
            n = m.get("n", "?")
            r1 = m.get("rouge1", 0)
            rL = m.get("rougeL", 0)
            print(f"  {sub:16s}  n={n:>5}  ROUGE-1={r1:.4f}  ROUGE-L={rL:.4f}")
        print("=" * 72)

    def save_report(self, results: dict, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
