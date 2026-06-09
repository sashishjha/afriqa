"""Hybrid retriever: normalised BM25 + Dense score fusion."""

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        alpha: float = 0.5,
    ):
        self.bm25 = bm25
        self.dense = dense
        self.alpha = alpha          # weight for dense
        self.documents = bm25.documents if bm25.documents else dense.documents
        self.answers = bm25.answers if bm25.answers else dense.answers

    @staticmethod
    def _norm(scores: dict) -> dict:
        if not scores:
            return {}
        vals = list(scores.values())
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return {k: 1.0 for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

    def retrieve(self, query: str, top_k: int = 5) -> list:
        n = top_k * 3
        bm_res = self.bm25.retrieve(query, n)
        dn_res = self.dense.retrieve(query, n)

        bm_s = self._norm({r["index"]: r["score"] for r in bm_res})
        dn_s = self._norm({r["index"]: r["score"] for r in dn_res})

        combined = {}
        for idx in set(bm_s) | set(dn_s):
            combined[idx] = (
                (1 - self.alpha) * bm_s.get(idx, 0.0)
                + self.alpha * dn_s.get(idx, 0.0)
            )

        ranked = sorted(combined, key=combined.get, reverse=True)[:top_k]
        return [
            {
                "question": self.documents[i],
                "answer": self.answers[i],
                "score": combined[i],
                "index": i,
            }
            for i in ranked
        ]

    def batch_retrieve(self, queries: list, top_k: int = 5) -> list:
        return [self.retrieve(q, top_k) for q in queries]
