"""BM25 sparse retriever backed by rank-bm25."""

import numpy as np
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.index = None
        self.documents: list = []
        self.answers: list = []

    def build_index(self, questions: list, answers: list):
        self.documents = list(questions)
        self.answers = list(answers)
        tokenised = [q.lower().split() for q in self.documents]
        self.index = BM25Okapi(tokenised, k1=self.k1, b=self.b)

    def retrieve(self, query: str, top_k: int = 5) -> list:
        tok_q = query.lower().split()
        scores = self.index.get_scores(tok_q)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "question": self.documents[i],
                "answer": self.answers[i],
                "score": float(scores[i]),
                "index": int(i),
            }
            for i in top_idx
        ]

    def batch_retrieve(self, queries: list, top_k: int = 5) -> list:
        return [self.retrieve(q, top_k) for q in queries]
