#!/usr/bin/env python3
"""
TF-IDF Character N-gram Retriever
===================================
Competitor-disclosed approach: character-level TF-IDF for low-resource
African language retrieval. Character n-grams handle:
  - Akan special characters (ɛ, ɔ)
  - Amharic Ge'ez script
  - Agglutinative morphology (Luganda, Swahili)
"""

import os
import pickle
import logging
import numpy as np
from typing import List, Tuple

logger = logging.getLogger("afriqa")


class TFIDFRetriever:
    """
    Character n-gram TF-IDF retriever.
    Stores the training corpus and retrieves top-k most similar
    examples for a given query using cosine similarity.
    """

    def __init__(
        self,
        analyzer: str = "char_wb",
        ngram_range: Tuple[int, int] = (2, 5),
        max_features: int = 200_000,
        sublinear_tf: bool = True,
    ):
        """
        Args:
            analyzer: 'char_wb' = character n-grams within word boundaries (best for African langs)
            ngram_range: (min_n, max_n) character n-gram range
            max_features: vocabulary size cap
            sublinear_tf: apply log(1+tf) instead of raw tf (reduces impact of very frequent terms)
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            raise ImportError(
                "scikit-learn is required for TFIDFRetriever. "
                "Install it with: pip install scikit-learn"
            )

        self._TfidfVectorizer = TfidfVectorizer
        self._cosine_similarity = cosine_similarity

        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
        )
        self.corpus_questions: List[str] = []
        self.corpus_answers: List[str] = []
        self.corpus_matrix = None
        self._fitted = False

    def build_index(self, questions: List[str], answers: List[str]) -> None:
        """Fit the TF-IDF vectorizer on the question corpus and store Q+A pairs."""
        logger.info(f"Building TF-IDF index over {len(questions):,} examples …")
        self.corpus_questions = questions
        self.corpus_answers = answers
        self.corpus_matrix = self.vectorizer.fit_transform(questions)
        self._fitted = True
        logger.info(
            f"TF-IDF vocab size: {len(self.vectorizer.vocabulary_):,} | "
            f"Matrix: {self.corpus_matrix.shape}"
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        """
        Retrieve top-k most similar examples for a single query.

        Returns:
            List of dicts with keys: 'question', 'answer', 'score'
        """
        if not self._fitted:
            raise RuntimeError("Index not built. Call build_index() first.")

        query_vec = self.vectorizer.transform([query])
        scores = self._cosine_similarity(query_vec, self.corpus_matrix)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "question": self.corpus_questions[idx],
                "answer": self.corpus_answers[idx],
                "score": float(scores[idx]),
            })
        return results

    def retrieve_batch(self, queries: List[str], top_k: int = 5) -> List[List[dict]]:
        """
        Retrieve top-k examples for a batch of queries.

        Returns:
            List of retrieval results (one per query)
        """
        if not self._fitted:
            raise RuntimeError("Index not built. Call build_index() first.")

        query_matrix = self.vectorizer.transform(queries)
        all_scores = self._cosine_similarity(query_matrix, self.corpus_matrix)

        results = []
        for scores in all_scores:
            top_indices = np.argsort(scores)[::-1][:top_k]
            retrieved = []
            for idx in top_indices:
                retrieved.append({
                    "question": self.corpus_questions[idx],
                    "answer": self.corpus_answers[idx],
                    "score": float(scores[idx]),
                })
            results.append(retrieved)
        return results

    def save(self, path: str) -> None:
        """Save the fitted retriever to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "corpus_questions": self.corpus_questions,
                "corpus_answers": self.corpus_answers,
                "corpus_matrix": self.corpus_matrix,
            }, f)
        logger.info(f"TF-IDF retriever saved to {path}")

    @classmethod
    def load(cls, path: str) -> "TFIDFRetriever":
        """Load a saved retriever from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        retriever = cls.__new__(cls)
        retriever.vectorizer = data["vectorizer"]
        retriever.corpus_questions = data["corpus_questions"]
        retriever.corpus_answers = data["corpus_answers"]
        retriever.corpus_matrix = data["corpus_matrix"]
        retriever._fitted = True

        # Re-import sklearn dependencies
        from sklearn.metrics.pairwise import cosine_similarity
        retriever._cosine_similarity = cosine_similarity
        logger.info(f"TF-IDF retriever loaded from {path} "
                    f"({len(retriever.corpus_questions):,} examples)")
        return retriever
