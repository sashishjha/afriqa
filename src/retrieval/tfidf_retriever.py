"""
TF-IDF Character N-gram Retriever (v2 — Amharic-aware)

Competitor-disclosed approach: character-level TF-IDF for low-resource
African language retrieval. Character n-grams handle:
- Akan special characters (ɛ, ɔ)
- Amharic Ge'ez script  ← NOW FIXED with transliteration
- Agglutinative morphology (Luganda, Swahili)

v2 changes:
  - Ge'ez → Latin transliteration (via Unicode character names)
  - Auto-detection of Amharic/Ge'ez text
  - Per-subset retriever support
  - Configurable n-gram range
"""

import os
import re
import pickle
import logging
import unicodedata
import numpy as np
from typing import List, Tuple, Dict, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("afriqa")


# ═══════════════════════════════════════════════════════════════════
#  Ge'ez (Ethiopic) → Latin Transliteration
# ═══════════════════════════════════════════════════════════════════

def _is_geez_char(ch: str) -> bool:
    """Check if a character is in the Ethiopic Unicode block (U+1200–U+137F)."""
    cp = ord(ch)
    return 0x1200 <= cp <= 0x137F


def _has_geez(text: str, threshold: float = 0.15) -> bool:
    """
    Detect if text contains significant Ge'ez script content.
    Returns True if > threshold fraction of alphabetic chars are Ge'ez.
    """
    if not text:
        return False
    alpha_count = 0
    geez_count = 0
    for ch in text:
        if ch.isalpha() or _is_geez_char(ch):
            alpha_count += 1
            if _is_geez_char(ch):
                geez_count += 1
    if alpha_count == 0:
        return False
    return (geez_count / alpha_count) > threshold


# Build a one-time lookup dict: Ge'ez codepoint → Latin string
# Uses Unicode character names: "ETHIOPIC SYLLABLE HA" → "ha"
_GEEZ_TRANSLITERATION_CACHE: Dict[str, str] = {}


def _build_geez_cache():
    """Build the Ge'ez → Latin transliteration cache from Unicode names."""
    global _GEEZ_TRANSLITERATION_CACHE
    if _GEEZ_TRANSLITERATION_CACHE:
        return  # Already built

    for cp in range(0x1200, 0x1380):
        ch = chr(cp)
        try:
            name = unicodedata.name(ch, "")
        except ValueError:
            continue

        if "SYLLABLE" in name:
            # "ETHIOPIC SYLLABLE HA" → "ha"
            # "ETHIOPIC SYLLABLE GLOTTAL A" → "glottal a" → "a"
            syllable = name.split("SYLLABLE ")[-1].lower()
            # Clean up multi-word syllable names
            syllable = syllable.replace(" ", "")
            _GEEZ_TRANSLITERATION_CACHE[ch] = syllable
        elif "PUNCTUATION" in name or "MARK" in name:
            _GEEZ_TRANSLITERATION_CACHE[ch] = " "
        elif "DIGIT" in name or "NUMBER" in name:
            # Ethiopic numerals → keep as marker
            _GEEZ_TRANSLITERATION_CACHE[ch] = " "
        else:
            _GEEZ_TRANSLITERATION_CACHE[ch] = " "

    logger.debug(f"Built Ge'ez transliteration cache: {len(_GEEZ_TRANSLITERATION_CACHE)} entries")


def transliterate_geez(text: str) -> str:
    """
    Transliterate Ge'ez (Ethiopic) script to Latin approximation.

    Examples:
        ሀ → ha,  ለ → la,  ሐ → hha,  መ → ma,  ሰ → sa
        "ጤና ይስጥልኝ" → "tena yisitiliñi" (approx.)

    Non-Ge'ez characters are passed through unchanged.
    """
    _build_geez_cache()

    result = []
    for ch in text:
        if ch in _GEEZ_TRANSLITERATION_CACHE:
            result.append(_GEEZ_TRANSLITERATION_CACHE[ch])
        else:
            result.append(ch)
    return "".join(result)


def preprocess_text(text: str) -> str:
    """
    Language-aware text preprocessing for TF-IDF.
    - Transliterates Ge'ez → Latin if Ge'ez detected
    - Lowercases
    - Normalises whitespace
    """
    if not text:
        return ""
    # Transliterate Ge'ez if present
    if _has_geez(text):
        text = transliterate_geez(text)
    # Lowercase + normalise whitespace
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ═══════════════════════════════════════════════════════════════════
#  TF-IDF Retriever
# ═══════════════════════════════════════════════════════════════════

class TFIDFRetriever:
    """
    Character n-gram TF-IDF retriever.
    Stores the training corpus and retrieves top-k most similar
    examples for a given query using cosine similarity.

    v2: Automatically transliterates Ge'ez script before vectorisation.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (2, 5),
        max_features: int = 80000,
        sublinear_tf: bool = True,
    ):
        """
        Args:
            ngram_range: (min_n, max_n) for character n-grams.
                         (2,5) works well after Ge'ez transliteration.
            max_features: Max vocabulary size for the vectoriser.
            sublinear_tf: Use sublinear TF scaling (log(1 + tf)).
        """
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf

        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",      # char_wb = character n-grams within word boundaries
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            strip_accents=None,      # We handle normalisation ourselves
            dtype=np.float32,
        )

        self.corpus_questions: List[str] = []
        self.corpus_answers: List[str] = []
        self.corpus_questions_processed: List[str] = []
        self.tfidf_matrix = None

    def fit(self, questions: List[str], answers: List[str]):
        """
        Build the TF-IDF index from the training corpus.

        Args:
            questions: List of question strings.
            answers:   List of answer strings (parallel to questions).
        """
        assert len(questions) == len(answers), "Questions and answers must be parallel."
        self.corpus_questions = questions
        self.corpus_answers = answers

        # Preprocess all questions (transliterates Ge'ez automatically)
        self.corpus_questions_processed = [preprocess_text(q) for q in questions]

        n_geez = sum(1 for q in questions if _has_geez(q))
        logger.info(
            f"TFIDFRetriever.fit: {len(questions)} docs, "
            f"{n_geez} Ge'ez-script docs transliterated, "
            f"ngram_range={self.ngram_range}, max_features={self.max_features}"
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus_questions_processed)
        logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top-k most similar training examples for a query.

        Args:
            query: The input question string.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: question, answer, score
        """
        if self.tfidf_matrix is None:
            raise RuntimeError("Retriever not fitted. Call .fit() first.")

        # Apply same preprocessing as corpus (including Ge'ez transliteration)
        query_processed = preprocess_text(query)
        query_vec = self.vectorizer.transform([query_processed])

        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                "question": self.corpus_questions[idx],  # Return ORIGINAL (not transliterated)
                "answer": self.corpus_answers[idx],
                "score": float(scores[idx]),
            })
        return results

    def save(self, path: str):
        """Save the fitted retriever to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "vectorizer": self.vectorizer,
                    "corpus_questions": self.corpus_questions,
                    "corpus_answers": self.corpus_answers,
                    "corpus_questions_processed": self.corpus_questions_processed,
                    "tfidf_matrix": self.tfidf_matrix,
                    "ngram_range": self.ngram_range,
                    "max_features": self.max_features,
                },
                f,
            )
        logger.info(f"TFIDFRetriever saved to {path}")

    def load(self, path: str):
        """Load a previously fitted retriever from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.corpus_questions = data["corpus_questions"]
        self.corpus_answers = data["corpus_answers"]
        self.corpus_questions_processed = data.get("corpus_questions_processed", [])
        self.tfidf_matrix = data["tfidf_matrix"]
        self.ngram_range = data.get("ngram_range", (2, 5))
        self.max_features = data.get("max_features", 80000)
        logger.info(f"TFIDFRetriever loaded from {path}: {len(self.corpus_questions)} docs")


# ═══════════════════════════════════════════════════════════════════
#  Per-Subset Retriever Manager (for language-aware retrieval)
# ═══════════════════════════════════════════════════════════════════

class PerSubsetTFIDFRetriever:
    """
    Builds and manages separate TF-IDF retrievers per language subset.

    This ensures:
    - Amharic queries only retrieve from Amharic training data
    - No cross-language contamination in retrieval
    - Each subset gets optimally tuned n-gram features
    """

    def __init__(self, ngram_range=(2, 5), max_features=80000):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.retrievers: Dict[str, TFIDFRetriever] = {}
        self.fallback_retriever: Optional[TFIDFRetriever] = None

    def fit(self, questions: List[str], answers: List[str], subsets: List[str]):
        """
        Build separate TF-IDF indices per subset.

        Args:
            questions: List of question strings.
            answers:   List of answer strings.
            subsets:   List of subset labels (e.g., 'Amh_Eth', 'Eng_Uga').
        """
        import pandas as pd

        df = pd.DataFrame({"question": questions, "answer": answers, "subset": subsets})

        for subset_name, group in df.groupby("subset"):
            retriever = TFIDFRetriever(
                ngram_range=self.ngram_range,
                max_features=self.max_features,
            )
            retriever.fit(
                group["question"].tolist(),
                group["answer"].tolist(),
            )
            self.retrievers[subset_name] = retriever
            logger.info(f"Per-subset retriever for '{subset_name}': {len(group)} docs")

        # Also build a global fallback
        self.fallback_retriever = TFIDFRetriever(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
        )
        self.fallback_retriever.fit(questions, answers)
        logger.info(f"Global fallback retriever: {len(questions)} docs")

    def retrieve(self, query: str, subset: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve using the subset-specific retriever, falling back to global.
        """
        retriever = self.retrievers.get(subset, self.fallback_retriever)
        if retriever is None:
            raise RuntimeError("Retriever not fitted. Call .fit() first.")
        return retriever.retrieve(query, top_k=top_k)
