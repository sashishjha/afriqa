#!/usr/bin/env python3
"""
run_gemma_vllm.py  —  Dense-Retrieval + Per-Subset Routing Pipeline
====================================================================
Zindi: Multilingual Health QA in Low-Resource African Languages

Strategy (v3):
  1. Encode all Train questions with multilingual-e5-large (dense embeddings).
  2. Encode all Test questions.
  3. For each Test row, compute cosine similarity to Train rows *within the
     same subset*, retrieve top-k neighbours.
  4. Per-subset routing:
       - HIGH-REUSE subsets (Eng_Uga, Eng_Ken, Swa_Ken, Lug_Uga):
             If top-1 cosine sim >= threshold → return Train answer verbatim.
       - MEDIUM-REUSE subsets (Eng_Eth):
             Slightly higher threshold; otherwise generate.
       - LOW-REUSE subsets (Aka_Gha, Eng_Gha, Amh_Eth):
             High threshold; most rows go to Gemma generation with
             few-shot retrieved context.
  5. For the generation path, query a local vLLM server (Gemma-4-31B-it)
     with a carefully crafted few-shot prompt using retrieved examples.
  6. Post-process outputs and write Zindi-format submission.csv.

Author : Team Sashish (IISc Bengaluru)
Date   : June 2026
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Ensure we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.tfidf_retriever import TFIDFRetriever, preprocess_text

# ===========================================================================
# DEFAULT PATHS & CONSTANTS
# ===========================================================================
EMBED_MODEL_PATH = "/mnt/data/sashishj/models/intfloat_multilingual-e5-large"
VLLM_BASE_URL    = "http://localhost:8000/v1"
MODEL_NAME       = "gemma-4-31B-it"
TOP_K            = 5

# Per-subset retrieval thresholds (cosine similarity)
# Tuned to p90 of in-sample training data distribution (Experiment B)
SUBSET_THRESHOLDS = {
    "Aka_Gha": 0.979,
    "Amh_Eth": 0.988,
    "Eng_Eth": 0.997,  # Pulled back from 0.999 for safety
    "Eng_Gha": 0.966,
    "Eng_Ken": 0.989,
    "Eng_Uga": 0.995,
    "Lug_Uga": 0.983,
    "Swa_Ken": 0.988,
}

# Language names (used in prompts)
SUBSET_TO_LANG = {
    "Eng_Uga": "English",
    "Eng_Ken": "English",
    "Eng_Gha": "English",
    "Eng_Eth": "English",
    "Aka_Gha": "Akan (Twi)",
    "Amh_Eth": "Amharic",
    "Lug_Uga": "Luganda",
    "Swa_Ken": "Swahili",
}

# Native-language anchoring cues (instructing the model in-language)
LANG_CUES = {
    "Akan (Twi)": "Kyerɛ mmuaeɛ no wɔ Twi kasa mu.",
    "Amharic":    "መልሱን በአማርኛ ብቻ ስጥ።",
    "Luganda":    "Ddamu ekibuuzo mu Luganda.",
    "Swahili":    "Jibu kwa Kiswahili.",
    "English":    "",
}

# Token-per-word multipliers (African languages tokenize heavier)
TOKENS_PER_WORD = {
    "Akan (Twi)": 4.0,
    "Amharic":    4.5,
    "Luganda":    3.5,
    "Swahili":    3.0,
    "English":    1.5,
}

# ===========================================================================
# LOGGING
# ===========================================================================
def setup_logging(output_dir: str):
    """Configure logging to both console and file."""
    log_path = os.path.join(output_dir, "run.log")
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="w"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    return logging.getLogger(__name__)


# ===========================================================================
# DATA LOADING
# ===========================================================================
def detect_columns(df: pd.DataFrame):
    """Detect the question and answer column names flexibly."""
    q_col, a_col = None, None
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("input", "question"):
            q_col = c
        if cl in ("output", "answer"):
            a_col = c
    if q_col is None:
        raise ValueError(f"Cannot detect question column. Columns: {list(df.columns)}")
    return q_col, a_col


def extract_subset(row_id: str) -> str:
    """
    Extract subset code from Zindi IDs, e.g.
      'ID_TR_Aka_Gha_A3B1799D' -> 'Aka_Gha'
      'ID_TS_Eng_Uga_12345678' -> 'Eng_Uga'
    Strategy: split on '_', take parts at index 2 and 3.
    """
    parts = row_id.split("_")
    if len(parts) >= 4:
        return parts[2] + "_" + parts[3]
    return "Unknown"


def load_data(train_path, test_path, val_path=None, logger=None):
    """Load train, test (and optional val) CSVs."""
    log = logger or logging.getLogger(__name__)

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    log.info(f"Loaded Train: {len(train_df)} rows from {train_path}")
    log.info(f"Loaded Test : {len(test_df)} rows from {test_path}")

    # Merge validation into train for maximum retrieval coverage
    if val_path and os.path.exists(val_path):
        val_df = pd.read_csv(val_path)
        log.info(f"Loaded Val  : {len(val_df)} rows from {val_path}")
        train_df = pd.concat([train_df, val_df], ignore_index=True)
        log.info(f"Merged Train+Val: {len(train_df)} total rows")

    # Detect columns
    q_col_train, a_col_train = detect_columns(train_df)
    q_col_test, _            = detect_columns(test_df)

    # Ensure 'subset' column exists
    if "subset" not in train_df.columns:
        train_df["subset"] = train_df["ID"].apply(extract_subset)
    if "subset" not in test_df.columns:
        test_df["subset"] = test_df["ID"].apply(extract_subset)

    log.info(f"Train columns: q='{q_col_train}', a='{a_col_train}'")
    log.info(f"Test  columns: q='{q_col_test}'")
    log.info(f"Train subsets: {dict(train_df['subset'].value_counts())}")
    log.info(f"Test  subsets: {dict(test_df['subset'].value_counts())}")

    return train_df, test_df, q_col_train, a_col_train, q_col_test


# ===========================================================================
# DENSE RETRIEVER
# ===========================================================================
class DenseRetriever:
    """
    Dense semantic retriever using multilingual-e5-large.
    Encodes questions with 'query: ' prefix (required by mE5).
    Uses cosine similarity over L2-normalised embeddings.
    Caches train embeddings to disk as .npy for reuse.
    """

    def __init__(self, model_path: str, device: str = "cuda:0", logger=None):
        self.log = logger or logging.getLogger(__name__)
        self.log.info(f"Loading embedding model from: {model_path}")
        self.log.info(f"  Device: {device}")

        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_path, device=device)
        self.log.info(f"  Model loaded. Embedding dim: {self.model.get_sentence_embedding_dimension()}")

        self.train_embeddings = None
        self.train_texts = None

    def encode_texts(self, texts: list, prefix: str = "query: ",
                     batch_size: int = 128, desc: str = "Encoding"):
        """Encode a list of texts with the required prefix."""
        prefixed = [prefix + t for t in texts]
        self.log.info(f"  {desc}: {len(prefixed)} texts (batch_size={batch_size})")
        embeddings = self.model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2-norm for cosine = dot product
            convert_to_numpy=True,
        )
        self.log.info(f"  Done. Shape: {embeddings.shape}")
        return embeddings

    def build_index(self, train_questions: list, cache_path: str = None):
        """Encode all training questions; optionally save/load from cache."""
        if cache_path and os.path.exists(cache_path):
            self.log.info(f"  Loading cached train embeddings from {cache_path}")
            self.train_embeddings = np.load(cache_path)
            self.log.info(f"  Loaded shape: {self.train_embeddings.shape}")
        else:
            self.train_embeddings = self.encode_texts(
                train_questions, prefix="query: ",
                batch_size=128, desc="Encoding TRAIN questions"
            )
            if cache_path:
                np.save(cache_path, self.train_embeddings)
                self.log.info(f"  Saved train embeddings to {cache_path}")
        self.train_texts = train_questions

    def batch_retrieve(self, test_embeddings: np.ndarray,
                       test_subsets: list, train_subsets: np.ndarray,
                       top_k: int = 5):
        """
        Batch retrieval for all test rows.
        Returns list of lists of (train_index, similarity) tuples.
        """
        self.log.info(f"Computing full similarity matrix ({test_embeddings.shape[0]} x {self.train_embeddings.shape[0]})...")
        # Full sim matrix at once for speed
        sim_matrix = test_embeddings @ self.train_embeddings.T  # (T, N)
        self.log.info(f"  Sim matrix shape: {sim_matrix.shape}")

        # Precompute subset masks
        unique_subsets = np.unique(train_subsets)
        subset_masks = {s: (train_subsets == s) for s in unique_subsets}

        all_results = []
        for i in range(len(test_embeddings)):
            subset = test_subsets[i]
            mask = subset_masks.get(subset, np.zeros(len(train_subsets), dtype=bool))
            sims = sim_matrix[i].copy()
            sims[~mask] = -1.0
            top_idx = np.argsort(sims)[::-1][:top_k]
            results = [(int(idx), float(sims[idx])) for idx in top_idx]
            all_results.append(results)

        return all_results


# ===========================================================================
# LENGTH CALIBRATION
# ===========================================================================
def compute_length_stats(train_df, a_col, logger=None):
    """Compute median, p75, p90 answer word counts per subset."""
    log = logger or logging.getLogger(__name__)
    stats = {}
    for subset in train_df["subset"].unique():
        answers = train_df[train_df["subset"] == subset][a_col].dropna()
        word_counts = answers.str.split().str.len()
        s = {
            "median": int(word_counts.median()),
            "p75":    int(word_counts.quantile(0.75)),
            "p90":    int(word_counts.quantile(0.90)),
            "mean":   round(float(word_counts.mean()), 1),
        }
        stats[subset] = s
        log.info(f"  Length stats [{subset:10s}]: median={s['median']}, p75={s['p75']}, p90={s['p90']}, mean={s['mean']}")
    return stats


def compute_max_tokens(subset: str, length_stats: dict) -> int:
    """Compute max_tokens for vLLM based on subset length stats."""
    lang = SUBSET_TO_LANG.get(subset, "English")
    multiplier = TOKENS_PER_WORD.get(lang, 3.0)
    target_words = length_stats.get(subset, {}).get("p75", 150)
    max_tokens = int(target_words * multiplier)
    max_tokens = max(64, min(max_tokens, 2048))
    return max_tokens


# ===========================================================================
# PROMPT BUILDING
# ===========================================================================
def build_generation_prompt(test_question: str, retrieved_examples: list,
                            subset: str, target_words: int):
    """
    Build system + user prompts for Gemma using strict Opus formatting.
    retrieved_examples: list of (question, answer) tuples from training data.
    """
    lang = SUBSET_TO_LANG.get(subset, "English")
    native_cue = LANG_CUES.get(lang, "")

    system_prompt = (
        f"You are a multilingual health information assistant specialising in "
        f"maternal, sexual, and reproductive health in African communities.\n\n"
        f"INSTRUCTIONS:\n"
        f"1. LANGUAGE: Answer in the same language as the question ({lang}). "
        f"Preserve any English medical or technical terms exactly as they appear — do not translate them.\n"
        f"2. TERMINOLOGY: Reuse the EXACT medical and health terms that appear "
        f"in the question and in the reference examples below. Do not paraphrase technical terms.\n"
        f"3. LENGTH: Your answer should be approximately {target_words} words long. "
        f"Match the length and detail level of the reference answers provided below. Do not over-elaborate.\n"
        f"4. STYLE: Mirror the phrasing patterns and sentence structures from "
        f"the reference answers. Use similar connectors, list formats, and explanation styles.\n"
        f"5. ACCURACY: Answer the specific question asked. Use information from "
        f"the reference examples as a knowledge base, but adapt the answer to address the exact question.\n"
        f"6. COMPLETENESS: Cover all key points that a correct answer would "
        f"include, but do not add unnecessary filler.\n\n"
        f"Output ONLY the answer. Do NOT include any preamble, greeting, label, translation note, or meta-commentary."
    )
    if native_cue:
        system_prompt += f"\n\n{native_cue}"

    # Build few-shot examples
    examples_parts = []
    for i, (q, a) in enumerate(retrieved_examples, 1):
        examples_parts.append(f"Q{i}: {q}\nA{i}: {a}")
    examples_text = "\n\n".join(examples_parts)

    user_prompt = (
        f"Here are similar health Q&A examples for reference:\n\n"
        f"{examples_text}\n\n"
        f"Now answer the following question using the same style, "
        f"terminology, and level of detail as the examples above:\n\n"
        f"Q: {test_question}\n"
        f"A:"
    )

    return system_prompt, user_prompt


# ===========================================================================
# vLLM CLIENT
# ===========================================================================
def query_vllm(system_prompt: str, user_prompt: str,
               vllm_url: str, model_name: str,
               max_tokens: int = 512, temperature: float = 0.1,
               logger=None):
    """
    Query the local vLLM server using the OpenAI-compatible chat API.
    Uses the requests library to POST to /v1/chat/completions.
    """
    log = logger or logging.getLogger(__name__)
    endpoint = f"{vllm_url}/chat/completions"

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "stop": [
            "\n\nQuestion:",
            "\n\nExample ",
            "\n---",
            "Question:",
            "\nNote:",
        ],
    }

    try:
        resp = requests.post(endpoint, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content.strip()
    except requests.exceptions.Timeout:
        log.warning("  vLLM request timed out (180s). Returning empty string.")
        return ""
    except requests.exceptions.ConnectionError as e:
        log.error(f"  vLLM connection error: {e}")
        return ""
    except Exception as e:
        log.error(f"  vLLM query error: {e}")
        return ""


def wait_for_vllm(vllm_url: str, timeout: int = 600, logger=None):
    """Wait for the vLLM server to become ready."""
    log = logger or logging.getLogger(__name__)
    log.info(f"Waiting for vLLM at {vllm_url} (timeout={timeout}s)...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{vllm_url}/models", timeout=5)
            if r.status_code == 200:
                models = r.json().get("data", [])
                log.info(f"  vLLM ready! Available models: {[m.get('id') for m in models]}")
                return True
        except Exception:
            pass
        time.sleep(5)
    log.error(f"  vLLM not ready after {timeout}s!")
    return False


# ===========================================================================
# POST-PROCESSING
# ===========================================================================
# Patterns for common LLM preambles to strip
PREAMBLE_PATTERNS = [
    # English preambles
    r"^(?:Here(?:'s| is) (?:the |my |an? )?(?:answer|response|translation)[:\-\.\s]*)",
    r"^(?:Sure[!,.]?\s*(?:Here(?:'s| is)[:\s]*)?)",
    r"^(?:Certainly[!,.]?\s*(?:Here(?:'s| is)[:\s]*)?)",
    r"^(?:Of course[!,.]?\s*)",
    r"^(?:The answer is[:\s]*)",
    r"^(?:Answer[:\s]*)",
    r"^(?:Response[:\s]*)",
    # Language-specific preambles
    r"^(?:Here is the answer in (?:Akan|Twi|Amharic|Luganda|Swahili|English)[:\s]*)",
    r"^(?:In (?:Akan|Twi|Amharic|Luganda|Swahili|English)[,:\s]*)",
    r"^(?:Translation[:\s]*)",
    r"^(?:Below is[:\s]*)",
]
COMPILED_PREAMBLES = [re.compile(p, re.IGNORECASE) for p in PREAMBLE_PATTERNS]

# Markdown artifacts
MARKDOWN_PATTERNS = [
    (re.compile(r"\*\*(.+?)\*\*"), r"\1"),        # **bold** -> bold
    (re.compile(r"\*(.+?)\*"), r"\1"),              # *italic* -> italic
    (re.compile(r"^#{1,6}\s+", re.MULTILINE), ""),  # headings
    (re.compile(r"^[-*]\s+", re.MULTILINE), ""),     # bullet points
    (re.compile(r"^\d+\.\s+", re.MULTILINE), ""),    # numbered lists
]


def postprocess_answer(text: str) -> str:
    """Clean up LLM-generated answer."""
    if not text:
        return ""

    # Strip whitespace
    text = text.strip()

    # Remove preambles (apply iteratively up to 3 times)
    for _ in range(3):
        changed = False
        for pattern in COMPILED_PREAMBLES:
            new_text = pattern.sub("", text).strip()
            if new_text != text:
                text = new_text
                changed = True
        if not changed:
            break

    # Remove markdown artifacts
    for pattern, replacement in MARKDOWN_PATTERNS:
        text = pattern.sub(replacement, text)

    # Remove leading/trailing quotes
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]

    # Collapse multiple newlines/spaces
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Dense-Retrieval + Per-Subset Routing for Zindi Multilingual Health QA"
    )
    parser.add_argument("--train-path", type=str, required=True,
                        help="Path to Train.csv")
    parser.add_argument("--val-path", type=str, default=None,
                        help="Path to Val.csv (optional, merged into train for retrieval)")
    parser.add_argument("--test-path", type=str, required=True,
                        help="Path to Test.csv")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Directory for outputs")
    parser.add_argument("--vllm-url", type=str, default=VLLM_BASE_URL,
                        help="vLLM server base URL")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME,
                        help="Model name served by vLLM")
    parser.add_argument("--embed-model-path", type=str, default=EMBED_MODEL_PATH,
                        help="Local path to multilingual-e5-large")
    parser.add_argument("--embed-gpu", type=int, default=0,
                        help="GPU index for embedding model (default: 0)")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help="Number of retrieved examples for few-shot (default: 5)")
    parser.add_argument("--reuse-threshold", type=float, default=None,
                        help="(Legacy/ignored) Global reuse threshold. Per-subset thresholds are used instead.")
    parser.add_argument("--skip-vllm-wait", action="store_true",
                        help="Skip waiting for vLLM (for debug/retrieval-only runs)")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Use retrieval for ALL rows (no generation). Good for baseline.")
    args = parser.parse_args()

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)

    logger.info("=" * 70)
    logger.info("  DENSE RETRIEVAL + PER-SUBSET ROUTING PIPELINE v3")
    logger.info("=" * 70)
    logger.info(f"  Embed model : {args.embed_model_path}")
    logger.info(f"  vLLM URL    : {args.vllm_url}")
    logger.info(f"  LLM model   : {args.model_name}")
    logger.info(f"  Top-K       : {args.top_k}")
    logger.info(f"  Output dir  : {args.output_dir}")
    logger.info(f"  Retrieval-only mode: {args.retrieval_only}")
    logger.info("")

    # ----- 1. LOAD DATA ----- #
    logger.info("[1/6] Loading data...")
    train_df, test_df, q_col, a_col, q_col_test = load_data(
        args.train_path, args.test_path, args.val_path, logger
    )

    # Ensure no NaN in question/answer
    train_df[q_col] = train_df[q_col].fillna("")
    train_df[a_col] = train_df[a_col].fillna("")
    test_df[q_col_test] = test_df[q_col_test].fillna("")

    # ----- 2. BUILD DENSE INDEX ----- #
    logger.info("[2/6] Building dense retrieval index...")
    embed_device = f"cuda:{args.embed_gpu}"
    retriever = DenseRetriever(args.embed_model_path, device=embed_device, logger=logger)

    # Cache path for train embeddings
    n_train = len(train_df)
    cache_name = f"train_embeddings_{n_train}.npy"
    cache_path = os.path.join(args.output_dir, cache_name)

    train_questions = train_df[q_col].tolist()
    retriever.build_index(train_questions, cache_path=cache_path)

    # Encode test questions
    test_questions = test_df[q_col_test].tolist()
    test_embeddings = retriever.encode_texts(
        test_questions, prefix="query: ",
        batch_size=128, desc="Encoding TEST questions"
    )

    logger.info("[2.5/6] Building TF-IDF sparse index for Hybrid Re-ranking...")
    tfidf = TFIDFRetriever(ngram_range=(3, 5), max_features=80000, min_df=2, max_df=0.95)
    tfidf.fit(train_questions, train_df[a_col].tolist())
    
    logger.info("  Encoding TEST questions with TF-IDF...")
    test_questions_proc = [preprocess_text(q) for q in test_questions]
    test_sparse_matrix = tfidf.vectorizer.transform(test_questions_proc)
    
    logger.info("  Computing full TF-IDF similarity matrix...")
    # This is fast: (test_size, train_size)
    tfidf_sim_matrix = (test_sparse_matrix * tfidf.tfidf_matrix.T).toarray()

    # ----- 3. BATCH RETRIEVAL ----- #
    logger.info("[3/6] Performing batch retrieval (same-subset filtering)...")
    test_subsets = test_df["subset"].tolist()
    train_subsets = train_df["subset"].values

    # We fetch top 25 from dense to allow robust RRF re-ranking
    all_retrievals = retriever.batch_retrieve(
        test_embeddings, test_subsets, train_subsets, top_k=25
    )
    logger.info(f"  Retrieval complete for {len(all_retrievals)} test rows.")

    # Log similarity statistics per subset
    for subset in sorted(set(test_subsets)):
        subset_sims = [r[0][1] for i, r in enumerate(all_retrievals) if test_subsets[i] == subset and len(r) > 0]
        if subset_sims:
            logger.info(
                f"  [{subset:10s}] top-1 sim: mean={np.mean(subset_sims):.4f}, "
                f"median={np.median(subset_sims):.4f}, "
                f"min={np.min(subset_sims):.4f}, max={np.max(subset_sims):.4f}"
            )

    # ----- 4. COMPUTE LENGTH STATS ----- #
    logger.info("[4/6] Computing answer length statistics per subset...")
    length_stats = compute_length_stats(train_df, a_col, logger)

    # ----- 5. WAIT FOR vLLM (if needed) ----- #
    need_generation = not args.retrieval_only
    if need_generation and not args.skip_vllm_wait:
        logger.info("[5/6] Waiting for vLLM server...")
        vllm_ready = wait_for_vllm(args.vllm_url, timeout=600, logger=logger)
        if not vllm_ready:
            logger.warning("  vLLM not ready — falling back to retrieval-only mode!")
            need_generation = False
    elif args.retrieval_only:
        logger.info("[5/6] Skipping vLLM (retrieval-only mode).")
    else:
        logger.info("[5/6] Skipping vLLM wait (--skip-vllm-wait).")

    # ----- 6. GENERATE PREDICTIONS ----- #
    logger.info("[6/6] Generating predictions with per-subset routing...")
    predictions = []
    routing_log = []  # For debug output

    total = len(test_df)
    n_retrieved = 0
    n_generated = 0
    n_fallback  = 0

    for i in range(total):
        row = test_df.iloc[i]
        test_id   = row["ID"]
        subset    = row["subset"]
        question  = row[q_col_test]
        retrievals = all_retrievals[i]

        # Top-1 info
        if len(retrievals) > 0:
            top1_idx, top1_sim = retrievals[0]
            top1_answer = train_df.iloc[top1_idx][a_col]
        else:
            top1_idx, top1_sim = -1, 0.0
            top1_answer = ""

        # Get threshold for this subset
        threshold = SUBSET_THRESHOLDS.get(subset, 0.96)

        # Routing decision uses DENSE ONLY — do not use hybrid score here!
        if top1_sim >= threshold or args.retrieval_only:
            # ----- RETRIEVAL PATH -----
            prediction = str(top1_answer) if top1_answer else ""
            route = "retrieval"
            n_retrieved += 1
        elif need_generation:
            # ----- GENERATION PATH -----
            # Re-rank the top 25 dense candidates using Reciprocal Rank Fusion (RRF)
            candidate_indices = [idx for idx, sim in retrievals]
            
            # Dense ranks are implicitly the order of candidate_indices
            dense_ranks = {idx: rank for rank, idx in enumerate(candidate_indices)}
            
            # Get TF-IDF scores for these candidates and rank them
            sparse_scores = tfidf_sim_matrix[i, candidate_indices]
            sparse_order = np.argsort(-sparse_scores)
            sparse_ranks = {candidate_indices[pos]: rank for rank, pos in enumerate(sparse_order)}
            
            # Combine via RRF
            k_rrf = 60
            hybrid_scores = {}
            for idx in candidate_indices:
                hybrid_scores[idx] = 1.0 / (k_rrf + dense_ranks[idx]) + 1.0 / (k_rrf + sparse_ranks[idx])
                
            hybrid_top_indices = sorted(candidate_indices, key=lambda idx: hybrid_scores[idx], reverse=True)

            # Build few-shot context from hybrid top-k
            # Filter low-quality dense retrievals (< 0.70) and keep top 3
            few_shot_examples = []
            for idx in hybrid_top_indices:
                # Look up the dense sim to ensure semantic safety
                sim = next(s for d_idx, s in retrievals if d_idx == idx)
                if sim > 0.70 and len(few_shot_examples) < 3:
                    ex_q = train_df.iloc[idx][q_col]
                    ex_a = train_df.iloc[idx][a_col]
                    if ex_q and ex_a:
                        few_shot_examples.append((str(ex_q), str(ex_a)))

            # Dynamically compute target length based on retrieved examples
            if few_shot_examples:
                retrieved_lengths_words = [len(a.split()) for _, a in few_shot_examples]
                target_words = int(np.median(retrieved_lengths_words))
            else:
                target_words = length_stats.get(subset, {}).get("p75", 100)
            
            # Ensure target_words isn't zero
            target_words = max(target_words, 10)

            # Compute max_tokens dynamically with 40% headroom
            lang = SUBSET_TO_LANG.get(subset, "English")
            multiplier = TOKENS_PER_WORD.get(lang, 3.0)
            max_tokens = int(target_words * multiplier * 1.4)
            max_tokens = max(64, min(max_tokens, 2048))

            system_prompt, user_prompt = build_generation_prompt(
                test_question=question,
                retrieved_examples=few_shot_examples,
                subset=subset,
                target_words=target_words,
            )

            raw_answer = query_vllm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                vllm_url=args.vllm_url,
                model_name=args.model_name,
                max_tokens=max_tokens,
                temperature=0.15,
                logger=logger,
            )

            prediction = postprocess_answer(raw_answer)

            # Fallback: if generation is empty or very short, use retrieval
            if len(prediction.strip()) < 10 and top1_answer:
                prediction = str(top1_answer)
                route = "fallback_to_retrieval"
                n_fallback += 1
            else:
                route = "generation"
                n_generated += 1
        else:
            # vLLM not available and not retrieval-only — forced retrieval
            prediction = str(top1_answer) if top1_answer else ""
            route = "forced_retrieval"
            n_retrieved += 1

        predictions.append(prediction)
        routing_log.append({
            "ID": test_id,
            "subset": subset,
            "route": route,
            "top1_sim": round(top1_sim, 5),
            "threshold": threshold,
            "top1_train_idx": top1_idx,
            "pred_length_words": len(prediction.split()) if prediction else 0,
        })

        # Progress logging
        if (i + 1) % 100 == 0 or (i + 1) == total:
            logger.info(
                f"  Progress: {i+1}/{total} | "
                f"retrieved={n_retrieved} generated={n_generated} fallback={n_fallback}"
            )

    # ----- FINAL STATS ----- #
    logger.info("")
    logger.info("=" * 50)
    logger.info("  ROUTING SUMMARY")
    logger.info("=" * 50)
    logger.info(f"  Total rows     : {total}")
    logger.info(f"  Retrieved      : {n_retrieved} ({100*n_retrieved/total:.1f}%)")
    logger.info(f"  Generated      : {n_generated} ({100*n_generated/total:.1f}%)")
    logger.info(f"  Fallback->Retr : {n_fallback} ({100*n_fallback/total:.1f}%)")

    # Per-subset breakdown
    logger.info("")
    logger.info("  Per-subset breakdown:")
    for subset in sorted(set(test_subsets)):
        sub_routes = [r for r in routing_log if r["subset"] == subset]
        n_sub = len(sub_routes)
        n_ret = sum(1 for r in sub_routes if "retrieval" in r["route"])
        n_gen = sum(1 for r in sub_routes if r["route"] == "generation")
        n_fb  = sum(1 for r in sub_routes if r["route"] == "fallback_to_retrieval")
        avg_sim = np.mean([r["top1_sim"] for r in sub_routes]) if sub_routes else 0
        gen_ratio = (n_gen / n_sub * 100) if n_sub > 0 else 0.0
        logger.info(
            f"    {subset:10s}: n={n_sub:4d} | ret={n_ret:4d} gen={n_gen:4d} fb={n_fb:3d} | "
            f"Ratio={gen_ratio:.1f}% gen | avg_sim={avg_sim:.4f}"
        )

    # ----- WRITE SUBMISSION CSV ----- #
    logger.info("")
    logger.info("Writing submission files...")

    # Ensure no NaN/None predictions
    clean_predictions = []
    for p in predictions:
        if p is None or (isinstance(p, float) and np.isnan(p)) or str(p).strip() == "" or str(p).lower() == "nan":
            clean_predictions.append("No answer available.")
        else:
            clean_predictions.append(str(p))

    # Generate timestamp for unique filenames
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Submission CSV (Zindi format)
    submission_path = os.path.join(args.output_dir, f"submission_{ts}.csv")
    with open(submission_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["ID", "TargetRLF1", "TargetR1F1", "TargetLLM"])
        for idx in range(len(test_df)):
            row = test_df.iloc[idx]
            pred = clean_predictions[idx]
            writer.writerow([row["ID"], pred, pred, pred])
    logger.info(f"  Submission written: {submission_path}")

    # Debug predictions CSV
    debug_path = os.path.join(args.output_dir, f"predictions_debug_{ts}.csv")
    debug_df = pd.DataFrame(routing_log)
    debug_df["prediction"] = clean_predictions
    debug_df["question"] = test_df[q_col_test].tolist()
    debug_df.to_csv(debug_path, index=False)
    logger.info(f"  Debug predictions: {debug_path}")

    # Routing stats JSON
    stats_path = os.path.join(args.output_dir, f"routing_stats_{ts}.json")
    summary = {
        "total": total,
        "n_retrieved": n_retrieved,
        "n_generated": n_generated,
        "n_fallback": n_fallback,
        "pct_retrieved": round(100 * n_retrieved / total, 2),
        "pct_generated": round(100 * n_generated / total, 2),
        "thresholds": SUBSET_THRESHOLDS,
        "length_stats": length_stats,
    }
    with open(stats_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Routing stats   : {stats_path}")

    logger.info("")
    logger.info("Pipeline complete!")
    logger.info(f"  Submit: {submission_path}")


if __name__ == "__main__":
    main()
