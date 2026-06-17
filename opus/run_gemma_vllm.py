#!/usr/bin/env python3
"""
Gemma 4 31B-it Inference with Per-Subset Routing
=================================================
Zindi Multilingual Health QA Competition - Day 1 & Day 2 Combined

Architecture:
  HIGH-REUSE subsets → Direct TF-IDF retrieval (no LLM)
  LOW-REUSE subsets  → Gemma 4 31B-it generation via vLLM + RAG context

Fully standalone — no dependency on existing src/ modules.
No torch, no HuggingFace imports. Works offline on compute nodes.

Usage:
  python run_gemma_vllm.py \
    --train-path data/Train.csv \
    --test-path data/Test.csv \
    --output-dir outputs/run_001 \
    --vllm-url http://localhost:8000/v1

Author: Sashish Jha
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

# Optional imports with fallbacks
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        """Fallback tqdm that just returns the iterable."""
        desc = kwargs.get("desc", "")
        total = kwargs.get("total", None)
        for i, item in enumerate(iterable):
            if total and (i % max(1, total // 20) == 0):
                print(f"  [{desc}] {i}/{total} ({100*i/total:.0f}%)")
            yield item

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUBSET_TO_LANG = {
    "Eng_Uga": "English",
    "Aka_Gha": "Akan",
    "Eng_Gha": "English",
    "Eng_Eth": "English",
    "Lug_Uga": "Luganda",
    "Eng_Ken": "English",
    "Swa_Ken": "Swahili",
    "Amh_Eth": "Amharic",
}

# Per-language system prompts with native language anchoring cues
SYSTEM_PROMPTS = {
    "Amharic": (
        "You are a helpful health assistant. "
        "በአማርኛ ብቻ መልስ ስጥ። "
        "Answer the following health question accurately and completely in Amharic. "
        "Do not switch to any other language. "
        "Provide ONLY the direct answer. Do not use conversational preambles like 'Here is the answer'. Be concise."
    ),
    "Swahili": (
        "You are a helpful health assistant. "
        "Jibu swali hili la afya kwa Kiswahili pekee. "
        "Answer the following health question accurately and completely in Swahili. "
        "Do not switch to any other language. "
        "Provide ONLY the direct answer. Do not use conversational preambles like 'Here is the answer'. Be concise."
    ),
    "Luganda": (
        "You are a helpful health assistant. "
        "Ddamu ekibuuzo kino mu Luganda bwokka. "
        "Answer the following health question accurately and completely in Luganda. "
        "Do not switch to any other language. "
        "Provide ONLY the direct answer. Do not use conversational preambles like 'Here is the answer'. Be concise."
    ),
    "Akan": (
        "You are a helpful health assistant. "
        "Bua wɔ Twi kasa mu nkoa. "
        "Answer the following health question accurately and completely in Akan (Twi). "
        "Do not switch to any other language. "
        "Provide ONLY the direct answer. Do not use conversational preambles like 'Here is the answer'. Be concise."
    ),
    "English": (
        "You are a helpful health assistant. "
        "Answer the following health question accurately and completely in English. "
        "Do not switch to any other language. "
        "Provide ONLY the direct answer. Do not use conversational preambles like 'Here is the answer'. Be concise."
    ),
}


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def setup_logging(output_dir: str) -> logging.Logger:
    """Configure logging to both console and file."""
    log = logging.getLogger("gemma_routing")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(ch)

    # File handler
    os.makedirs(output_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(output_dir, "run.log"), mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)

    return log


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_data(train_path: str, val_path: str, test_path: str, log: logging.Logger):
    """Load and merge training data; load test data separately."""

    log.info(f"Loading training data from: {train_path}")
    train_df = pd.read_csv(train_path)

    if val_path and os.path.exists(val_path):
        log.info(f"Loading validation data from: {val_path}")
        val_df = pd.read_csv(val_path)
        train_df = pd.concat([train_df, val_df], ignore_index=True)
        log.info(f"Merged train+val: {len(train_df)} rows")
    else:
        log.info(f"No validation file; using train only: {len(train_df)} rows")

    log.info(f"Loading test data from: {test_path}")
    test_df = pd.read_csv(test_path)

    # Handle column name variations
    col_renames = {"question": "input", "answer": "output"}
    for old, new in col_renames.items():
        if old in train_df.columns and new not in train_df.columns:
            train_df = train_df.rename(columns={old: new})
        if old in test_df.columns and new not in test_df.columns:
            test_df = test_df.rename(columns={old: new})

    # Validate columns
    for col in ["input", "output", "subset"]:
        if col not in train_df.columns:
            raise ValueError(f"Training data missing required column: '{col}'. "
                             f"Found: {list(train_df.columns)}")
    for col in ["ID", "input", "subset"]:
        if col not in test_df.columns:
            raise ValueError(f"Test data missing required column: '{col}'. "
                             f"Found: {list(test_df.columns)}")

    # Drop rows with NaN in critical columns
    before = len(train_df)
    train_df = train_df.dropna(subset=["input", "output", "subset"])
    if len(train_df) < before:
        log.warning(f"Dropped {before - len(train_df)} training rows with NaN values")

    log.info(f"Train subsets: {sorted(train_df['subset'].unique())}")
    log.info(f"Test subsets:  {sorted(test_df['subset'].unique())}")
    log.info(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    return train_df, test_df


# ---------------------------------------------------------------------------
# Subset Reuse Analysis
# ---------------------------------------------------------------------------
def analyse_subset_reuse(train_df: pd.DataFrame, threshold: float, log: logging.Logger):
    """Analyse answer reuse per subset and classify into high/low reuse."""

    stats = {}
    log.info("")
    log.info("=" * 75)
    log.info("PER-SUBSET REUSE ANALYSIS")
    log.info("=" * 75)
    log.info(f"{'Subset':<12} {'Lang':<10} {'Total':>7} {'Unique':>7} "
             f"{'Ratio':>7} {'Route':<10}")
    log.info("-" * 75)

    high_reuse = set()
    low_reuse = set()

    for subset in sorted(train_df["subset"].unique()):
        answers = train_df[train_df["subset"] == subset]["output"]
        total = len(answers)
        unique = answers.nunique()
        ratio = unique / total if total > 0 else 1.0
        lang = SUBSET_TO_LANG.get(subset, "Unknown")

        if ratio < threshold:
            route = "RETRIEVE"
            high_reuse.add(subset)
        else:
            route = "GENERATE"
            low_reuse.add(subset)

        stats[subset] = {
            "language": lang,
            "total": total,
            "unique": unique,
            "ratio": round(ratio, 4),
            "route": route,
        }

        log.info(f"{subset:<12} {lang:<10} {total:>7} {unique:>7} "
                 f"{ratio:>7.4f} {route:<10}")

    log.info("-" * 75)
    log.info(f"Threshold: {threshold}")
    log.info(f"HIGH-REUSE (direct retrieval): {sorted(high_reuse)}")
    log.info(f"LOW-REUSE  (LLM generation):   {sorted(low_reuse)}")
    log.info("=" * 75)
    log.info("")

    return high_reuse, low_reuse, stats


# ---------------------------------------------------------------------------
# Output Length Calibration
# ---------------------------------------------------------------------------
def compute_target_lengths(train_df: pd.DataFrame, log: logging.Logger):
    """Compute target max_tokens per subset based on training answer lengths."""
    target_tokens = {}

    log.info("OUTPUT LENGTH CALIBRATION")
    log.info(f"{'Subset':<12} {'Median Words':>14} {'Max Tokens':>12}")
    log.info("-" * 42)

    for subset in sorted(train_df["subset"].unique()):
        answers = train_df[train_df["subset"] == subset]["output"]
        word_counts = answers.str.split().str.len()
        median_words = word_counts.median()
        # Heuristic: African languages tokenize heavily (up to 3-4 tokens/word).
        # We also need to give the model room to finish its thought.
        max_tokens = int(median_words * 4.0)
        max_tokens = max(128, min(max_tokens, 1024))
        target_tokens[subset] = max_tokens
        log.info(f"{subset:<12} {median_words:>14.0f} {max_tokens:>12}")

    log.info("")
    return target_tokens


# ---------------------------------------------------------------------------
# Per-Subset TF-IDF Retriever
# ---------------------------------------------------------------------------
class PerSubsetTFIDFRetriever:
    """Builds one TF-IDF index per subset with Amharic Ge'ez handling."""

    def __init__(self, train_df: pd.DataFrame, log: logging.Logger):
        self.log = log
        self.indices = {}   # subset -> {vectorizer, tfidf_matrix, questions, answers}

        for subset in sorted(train_df["subset"].unique()):
            subset_data = train_df[train_df["subset"] == subset].reset_index(drop=True)
            questions = subset_data["input"].astype(str).tolist()
            answers = subset_data["output"].astype(str).tolist()
            lang = SUBSET_TO_LANG.get(subset, "English")

            # Choose vectorizer settings based on script type
            if lang == "Amharic":
                # Ge'ez script: character n-grams work better
                vectorizer = TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    max_features=50000,
                    sublinear_tf=True,
                )
            else:
                # Latin script: word-level TF-IDF
                vectorizer = TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    max_features=50000,
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b\w+\b",
                )

            tfidf_matrix = vectorizer.fit_transform(questions)

            self.indices[subset] = {
                "vectorizer": vectorizer,
                "tfidf_matrix": tfidf_matrix,
                "questions": questions,
                "answers": answers,
            }

            log.debug(f"Built TF-IDF index for {subset}: {len(questions)} docs, "
                      f"vocab={len(vectorizer.vocabulary_)}")

        log.info(f"TF-IDF retriever ready: {len(self.indices)} subset indices built")

    def retrieve(self, query: str, subset: str, top_k: int = 3):
        """Retrieve top-k similar Q&A pairs from the given subset.

        Returns:
            List of (answer, question, similarity_score) tuples.
        """
        if subset not in self.indices:
            self.log.warning(f"No index for subset '{subset}'; returning empty")
            return []

        idx = self.indices[subset]
        query_vec = idx["vectorizer"].transform([query])
        sims = sklearn_cosine_similarity(query_vec, idx["tfidf_matrix"]).flatten()

        # Get top-k indices
        k = min(top_k, len(sims))
        top_indices = sims.argsort()[-k:][::-1]

        results = []
        for i in top_indices:
            results.append((
                idx["answers"][i],
                idx["questions"][i],
                float(sims[i]),
            ))
        return results


# ---------------------------------------------------------------------------
# vLLM Client
# ---------------------------------------------------------------------------
class VLLMClient:
    """Minimal HTTP client for vLLM's OpenAI-compatible API."""

    def __init__(self, base_url: str, model_name: str, timeout: int = 120,
                 log: logging.Logger = None):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.log = log or logging.getLogger("vllm_client")

    def _post_json(self, url: str, payload: dict) -> dict:
        """Send POST request, using requests if available, else urllib."""
        data_bytes = json.dumps(payload).encode("utf-8")

        if HAS_REQUESTS:
            resp = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            return resp.json()
        else:
            import urllib.request
            req = urllib.request.Request(
                url,
                data=data_bytes,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))

    def generate(self, system_prompt: str, user_prompt: str,
                 max_tokens: int = 512, temperature: float = 0.0) -> str:
        """Generate a response from the vLLM server.

        Retries up to 3 times with exponential backoff on failure.
        """
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1.0,
        }

        last_error = None
        for attempt in range(3):
            try:
                result = self._post_json(url, payload)
                content = result["choices"][0]["message"]["content"]
                return content.strip()
            except Exception as e:
                last_error = e
                wait = 2 ** (attempt + 1)
                self.log.warning(f"vLLM request failed (attempt {attempt+1}/3): {e}. "
                                 f"Retrying in {wait}s...")
                time.sleep(wait)

        self.log.error(f"vLLM request failed after 3 attempts: {last_error}")
        raise last_error

    def health_check(self) -> bool:
        """Check if vLLM server is reachable."""
        try:
            if HAS_REQUESTS:
                resp = requests.get(
                    f"{self.base_url.rsplit('/v1', 1)[0]}/health",
                    timeout=10,
                )
                return resp.status_code == 200
            else:
                import urllib.request
                url = f"{self.base_url.rsplit('/v1', 1)[0]}/health"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    return resp.status == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------
def build_generation_prompt(question: str, subset: str,
                            retrieved_examples: list) -> str:
    """Build the user prompt for LLM generation, with RAG context."""
    parts = []

    if retrieved_examples:
        parts.append(
            "Here are some reference answers to similar health questions for context:"
        )
        parts.append("")
        for i, (ans, q, score) in enumerate(retrieved_examples, 1):
            parts.append(f"Reference {i} (similarity: {score:.2f}):")
            parts.append(f"  Q: {q}")
            parts.append(f"  A: {ans}")
            parts.append("")

        parts.append(
            "Now answer the following question. Use the reference answers to "
            "inform your terminology. You MUST provide ONLY the direct answer. "
            "Do not output any conversational text or preambles."
        )
        parts.append("")

    parts.append(f"Question: {question}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main Inference Pipeline
# ---------------------------------------------------------------------------
def run_inference(args):
    """Main entry point: load data, route subsets, generate predictions."""

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    log = setup_logging(args.output_dir)

    log.info("=" * 75)
    log.info("GEMMA 4 31B-IT + PER-SUBSET ROUTING INFERENCE")
    log.info("=" * 75)
    log.info(f"Timestamp:       {datetime.now().isoformat()}")
    log.info(f"Model:           {args.model_name}")
    log.info(f"vLLM URL:        {args.vllm_url}")
    log.info(f"Reuse threshold: {args.reuse_threshold}")
    log.info(f"Top-K retrieval: {args.top_k}")
    log.info(f"Output dir:      {args.output_dir}")
    log.info("")

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    train_df, test_df = load_data(
        args.train_path, args.val_path, args.test_path, log
    )

    # ------------------------------------------------------------------
    # Step 2: Analyse subset reuse
    # ------------------------------------------------------------------
    high_reuse, low_reuse, reuse_stats = analyse_subset_reuse(
        train_df, args.reuse_threshold, log
    )

    # ------------------------------------------------------------------
    # Step 3: Compute target output lengths
    # ------------------------------------------------------------------
    target_tokens = compute_target_lengths(train_df, log)

    # ------------------------------------------------------------------
    # Step 4: Build retriever
    # ------------------------------------------------------------------
    log.info("Building per-subset TF-IDF retriever...")
    retriever = PerSubsetTFIDFRetriever(train_df, log)

    # ------------------------------------------------------------------
    # Step 5: Connect to vLLM (only needed if there are low-reuse subsets)
    # ------------------------------------------------------------------
    llm = None
    need_llm = any(row["subset"] in low_reuse for _, row in test_df.iterrows())

    if need_llm:
        log.info(f"Connecting to vLLM server at {args.vllm_url}...")
        llm = VLLMClient(
            base_url=args.vllm_url,
            model_name=args.model_name,
            timeout=args.timeout,
            log=log,
        )
        if llm.health_check():
            log.info("vLLM server is healthy and ready!")
        else:
            log.warning("vLLM health check failed — will attempt generation anyway")
    else:
        log.info("All test subsets are HIGH-REUSE; skipping vLLM connection")

    # ------------------------------------------------------------------
    # Step 6: Run inference with routing
    # ------------------------------------------------------------------
    log.info("")
    log.info("Starting inference...")
    log.info(f"Total test rows: {len(test_df)}")

    predictions = []
    
    # ------------------------------------------------------------------
    # Checkpoint setup
    # ------------------------------------------------------------------
    checkpoint_path = os.path.join(args.output_dir, "predictions_checkpoint.jsonl")
    completed_ids = set()
    
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data)
                completed_ids.add(data["ID"])
        log.info(f"Loaded {len(predictions)} completed predictions from checkpoint.")

    route_counts = defaultdict(int)
    subset_counts = defaultdict(int)
    errors = 0

    # Initialize counters from loaded predictions
    for p in predictions:
        route_counts[p["route"]] += 1
        subset_counts[p["subset"]] += 1
        if p["route"] in ["ERROR", "FALLBACK"]:
            errors += 1

    test_rows = list(test_df.iterrows())
    
    # Open checkpoint file in append mode
    with open(checkpoint_path, "a") as chk_f:
        for idx, row in tqdm(test_rows, desc="Inference", total=len(test_rows)):
            row_id = row["ID"]
            
            # Skip if already completed in checkpoint
            if row_id in completed_ids:
                continue

            question = str(row["input"])
            subset = row["subset"]
            lang = SUBSET_TO_LANG.get(subset, "English")

            subset_counts[subset] += 1

            try:
                if subset in high_reuse:
                    # ----- HIGH-REUSE: Direct retrieval -----
                    results = retriever.retrieve(question, subset, top_k=1)
                    if results:
                        prediction = results[0][0]  # Top-1 answer
                    else:
                        prediction = ""
                        log.warning(f"Row {row_id}: No retrieval result for {subset}")
                    route = "RETRIEVE"
                    route_counts["RETRIEVE"] += 1

                else:
                    # ----- LOW-REUSE: LLM generation with RAG context -----
                    top_k = args.top_k
                    retrieved = retriever.retrieve(question, subset, top_k=top_k)

                    system_prompt = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["English"])
                    user_prompt = build_generation_prompt(question, subset, retrieved)
                    max_tok = target_tokens.get(subset, 512)

                    try:
                        prediction = llm.generate(
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            max_tokens=max_tok,
                            temperature=0.0,
                        )
                        route = "GENERATE"
                        route_counts["GENERATE"] += 1
                    except Exception as e:
                        log.error(f"Row {row_id}: Generation failed ({e}); "
                                  f"falling back to retrieval")
                        results = retriever.retrieve(question, subset, top_k=1)
                        prediction = results[0][0] if results else ""
                        route = "FALLBACK"
                        route_counts["FALLBACK"] += 1
                        errors += 1

            except Exception as e:
                log.error(f"Row {row_id}: Unexpected error: {e}")
                prediction = ""
                route = "ERROR"
                route_counts["ERROR"] += 1
                errors += 1

            pred_dict = {
                "ID": row_id,
                "input": question,
                "subset": subset,
                "output": prediction,
                "route": route,
                "pred_word_count": len(prediction.split()) if prediction else 0,
            }
            predictions.append(pred_dict)
            
            # Save to checkpoint immediately
            chk_f.write(json.dumps(pred_dict) + "\n")
            chk_f.flush()

    log.info(f"\nInference complete! Errors: {errors}")

    # ------------------------------------------------------------------
    # Step 7: Save outputs
    # ------------------------------------------------------------------
    log.info("\nSaving outputs...")

    pred_df = pd.DataFrame(predictions)

    # 7a. Competition submission file (Zindi expects 3 target columns)
    pred_df["TargetRLF1"] = pred_df["output"]
    pred_df["TargetR1F1"] = pred_df["output"]
    pred_df["TargetLLM"] = pred_df["output"]
    
    submission_path = os.path.join(args.output_dir, "submission.csv")
    pred_df[["ID", "TargetRLF1", "TargetR1F1", "TargetLLM"]].to_csv(submission_path, index=False)
    log.info(f"Submission saved: {submission_path} ({len(pred_df)} rows)")

    # 7b. Debug predictions file
    debug_path = os.path.join(args.output_dir, "predictions_debug.csv")
    pred_df.to_csv(debug_path, index=False)
    log.info(f"Debug predictions saved: {debug_path}")

    # 7c. Routing statistics
    stats_path = os.path.join(args.output_dir, "routing_stats.json")
    stats = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model_name,
        "reuse_threshold": args.reuse_threshold,
        "top_k": args.top_k,
        "high_reuse_subsets": sorted(high_reuse),
        "low_reuse_subsets": sorted(low_reuse),
        "route_counts": dict(route_counts),
        "subset_counts": dict(subset_counts),
        "reuse_stats": reuse_stats,
        "target_tokens": target_tokens,
        "total_test_rows": len(test_df),
        "errors": errors,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Routing stats saved: {stats_path}")

    # ------------------------------------------------------------------
    # Step 8: Print summary table
    # ------------------------------------------------------------------
    log.info("")
    log.info("=" * 75)
    log.info("PER-SUBSET PREDICTION SUMMARY")
    log.info("=" * 75)
    log.info(f"{'Subset':<12} {'Count':>7} {'Route':<10} {'Avg Words':>10}")
    log.info("-" * 45)

    for subset in sorted(pred_df["subset"].unique()):
        sub_df = pred_df[pred_df["subset"] == subset]
        count = len(sub_df)
        route = sub_df["route"].mode()[0] if len(sub_df) > 0 else "N/A"
        avg_words = sub_df["pred_word_count"].mean()
        log.info(f"{subset:<12} {count:>7} {route:<10} {avg_words:>10.1f}")

    log.info("-" * 45)
    log.info(f"{'TOTAL':<12} {len(pred_df):>7}")
    log.info(f"\nRoute distribution: {dict(route_counts)}")
    log.info(f"Output directory: {args.output_dir}")
    log.info("=" * 75)
    log.info("\nDone! Submit submission.csv to Zindi.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Gemma 4 31B-it inference with per-subset routing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Day 1: Pure zero-shot (all subsets go to LLM, no routing)
  python run_gemma_vllm.py \\
    --train-path data/Train.csv --test-path data/Test.csv \\
    --output-dir outputs/day1_zeroshot \\
    --reuse-threshold 0.0

  # Day 2: Per-subset routing (high-reuse retrieved, low-reuse generated)
  python run_gemma_vllm.py \\
    --train-path data/Train.csv --val-path data/Val.csv \\
    --test-path data/Test.csv \\
    --output-dir outputs/day2_routing \\
    --reuse-threshold 0.50 --top-k 3
        """,
    )

    parser.add_argument("--train-path", required=True,
                        help="Path to Train.csv")
    parser.add_argument("--val-path", default=None,
                        help="Path to Val.csv (optional, merged with train)")
    parser.add_argument("--test-path", required=True,
                        help="Path to Test.csv")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for output files")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1",
                        help="vLLM server base URL (default: http://localhost:8000/v1)")
    parser.add_argument("--model-name", default="gemma-4-31B-it",
                        help="Model name served by vLLM (default: gemma-4-31B-it)")
    parser.add_argument("--reuse-threshold", type=float, default=0.50,
                        help="Uniqueness ratio threshold for high/low reuse (default: 0.50)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of retrieved examples for RAG context (default: 3)")
    parser.add_argument("--timeout", type=int, default=120,
                        help="HTTP timeout for vLLM requests in seconds (default: 120)")

    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
