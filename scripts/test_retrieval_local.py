#!/usr/bin/env python3
"""
Local Retrieval Quality Test — runs on MacBook M1 (CPU only)
=============================================================
Tests BM25 vs TF-IDF character n-gram retrieval on validation data.
Measures: what % of val questions retrieve a training example
whose answer overlaps with the gold answer (proxy for RAG quality).

Usage:
    cd ~/Downloads/afriqa
    pip install pandas scikit-learn rank-bm25 rouge-score
    python scripts/test_retrieval_local.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import pandas as pd
import numpy as np
from rouge_score import rouge_scorer

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.tfidf_retriever import TFIDFRetriever


def test_retrieval(retriever, name, val_df, top_k=3):
    """Score retrieval quality: does the retrieved answer overlap with gold?"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)
    
    questions = val_df['input'].tolist()
    gold_answers = val_df['output'].tolist()
    
    r1_scores, rl_scores = [], []
    
    t0 = time.time()
    for i, (q, gold) in enumerate(zip(questions, gold_answers)):
        retrieved = retriever.retrieve(q, top_k=top_k)
        # Concatenate retrieved answers as "pseudo-answer"
        retrieved_text = " ".join([d['answer'] for d in retrieved])
        
        s = scorer.score(gold.strip(), retrieved_text.strip())
        r1_scores.append(s['rouge1'].fmeasure)
        rl_scores.append(s['rougeL'].fmeasure)
        
        if (i + 1) % 500 == 0:
            print(f"  [{name}] {i+1}/{len(questions)} ...")
    
    elapsed = time.time() - t0
    
    # Per-subset breakdown
    subsets = val_df['subset'].unique()
    print(f"\n{'='*60}")
    print(f"  {name} Retrieval Quality (top_k={top_k})")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"  {'Subset':<15} {'ROUGE-1':>8} {'ROUGE-L':>8} {'N':>6}")
    print(f"  {'-'*40}")
    
    for subset in sorted(subsets):
        mask = val_df['subset'] == subset
        idx = [i for i, m in enumerate(mask) if m]
        sub_r1 = np.mean([r1_scores[i] for i in idx])
        sub_rl = np.mean([rl_scores[i] for i in idx])
        print(f"  {subset:<15} {sub_r1:>8.4f} {sub_rl:>8.4f} {len(idx):>6}")
    
    global_r1 = np.mean(r1_scores)
    global_rl = np.mean(rl_scores)
    print(f"  {'-'*40}")
    print(f"  {'GLOBAL':<15} {global_r1:>8.4f} {global_rl:>8.4f} {len(r1_scores):>6}")
    
    return {'rouge1': global_r1, 'rougeL': global_rl}


def main():
    print("Loading data ...")
    train_df = pd.read_csv('Train.csv').fillna('')
    val_df = pd.read_csv('Val.csv').fillna('')
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,}")
    
    corpus_q = train_df['input'].tolist()
    corpus_a = train_df['output'].tolist()
    
    # ── Test 1: BM25 ─────────────────────────────────────────
    print("\n▶️  Building BM25 index ...")
    bm25 = BM25Retriever(k1=1.5, b=0.75)
    bm25.build_index(corpus_q, corpus_a)
    bm25_results = test_retrieval(bm25, "BM25", val_df, top_k=3)
    
    # ── Test 2: TF-IDF char n-grams ──────────────────────────
    print("\n▶️  Building TF-IDF char n-gram index ...")
    tfidf = TFIDFRetriever(analyzer='char_wb', ngram_range=(2, 5))
    tfidf.build_index(corpus_q, corpus_a)
    tfidf_results = test_retrieval(tfidf, "TF-IDF", val_df, top_k=3)
    
    # ── Comparison ────────────────────────────────────────────
    print("\n" + "="*60)
    print("  COMPARISON: BM25 vs TF-IDF")
    print("="*60)
    for metric in ['rouge1', 'rougeL']:
        b = bm25_results[metric]
        t = tfidf_results[metric]
        winner = "TF-IDF" if t > b else "BM25"
        print(f"  {metric}: BM25={b:.4f}  TF-IDF={t:.4f}  Winner={winner} (Δ={abs(t-b):+.4f})")
    
    print("\n✅ Done! Use the winning retriever in your RAG pipeline.")
    print("   If TF-IDF wins → use --retriever_type tfidf on server")
    print("   If BM25 wins   → use --retriever_type bm25 on server")


if __name__ == "__main__":
    main()
