# AfriQA — Multilingual Health QA Competition

**Goal:** Score 0.72+ on Zindi leaderboard | **Deadline:** June 22, 2026

## Quick Start (Server)

```bash
cd /mnt/data/sashishj/projects/afriqa
sbatch slurm/run_exp10_best.sbatch        # THE main experiment
tail -f logs/exp10_best_*.log             # Monitor
```

## What This Does

**Experiment 10** (our best combo):
1. **Phase 1:** Trains `mT5-Large` (1.2B) with LoRA on merged train+val (36k samples), 3 GPUs, 5 epochs
2. **Phase 2:** Runs RAG inference (BM25 retrieval + generation) using the trained model
3. **Output:** Two submission CSVs in `submissions/`

## Repo Structure

```
afriqa/
├── configs/
│   └── experiment_10_best.yaml      # THE config (mT5-Large + merged + RAG)
├── scripts/
│   ├── run_exp1_generation.py       # Main training script (all experiments use this)
│   └── run_exp3_rag_baseline.py     # RAG inference script
├── slurm/
│   ├── run_exp10_best.sbatch        # THE sbatch to run
│   └── run_exp6.sbatch              # Previous mT5-Large experiment (kept for reference)
├── src/
│   ├── data_loader.py               # CSV loading + tokenization
│   ├── metrics.py                   # ROUGE scoring
│   ├── utils.py                     # Seed, config, logging helpers
│   ├── training/trainer.py          # LoRA + Seq2SeqTrainer setup
│   ├── inference/predictor.py       # Batch prediction
│   ├── evaluation/evaluator.py      # Per-subset evaluation
│   ├── retrieval/                   # BM25, Dense (LaBSE), Hybrid retrievers
│   └── rag/pipeline.py             # RAG: retrieve examples → generate answer
├── notebooks/                       # (empty — Kaggle/Colab were unstable)
├── submissions/                     # Output CSVs for Zindi
├── Train.csv / Val.csv / Test.csv   # Competition data
├── SampleSubmission.csv             # Zindi format reference
└── requirements.txt                 # Python dependencies
```

## Scoring History

| # | Experiment | Score | Key Insight |
|---|---|---|---|
| 1 | mT5-base, train only | 0.231 | Baseline |
| 2 | mT5-base + RAG | **0.255** | Retrieval helps more than model size |
| 3 | mT5-Large, train only | 0.250 | Bigger model alone ≠ better |
| 4 | **Exp10 (pending)** | **???** | mT5-Large + merged data + RAG |

## Competition Intelligence

- Score = 0.37×ROUGE-L + 0.37×ROUGE-1 + 0.26×LLM-Judge
- Competitor setup: mT5-Large, AdaFactor, beam=4, length_penalty=0.8, min_length=15
- Retrieval (TF-IDF/BM25) is critical — beats raw model scaling
- External data is allowed (must be documented)
