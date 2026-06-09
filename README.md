# AfriQA: Multilingual Health Question Answering

This repository contains code and scripts for training and evaluating multilingual retrieval and generation systems on the AfriQA dataset.

## Installation

Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running Experiments

### Experiment 1: Pure Generation Baseline
Train mT5-base with LoRA directly on the question-answering task:
```bash
python scripts/run_exp1_generation.py \
    --config configs/config.yaml \
    --output_dir outputs/exp1_generation
```

*To skip training and perform evaluation/submission generation using an existing best model checkpoint:*
```bash
python scripts/run_exp1_generation.py --skip_train
```

---

### Experiment 2: Retrieval Benchmark
Compare sparse (BM25), dense (LaBSE, MPNet), and hybrid retrieval architectures:
```bash
python scripts/run_exp2_retrieval.py \
    --config configs/config.yaml \
    --output_dir outputs/exp2_retrieval
```

---

### Experiment 3: RAG Baseline
Run a Retrieval-Augmented Generation pipeline using the best retriever and the trained generator:
```bash
python scripts/run_exp3_rag_baseline.py \
    --config configs/config.yaml \
    --exp1_model_dir outputs/exp1_generation/best_model
```

---

### Experiment 5: Foundation Model Benchmark
Compare different multilingual architectures (mT5-base, mT5-large, ByT5, NLLB-600M, Aya-101, Qwen2.5-1.5B):

**Run all registry models:**
```bash
python scripts/run_exp5_foundation.py \
    --config configs/config.yaml
```

**Fast sweep (subset of models, custom epochs, subsampled data):**
```bash
python scripts/run_exp5_foundation.py \
    --models mt5-base byt5-base qwen2.5-1.5B \
    --epochs 2 \
    --max_train_samples 5000
```
