#!/bin/bash
#SBATCH --job-name=afriqa_BEST
#SBATCH --partition=debug
#SBATCH --account=mtech_ras
#SBATCH --qos=nonphd_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:3
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/exp10_best_%j.log
#SBATCH --error=logs/exp10_best_%j.log

# ============================================================
#  EXPERIMENT 10: THE BEST COMBO
#  mT5-Large + Merged Train+Val + RAG Inference
#  + Competitor Decoding (length_penalty=0.8, min_length=15)
# ============================================================
# Phase 1: Train mT5-Large on ALL data (train+val merged) — 3 GPUs
# Phase 2: RAG inference using trained model — 1 GPU
# Phase 3: Generate submission CSV

set -uo pipefail
mkdir -p logs

echo "======================================================"
echo "  EXP10: mT5-Large + Merged + RAG (THE BEST COMBO)"
echo "  Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "  Start: $(date)"
echo "======================================================"

# ── GPU setup ──────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0,1,2
export HIP_VISIBLE_DEVICES=0,1,2
export ROCR_VISIBLE_DEVICES=0,1,2
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export NCCL_DEBUG=WARN
export HSA_FORCE_FINE_GRAIN_PCIE=1
export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=1

cd /mnt/data/sashishj/projects/afriqa
source .venv/bin/activate

echo "Python: $(which python)"
echo "GPUs:"
python -c "
import torch
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} | {torch.cuda.get_device_properties(i).total_memory//1024**3} GB')
"

# ── Phase 1: Pre-download model ───────────────────────────
echo ""
echo "▶️  Phase 0: Pre-downloading model ..."
python -c "
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_name = 'google/mt5-large'
print(f'Downloading {model_name} ...')
AutoTokenizer.from_pretrained(model_name, use_fast=False)
AutoModelForSeq2SeqLM.from_pretrained(model_name)
print('✅ Model cached')
" 2>&1

# ── Phase 1: Train mT5-Large on MERGED data ──────────────
echo ""
echo "======================================================"
echo "  PHASE 1: Training mT5-Large on merged train+val"
echo "  3 GPUs | ~6 hours"
echo "  $(date)"
echo "======================================================"

torchrun \
    --nproc_per_node=3 \
    --master_port=29500 \
    scripts/run_exp1_generation.py \
        --config   configs/experiment_10_best.yaml \
        --output_dir outputs/exp10_best \
        --merge_val \
    2>&1

TRAIN_EXIT=$?
echo ""
echo "Phase 1 (Training): exit=$TRAIN_EXIT | $(date)"

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "❌ Training FAILED! Aborting."
    exit 1
fi

# ── Phase 2: RAG Inference with trained mT5-Large ────────
echo ""
echo "======================================================"
echo "  PHASE 2: RAG Inference using trained mT5-Large"
echo "  1 GPU | ~1 hour"
echo "  $(date)"
echo "======================================================"

# RAG only needs 1 GPU for inference
CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
python scripts/run_exp3_rag_baseline.py \
    --config configs/experiment_10_best.yaml \
    --output_dir outputs/exp10_rag \
    --exp1_model_dir outputs/exp10_best/best_model \
    --retriever_type tfidf \
    --top_k 3 \
    --batch_size 4 \
    2>&1

RAG_EXIT=$?
echo ""
echo "Phase 2 (RAG): exit=$RAG_EXIT | $(date)"

# ── Phase 3: Also generate NON-RAG submission ────────────
# (Sometimes pure generation beats RAG — we want both CSVs)
echo ""
echo "======================================================"
echo "  PHASE 3: Also copying pure-generation submission"
echo "  $(date)"
echo "======================================================"

# The training script already generated a submission in Phase 1
# Rename it so we have both
if [ -f "submissions/exp10_best_submission.csv" ]; then
    cp submissions/exp10_best_submission.csv submissions/exp10_pure_generation.csv
    echo "✅ Pure generation submission: exp10_pure_generation.csv"
fi

# ── Summary ───────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  ALL DONE"
echo "  $(date)"
echo "======================================================"
echo "  Phase 1 (Training): exit=$TRAIN_EXIT"
echo "  Phase 2 (RAG):      exit=$RAG_EXIT"
echo ""
echo "Submissions:"
ls -la submissions/exp10_*.csv 2>/dev/null || echo "  No exp10 submissions found"
ls -la submissions/exp3_submission.csv 2>/dev/null || echo "  No exp3 RAG submission found"
echo ""
echo "Submit BOTH to Zindi and compare!"
