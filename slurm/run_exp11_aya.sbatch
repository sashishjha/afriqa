#!/bin/bash
#SBATCH --job-name=afriqa_AYA101
#SBATCH --partition=debug
#SBATCH --account=mtech_ras
#SBATCH --qos=nonphd_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:3
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --output=logs/exp11_aya_%j.log
#SBATCH --error=logs/exp11_aya_%j.log

# ============================================================
#  EXPERIMENT 11: AYA-101 (CohereForAI)
#  13B seq2seq + LoRA r=8 + Merged Train+Val + RAG Inference
#  + Competitor Decoding (length_penalty=0.8, min_length=15)
# ============================================================
# Phase 0: Pre-download 13B model (~26GB)
# Phase 1: Train Aya-101 with LoRA on ALL data — 3 GPUs
# Phase 2: RAG inference using trained model — 1 GPU
# Phase 3: Generate submission CSV

set -uo pipefail
mkdir -p logs submissions

echo "======================================================"
echo "  EXP11: Aya-101 (13B) + LoRA + Merged + RAG"
echo "  Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "  Start: $(date)"
echo "======================================================"

# ── GPU setup (ROCm) ──────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0,1,2
export HIP_VISIBLE_DEVICES=0,1,2
export ROCR_VISIBLE_DEVICES=0,1,2
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export NCCL_DEBUG=WARN
export HSA_FORCE_FINE_GRAIN_PCIE=1
export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=1

# ── ROCm memory optimisations for 13B model ───────────────
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
export HSA_ENABLE_SDMA=0

cd /mnt/data/sashishj/projects/afriqa
source .venv/bin/activate

echo "Python: $(which python)"
echo "GPUs:"
python -c "
import torch
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} | {props.total_memory//1024**3} GB')
"

# ── Phase 0: Pre-download model ───────────────────────────
echo ""
echo "▶️  Phase 0: Pre-downloading Aya-101 (13B) — this may take a while ..."
python -c "
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = 'CohereForAI/aya-101'
print(f'Downloading {model_name} ...')
try:
    AutoTokenizer.from_pretrained(model_name, use_fast=False)
    AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16)
    print('✅ Aya-101 cached successfully')
except Exception as e:
    print(f'⚠️  CohereForAI/aya-101 failed: {e}')
    print('Trying CohereLabs/aya-101 ...')
    model_name = 'CohereLabs/aya-101'
    AutoTokenizer.from_pretrained(model_name, use_fast=False)
    AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16)
    print('✅ CohereLabs/aya-101 cached successfully')
" 2>&1

# ── Phase 1: Train Aya-101 with LoRA on MERGED data ──────
echo ""
echo "======================================================"
echo "  PHASE 1: Training Aya-101 (13B) with LoRA on merged data"
echo "  3 GPUs | batch=2/gpu | grad_accum=8 | ~12-24 hours"
echo "  $(date)"
echo "======================================================"

torchrun \
    --nproc_per_node=3 \
    --master_port=29501 \
    scripts/run_exp1_generation.py \
        --config   configs/experiment_11_aya.yaml \
        --output_dir outputs/exp11_aya \
        --merge_val \
        --batch_size 2 \
    2>&1

TRAIN_EXIT=$?
echo ""
echo "Phase 1 (Training): exit=$TRAIN_EXIT | $(date)"

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "❌ Aya-101 Training FAILED! Check logs."
    echo "Common fixes:"
    echo "  - Reduce batch_size to 1 in the yaml"
    echo "  - Check if model ID changed: try CohereLabs/aya-101"
    echo "  - Increase --mem to 256G"
    exit 1
fi

# ── Phase 2: RAG Inference ────────────────────────────────
echo ""
echo "======================================================"
echo "  PHASE 2: RAG Inference using trained Aya-101"
echo "  1 GPU | per-subset retrieval (Amharic fix)"
echo "  $(date)"
echo "======================================================"

CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
python scripts/run_exp3_rag_baseline.py \
    --config configs/experiment_11_aya.yaml \
    --output_dir outputs/exp11_rag \
    --exp1_model_dir outputs/exp11_aya/best_model \
    --retriever_type tfidf \
    --per_subset_retrieval \
    --top_k 3 \
    --batch_size 2 \
    2>&1

RAG_EXIT=$?
echo ""
echo "Phase 2 (RAG): exit=$RAG_EXIT | $(date)"

# ── Phase 3: Copy pure-generation submission ──────────────
echo ""
echo "======================================================"
echo "  PHASE 3: Collecting submissions"
echo "  $(date)"
echo "======================================================"

if [ -f "submissions/exp11_aya_submission.csv" ]; then
    cp submissions/exp11_aya_submission.csv submissions/exp11_pure_generation.csv
    echo "✅ Pure generation submission: exp11_pure_generation.csv"
fi

# ── Summary ───────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  EXP11 COMPLETE"
echo "  $(date)"
echo "======================================================"
echo "  Phase 1 (Training): exit=$TRAIN_EXIT"
echo "  Phase 2 (RAG):      exit=$RAG_EXIT"
echo ""
echo "Submissions:"
ls -la submissions/exp11_*.csv 2>/dev/null || echo "  No exp11 submissions found"
echo ""
echo "Submit BOTH (pure gen + RAG) to Zindi and compare!"
