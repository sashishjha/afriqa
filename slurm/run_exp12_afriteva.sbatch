#!/bin/bash
#SBATCH --job-name=afriqa_AFRITEVA
#SBATCH --partition=debug
#SBATCH --account=mtech_ras
#SBATCH --qos=nonphd_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:3
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/exp12_afriteva_%j.log
#SBATCH --error=logs/exp12_afriteva_%j.log

# ============================================================
#  EXPERIMENT 12: AfriTeVa V2 Large (castorini)
#  1B T5-v1.1 + LoRA + Merged Train+Val + RAG Inference
#  African-specific model — 2x better than mT5 on XL-QA
# ============================================================
# FIXES for previous ROCm crash:
#   1. use_fast=False for tokenizer (SentencePiece + 150K vocab)
#   2. Dropout re-enabled (was OFF during pretraining)
#   3. HSA_ENABLE_SDMA=0 for ROCm stability
#   4. transformers>=4.30.0 required (gated-gelu support)
# ============================================================

set -uo pipefail
mkdir -p logs submissions

echo "======================================================"
echo "  EXP12: AfriTeVa V2 Large (1B) + LoRA + Merged + RAG"
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

# ── ROCm stability fixes (CRITICAL for AfriTeVa) ─────────
export HSA_ENABLE_SDMA=0
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512

cd /mnt/data/sashishj/projects/afriqa
source .venv/bin/activate

echo "Python: $(which python)"

# ── Pre-flight: Verify transformers version ───────────────
echo ""
echo "▶️  Pre-flight checks..."
python -c "
import transformers
print(f'transformers version: {transformers.__version__}')
v = tuple(int(x) for x in transformers.__version__.split('.')[:2])
assert v >= (4, 30), f'Need transformers>=4.30.0 for T5v1.1 gated-gelu, got {transformers.__version__}'
print('✅ transformers version OK')

# Test that T5 v1.1 config works
from transformers import T5Config
cfg = T5Config(feed_forward_proj='gated-gelu')
print('✅ gated-gelu activation supported')

# Test tokenizer loads with use_fast=False
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('castorini/afriteva_v2_large', use_fast=False)
print(f'✅ Tokenizer loaded: vocab_size={tok.vocab_size}')
" 2>&1

PREFLIGHT_EXIT=$?
if [ $PREFLIGHT_EXIT -ne 0 ]; then
    echo "❌ Pre-flight checks FAILED!"
    echo "Fix: pip install transformers>=4.30.0"
    exit 1
fi

echo ""
echo "GPUs:"
python -c "
import torch
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} | {props.total_memory//1024**3} GB')
"

# ── Phase 0: Pre-download model ───────────────────────────
echo ""
echo "▶️  Phase 0: Pre-downloading AfriTeVa V2 Large ..."
python -c "
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_name = 'castorini/afriteva_v2_large'
print(f'Downloading {model_name} ...')
AutoTokenizer.from_pretrained(model_name, use_fast=False)
AutoModelForSeq2SeqLM.from_pretrained(model_name)
print('✅ AfriTeVa V2 Large cached')
" 2>&1

# ── Phase 1: Train AfriTeVa V2 on MERGED data ────────────
echo ""
echo "======================================================"
echo "  PHASE 1: Training AfriTeVa V2 Large (1B) on merged data"
echo "  3 GPUs | dropout RE-ENABLED | ~6-8 hours"
echo "  $(date)"
echo "======================================================"

torchrun \
    --nproc_per_node=3 \
    --master_port=29502 \
    scripts/run_exp1_generation.py \
        --config   configs/experiment_12_afriteva.yaml \
        --output_dir outputs/exp12_afriteva \
        --merge_val \
    2>&1

TRAIN_EXIT=$?
echo ""
echo "Phase 1 (Training): exit=$TRAIN_EXIT | $(date)"

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "❌ AfriTeVa Training FAILED!"
    echo "Common ROCm fixes:"
    echo "  1. Ensure HSA_ENABLE_SDMA=0 is set"
    echo "  2. Check: pip install transformers>=4.30.0"
    echo "  3. Reduce batch_size to 4 in yaml"
    echo "  4. Try: export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:256"
    exit 1
fi

# ── Phase 2: RAG Inference ────────────────────────────────
echo ""
echo "======================================================"
echo "  PHASE 2: RAG Inference using trained AfriTeVa V2"
echo "  1 GPU | per-subset retrieval (Amharic fix)"
echo "  $(date)"
echo "======================================================"

CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
python scripts/run_exp3_rag_baseline.py \
    --config configs/experiment_12_afriteva.yaml \
    --output_dir outputs/exp12_rag \
    --exp1_model_dir outputs/exp12_afriteva/best_model \
    --retriever_type tfidf \
    --per_subset_retrieval \
    --top_k 3 \
    --batch_size 4 \
    2>&1

RAG_EXIT=$?
echo ""
echo "Phase 2 (RAG): exit=$RAG_EXIT | $(date)"

# ── Phase 3: Collect submissions ──────────────────────────
echo ""
echo "======================================================"
echo "  PHASE 3: Collecting submissions"
echo "  $(date)"
echo "======================================================"

if [ -f "submissions/exp12_afriteva_submission.csv" ]; then
    cp submissions/exp12_afriteva_submission.csv submissions/exp12_pure_generation.csv
    echo "✅ Pure generation submission: exp12_pure_generation.csv"
fi

# ── Summary ───────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  EXP12 COMPLETE"
echo "  $(date)"
echo "======================================================"
echo "  Phase 1 (Training): exit=$TRAIN_EXIT"
echo "  Phase 2 (RAG):      exit=$RAG_EXIT"
echo ""
echo "Submissions:"
ls -la submissions/exp12_*.csv 2>/dev/null || echo "  No exp12 submissions found"
echo ""
echo "Submit BOTH (pure gen + RAG) to Zindi and compare!"
