#!/bin/bash
#SBATCH --job-name=afriqa_exp6
#SBATCH --partition=debug
#SBATCH --account=mtech_ras
#SBATCH --qos=nonphd_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24          # 8 CPU cores per GPU
#SBATCH --gres=gpu:3                # ALL 3 GPUs
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/exp6_mt5large_%j.log
#SBATCH --error=logs/exp6_mt5large_%j.log

set -uo pipefail
mkdir -p logs

echo "====================================================="
echo "  Exp6: mT5-Large + LoRA  [3-GPU torchrun]"
echo "  Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "  Start: $(date)"
echo "====================================================="

# ── All 3 MI300X GPUs ─────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0,1,2
export HIP_VISIBLE_DEVICES=0,1,2
export ROCR_VISIBLE_DEVICES=0,1,2

# ── ROCm multi-GPU communication (RCCL) ───────────────────
export NCCL_SOCKET_IFNAME=lo          # use loopback for single-node
export GLOO_SOCKET_IFNAME=lo
export NCCL_DEBUG=WARN                # set INFO to debug RCCL issues
export HSA_FORCE_FINE_GRAIN_PCIE=1   # required for MI300X peer access

# ── ROCm safety (NO TUNABLEOP — causes deadlock on MI300X) ─
# NO HSA_OVERRIDE_GFX_VERSION — causes SIGABRT on gfx942
export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=1

cd /mnt/data/sashishj/projects/afriqa
source .venv/bin/activate

echo "Python:    $(which python)"
echo "torchrun:  $(which torchrun)"
echo "GPUs:"
python -c "
import torch
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} | {torch.cuda.get_device_properties(i).total_memory//1024**3} GB')
"

echo ""
echo "▶️  Pre-downloading model (single process, avoids cache race) ..."
python -c "
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_name = 'google/mt5-large'
print(f'Downloading {model_name} ...')
AutoTokenizer.from_pretrained(model_name, use_fast=False)
AutoModelForSeq2SeqLM.from_pretrained(model_name)
print('✅ Model cached successfully')
" 2>&1

echo ""
echo "▶️  Launching 3-GPU training with torchrun ..."
echo ""

# torchrun launches one process per GPU automatically
# Seq2SeqTrainer detects multiple processes and uses DDP
torchrun \
    --nproc_per_node=3 \
    --master_port=29500 \
    scripts/run_exp1_generation.py \
        --config   configs/experiment_6_mt5large.yaml \
        --output_dir outputs/exp6_mt5large \
    2>&1

EXIT_CODE=$?
echo ""
echo "====================================================="
echo "  Exp6 finished | Exit: $EXIT_CODE | $(date)"
echo "====================================================="
exit $EXIT_CODE
