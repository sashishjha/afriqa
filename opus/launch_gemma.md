#!/bin/bash
#SBATCH --job-name=gemma-routing
#SBATCH --partition=debug
#SBATCH --account=mtech_ras
#SBATCH --qos=nonphd_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/gemma_routing_%j.log
#SBATCH --error=logs/gemma_routing_%j.log

#=============================================================================
# SLURM Batch Script: Gemma 4 31B-it Inference with Per-Subset Routing
#=============================================================================
# Cluster:  RBCCPS (IISc Bangalore)
# Hardware: 3x AMD Instinct MI300X (192GB HBM3 each, 576GB total)
# Software: ROCm 6.2.4, Singularity (no raw Docker)
# Internet: NONE on compute nodes (all models must be pre-downloaded)
#
# QUICK START (run these on the LOGIN NODE first):
# -----------------------------------------------
# 1. Download the model (login node has internet):
#      huggingface-cli download google/gemma-4-31B-it \
#        --local-dir /mnt/data/${USER}/models/gemma-4-31B-it
#
# 2. Build Singularity image (login node has internet):
#      singularity pull /mnt/data/${USER}/containers/vllm_rocm.sif \
#        docker://vllm/vllm-openai-rocm:latest
#
# 3. Install Python dependencies on host (for the inference script):
#      pip install --user pandas scikit-learn requests tqdm numpy
#
# 4. Submit this job:
#      mkdir -p logs
#      sbatch launch_gemma.sbatch
#
# SWITCHING MODES:
# ----------------
# Day 1 (pure zero-shot, no routing):
#   Set REUSE_THRESHOLD=0.0 below. All subsets go to LLM.
#
# Day 2 (per-subset routing):
#   Set REUSE_THRESHOLD=0.50 below. High-reuse subsets use retrieval.
#
# TENSOR PARALLELISM:
# -------------------
# Gemma 4 31B in bf16 ≈ 62GB. Fits on 1x MI300X (192GB) easily.
# TP_SIZE=1 is recommended for simplicity and fastest per-request latency.
# If you want to use all 3 GPUs for faster throughput:
#   Set TP_SIZE=3 — but note that 31B's num_attention_heads (32) is not
#   divisible by 3. Use TP_SIZE=2 instead if needed, or stick with TP_SIZE=1.
#   For TP_SIZE=1, the other 2 GPUs sit idle (fine for this competition).
#
#=============================================================================

# ==========================
# USER-CONFIGURABLE PATHS
# ==========================
# >>> EDIT THESE to match your cluster layout <<<

# Day 1: set to 0.0 (all LLM generation, no routing)
# Day 2: set to 0.50 (high-reuse subsets use direct retrieval)
REUSE_THRESHOLD=0.0
TOP_K=3

MODEL_PATH="/mnt/data/${USER}/models/gemma-4-31B-it"
SIF_IMAGE="/mnt/data/${USER}/containers/vllm_rocm.sif"
WORK_DIR="/mnt/data/${USER}/projects/afriqa"
DATA_DIR="${WORK_DIR}"
SCRIPT_PATH="${WORK_DIR}/scripts/run_gemma_vllm.py"
OUTPUT_DIR="${WORK_DIR}/outputs/gemma_routing_t${REUSE_THRESHOLD}"

VLLM_PORT=8000
TP_SIZE=1         # Tensor parallel size (1 = single GPU, fits 31B easily)

# ==========================
# ENVIRONMENT SETUP
# ==========================
echo "========================================================"
echo "JOB START: $(date)"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "GPUs:      ${SLURM_GPUS_ON_NODE:-3}"
echo "========================================================"

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

module load rocm/6.2.4
module load python/3.11

cd "${WORK_DIR}"
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi
# Just in case requests is missing
pip install requests > /dev/null 2>&1

# Create output and log directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

# Verify critical files exist
if [ ! -f "${SIF_IMAGE}" ]; then
    echo "ERROR: Singularity image not found: ${SIF_IMAGE}"
    echo "Build it on the login node with:"
    echo "  singularity pull ${SIF_IMAGE} docker://vllm/vllm-openai-rocm:latest"
    exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model directory not found: ${MODEL_PATH}"
    echo "Download it on the login node with:"
    echo "  huggingface-cli download google/gemma-4-31B-it --local-dir ${MODEL_PATH}"
    exit 1
fi

if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "ERROR: Inference script not found: ${SCRIPT_PATH}"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Model:           ${MODEL_PATH}"
echo "  SIF Image:       ${SIF_IMAGE}"
echo "  Output:          ${OUTPUT_DIR}"
echo "  TP Size:         ${TP_SIZE}"
echo "  Reuse Threshold: ${REUSE_THRESHOLD}"
echo "  Top-K:           ${TOP_K}"
echo "  vLLM Port:       ${VLLM_PORT}"
echo ""

# ==========================
# STEP 1: Launch vLLM Server
# ==========================
echo "[$(date +%H:%M:%S)] Step 1: Launching vLLM server in background..."

singularity exec --rocm \
  --bind "/mnt/data:/mnt/data,/tmp:/tmp" \
  --pwd "${WORK_DIR}" \
  "${SIF_IMAGE}" \
  bash -c "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/opt/rocm/lib:\$LD_LIBRARY_PATH && python -m vllm.entrypoints.openai.api_server \
    --model \"${MODEL_PATH}\" \
    --served-model-name \"gemma-4-31B-it\" \
    --dtype bfloat16 \
    --tensor-parallel-size \"${TP_SIZE}\" \
    --max-model-len 8192 \
    --port \"${VLLM_PORT}\" \
    --trust-remote-code \
    --download-dir /tmp/vllm_cache \
    --disable-log-stats" \
    2>&1 | tee "${OUTPUT_DIR}/vllm_server.log" &

VLLM_PID=$!
echo "[$(date +%H:%M:%S)] vLLM server PID: ${VLLM_PID}"

# Ensure cleanup on exit (kill vLLM server when job finishes or is cancelled)
trap "echo 'Cleaning up vLLM server (PID ${VLLM_PID})...'; kill -9 ${VLLM_PID} 2>/dev/null; sleep 2; echo 'Cleanup done.'" EXIT

# ==========================
# STEP 2: Wait for Server
# ==========================
echo "[$(date +%H:%M:%S)] Step 2: Waiting for vLLM server to become ready..."

MAX_WAIT=1200   # 20 minutes max (model loading can take ~10-15 min)
POLL_INTERVAL=15
ELAPSED=0

while [ ${ELAPSED} -lt ${MAX_WAIT} ]; do
    # Use Python with urllib (no curl dependency, no requests needed)
    HEALTH_OK=$(python3 -c "
import urllib.request, sys
try:
    r = urllib.request.urlopen('http://localhost:${VLLM_PORT}/health', timeout=5)
    print('OK' if r.status == 200 else 'FAIL')
except Exception:
    print('FAIL')
" 2>/dev/null)

    if [ "${HEALTH_OK}" = "OK" ]; then
        echo "[$(date +%H:%M:%S)] vLLM server is READY! (took ${ELAPSED}s)"
        break
    fi

    # Check if vLLM process is still alive
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] ERROR: vLLM server process died unexpectedly!"
        echo "Check logs: ${OUTPUT_DIR}/vllm_server.log"
        exit 1
    fi

    echo "[$(date +%H:%M:%S)] Server not ready yet... (${ELAPSED}s / ${MAX_WAIT}s)"
    sleep ${POLL_INTERVAL}
    ELAPSED=$((ELAPSED + POLL_INTERVAL))
done

if [ ${ELAPSED} -ge ${MAX_WAIT} ]; then
    echo "[$(date +%H:%M:%S)] ERROR: vLLM server failed to start within ${MAX_WAIT}s"
    echo "Check logs: ${OUTPUT_DIR}/vllm_server.log"
    exit 1
fi

# ==========================
# STEP 3: Run Inference
# ==========================
echo ""
echo "[$(date +%H:%M:%S)] Step 3: Running inference script..."
echo "========================================================"

python "${SCRIPT_PATH}" \
    --train-path "${DATA_DIR}/Train.csv" \
    --val-path "${DATA_DIR}/Val.csv" \
    --test-path "${DATA_DIR}/Test.csv" \
    --output-dir "${OUTPUT_DIR}" \
    --vllm-url "http://localhost:${VLLM_PORT}/v1" \
    --model-name "gemma-4-31B-it" \
    --reuse-threshold "${REUSE_THRESHOLD}" \
    --top-k "${TOP_K}"

INFERENCE_EXIT=$?

echo ""
echo "========================================================"
echo "[$(date +%H:%M:%S)] Step 4: Inference complete!"
echo "  Exit code:    ${INFERENCE_EXIT}"
echo "  Output dir:   ${OUTPUT_DIR}"
echo "  Submission:   ${OUTPUT_DIR}/submission.csv"
echo "  Debug preds:  ${OUTPUT_DIR}/predictions_debug.csv"
echo "  Stats:        ${OUTPUT_DIR}/routing_stats.json"
echo "  Run log:      ${OUTPUT_DIR}/run.log"
echo "  vLLM log:     ${OUTPUT_DIR}/vllm_server.log"
echo "========================================================"
echo "JOB END: $(date)"

exit ${INFERENCE_EXIT}
