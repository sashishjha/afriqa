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
#SBATCH --output=logs/gemma_routing_%j.out
#SBATCH --error=logs/gemma_routing_%j.err

# ============================================================================
#  Dense Retrieval + Per-Subset Routing for Zindi Multilingual Health QA
#  Cluster: IISc Slurm · 3x AMD MI300X · Singularity + vLLM (ROCm)
# ============================================================================
set -euo pipefail

echo "====================================================================="
echo "  Job ID     : ${SLURM_JOB_ID}"
echo "  Node       : $(hostname)"
echo "  Started    : $(date)"
echo "  GPUs       : ${SLURM_JOB_GPUS:-N/A}"
echo "====================================================================="

# ---------- Directories & Paths ----------
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
DATA_DIR="${PROJECT_DIR}"
SCRIPT_DIR="${PROJECT_DIR}/scripts"
OUTPUT_DIR="${PROJECT_DIR}/output/run_${SLURM_JOB_ID}"
LOG_DIR="${PROJECT_DIR}/logs"

MODEL_DIR="/mnt/data/${USER}/models"
CONTAINER_DIR="/mnt/data/${USER}/containers"
GEMMA_MODEL="${MODEL_DIR}/gemma-4-31B-it"                       # local Gemma weights
EMBED_MODEL="${MODEL_DIR}/intfloat_multilingual-e5-large"       # local mE5 weights
SIF_IMAGE="${CONTAINER_DIR}/vllm_rocm.sif"                      # Singularity container

# ---------- Configurable Parameters ----------
# Per-subset thresholds are hardcoded in the Python script.
# These legacy variables are kept for backward compat / logging.
REUSE_THRESHOLD=0.85     # (legacy — not used by the new script)
TOP_K=5
VLLM_PORT=8000

# ---------- Create Directories ----------
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# ---------- Verify Paths ----------
echo ""
echo "[CHECK] Verifying critical paths..."
for P in "${DATA_DIR}/Train.csv" "${DATA_DIR}/Test.csv" "${GEMMA_MODEL}" "${EMBED_MODEL}" "${SIF_IMAGE}" "${SCRIPT_DIR}/run_gemma_vllm.py"; do
    if [ -e "$P" ]; then
        echo "  [OK]   $P"
    else
        echo "  [FAIL] $P — NOT FOUND!"
        exit 1
    fi
done
echo ""

# ---------- Environment Variables ----------
# GPU visibility for ROCm / AMD MI300X
export CUDA_VISIBLE_DEVICES=0,1,2
export HIP_VISIBLE_DEVICES=0,1,2
export ROCR_VISIBLE_DEVICES=0,1,2

# Networking (local only)
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export NCCL_DEBUG=WARN

# AMD-specific workaround
export HSA_FORCE_FINE_GRAIN_PCIE=1

# Tokenizer / parallelism safety
export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=1

# *** CRITICAL: Force offline mode to prevent ANY HuggingFace internet access ***
export HF_HUB_OFFLINE=1
export HF_HOME="${MODEL_DIR}/hf_cache"
export SENTENCE_TRANSFORMERS_HOME="${MODEL_DIR}"
export TRANSFORMERS_OFFLINE=1

# ---------- Modules ----------
module load rocm/6.2.4
module load python/3.11

# Activate the virtual environment
echo "[SETUP] Activating virtual environment..."
source "${PROJECT_DIR}/.venv/bin/activate"

# ---------- Start vLLM Server ----------
echo ""
echo "[vLLM] Starting Gemma-4-31B-it server on port ${VLLM_PORT}..."
echo "  Model: ${GEMMA_MODEL}"
echo "  Image: ${SIF_IMAGE}"

TP_SIZE=2  # tensor-parallel across 2 MI300X GPUs

singularity exec \
    --rocm \
    --bind "/mnt/data:/mnt/data,/tmp:/tmp" \
    "${SIF_IMAGE}" \
    bash -c "
        export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/opt/rocm/lib:\$LD_LIBRARY_PATH
        python3 -m vllm.entrypoints.openai.api_server \
            --model '${GEMMA_MODEL}' \
            --served-model-name 'gemma-4-31B-it' \
            --dtype bfloat16 \
            --tensor-parallel-size ${TP_SIZE} \
            --max-model-len 8192 \
            --trust-remote-code \
            --disable-log-stats \
            --port ${VLLM_PORT} \
            --host 0.0.0.0
    " &
VLLM_PID=$!
echo "  vLLM PID: ${VLLM_PID}"

# Cleanup trap: kill vLLM server on exit
cleanup() {
    echo ""
    echo "[CLEANUP] Killing vLLM server (PID=${VLLM_PID})..."
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
    echo "[CLEANUP] Done. Job finished at $(date)"
}
trap cleanup EXIT INT TERM

# ---------- Wait for vLLM Health Check ----------
echo "[vLLM] Waiting for server to become ready..."
MAX_WAIT=600
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if python3 -c "
import urllib.request, json, sys
try:
    r = urllib.request.urlopen('http://localhost:${VLLM_PORT}/v1/models', timeout=5)
    data = json.loads(r.read())
    models = [m['id'] for m in data.get('data', [])]
    print(f'  Models available: {models}')
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
        echo "  vLLM is ready!"
        break
    fi
    sleep 10
    ELAPSED=$((ELAPSED + 10))
    echo "  Waiting... ${ELAPSED}s / ${MAX_WAIT}s"
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "[ERROR] vLLM did not become ready within ${MAX_WAIT}s!"
    echo "  Continuing anyway — script will fall back to retrieval-only mode."
fi

# ---------- Run Inference Pipeline ----------
echo ""
echo "====================================================================="
echo "[INFERENCE] Launching Dense Retrieval + Per-Subset Routing pipeline..."
echo "====================================================================="
echo ""

python3 "${SCRIPT_DIR}/run_gemma_vllm.py" \
    --train-path "${DATA_DIR}/Train.csv" \
    --val-path   "${DATA_DIR}/Val.csv" \
    --test-path  "${DATA_DIR}/Test.csv" \
    --output-dir "${OUTPUT_DIR}" \
    --vllm-url   "http://localhost:${VLLM_PORT}/v1" \
    --model-name "gemma-4-31B-it" \
    --embed-model-path "${EMBED_MODEL}" \
    --embed-gpu 2 \
    --top-k ${TOP_K}

echo ""
echo "====================================================================="
echo "  INFERENCE COMPLETE"
echo "  Output dir : ${OUTPUT_DIR}"
echo "  Submission : ${OUTPUT_DIR}/submission.csv"
echo "  Debug      : ${OUTPUT_DIR}/predictions_debug.csv"
echo "  Stats      : ${OUTPUT_DIR}/routing_stats.json"
echo "  Log        : ${OUTPUT_DIR}/run.log"
echo "  Finished   : $(date)"
echo "====================================================================="
