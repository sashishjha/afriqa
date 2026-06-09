# AfriQA — RBCCPS Cluster Migration Guide (AMD MI210 / ROCm)

## Problem

The **RBCCPS HPC cluster at IISc Bangalore** uses **AMD Instinct MI210 GPUs**, which do
**NOT** support **NVIDIA CUDA**. Instead, they use **AMD ROCm** (Radeon Open Compute),
which is AMD's open-source GPU computing platform.

Your original codebase was written assuming NVIDIA CUDA GPUs. This migration guide and
the modified files make the code **fully compatible with the RBCCPS AMD ROCm environment**.

### Key differences

| Feature | NVIDIA (CUDA) | AMD MI210 (ROCm) |
|---|---|---|
| GPU API | CUDA | ROCm / HIP |
| PyTorch build | `torch` (CUDA) | `torch` (ROCm) |
| Mixed precision | `fp16` (float16) | `bf16` (bfloat16) — native support |
| FAISS GPU | `faiss-gpu` (CUDA only) | **Not available** — use `faiss-cpu` |
| Device string | `"cuda"` | `"cuda"` (ROCm PyTorch maps this to HIP) |

> **Good news:** ROCm-enabled PyTorch maps `torch.cuda.*` calls to HIP internally, so
> most PyTorch code works without changes. The modifications below handle the edge cases.

---

## File Replacement Map

Replace the following **7 files** in your project with the modified versions from this
directory:

| # | Original File (your repo) | Replace With | What Changed |
|---|---|---|---|
| 1 | `src/utils.py` | `utils.py` | Added `is_rocm()`, `gpu_memory_mb()`, ROCm-aware logging in `get_device()` |
| 2 | `src/retrieval/dense_retriever.py` | `dense_retriever.py` | Default device `"cuda"` → auto-detect; uses `faiss-cpu` |
| 3 | `src/training/trainer.py` | `trainer.py` | Uses `bf16` on ROCm instead of `fp16`; `_is_rocm()` helper; `bfloat16` dtype |
| 4 | `src/predictor.py` | `predictor.py` | Default device `"cuda"` → auto-detect |
| 5 | `src/rag/pipeline.py` | `pipeline.py` | Default device `"cuda"` → auto-detect |
| 6 | `experiments/run_exp5_foundation.py` | `run_exp5_foundation.py` | ROCm-aware `gpu_memory_mb()`, ROCm detection logging |
| 7 | `src/metrics.py` | `metrics.py` | `compute_bertscore` auto-detects device for ROCm/CUDA/CPU |

### Also add

| File | Location | Purpose |
|---|---|---|
| `requirements_rocm.txt` | Project root (as `requirements.txt`) | ROCm-compatible dependencies (`faiss-cpu`, not `faiss-gpu`) |

### Files that do NOT need changes

These files are already compatible with ROCm and should be kept as-is:

- `src/__init__.py`
- `src/data_loader.py`
- `src/evaluation/evaluator.py`
- `src/retrieval/bm25_retriever.py`
- `src/retrieval/hybrid_retriever.py`
- `experiments/run_exp1_generation.py`
- `experiments/run_exp2_retrieval.py`
- `experiments/run_exp3_rag_baseline.py`
- `experiments/run_exp4_retrieval_arch.py` (placeholder)
- `experiments/run_exp6_training_rag.py` (placeholder)
- `experiments/run_exp7_ensemble.py` (placeholder)
- `experiments/generate_submission.py` (placeholder)

---

## Step-by-Step Setup on RBCCPS Cluster

### Step 1: SSH into the cluster

```bash
ssh <your_username>@<rbccps-cluster-address>
```

### Step 2: Create a virtual environment

```bash
module load python/3.10   # or whatever module system the cluster uses
python3 -m venv ~/afriqa_rocm_env
source ~/afriqa_rocm_env/bin/activate
```

### Step 3: Install PyTorch with ROCm support

```bash
# Option A: From PyTorch official (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Option B: From AMD repo (if Option A doesn't work)
# Check https://repo.radeon.com/rocm/manylinux/ for latest wheels
# wget <appropriate wheel URL>
# pip install <wheel_file>.whl
```

### Step 4: Verify PyTorch + ROCm

```bash
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'ROCm available:  {torch.cuda.is_available()}')
print(f'HIP version:     {torch.version.hip}')
print(f'GPU count:       {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name:        {torch.cuda.get_device_name(0)}')
"
```

Expected output:

```
PyTorch version: 2.x.x+rocmX.X
ROCm available:  True
HIP version:     X.X.XXXXX-XXXXXXX
GPU count:       1
GPU name:        AMD Instinct MI210
```

### Step 5: Install remaining dependencies

```bash
pip install -r requirements_rocm.txt
```

### Step 6: Replace the files

```bash
# From your project root directory:

# 1. Utils
cp rocm_modified/utils.py src/utils.py

# 2. Dense retriever
cp rocm_modified/dense_retriever.py src/retrieval/dense_retriever.py

# 3. Trainer
cp rocm_modified/trainer.py src/training/trainer.py

# 4. Predictor
cp rocm_modified/predictor.py src/predictor.py

# 5. Pipeline
cp rocm_modified/pipeline.py src/rag/pipeline.py

# 6. Experiment 5
cp rocm_modified/run_exp5_foundation.py experiments/run_exp5_foundation.py

# 7. Metrics
cp rocm_modified/metrics.py src/metrics.py

# 8. Requirements
cp rocm_modified/requirements_rocm.txt requirements.txt
```

### Step 7: Run the experiments

```bash
# Example: Run Experiment 1
python experiments/run_exp1_generation.py --config configs/config.yaml

# Example: Run Experiment 5 (Foundation Model Benchmark)
python experiments/run_exp5_foundation.py --config configs/config.yaml

# Run a single model only:
python experiments/run_exp5_foundation.py --config configs/config.yaml --models mt5-base
```

---

## Verification Checklist

After setup, run these checks:

```bash
# 1. GPU is detected
python3 -c "import torch; print('GPU OK' if torch.cuda.is_available() else 'NO GPU')"

# 2. ROCm backend is active
python3 -c "import torch; print(f'ROCm HIP: {torch.version.hip}')"

# 3. FAISS (CPU) works
python3 -c "import faiss; idx = faiss.IndexFlatIP(768); print(f'FAISS OK, dim={idx.d}')"

# 4. SentenceTransformers uses GPU
python3 -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('sentence-transformers/LaBSE', device='cuda')
print('SentenceTransformers on GPU: OK')
"

# 5. bfloat16 works on MI210
python3 -c "
import torch
t = torch.randn(2, 2, dtype=torch.bfloat16, device='cuda')
print(f'bf16 tensor on GPU: {t.device}, dtype: {t.dtype}')
"
```

---

## Troubleshooting

### `torch.cuda.is_available()` returns `False`

- Ensure ROCm is installed: run `rocm-smi` — it should show your GPU
- Ensure you installed the ROCm build of PyTorch (not the CPU or CUDA build)
- Check: `python3 -c "import torch; print(torch.version.hip)"` — should not be `None`

### ImportError: faiss / FAISS GPU errors

- You **must** use `faiss-cpu`, not `faiss-gpu` (which requires CUDA)
- Install: `pip install faiss-cpu`
- If both are installed: `pip uninstall faiss-gpu && pip install faiss-cpu`

### Out-of-memory (OOM) errors

- MI210 has 64 GB HBM2e — generous, but large models may still OOM
- Reduce `batch_size` in config (try 4 or 2)
- Increase `gradient_accumulation_steps` to compensate

### RuntimeError: HIP error / hipErrorNoBinaryForGpu

- MI210 uses architecture `gfx90a`
- Ensure your ROCm + PyTorch versions support `gfx90a`
- Try: `export HSA_OVERRIDE_GFX_VERSION=9.0.10`
- If building from source: `export PYTORCH_ROCM_ARCH=gfx90a`

### Training is slower than expected

- Enable TunableOp for optimized GEMM kernels:
  ```bash
  export PYTORCH_TUNABLEOP_ENABLED=1
  ```
- Ensure you're using `bf16` — the modified trainer handles this automatically

---

## Summary of All Changes

| Change | Why |
|---|---|
| `fp16` → `bf16` in trainer | AMD MI210 has native bf16 support; fp16 can be unstable on ROCm |
| `device="cuda"` → auto-detect | Graceful fallback; works on any hardware |
| `faiss-gpu` → `faiss-cpu` | FAISS GPU requires CUDA; CPU version works everywhere |
| `torch.float16` → `torch.bfloat16` | Better numerical stability on MI210 |
| Added `is_rocm()` utility | Detect ROCm vs CUDA at runtime |
| Added `gpu_memory_mb()` utility | Monitor GPU memory (works on both backends) |
| ROCm info logging | Clear startup messages about detected backend |

---

## References

- [ROCm PyTorch Compatibility](https://rocm.docs.amd.com/en/latest/compatibility/ml-compatibility/pytorch-compatibility.html)
- [AMD MI210 Datasheet](https://www.amd.com/en/products/accelerators/instinct/mi200/mi210.html)
- [PyTorch ROCm Installation](https://pytorch.org/get-started/locally/)
- [RBCCPS IISc](https://cps.iisc.ac.in/)
