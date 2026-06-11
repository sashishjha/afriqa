"""Shared utilities: seeding, device, config loading, I/O.
Modified for AMD ROCm (RBCCPS cluster, IISc - AMD MI210 GPUs).
"""

import os
import json
import random
import logging
import yaml
import numpy as np
import torch


def is_rocm() -> bool:
    """Return True if PyTorch was built with ROCm (HIP) support."""
    return hasattr(torch.version, "hip") and torch.version.hip is not None


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():          # works for both CUDA and ROCm
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    """Auto-detect the best available device (ROCm / CUDA / CPU)."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        if is_rocm():
            logging.getLogger("afriqa").info(
                f"ROCm detected (HIP {torch.version.hip}). Using device: {dev}"
            )
        else:
            logging.getLogger("afriqa").info(f"CUDA detected. Using device: {dev}")
        return dev
    logging.getLogger("afriqa").info("No GPU detected. Using CPU.")
    return torch.device("cpu")


def gpu_memory_mb() -> float:
    """Return peak allocated GPU memory in MB (works for both CUDA and ROCm)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_json(data, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )
    return logging.getLogger("afriqa")
