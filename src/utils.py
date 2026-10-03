"""
Utility functions for reproducibility, hardware introspection, and metrics persistence.
"""

import os
import sys
import json
import random
import platform
import psutil
import numpy as np
import torch
from typing import Dict, Any


def set_seed(seed: int = 42) -> None:
    """
    Sets deterministic random seeds across python random, numpy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def detect_hardware() -> Dict[str, Any]:
    """
    Detects and returns hardware and environment details dynamically without static assumptions.
    """
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "cpu": platform.processor() or "Unknown CPU",
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
    else:
        info["gpu_count"] = 0
        info["gpu_name"] = "None (CPU only)"
        info["gpu_memory_gb"] = 0.0

    return info


def save_json(data: Any, filepath: str) -> None:
    """
    Saves a dictionary or list to JSON with formatting.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Any:
    """
    Loads JSON file safely.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
