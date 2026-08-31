"""Shared utilities: seeding, config loading, device selection."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def set_seed(seed: int = 42) -> None:
    """Fix all RNGs we can reach for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # deterministic-friendly (may slow things down slightly)
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
