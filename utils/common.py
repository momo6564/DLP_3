"""Small shared helpers: config loading, seeding, device pick, timing, results table."""
from __future__ import annotations

import json
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def set_seed(seed: int = 42) -> None:
    """Make a run repeatable enough to compare against.

    Not bit-exact on GPU, and that is fine - just state the seed in your report.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model) -> tuple:
    """Return (total, trainable) parameter counts.

    Quote both in your report: the gap between them is the whole point of freezing layers.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@contextmanager
def timer(label: str):
    start = time.perf_counter()
    yield
    print("[time] {}: {:.1f}s".format(label, time.perf_counter() - start))


def save_results(record: dict, out_dir, name: str = "results.json") -> Path:
    """Append one experiment to a JSON file so the final comparison table writes itself."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    history = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    history.append(record)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("[saved] {}".format(path))
    return path
