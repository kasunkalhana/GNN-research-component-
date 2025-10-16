"""Self-supervised dataset utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class SSLExample:
    history: torch.Tensor
    target: torch.Tensor


class SSLDataset(Dataset):
    """Construct sequences from recorded junction features."""

    def __init__(self, dataframe: pd.DataFrame, history: int = 4):
        self.history = history
        grouped = dataframe.groupby("junction_id")
        samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _, group in grouped:
            features = torch.tensor(np.stack(group["features"].to_list()), dtype=torch.float32)
            for idx in range(len(features) - history):
                hist = features[idx : idx + history]
                target = features[idx + history]
                samples.append((hist, target))
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> SSLExample:
        hist, target = self.samples[idx]
        return SSLExample(history=hist, target=target)


def load_ssl_dataset(path: Path, history: int = 4) -> "SSLDataset":
    df = pd.read_parquet(path)
    return SSLDataset(df, history=history)


__all__ = ["SSLDataset", "SSLExample", "load_ssl_dataset"]
