"""Rollout recorder for self-supervised pretraining."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..sim.kpis import KPIState
from ..sim.state_extractor import StateExtractor
from ..sim.traci_client import TraciClient
from ..utils.fileio import write_parquet


@dataclass
class RecorderConfig:
    steps: int = 3600
    output: Path = Path("data/ssl_rollouts.parquet")


class RolloutRecorder:
    def __init__(self, client: TraciClient, extractor: StateExtractor, config: RecorderConfig):
        self.client = client
        self.extractor = extractor
        self.config = config

    def run(self) -> Path:
        rows: List[Dict[str, object]] = []
        with self.client:
            for _ in range(self.config.steps):
                raw = self.client.step()
                struct = self.extractor.extract(raw)
                for jid, feat in struct.junction_features.items():
                    rows.append({"time": raw.time, "junction_id": jid, "features": feat.tolist()})
        df = pd.DataFrame(rows)
        write_parquet(df, self.config.output)
        return self.config.output


__all__ = ["RolloutRecorder", "RecorderConfig"]
