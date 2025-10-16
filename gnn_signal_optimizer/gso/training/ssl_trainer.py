"""Self-supervised training loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.logging import logger
from .ssl_dataset import SSLDataset


@dataclass
class SSLTrainConfig:
    epochs: int = 5
    lr: float = 1e-3
    batch_size: int = 32
    history: int = 4


class SSLTrainer:
    """Train a model to predict next-step junction features."""

    def __init__(self, model: torch.nn.Module, config: SSLTrainConfig):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        self.loss_fn = torch.nn.MSELoss()

    def fit(self, dataset: SSLDataset) -> None:
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        self.model.train()
        for epoch in range(self.config.epochs):
            losses = []
            for batch in tqdm(loader, desc=f"ssl-epoch-{epoch}"):
                hist = batch.history.view(batch.history.size(0), -1)
                target = batch.target
                pred = self.model(hist)
                loss = self.loss_fn(pred, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            logger.info("SSL epoch {epoch} loss={loss}", epoch=epoch, loss=sum(losses) / max(len(losses), 1))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)


__all__ = ["SSLTrainer", "SSLTrainConfig"]
