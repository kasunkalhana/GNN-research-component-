"""Entry point for self-supervised pretraining."""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from gso.runtime.recorder import RecorderConfig, RolloutRecorder
from gso.sim.state_extractor import StateExtractor
from gso.sim.traci_client import TraciClient
from gso.training.ssl_dataset import load_ssl_dataset
from gso.training.ssl_trainer import SSLTrainer, SSLTrainConfig
from gso.utils.logging import setup_logging


@hydra.main(config_path="../configs", config_name="train_ssl", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()
    mode = cfg.get("mode", "train")
    if mode == "record":
        extractor = StateExtractor(cfg.phases)
        client = TraciClient(Path(cfg.env.sumocfg), gui=cfg.env.gui, step_length=cfg.env.step_length)
        recorder = RolloutRecorder(client, extractor, RecorderConfig(steps=cfg.steps, output=Path(cfg.output)))
        recorder.run()
        return

    dataset = load_ssl_dataset(Path(cfg.dataset), history=cfg.trainer.history)
    input_dim = dataset[0].history.numel()
    model = torch.nn.Sequential(torch.nn.Linear(input_dim, cfg.model.hidden_dim), torch.nn.ReLU(), torch.nn.Linear(cfg.model.hidden_dim, dataset[0].target.numel()))
    trainer = SSLTrainer(model, SSLTrainConfig(**cfg.trainer))
    trainer.fit(dataset)
    trainer.save(Path(cfg.checkpoint))


if __name__ == "__main__":
    main()
