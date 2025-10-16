"""Entry point for MARL PPO fine-tuning."""

from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig

from gso.training.marl_env import JunctionMARLEnv
from gso.training.marl_trainer import PPOConfig, PPOTrainer
from gso.utils.logging import setup_logging


@hydra.main(config_path="../configs", config_name="train_marl", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()
    env = JunctionMARLEnv(num_junctions=cfg.env.num_junctions, action_dim=cfg.model.action_dim)
    actor = torch.nn.Linear(cfg.env.state_dim, cfg.model.action_dim)
    critic = torch.nn.Linear(cfg.env.state_dim, 1)
    trainer = PPOTrainer(env, actor, critic, PPOConfig(**cfg.trainer))
    trainer.train(episodes=cfg.trainer.episodes)
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, cfg.checkpoint)


if __name__ == "__main__":
    main()
