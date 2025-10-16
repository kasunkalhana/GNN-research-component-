"""PPO-style trainer for MARL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.distributions import Categorical

from ..utils.logging import logger
from .marl_env import JunctionMARLEnv


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    epochs: int = 3
    batch_size: int = 32
    max_steps: int = 200


class PPOTrainer:
    def __init__(self, env: JunctionMARLEnv, actor: torch.nn.Module, critic: torch.nn.Module, config: PPOConfig):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.config = config
        self.actor_optim = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
        self.critic_optim = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)

    def run_episode(self) -> Dict[str, torch.Tensor]:
        obs = self.env.reset()
        observations: List[torch.Tensor] = []
        actions: List[torch.Tensor] = []
        log_probs: List[torch.Tensor] = []
        rewards: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        for _ in range(self.config.max_steps):
            logits = self.actor(obs)
            dist = Categorical(logits=logits)
            action = dist.sample()
            value = self.critic(obs)
            next_obs, reward, done, _ = self.env.step(action.float())
            observations.append(obs)
            actions.append(action)
            log_probs.append(dist.log_prob(action))
            rewards.append(reward)
            values.append(value.squeeze(-1))
            obs = next_obs
            if done:
                break
        return {
            "observations": torch.stack(observations),
            "actions": torch.stack(actions),
            "log_probs": torch.stack(log_probs),
            "rewards": torch.stack(rewards),
            "values": torch.stack(values),
        }

    def compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.config.gamma * (values[t + 1] if t + 1 < len(values) else 0) - values[t]
            last_gae = delta + self.config.gamma * self.config.gae_lambda * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        return advantages, returns

    def train(self, episodes: int = 10) -> None:
        for episode in range(episodes):
            batch = self.run_episode()
            advantages, returns = self.compute_advantages(batch["rewards"], batch["values"])
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            for _ in range(self.config.epochs):
                logits = self.actor(batch["observations"][-1])
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(batch["actions"][-1])
                ratio = torch.exp(log_probs - batch["log_probs"][-1])
                surr1 = ratio * advantages[-1]
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * advantages[-1]
                actor_loss = -torch.min(surr1, surr2)
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                value_pred = self.critic(batch["observations"][-1]).squeeze(-1)
                critic_loss = torch.nn.functional.mse_loss(value_pred, returns[-1])
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
            logger.info("ppo episode {episode} loss_actor={actor} loss_critic={critic}", episode=episode, actor=float(actor_loss), critic=float(critic_loss))


__all__ = ["PPOTrainer", "PPOConfig"]
