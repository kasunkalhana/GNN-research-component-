"""Recurrent heterogeneous graph attention network."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn


class AttentionAggregator(nn.Module):
    """Simple attention-like aggregator for message passing."""

    def __init__(self, in_src: int, in_dst: int, out: int, dropout: float = 0.1):
        super().__init__()
        self.src_lin = nn.Linear(in_src, out)
        self.dst_lin = nn.Linear(in_dst, out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, dst: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.zeros((dst.size(0), self.src_lin.out_features), device=dst.device)
        src_proj = self.src_lin(src)[edge_index[0]]
        dst_proj = self.dst_lin(dst)[edge_index[1]]
        scores = (src_proj * dst_proj).sum(dim=-1)
        weights = torch.softmax(scores, dim=0)
        weights = self.dropout(weights.unsqueeze(-1))
        messages = src_proj * weights
        out = torch.zeros((dst.size(0), src_proj.size(-1)), device=dst.device)
        out.index_add_(0, edge_index[1], messages)
        return out


@dataclass
class HGATConfig:
    hidden_dim: int = 64
    dropout: float = 0.1
    history: int = 4
    num_actions: int = 4


class RecurrentHGAT(nn.Module):
    """Minimal recurrent HGAT over junction nodes."""

    def __init__(self, input_dims: Dict[str, int], config: HGATConfig):
        super().__init__()
        self.config = config
        self.encoders = nn.ModuleDict({k: nn.Sequential(nn.Linear(dim, config.hidden_dim), nn.ReLU()) for k, dim in input_dims.items()})
        self.aggregators = nn.ModuleDict({
            "lane_to_junction": AttentionAggregator(config.hidden_dim, config.hidden_dim, config.hidden_dim, dropout=config.dropout),
            "junction_self": AttentionAggregator(config.hidden_dim, config.hidden_dim, config.hidden_dim, dropout=config.dropout),
        })
        self.gru = nn.GRUCell(config.hidden_dim, config.hidden_dim)
        self.head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(config.hidden_dim, config.num_actions))

    def forward(
        self,
        data,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        junction_x = data[("junction", "x")]
        lane_x = data[("lane", "x")]
        junction_emb = self.encoders["junction"](junction_x)
        lane_emb = self.encoders["lane"](lane_x)

        lane_to_junction_edges = data[("lane", "to", "junction", "edge_index")]
        lane_messages = self.aggregators["lane_to_junction"](lane_emb, junction_emb, lane_to_junction_edges)

        combined = junction_emb + lane_messages
        if hidden is None:
            hidden = torch.zeros_like(combined)
        new_hidden = self.gru(combined, hidden)
        logits = self.head(new_hidden)
        return logits, new_hidden


__all__ = ["RecurrentHGAT", "HGATConfig"]
