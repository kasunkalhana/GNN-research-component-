"""Runtime inference loop tying together simulation and policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from ..graph.builder import GraphBuilder
from ..model.hgat import RecurrentHGAT
from ..model.mc_dropout import mc_dropout_inference, predictive_entropy
from ..model.policy import select_actions
from ..sim.control_translator import ControlTranslator
from ..sim.kpis import KPIState
from ..sim.state_extractor import StateExtractor
from ..sim.traci_client import RawState, TraciClient


@dataclass
class InferenceConfig:
    mc_passes: int = 10


class InferenceEngine:
    def __init__(
        self,
        client: TraciClient,
        extractor: StateExtractor,
        builder: GraphBuilder,
        model: RecurrentHGAT,
        translator: ControlTranslator,
        kpis: KPIState,
        config: InferenceConfig,
    ):
        self.client = client
        self.extractor = extractor
        self.builder = builder
        self.model = model
        self.translator = translator
        self.kpis = kpis
        self.config = config
        self.hidden: Optional[torch.Tensor] = None

    def step(self) -> Dict[str, object]:
        raw = self.client.step()
        struct = self.extractor.extract(raw)
        data = self.builder.build(struct, raw.time)

        def forward_fn(model: RecurrentHGAT) -> torch.Tensor:
            logits, _ = model(data, self.hidden)
            return logits

        mean_logits, var_logits = mc_dropout_inference(self.model, forward_fn, passes=self.config.mc_passes)
        self.model.eval()
        logits, self.hidden = self.model(data, self.hidden)
        uncertainties = predictive_entropy(mean_logits)

        legal_masks = {jid: torch.tensor(mask, dtype=torch.float32) for jid, mask in struct.legal_actions.items()}
        actions = select_actions(logits, legal_masks, self.builder.node_order)
        action_dict = self.translator.translate(actions, {jid: float(uncertainties[idx]) for idx, jid in enumerate(self.builder.node_order["junction"])}, raw.junctions)

        self.kpis.update(raw)
        return {
            "time": raw.time,
            "actions": action_dict,
            "uncertainty": {jid: float(uncertainties[idx]) for idx, jid in enumerate(self.builder.node_order["junction"])},
            "kpis": self.kpis.as_dict(),
        }


__all__ = ["InferenceEngine", "InferenceConfig"]
