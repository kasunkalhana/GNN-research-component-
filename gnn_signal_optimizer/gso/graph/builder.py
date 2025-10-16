"""Build PyTorch Geometric :class:`HeteroData` objects."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import torch

from ..sim.state_extractor import StructuredState
from .schema import GraphSchema, DEFAULT_SCHEMA

try:  # pragma: no cover - optional dependency
    from torch_geometric.data import HeteroData
except Exception:  # pragma: no cover - fallback implementation for tests
    class _Store(dict):  # type: ignore
        """Simple attribute container mimicking PyG's storage classes."""

        def __getattr__(self, item):
            try:
                return self[item]
            except KeyError as exc:  # pragma: no cover - clearer error
                raise AttributeError(item) from exc

        def __setattr__(self, key, value):
            self[key] = value

        @property
        def num_nodes(self) -> int:
            if "x" in self:
                return int(self["x"].size(0))
            raise AttributeError("num_nodes")

    class HeteroData(dict):  # type: ignore
        """Minimal stand-in for :class:`torch_geometric.data.HeteroData`."""

        def __init__(self):
            super().__init__()
            self._node_store: Dict[str, _Store] = {}
            self._edge_store: Dict[Tuple[str, str, str], _Store] = {}

        def __getitem__(self, key):
            if isinstance(key, tuple):
                store = self._edge_store.setdefault(tuple(key), _Store())
                return store
            store = self._node_store.setdefault(key, _Store())
            return store

        def __setitem__(self, key, value):
            if isinstance(key, tuple):
                self._edge_store[tuple(key)] = value
            else:
                self._node_store[key] = value

        def __getattr__(self, item):
            try:
                return super().__getattribute__(item)
            except AttributeError:
                return self[item]

        def __contains__(self, key):
            if isinstance(key, tuple):
                return tuple(key) in self._edge_store
            return key in self._node_store


class GraphBuilder:
    """Incrementally build :class:`HeteroData` objects from structured state."""

    def __init__(
        self,
        schema: GraphSchema = DEFAULT_SCHEMA,
        lane_to_junction: Dict[str, str] | None = None,
        lane_successors: Dict[str, Iterable[str]] | None = None,
    ):
        self.schema = schema
        self.lane_to_junction = lane_to_junction or {}
        self.lane_successors = lane_successors or {}
        self.node_to_index: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.node_order: Dict[str, List[str]] = defaultdict(list)

    def _ensure_nodes(self, node_type: str, ids: Iterable[str]) -> None:
        mapping = self.node_to_index[node_type]
        order = self.node_order[node_type]
        for node_id in ids:
            if node_id not in mapping:
                mapping[node_id] = len(order)
                order.append(node_id)

    def build(self, state: StructuredState, time: float = 0.0) -> HeteroData:
        """Create a :class:`HeteroData` snapshot."""

        self._ensure_nodes("junction", state.junction_features.keys())
        self._ensure_nodes("lane", state.lane_features.keys())
        self._ensure_nodes("vehicle", state.vehicle_features.keys())

        data = HeteroData()
        feature_dims = self.schema.feature_dims()
        for node_type in ["junction", "lane", "vehicle"]:
            dim = feature_dims.get(node_type, 0)
            order = self.node_order[node_type]
            feats = torch.zeros((len(order), dim), dtype=torch.float32)
            present_mask = torch.zeros(len(order), dtype=torch.bool)
            feat_dict = getattr(state, f"{node_type}_features")
            for node_id, arr in feat_dict.items():
                idx = self.node_to_index[node_type][node_id]
                feats[idx, : len(arr)] = torch.as_tensor(arr, dtype=torch.float32)
                present_mask[idx] = True
            node_store = data[node_type]
            node_store.x = feats
            node_store.present_mask = present_mask

        # attach legal action masks for junctions (variable action dimensions supported)
        junction_order = self.node_order["junction"]
        if junction_order:
            max_actions = max((len(state.legal_actions.get(jid, [])) for jid in junction_order), default=0)
            if max_actions > 0:
                action_mask = torch.zeros((len(junction_order), max_actions), dtype=torch.float32)
                for jid in junction_order:
                    mask_values = state.legal_actions.get(jid, [])
                    if len(mask_values) == 0:
                        continue
                    idx = self.node_to_index["junction"][jid]
                    values = torch.as_tensor(mask_values, dtype=torch.float32)
                    action_mask[idx, : len(values)] = values
                data["junction"].action_mask = action_mask

        lane_idx = self.node_to_index["lane"]
        junction_idx = self.node_to_index["junction"]

        lane_to_junc_edges = self._make_edges(
            [
                (lane_idx[lane], junction_idx[self.lane_to_junction[lane]])
                for lane in state.lane_features
                if lane in self.lane_to_junction and self.lane_to_junction[lane] in junction_idx
            ]
        )
        data[("lane", "to", "junction")].edge_index = lane_to_junc_edges

        junc_to_lane_edges = self._make_edges(
            [
                (junction_idx[junc], lane_idx[lane])
                for lane, junc in self.lane_to_junction.items()
                if lane in lane_idx and junc in junction_idx
            ]
        )
        data[("junction", "to", "lane")].edge_index = junc_to_lane_edges

        lane_to_lane_edges = self._make_edges(
            [
                (lane_idx[src], lane_idx[dst])
                for src, dsts in self.lane_successors.items()
                if src in lane_idx
                for dst in dsts
                if dst in lane_idx
            ]
        )
        data[("lane", "to", "lane")].edge_index = lane_to_lane_edges

        data[("vehicle", "to", "lane")].edge_index = self._make_edges([])
        data.time = torch.tensor([time], dtype=torch.float32)
        return data

    def _make_edges(self, pairs: List[Tuple[int, int]]) -> torch.Tensor:
        if not pairs:
            return torch.zeros((2, 0), dtype=torch.long)
        return torch.tensor(pairs, dtype=torch.long).t().contiguous()


__all__ = ["GraphBuilder"]
