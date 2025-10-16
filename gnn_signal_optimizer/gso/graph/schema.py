"""Graph schema definitions for heterogeneous graphs."""

from __future__ import annotations

from typing import Dict, List, Tuple

from pydantic import BaseModel, Field


class NodeSpec(BaseModel):
    """Specification for node feature layout."""

    name: str
    dim: int
    description: str


class EdgeSpec(BaseModel):
    """Specification for edge connectivity."""

    name: str
    src: str
    dst: str
    description: str = ""


class GraphSchema(BaseModel):
    """Container for node and edge specifications."""

    nodes: Dict[str, NodeSpec]
    edges: List[EdgeSpec]

    @property
    def node_types(self) -> List[str]:
        return list(self.nodes.keys())

    def feature_dims(self) -> Dict[str, int]:
        return {name: spec.dim for name, spec in self.nodes.items()}


DEFAULT_SCHEMA = GraphSchema(
    nodes={
        "junction": NodeSpec(name="junction", dim=6, description="Aggregated TLS state"),
        "lane": NodeSpec(name="lane", dim=5, description="Lane traffic state"),
        "vehicle": NodeSpec(name="vehicle", dim=3, description="Optional vehicle features"),
    },
    edges=[
        EdgeSpec(name="lane_to_junction", src="lane", dst="junction", description="Incoming lane"),
        EdgeSpec(name="junction_to_lane", src="junction", dst="lane", description="Outgoing movements"),
        EdgeSpec(name="lane_to_lane", src="lane", dst="lane", description="Successive lanes"),
        EdgeSpec(name="vehicle_to_lane", src="vehicle", dst="lane", description="Vehicle occupancy"),
    ],
)


__all__ = ["GraphSchema", "NodeSpec", "EdgeSpec", "DEFAULT_SCHEMA"]
