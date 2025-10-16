"""Top-level package for the GNN Signal Optimizer."""

from importlib.metadata import version

__all__ = ["__version__"]


def __getattr__(name: str) -> str:
    if name == "__version__":
        try:
            return version("gnn-signal-optimizer")
        except Exception:  # pragma: no cover - fallback when pkg metadata missing
            return "0.1.0"
    raise AttributeError(name)
