"""File IO helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: Path) -> Path:
    """Ensure that a directory exists."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a dataframe to Parquet."""

    ensure_dir(path.parent)
    df.to_parquet(path, index=False)


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a dataframe from Parquet."""

    return pd.read_parquet(path)


def write_yaml(data: Any, path: Path) -> None:
    """Serialize data to YAML file."""

    import yaml

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh)


__all__ = ["ensure_dir", "write_parquet", "read_parquet", "write_yaml"]
