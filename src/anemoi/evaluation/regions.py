"""Region masks; plain (N,) boolean arrays at the API boundary, they may overlap. YAML specs map onto them in `config`."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from anemoi.evaluation.frame import Grid


def all(grid: Grid) -> np.ndarray:  # noqa: A001
    """Every node."""
    return np.ones(grid.n, dtype=bool)


def bbox(grid: Grid, north: float, west: float, south: float, east: float) -> np.ndarray:
    """Nodes inside a latitude/longitude box, longitude wrap handled by anemoi-transform."""
    from anemoi.transform.spatial import cropping_mask

    return np.asarray(cropping_mask(grid.latitudes, grid.longitudes, north, west, south, east), dtype=bool)


def from_file(path: str | Path, grid: Grid) -> np.ndarray:
    """Boolean (N,) mask from a .npy file."""
    return grid.check_shape(np.load(path), str(path)).astype(bool)


def stack(regions: dict[str, np.ndarray], n: int) -> tuple[list[str], np.ndarray]:
    """Region names and the (N, R) boolean mask matrix."""
    if not regions:
        raise ValueError("at least one region is required")
    masks = []
    for name, mask in regions.items():
        mask = np.asarray(mask)
        if mask.shape != (n,) or mask.dtype != bool:
            raise ValueError(f"region {name!r} must be a boolean array of shape {(n,)}, got {mask.dtype} {mask.shape}")
        masks.append(mask)
    return list(regions), np.stack(masks, axis=1)
