"""Node weight providers; plain (N,) float64 arrays at the API boundary, YAML specs map onto them in `config`."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from anemoi.evaluation.frame import Grid


def uniform(grid: Grid) -> np.ndarray:
    """Equal weight for every node."""
    return np.ones(grid.n, dtype=np.float64)


def spherical_voronoi(grid: Grid) -> np.ndarray:
    """Spherical Voronoi cell areas from anemoi-graphs."""
    from anemoi.graphs.nodes.attributes import SphericalAreaWeights

    return np.asarray(SphericalAreaWeights().compute_area_weights(grid.latlons_rad()), dtype=np.float64)


def from_file(path: str | Path, grid: Grid) -> np.ndarray:
    """Precomputed (N,) weights from a .npy file."""
    return grid.check_shape(np.load(path), str(path)).astype(np.float64)


def graph_node_attribute(forecast: object, name: str, grid: Grid) -> np.ndarray:
    """Per-node attribute of the forecast source's model graph, shape-checked against `grid`."""
    if forecast is None:
        raise ValueError(f"a forecast source with a model graph is needed for the node attribute {name!r}")
    return grid.check_shape(np.asarray(forecast.graph_node_attribute(name)), name)
