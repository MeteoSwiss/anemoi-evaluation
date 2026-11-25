"""Nearest-node search on the model grid, and `Nearest`: each point takes the values of its nearest node."""

from __future__ import annotations

import datetime
import math
import numbers
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from anemoi.graphs import EARTH_RADIUS

from anemoi.evaluation.frame import Frame
from anemoi.evaluation.frame import Grid
from anemoi.evaluation.operators.base import GRID
from anemoi.evaluation.operators.base import POINTS
from anemoi.evaluation.operators.base import Constant
from anemoi.evaluation.operators.base import ModelSide
from anemoi.evaluation.operators.base import ObservationOperatorBase
from anemoi.evaluation.points import PointSet


def _xyz(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    from anemoi.transform.spatial import latlon_to_xyz

    return np.stack(latlon_to_xyz(np.asarray(latitudes, np.float64), np.asarray(longitudes, np.float64)), axis=1)


def chord(distance_km: float) -> float:
    """Straight-line distance on the unit sphere between two points `distance_km` apart along the surface."""
    return 2.0 * math.sin(min(distance_km / EARTH_RADIUS, math.pi) / 2.0)


def _check_distance(max_distance_km: float | None) -> float | None:
    if max_distance_km is None:
        return None
    if isinstance(max_distance_km, bool) or not isinstance(max_distance_km, numbers.Real):
        raise ValueError(f"max_distance_km must be a number of km or None, got {max_distance_km!r}")
    if not (max_distance_km > 0 and math.isfinite(max_distance_km)):
        raise ValueError(f"max_distance_km must be positive and finite, got {max_distance_km!r}")
    return float(max_distance_km)


class NearestNodes:
    """Nearest node of a grid for any points, from a KD-tree on the unit sphere built once (for the operator, and for
    reading node attributes such as region masks at the nearest node)."""

    def __init__(self, grid: Grid) -> None:
        from scipy.spatial import KDTree

        self.n = grid.n
        self.tree = KDTree(_xyz(grid.latitudes, grid.longitudes))

    def query(self, points: PointSet, max_distance_km: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per point: the nearest node (0 where out of the domain, so mask with `in_domain` before use), the
        great-circle distance in km (inf there) and whether a node lies within `max_distance_km` (None: any)."""
        max_distance_km = _check_distance(max_distance_km)
        bound = np.inf if max_distance_km is None else chord(max_distance_km)
        distance, index = self.tree.query(_xyz(points.latitudes, points.longitudes), k=1, distance_upper_bound=bound)
        index = np.asarray(index, dtype=np.int64).reshape(-1)
        in_domain = index < self.n
        distance = np.asarray(distance, np.float64).reshape(-1)
        distance_km = np.full(index.shape, np.inf)
        distance_km[in_domain] = 2.0 * EARTH_RADIUS * np.arcsin(np.minimum(distance[in_domain] / 2.0, 1.0))
        return np.where(in_domain, index, 0), distance_km, in_domain


@dataclass(frozen=True)
class NearestPlan:
    """Per point: the nearest node (0 where out of the domain), great-circle distance in km (inf there) and the
    in-domain mask; the node index and the out-of-domain mask also as tensors on the device."""

    nodes: np.ndarray
    distance_km: np.ndarray
    in_domain: np.ndarray
    index: torch.Tensor
    outside: torch.Tensor


class Nearest(ObservationOperatorBase):
    """The value of the nearest model node at each point.

    `max_distance_km` is required and keyword-only: a point farther than that along the surface from every node
    (outside a limited-area domain) gets a NaN forecast and is reported by `out_of_domain`. None maps every point, which
    suits a global model only.
    """

    input_space = GRID
    output_space = POINTS

    def __init__(self, *, max_distance_km: float | None) -> None:
        self.max_distance_km = _check_distance(max_distance_km)
        self._search: NearestNodes | None = None

    def bind(self, model: ModelSide) -> None:
        """Build the tree on the model grid."""
        super().bind(model)
        self._device = torch.device(model.device)
        self._search = NearestNodes(model.grid)

    def prepare(self, points: PointSet, constants: Mapping[str, Constant]) -> NearestPlan:
        """Nearest node of every point."""
        if self._search is None:
            raise RuntimeError("bind the operator to a model first")
        nodes, distance_km, in_domain = self._search.query(points, self.max_distance_km)
        return NearestPlan(
            nodes=nodes,
            distance_km=distance_km,
            in_domain=in_domain,
            index=torch.from_numpy(nodes).to(self._device),
            outside=torch.from_numpy(~in_domain).to(self._device),
        )

    def apply(self, frames: Sequence[Frame], plan: NearestPlan, valid_time: datetime.datetime) -> Frame:
        """Gather the nodes; NaN where out of the domain. Exact, so float32 throughout."""
        frame = frames[0]
        if frame.data.shape[-1] != self._search.n:
            raise ValueError(f"the frame has {frame.data.shape[-1]} nodes, the operator was bound to {self._search.n}")
        data = frame.data.index_select(2, plan.index.to(frame.data.device))
        if not plan.in_domain.all():
            data = data.masked_fill(plan.outside.to(frame.data.device), float("nan"))
        return self.output_frame(frames, valid_time, list(frame.variables), data)

    def map_constant(self, values: np.ndarray, plan: NearestPlan) -> np.ndarray:
        """The constant at the nearest node; NaN where out of the domain."""
        return np.where(plan.in_domain, np.asarray(values, dtype=np.float64)[plan.nodes], np.nan)

    def out_of_domain(self, plan: NearestPlan) -> np.ndarray:
        """Points with no node within `max_distance_km`."""
        return ~plan.in_domain

    def diagnostics(self, plan: NearestPlan) -> dict:
        """`nearest_node_max_km`, the largest distance from a point in the domain to its node (none when no point is
        in the domain)."""
        if not plan.in_domain.any():
            return {}
        return {"nearest_node_max_km": float(plan.distance_km[plan.in_domain].max())}

    def to_config(self) -> dict:
        """`nearest` with its parameters (provisional, see `ObservationOperator.to_config`)."""
        return {"nearest": self.parameters()}
