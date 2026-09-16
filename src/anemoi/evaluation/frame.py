"""Grid and Frame, the data model shared by sources and the aggregator."""

from __future__ import annotations

import datetime
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Grid:
    """Node coordinates in degrees, shape (N,)."""

    latitudes: np.ndarray
    longitudes: np.ndarray

    def __post_init__(self) -> None:
        lats = np.asarray(self.latitudes, dtype=np.float64).reshape(-1)
        lons = np.asarray(self.longitudes, dtype=np.float64).reshape(-1)
        if lats.shape != lons.shape:
            raise ValueError(f"latitudes {lats.shape} and longitudes {lons.shape} differ in length")
        object.__setattr__(self, "latitudes", lats)
        object.__setattr__(self, "longitudes", lons)

    @property
    def n(self) -> int:
        """Number of nodes."""
        return self.latitudes.shape[0]

    def latlons_rad(self) -> np.ndarray:
        """Coordinates as an (N, 2) array of [lat, lon] in radians."""
        return np.deg2rad(np.stack([self.latitudes, self.longitudes], axis=1))

    def check_shape(self, array: np.ndarray, what: str) -> np.ndarray:
        """Return `array` if it has one value per node, else raise naming `what`."""
        if array.shape != (self.n,):
            raise ValueError(f"{what} has shape {array.shape}, expected {(self.n,)}")
        return array

    def check_compatible(self, other: Grid, tolerance: float = 1e-5) -> None:
        """Raise unless both grids have the same nodes within `tolerance` degrees."""
        if self.n != other.n:
            raise ValueError(f"grids differ in size: {self.n} vs {other.n} nodes")
        if not (
            np.allclose(self.latitudes, other.latitudes, atol=tolerance)
            and np.allclose(self.longitudes, other.longitudes, atol=tolerance)
        ):
            raise ValueError(f"grid coordinates differ by more than {tolerance} degrees")


@dataclass(frozen=True)
class Frame:
    """One forecast step: `data` is float32 (M members, V variables, N nodes)."""

    init_time: datetime.datetime
    lead_time: datetime.timedelta
    valid_time: datetime.datetime
    variables: list[str]
    data: torch.Tensor

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise ValueError(f"frame data must be (M, V, N), got shape {tuple(self.data.shape)}")
        if self.data.dtype != torch.float32:
            raise ValueError(f"frame data must be float32, got {self.data.dtype}")
        if self.data.shape[1] != len(self.variables):
            raise ValueError(f"frame has {self.data.shape[1]} variable columns for {len(self.variables)} names")
        if self.valid_time != self.init_time + self.lead_time:
            raise ValueError(f"valid_time {self.valid_time} != init_time {self.init_time} + lead_time {self.lead_time}")

    @property
    def members(self) -> int:
        """Number of ensemble members."""
        return self.data.shape[0]
