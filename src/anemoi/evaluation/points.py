"""PointSet, the observation points of one valid time; names follow anemoi-core's tabular payload keys."""

from __future__ import annotations

import hashlib
from collections.abc import Hashable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import numpy as np


def _readonly(array: np.ndarray) -> np.ndarray:
    array = np.array(array, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class PointSet:
    """P observation points: coordinates in degrees, float64 (P,); `timedeltas` in float32 seconds (P,), signed,
    relative to the valid time (zeros by default); named per-point `attributes` (P,) such as an elevation or an
    observation type. The driver refuses non-zero timedeltas for now: scoring a time offset needs a time stencil, which
    the driver does not run yet.

    `key` identifies the set: a plan prepared for one point set is reused for any other with the same key. It
    defaults to a digest of the content, so a fixed station network gets one key without saying so; a source that
    knows its set is fixed may pass a cheaper explicit key, and then vouches that sets with that key have the same
    coordinates, timedeltas and attributes (the driver checks it on reuse and raises otherwise). Arrays are copied and
    read-only, so the key stays true.
    """

    latitudes: np.ndarray
    longitudes: np.ndarray
    timedeltas: np.ndarray | None = None
    attributes: Mapping[str, np.ndarray] = field(default_factory=dict)
    key: Hashable | None = None

    def __post_init__(self) -> None:
        lats = _readonly(_real(self.latitudes, "latitudes").astype(np.float64))
        lons = _readonly(_real(self.longitudes, "longitudes").astype(np.float64))
        if lats.ndim != 1 or lats.shape != lons.shape:
            raise ValueError(f"latitudes {lats.shape} and longitudes {lons.shape} must be one-dimensional, same length")
        if not (np.isfinite(lats).all() and np.isfinite(lons).all()):
            raise ValueError("point coordinates must be finite")
        if np.abs(lats).max(initial=0.0) > 90.0:
            raise ValueError("latitudes must lie in [-90, 90] degrees")
        n = lats.shape[0]
        if self.timedeltas is None:
            timedeltas = np.zeros(n, dtype=np.float32)
        else:
            timedeltas = _real(self.timedeltas, "timedeltas").astype(np.float32)
            if timedeltas.shape != (n,) or not np.isfinite(timedeltas).all():
                raise ValueError(f"timedeltas must be finite with shape {(n,)}, got {timedeltas.shape}")
        attributes = {}
        for name, values in self.attributes.items():
            values = np.asarray(values)
            if values.shape != (n,):
                raise ValueError(f"point attribute {name!r} has shape {values.shape}, expected {(n,)}")
            attributes[str(name)] = _readonly(values)
        if self.key is not None:
            hash(self.key)
        object.__setattr__(self, "latitudes", lats)
        object.__setattr__(self, "longitudes", lons)
        object.__setattr__(self, "timedeltas", _readonly(timedeltas))
        object.__setattr__(self, "attributes", attributes)

    def __len__(self) -> int:
        return self.latitudes.shape[0]

    @property
    def n(self) -> int:
        """Number of points."""
        return len(self)

    @property
    def identity(self) -> Hashable:
        """The key, or a digest of coordinates, timedeltas and attributes (computed once) when none was given."""
        if self.key is not None:
            return self.key
        if "_digest" not in self.__dict__:
            digest = hashlib.sha256()
            for array in (self.latitudes, self.longitudes, self.timedeltas):
                digest.update(array.tobytes())
            for name in sorted(self.attributes):
                values = self.attributes[name]
                digest.update(f"\0{name}\0{values.dtype}\0".encode())
                digest.update((values.astype(str) if values.dtype == object else values).tobytes())
            object.__setattr__(self, "_digest", ("sha256", digest.hexdigest()))
        return self.__dict__["_digest"]

    def attribute(self, name: str) -> np.ndarray:
        """One per-point attribute; raises naming the ones present."""
        try:
            return self.attributes[name]
        except KeyError:
            raise KeyError(f"the point set has no attribute {name!r}, only {sorted(self.attributes)}") from None


def _real(values: Any, what: str) -> np.ndarray:
    array = np.asarray(values)
    if not (np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer)):
        raise ValueError(f"{what} must be real numbers, got {array.dtype}")
    return array
