"""Variable helpers shared by the sources and the observation operators: typed-variable lookup and the linear
conversion `value * scale + offset` of a source variable."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any

import numpy as np


def variable_info(variable: object | None) -> dict:
    """`param` and `level` of an anemoi typed variable (anemoi-transform `Variable`); None when unknown."""
    if variable is None:
        return {"param": None, "level": None}
    return {"param": getattr(variable, "param", None), "level": getattr(variable, "level", None)}


def linear_conversion(spec: object, keys: tuple[str, ...], where: str) -> tuple[float, float]:
    """`scale` (default 1) and `offset` (default 0) of the conversion `value * scale + offset` in the mapping `spec`,
    checked to be finite numbers; raises on a key other than `keys`, `scale` and `offset`, naming `where`."""
    allowed = (*keys, "scale", "offset")
    if not isinstance(spec, Mapping):
        raise ValueError(f"{where}: expected a mapping of {', '.join(allowed)}, got {spec!r}")
    unknown = set(spec) - set(allowed)
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}, use {', '.join(allowed)}")
    values = []
    for what, default in (("scale", 1.0), ("offset", 0.0)):
        value = spec.get(what, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{where}: {what} must be a finite number, got {value!r}")
        values.append(float(value))
    return values[0], values[1]


@dataclass(frozen=True)
class Conversion:
    """A variable of a source as another quantity: `values = raw[variable] * scale + offset` in float64, declared in
    `units` (None when not stated; nothing checks them, the sources carry none)."""

    variable: str
    scale: float = 1.0
    offset: float = 0.0
    units: str | None = None

    def apply(self, values: Any) -> np.ndarray:
        """The converted values, float64."""
        return np.asarray(values, dtype=np.float64) * self.scale + self.offset

    def describe(self) -> dict:
        """`variable`, `scale` and `offset`, and `units` when stated."""
        facts = asdict(self)
        if self.units is None:
            facts.pop("units")
        return facts
