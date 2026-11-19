"""`LapseRate`: a fixed-lapse-rate height correction of a temperature from the model orography to the station."""

from __future__ import annotations

import datetime
import math
from collections.abc import Mapping
from collections.abc import Sequence

import numpy as np
import torch

from anemoi.evaluation.frame import Frame
from anemoi.evaluation.operators.base import POINTS
from anemoi.evaluation.operators.base import Constant
from anemoi.evaluation.operators.base import ConstantNeed
from anemoi.evaluation.operators.base import ModelSide
from anemoi.evaluation.operators.base import ObservationOperatorBase
from anemoi.evaluation.points import PointSet


class LapseRate(ObservationOperatorBase):
    """`variable - rate * (elevation - orography)`, in float64: a station above the model orography is colder.

    `rate` is in K/m, `elevation` names the point attribute holding the station elevation in metres and `orography`
    the model constant in metres, mapped to the points by the spatial step before this one. Touches `variable` only.
    """

    input_space = POINTS
    output_space = POINTS

    def __init__(
        self,
        variable: str = "2t",
        *,
        rate: float = 0.0065,
        elevation: str = "elevation",
        orography: str = "orography",
    ) -> None:
        if not math.isfinite(rate):
            raise ValueError(f"rate must be a finite number of K/m, got {rate!r}")
        self.variable = variable
        self.rate = float(rate)
        self.elevation = elevation
        self.orography = orography
        self.touches = (variable,)
        self.point_attributes = (elevation,)
        self.constants = (ConstantNeed(orography, units="m"),)

    def bind(self, model: ModelSide) -> None:
        """Keep the device."""
        super().bind(model)
        self._device = torch.device(model.device)

    def prepare(self, points: PointSet, constants: Mapping[str, Constant]) -> torch.Tensor:
        """Station elevation minus model orography (P,), float64 on the device; NaN, so a NaN forecast, where the
        elevation is not finite (unknown) or the orography is NaN (points out of the domain)."""
        elevation = points.attribute(self.elevation)
        if not (np.issubdtype(elevation.dtype, np.floating) or np.issubdtype(elevation.dtype, np.integer)):
            raise ValueError(f"point attribute {self.elevation!r} must be numeric metres, got {elevation.dtype}")
        elevation = elevation.astype(np.float64)
        elevation = np.where(np.isfinite(elevation), elevation, np.nan)
        return torch.from_numpy(elevation - constants[self.orography].values).to(self._device)

    def apply(self, frames: Sequence[Frame], plan: torch.Tensor, valid_time: datetime.datetime) -> Frame:
        """Correct `variable` per member and point; the other variables are passed through."""
        frame = frames[0]
        i = frame.variables.index(self.variable)
        data = frame.data.clone()
        data[:, i] = (frame.data[:, i].double() - self.rate * plan.to(frame.data.device)).float()
        return self.output_frame(frames, valid_time, list(frame.variables), data)

    def to_config(self) -> dict:
        """`lapse_rate` with its parameters (provisional, see `ObservationOperator.to_config`)."""
        return {"lapse_rate": self.parameters()}
