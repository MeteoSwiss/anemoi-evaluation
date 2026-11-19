"""In-memory synthetic sources for tests and notebooks."""

from __future__ import annotations

import datetime
from collections.abc import Iterator

import numpy as np
import torch

from anemoi.evaluation.frame import Frame
from anemoi.evaluation.frame import Grid
from anemoi.evaluation.points import PointSet
from anemoi.evaluation.sources.base import ForecastSourceBase
from anemoi.evaluation.sources.base import MissingTargetError
from anemoi.evaluation.sources.base import Observations
from anemoi.evaluation.sources.base import PointTargetSourceBase
from anemoi.evaluation.sources.base import TargetSourceBase
from anemoi.evaluation.sources.base import lead_time_steps
from anemoi.evaluation.sources.base import select_variables


class ArrayTargets(TargetSourceBase):
    """Targets from a dict of valid time -> float32 (V, N) array."""

    def __init__(self, grid: Grid, variables: list[str], fields: dict[datetime.datetime, np.ndarray]) -> None:
        self.grid = grid
        self.variables = list(variables)
        self.fields = fields
        for date, field in fields.items():
            if field.shape != (len(variables), grid.n):
                raise ValueError(f"field at {date} has shape {field.shape}, expected {(len(variables), grid.n)}")

    def frame(self, valid_time: datetime.datetime, variables: list[str], device: torch.device) -> torch.Tensor:
        """Target for `valid_time` as (1, V, N)."""
        if valid_time not in self.fields:
            raise MissingTargetError(f"no target for {valid_time}")
        idx = select_variables(variables, self.variables)
        return torch.from_numpy(np.ascontiguousarray(self.fields[valid_time][idx], dtype=np.float32)).to(device)[None]


class ArrayPointTargets(PointTargetSourceBase):
    """Observations from a dict of valid time -> (`PointSet`, float32 (V, P) array), the point set varying from one
    valid time to the next; `fixed` builds one whose every valid time has the same set. A valid time absent from the
    dict raises MissingTargetError. `point_attributes` defaults to the attributes every point set carries.
    """

    def __init__(
        self,
        variables: list[str],
        observations: dict[datetime.datetime, tuple[PointSet, np.ndarray]],
        point_attributes: tuple[str, ...] | None = None,
    ) -> None:
        self.variables = list(variables)
        self.data = {}
        for date, (points, values) in observations.items():
            values = np.asarray(values, dtype=np.float32)
            if values.shape != (len(self.variables), len(points)):
                raise ValueError(
                    f"values at {date} have shape {values.shape}, expected {(len(variables), len(points))}"
                )
            self.data[date] = (points, values)
        if point_attributes is None:
            sets = [set(points.attributes) for points, _ in self.data.values()]
            point_attributes = tuple(sorted(set.intersection(*sets))) if sets else ()
        self.point_attributes = tuple(point_attributes)
        self.reads = 0

    @classmethod
    def fixed(
        cls, points: PointSet, variables: list[str], fields: dict[datetime.datetime, np.ndarray]
    ) -> ArrayPointTargets:
        """A fixed point set: the same `points` at every valid time of `fields` (valid time -> (V, P))."""
        source = cls(variables, {date: (points, values) for date, values in fields.items()}, tuple(points.attributes))
        source.fixed_points = points
        return source

    def observations(self, valid_time: datetime.datetime, variables: list[str], device: torch.device) -> Observations:
        """The point set and values (1, V, P) at `valid_time`."""
        if valid_time not in self.data:
            raise MissingTargetError(f"no observations at {valid_time}")
        self.reads += 1
        points, values = self.data[valid_time]
        idx = select_variables(variables, self.variables)
        return Observations(points, torch.from_numpy(np.ascontiguousarray(values[idx])).to(device)[None])


class FakeForecastSource(ForecastSourceBase):
    """Persistence of the target at init time plus `drift` per step and member offsets of size `spread`."""

    supports_lead_zero = True

    def __init__(
        self,
        targets: ArrayTargets,
        timestep: datetime.timedelta,
        members: int = 1,
        drift: float = 0.0,
        spread: float = 0.0,
        frames_per_pass: int = 1,
        device: str | torch.device | None = None,
    ) -> None:
        self.targets = targets
        self.grid = targets.grid
        self.variables = list(targets.variables)
        self.timestep = timestep
        self.members = members
        self.drift = drift
        self.spread = spread
        self.frames_per_pass = frames_per_pass
        self.device = device

    def lead_times(self, lead_time: datetime.timedelta) -> list[datetime.timedelta]:
        """Lead times k * timestep up to `lead_time`."""
        return lead_time_steps(lead_time, self.timestep)

    def initial_frame(self, init_time: datetime.datetime, variables: list[str], device: torch.device) -> Frame:
        """The init-time target for every member."""
        base = self.targets.frame(init_time, variables, device)
        return Frame(init_time, datetime.timedelta(0), init_time, list(variables), base.repeat(self.members, 1, 1))

    def frames(
        self,
        init_time: datetime.datetime,
        lead_time: datetime.timedelta,
        variables: list[str],
        device: torch.device,
    ) -> Iterator[Frame]:
        """Yield the frames of one forecast, computed `frames_per_pass` steps at a time."""
        base = self.targets.frame(init_time, variables, device)
        offsets = self.spread * (
            torch.arange(self.members, dtype=torch.float32, device=device) - (self.members - 1) / 2
        )
        steps = self.lead_times(lead_time)
        for start in range(0, len(steps), self.frames_per_pass):
            chunk = steps[start : start + self.frames_per_pass]
            outputs = [base + (start + j + 1) * self.drift + offsets[:, None, None] for j in range(len(chunk))]
            for step, data in zip(chunk, outputs):
                yield Frame(init_time, step, init_time + step, list(variables), data.float())
