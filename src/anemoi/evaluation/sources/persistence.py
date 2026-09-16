"""Persistence baseline: the target at init time, repeated at every lead."""

from __future__ import annotations

import datetime
import logging
from collections.abc import Iterator

import torch
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta

from anemoi.evaluation.frame import Frame
from anemoi.evaluation.sources.base import ForecastSourceBase
from anemoi.evaluation.sources.base import MissingTargetError
from anemoi.evaluation.sources.base import TargetSource
from anemoi.evaluation.sources.base import lead_time_steps

LOG = logging.getLogger(__name__)


class PersistenceForecastSource(ForecastSourceBase):
    """Persists the target at init time: the baseline a model must beat, and an end-to-end check that needs no model."""

    supports_lead_zero = True

    def __init__(self, targets: TargetSource, timestep: datetime.timedelta | str, members: int = 1) -> None:
        if members < 1:
            raise ValueError(f"members must be at least 1, got {members}")
        self.targets = targets
        self.grid = targets.grid
        self.variables = list(targets.variables)
        self.timestep = frequency_to_timedelta(timestep)
        self.members = members

    def lead_times(self, lead_time: datetime.timedelta) -> list[datetime.timedelta]:
        """Lead times k * timestep up to `lead_time`."""
        return lead_time_steps(lead_time, self.timestep)

    def initial_frame(self, init_time: datetime.datetime, variables: list[str], device: torch.device) -> Frame:
        """The init-time target for every member; MissingTargetError when there is none."""
        base = self.targets.frame(init_time, variables, device)
        return Frame(init_time, datetime.timedelta(0), init_time, list(variables), base.repeat(self.members, 1, 1))

    def frames(
        self,
        init_time: datetime.datetime,
        lead_time: datetime.timedelta,
        variables: list[str],
        device: torch.device,
    ) -> Iterator[Frame]:
        """One frame per lead holding the init-time target; an init time without a target yields nothing."""
        try:
            base = self.targets.frame(init_time, variables, device)
        except MissingTargetError as error:
            LOG.warning("%s: no persistence forecast from %s", error, init_time)
            return
        for lead in self.lead_times(lead_time):
            yield Frame(init_time, lead, init_time + lead, list(variables), base.repeat(self.members, 1, 1))

    def check_init_times(self, init_times: list[datetime.datetime], lead_time: datetime.timedelta) -> None:
        """Warn about init times whose target is known to be unavailable; they will be skipped."""
        absent = [init for init in init_times if not self.targets.available(init)]
        if absent:
            LOG.warning("%d init times have no target to persist and will be skipped, e.g. %s", len(absent), absent[0])

    def variable_info(self, name: str) -> dict:
        """The targets' `param` and `level`."""
        return self.targets.variable_info(name)

    def provenance(self) -> dict:
        """Names the baseline."""
        return {"forecast": "persistence"}

    def describe(self) -> dict:
        """Timestep on top of the base facts."""
        return {**super().describe(), "source": "persistence", "timestep": frequency_to_string(self.timestep)}

    def to_config(self) -> dict:
        """The `forecast:` block that rebuilds this source (with an explicit `targets:` block)."""
        return {"persistence": {"timestep": frequency_to_string(self.timestep), "members": self.members}}
