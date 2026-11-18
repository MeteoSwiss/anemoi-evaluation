"""Linear interpolation between two targets: the baseline of a temporal downscaler."""

from __future__ import annotations

import datetime
import logging
from collections.abc import Iterator

import torch
from anemoi.utils.dates import frequency_to_string

from anemoi.evaluation.frame import Frame
from anemoi.evaluation.sources.base import MissingTargetError
from anemoi.evaluation.sources.base import lead_time_steps
from anemoi.evaluation.sources.persistence import TargetBaseline

LOG = logging.getLogger(__name__)


class LinearInterpolationForecastSource(TargetBaseline):
    """Interpolates the targets linearly in time between the bounds of a window: an init time is the left bound `t`,
    the run's lead time the window `T`, and the frame at lead `k * timestep` (`0 < k * timestep < T`) is
    `(1 - a) * x(t) + a * x(t + T)` with `a = k * timestep / T`, computed in float64 and rounded to float32.

    The bounds themselves are not yielded: they are the inputs, scored zero."""

    key = "linear_interpolation"
    temporal_downscaling = True

    def lead_times(self, lead_time: datetime.timedelta) -> list[datetime.timedelta]:
        """Lead times k * timestep strictly inside the window `lead_time`."""
        steps = lead_time_steps(lead_time, self.timestep)[:-1]
        if not steps:
            raise ValueError(
                f"the window {frequency_to_string(lead_time)} has no time strictly inside it at the timestep "
                f"{frequency_to_string(self.timestep)}"
            )
        return steps

    def model_calls(self, lead_time: datetime.timedelta) -> tuple[int, datetime.timedelta]:
        """One pass per lead; the bounds read cover the whole window `lead_time`."""
        return len(self.lead_times(lead_time)), lead_time

    def frames(
        self,
        init_time: datetime.datetime,
        lead_time: datetime.timedelta,
        variables: list[str],
        device: torch.device,
    ) -> Iterator[Frame]:
        """One frame per lead inside the window; a window without a target at either bound yields nothing."""
        steps = self.lead_times(lead_time)
        try:
            left = self.targets.frame(init_time, variables, device).double()
            right = self.targets.frame(init_time + lead_time, variables, device).double()
        except MissingTargetError as error:
            LOG.warning("%s: no linear interpolation from %s", error, init_time)
            return
        for lead in steps:
            a = lead / lead_time
            data = ((1.0 - a) * left + a * right).float()
            yield Frame(init_time, lead, init_time + lead, list(variables), data.repeat(self.members, 1, 1))

    def check_init_times(self, init_times: list[datetime.datetime], lead_time: datetime.timedelta) -> None:
        """Warn about init times whose window has a bound known to be unavailable; they will be skipped."""
        absent = [
            init
            for init in init_times
            if not (self.targets.available(init) and self.targets.available(init + lead_time))
        ]
        if absent:
            LOG.warning(
                "%d init times have no target at a bound of their window and will be skipped, e.g. %s",
                len(absent),
                absent[0],
            )
