"""Time binning: the second state axis, one bin per season, month or date of a frame's init or valid time."""

from __future__ import annotations

import datetime
from collections.abc import Iterable
from typing import Protocol

from anemoi.evaluation.frame import Frame

SEASONS = ("DJF", "MAM", "JJA", "SON")
MONTHS = tuple(f"{month:02d}" for month in range(1, 13))
KINDS = ("season", "month", "init_time", "none")
BY = ("init_time", "valid_time")
_SEASON_OF_MONTH = (0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0)


def season_of(date: datetime.datetime) -> int:
    """Index of the meteorological season of `date` in SEASONS."""
    return _SEASON_OF_MONTH[date.month - 1]


class Binning(Protocol):
    """Maps a frame to a bin index; `coords` are the bin labels, `kind` and `by` name the rule."""

    kind: str
    by: str
    coords: list[str]

    def index(self, frame: Frame) -> int:
        """Bin index of `frame`."""
        ...


class _DateBinning:
    kind: str
    coords: list[str]

    def __init__(self, by: str = "init_time") -> None:
        if by not in BY:
            raise ValueError(f"bins can be by {BY}, got {by!r}")
        self.by = by

    def date(self, frame: Frame) -> datetime.datetime:
        """The frame's init or valid time, whichever the binning is by."""
        return getattr(frame, self.by)


class SeasonBinning(_DateBinning):
    """One bin per meteorological season (DJF, MAM, JJA, SON)."""

    kind = "season"
    coords = list(SEASONS)

    def index(self, frame: Frame) -> int:
        return season_of(self.date(frame))


class MonthBinning(_DateBinning):
    """One bin per calendar month, labelled 01 to 12."""

    kind = "month"
    coords = list(MONTHS)

    def index(self, frame: Frame) -> int:
        return self.date(frame).month - 1


class InitTimeBinning(_DateBinning):
    """One bin per date: the init times, or every valid time they reach with `lead_times` when by valid time."""

    kind = "init_time"

    def __init__(
        self,
        init_times: Iterable[datetime.datetime],
        lead_times: Iterable[datetime.timedelta] = (),
        by: str = "init_time",
    ) -> None:
        super().__init__(by)
        init_times = list(init_times)
        dates = init_times if by == "init_time" else [init + lead for init in init_times for lead in lead_times]
        self.dates = sorted(set(dates))
        self._index = {date: i for i, date in enumerate(self.dates)}
        self.coords = [date.isoformat() for date in self.dates]

    def index(self, frame: Frame) -> int:
        date = self.date(frame)
        try:
            return self._index[date]
        except KeyError:
            raise ValueError(f"{self.by} {date} is not one of the binned dates") from None


class NoBinning(_DateBinning):
    """A single bin labelled `all`."""

    kind = "none"
    coords = ["all"]

    def index(self, frame: Frame) -> int:
        return 0


def build(
    kind: str = "season",
    by: str = "init_time",
    init_times: Iterable[datetime.datetime] = (),
    lead_times: Iterable[datetime.timedelta] = (),
) -> Binning:
    """Binning from the config names; `init_times` and `lead_times` are needed for `init_time` bins only."""
    if kind == "season":
        return SeasonBinning(by)
    if kind == "month":
        return MonthBinning(by)
    if kind == "init_time":
        return InitTimeBinning(init_times, lead_times, by)
    if kind == "none":
        return NoBinning(by)
    raise ValueError(f"unknown time binning {kind!r}, expected one of {KINDS}")
