"""Weighted, masked, binned sums of statistics over nodes and init times."""

from __future__ import annotations

import datetime
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field

import numpy as np
import torch

from anemoi.evaluation.binning import Binning
from anemoi.evaluation.frame import Frame
from anemoi.evaluation.statistics import Statistic
from anemoi.evaluation.statistics import anomalies
from anemoi.evaluation.statistics import ensemble_mean


@dataclass
class AggregationState:
    """Sums over (lead_time, bin, variable, region) of `members`-member frames; additive, so states merge exactly.

    `variable_coords` are auxiliary per-variable labels (`param`, `level`) written next to the variable axis;
    `init_times` are the init times whose frames were added, so that shards can refuse to merge twice.
    """

    lead_times: list[datetime.timedelta]
    bins: list[str]
    variables: list[str]
    regions: list[str]
    sums: dict[str, torch.Tensor]
    weights: torch.Tensor
    n_init: torch.Tensor
    members: int
    variable_coords: dict[str, list] = field(default_factory=dict)
    init_times: list[datetime.datetime] = field(default_factory=list)
    attrs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._lead_index = {lead: i for i, lead in enumerate(self.lead_times)}
        shape = (len(self.lead_times), len(self.bins), len(self.variables), len(self.regions))
        for name, tensor in [("weights", self.weights), *self.sums.items()]:
            if tuple(tensor.shape) != shape or tensor.dtype != torch.float64:
                raise ValueError(f"{name} must be float64 {shape}, got {tensor.dtype} {tuple(tensor.shape)}")
        if tuple(self.n_init.shape) != shape[:2] or self.n_init.dtype != torch.int64:
            raise ValueError(f"n_init must be int64 {shape[:2]}, got {self.n_init.dtype} {tuple(self.n_init.shape)}")
        if self.members < 1:
            raise ValueError(f"members must be at least 1, got {self.members}")
        for name, values in self.variable_coords.items():
            if len(values) != len(self.variables):
                raise ValueError(
                    f"variable coordinate {name!r} has {len(values)} values for {len(self.variables)} variables"
                )

    @classmethod
    def zeros(
        cls,
        lead_times: list[datetime.timedelta],
        bins: list[str],
        variables: list[str],
        regions: list[str],
        statistics: Iterable[str],
        members: int,
        device: torch.device | str | None = None,
        variable_coords: dict[str, list] | None = None,
    ) -> AggregationState:
        """Empty state for the given coordinates, statistic names and ensemble size."""
        shape = (len(lead_times), len(bins), len(variables), len(regions))
        return cls(
            list(lead_times),
            list(bins),
            list(variables),
            list(regions),
            {name: torch.zeros(shape, dtype=torch.float64, device=device) for name in statistics},
            torch.zeros(shape, dtype=torch.float64, device=device),
            torch.zeros(shape[:2], dtype=torch.int64, device=device),
            members,
            dict(variable_coords or {}),
        )

    @property
    def coords(self) -> tuple:
        """Coordinates as hashable tuples, for merge compatibility checks."""
        return (tuple(self.lead_times), tuple(self.bins), tuple(self.variables), tuple(self.regions))

    def lead_index(self, lead_time: datetime.timedelta) -> int:
        """Index of `lead_time` along the first axis."""
        try:
            return self._lead_index[lead_time]
        except KeyError:
            raise ValueError(f"lead time {lead_time} is not one of {self.lead_times}") from None

    def means(self) -> dict[str, torch.Tensor]:
        """Weighted means, NaN where nothing was accumulated."""
        return {name: total / self.weights for name, total in self.sums.items()}

    def merge(self, other: AggregationState) -> AggregationState:
        """Elementwise sum of two states with identical coordinates, statistics, ensemble size and disjoint init times."""
        if self.coords != other.coords or self.sums.keys() != other.sums.keys():
            raise ValueError("cannot merge states with different coordinates or statistics")
        if self.members != other.members:
            raise ValueError(f"cannot merge states with {self.members} and {other.members} members")
        shared = sorted(set(self.init_times) & set(other.init_times))
        if shared:
            raise ValueError(f"cannot merge states that share init times, e.g. {shared[0]} ({len(shared)} shared)")
        return AggregationState(
            list(self.lead_times),
            list(self.bins),
            list(self.variables),
            list(self.regions),
            {name: total + other.sums[name].to(total.device) for name, total in self.sums.items()},
            self.weights + other.weights.to(self.weights.device),
            self.n_init + other.n_init.to(self.n_init.device),
            self.members,
            dict(self.variable_coords),
            sorted(set(self.init_times) | set(other.init_times)),
            dict(self.attrs),
        )

    def to(self, device: torch.device | str) -> AggregationState:
        """Copy of the state on `device`."""
        return AggregationState(
            list(self.lead_times),
            list(self.bins),
            list(self.variables),
            list(self.regions),
            {name: total.to(device) for name, total in self.sums.items()},
            self.weights.to(device),
            self.n_init.to(device),
            self.members,
            dict(self.variable_coords),
            list(self.init_times),
            dict(self.attrs),
        )

    def cpu(self) -> AggregationState:
        """Copy of the state on the CPU."""
        return self.to("cpu")

    def to_xarray(self, metrics: list, attrs: dict | None = None):
        """Metrics and raw state as an xarray Dataset (see `anemoi.evaluation.output`)."""
        from anemoi.evaluation.output import to_xarray

        return to_xarray(self, metrics, attrs)


class Aggregator:
    """Accumulates node-weighted, region-masked statistics of frames into an AggregationState."""

    def __init__(
        self,
        weights: np.ndarray,
        masks: np.ndarray,
        statistics: dict[str, Statistic],
        binning: Binning,
        device: torch.device | str | None = None,
    ) -> None:
        weights = np.asarray(weights, dtype=np.float64)
        masks = np.asarray(masks, dtype=bool)
        if weights.ndim != 1 or masks.ndim != 2 or masks.shape[0] != weights.shape[0]:
            raise ValueError(f"weights must be (N,) and masks (N, R), got {weights.shape} and {masks.shape}")
        self.W = torch.from_numpy(weights[:, None] * masks).to(device)
        self.statistics = dict(statistics)
        self.binning = binning
        self.device = device

    @property
    def n_regions(self) -> int:
        """Number of region masks."""
        return self.W.shape[1]

    def new_state(
        self,
        lead_times: list[datetime.timedelta],
        variables: list[str],
        regions: list[str],
        members: int,
        variable_coords: dict[str, list] | None = None,
    ) -> AggregationState:
        """Empty state matching this aggregator's bins and statistics, for `members`-member frames."""
        if len(regions) != self.n_regions:
            raise ValueError(f"{len(regions)} region names for {self.n_regions} masks")
        state = AggregationState.zeros(
            lead_times, self.binning.coords, variables, regions, self.statistics, members, self.device, variable_coords
        )
        state.attrs.update(bin_kind=self.binning.kind, bin_by=self.binning.by)
        return state

    def add(
        self, state: AggregationState, frame: Frame, target: torch.Tensor, aux: dict[str, torch.Tensor] | None = None
    ) -> None:
        """Accumulate one frame; a node is excluded from every statistic where any member or the target is non-finite.

        `aux` holds per-frame fields handed to the statistics; the ensemble mean is added here and, when `aux` carries a
        (V, N) `climatology`, the float64 anomalies too. The climatology never touches the shared weights: the anomaly
        statistics zero themselves where it is non-finite. A statistic whose `aux` names a missing field raises.
        """
        pred = frame.data
        if list(frame.variables) != state.variables:
            raise ValueError(f"frame variables {frame.variables} do not match the state's {state.variables}")
        if frame.members != state.members:
            raise ValueError(f"frame has {frame.members} members, the state expects {state.members}")
        if tuple(target.shape) != (1, *pred.shape[1:]):
            raise ValueError(f"target shape {tuple(target.shape)} does not match prediction {tuple(pred.shape)}")
        lead = state.lead_index(frame.lead_time)
        time_bin = self.binning.index(frame)
        aux = dict(aux or {})
        aux.setdefault("ensemble_mean", ensemble_mean(pred))
        climatology = aux.get("climatology")
        if climatology is not None:
            if tuple(climatology.shape) != tuple(pred.shape[1:]):
                raise ValueError(f"climatology shape {tuple(climatology.shape)} does not match {tuple(pred.shape[1:])}")
            forecast, truth, finite = anomalies(aux["ensemble_mean"], target[0], climatology)
            aux.update(forecast_anomaly=forecast, target_anomaly=truth, climatology_finite=finite)
        valid = torch.isfinite(pred).all(0) & torch.isfinite(target[0])
        state.weights[lead, time_bin] += valid.double() @ self.W
        for name, statistic in self.statistics.items():
            missing = [key for key in statistic.aux if key not in aux]
            if missing:
                raise ValueError(f"statistic {name} needs {missing} in aux")
            values = torch.where(valid, statistic.compute(pred, target, aux), 0.0)
            state.sums[name][lead, time_bin] += values.double() @ self.W
        state.n_init[lead, time_bin] += 1
        if frame.init_time not in state.init_times:
            state.init_times.append(frame.init_time)
