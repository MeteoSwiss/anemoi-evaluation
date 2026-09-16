"""Source protocols and base classes, the BYOD boundary of the framework.

`ForecastSource` and `TargetSource` are the full contracts; `ForecastSourceBase` and `TargetSourceBase` implement
the optional capabilities with defaults (no device preference, no checkpoint graph, no date check, ...), so a new
source subclasses a base and overrides what it can provide.
"""

from __future__ import annotations

import datetime
from collections.abc import Iterator
from typing import Any
from typing import Protocol

import numpy as np
import torch

from anemoi.evaluation.frame import Frame
from anemoi.evaluation.frame import Grid


class MissingTargetError(LookupError):
    """Raised by a target source that has no data for the requested valid time."""


class ForecastSource(Protocol):
    """Yields forecast frames for an init time; one member for deterministic models."""

    grid: Grid
    variables: list[str]
    members: int
    device: torch.device | str | None
    grid_indices: Any
    has_graph: bool
    frames_per_pass: int
    supports_lead_zero: bool

    def lead_times(self, lead_time: datetime.timedelta) -> list[datetime.timedelta]:
        """Lead times this source yields for a forecast up to `lead_time`."""
        ...

    def initial_frame(self, init_time: datetime.datetime, variables: list[str], device: torch.device) -> Frame:
        """The lead-0 frame (the initial state); raises when `supports_lead_zero` is False."""
        ...

    def frames(
        self,
        init_time: datetime.datetime,
        lead_time: datetime.timedelta,
        variables: list[str],
        device: torch.device,
    ) -> Iterator[Frame]:
        """Yield frames with strictly increasing lead times; each `data` is owned by the caller."""
        ...

    def check_init_times(self, init_times: list[datetime.datetime], lead_time: datetime.timedelta) -> None:
        """Raise early when a forecast from one of `init_times` cannot be produced."""
        ...

    def graph_node_attribute(self, name: str, nodes: str = "data") -> np.ndarray:
        """Per-node attribute of the model's graph, e.g. area weights; raises when `has_graph` is False."""
        ...

    def variable_info(self, name: str) -> dict:
        """`param` and `level` of a variable, None when unknown."""
        ...

    def provenance(self) -> dict:
        """Result-file attrs identifying the model, e.g. the checkpoint path."""
        ...

    @property
    def stats(self) -> dict:
        """Counters for the result attrs, e.g. forcings computed and served from a cache; may be empty."""
        ...

    def describe(self) -> dict:
        """Facts about the source for a dry run; must not load a model."""
        ...

    def to_config(self) -> dict:
        """The `forecast:` config block that rebuilds this source; raises when not serialisable."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...


class TargetSource(Protocol):
    """Returns the target field at a valid time as float32 (1, V, N)."""

    grid: Grid
    variables: list[str]

    def frame(self, valid_time: datetime.datetime, variables: list[str], device: torch.device) -> torch.Tensor:
        """Target for `valid_time`; raises MissingTargetError when unavailable."""
        ...

    def available(self, valid_time: datetime.datetime) -> bool:
        """Whether `frame` can deliver `valid_time`, when known without reading; True by default."""
        ...

    def prefetch(self, valid_times: list[datetime.datetime], variables: list[str]) -> None:
        """Announce the valid times the driver will request next, in order, so a source can read ahead."""
        ...

    def prefetch_hint(self, frames_per_pass: int) -> None:
        """Tell the source how many frames the forecast source delivers per model call, to size its read-ahead."""
        ...

    @property
    def stats(self) -> dict:
        """Read counters for the result attrs (`reads`, `read_seconds`, `cache_hits`); empty when none are kept."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...

    def grid_mask(self, i: int) -> np.ndarray:
        """Nodes of sub-grid `i` of a composite grid; raises when the source has no sub-grids."""
        ...

    def variable_info(self, name: str) -> dict:
        """`param` and `level` of a variable, None when unknown."""
        ...

    def to_config(self) -> dict:
        """The `targets:` config block that rebuilds this source; raises when not serialisable."""
        ...

    def describe(self) -> dict:
        """Facts about the source for a dry run."""
        ...


class ClimatologySource(Protocol):
    """Returns the climatology at a valid time as float32 (V, N) on the target grid, for the anomaly statistics."""

    grid: Grid
    variables: list[str]

    def frame(self, valid_time: datetime.datetime, variables: list[str], device: torch.device) -> torch.Tensor:
        """Climatology for `valid_time`; read-only for the caller, raises when it cannot be delivered."""
        ...

    def nonfinite_nodes(self, variables: list[str]) -> dict[str, int]:
        """Per variable, how many nodes are non-finite (they drop out of the anomaly statistics only)."""
        ...

    def describe(self) -> dict:
        """Facts about the source for a dry run and the result attrs."""
        ...

    def to_config(self) -> dict:
        """The `climatology:` config block that rebuilds this source; raises when not serialisable."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...


class ForecastSourceBase:
    """Defaults for the optional capabilities of a forecast source."""

    device: torch.device | str | None = None
    grid_indices: Any = None
    has_graph: bool = False
    frames_per_pass: int = 1
    supports_lead_zero: bool = False

    def initial_frame(self, init_time: datetime.datetime, variables: list[str], device: torch.device) -> Frame:
        """No initial state."""
        raise ValueError(f"{type(self).__name__} cannot produce lead-0 frames")

    def check_init_times(self, init_times: list[datetime.datetime], lead_time: datetime.timedelta) -> None:
        """Nothing to check."""

    def graph_node_attribute(self, name: str, nodes: str = "data") -> np.ndarray:
        """No graph."""
        raise ValueError(f"{type(self).__name__} has no model graph to read the node attribute {name!r} from")

    def variable_info(self, name: str) -> dict:
        """Unknown."""
        return variable_info(None)

    def provenance(self) -> dict:
        """Nothing to record."""
        return {}

    @property
    def stats(self) -> dict:
        """Counters for the result attrs; none by default."""
        return {}

    def describe(self) -> dict:
        """Class, grid size, variable and member counts."""
        return {
            "source": type(self).__name__,
            "nodes": self.grid.n,
            "variables": len(self.variables),
            "members": self.members,
        }

    def to_config(self) -> dict:
        """Not serialisable."""
        raise ValueError(f"{type(self).__name__} cannot be written to a config")

    def close(self) -> None:
        """Nothing to release."""


class TargetSourceBase:
    """Defaults for the optional capabilities of a target source."""

    def available(self, valid_time: datetime.datetime) -> bool:
        """Unknown until read, so assumed available."""
        return True

    def prefetch(self, valid_times: list[datetime.datetime], variables: list[str]) -> None:
        """Nothing to read ahead."""

    def prefetch_hint(self, frames_per_pass: int) -> None:
        """Nothing to size."""

    @property
    def stats(self) -> dict:
        """No counters."""
        return {}

    def close(self) -> None:
        """Nothing to release."""

    def describe(self) -> dict:
        """Class, grid size and variable count."""
        return {"source": type(self).__name__, "nodes": self.grid.n, "variables": len(self.variables)}

    def grid_mask(self, i: int) -> np.ndarray:
        """No sub-grids."""
        raise ValueError(f"{type(self).__name__} has no sub-grids for the 'grid' region {i}")

    def variable_info(self, name: str) -> dict:
        """Unknown."""
        return variable_info(None)

    def to_config(self) -> dict:
        """Not serialisable."""
        raise ValueError(f"{type(self).__name__} cannot be written to a config")


class ClimatologySourceBase:
    """Defaults for the optional capabilities of a climatology source."""

    def nonfinite_nodes(self, variables: list[str]) -> dict[str, int]:
        """Unknown, assumed finite."""
        return {}

    def describe(self) -> dict:
        """Class, grid size and variable count."""
        return {"source": type(self).__name__, "nodes": self.grid.n, "variables": len(self.variables)}

    def to_config(self) -> dict:
        """Not serialisable."""
        raise ValueError(f"{type(self).__name__} cannot be written to a config")

    def close(self) -> None:
        """Nothing to release."""


def lead_time_steps(lead_time: datetime.timedelta, timestep: datetime.timedelta) -> list[datetime.timedelta]:
    """Lead times k * timestep up to `lead_time`; raises unless `lead_time` is a positive multiple."""
    if lead_time <= datetime.timedelta(0) or lead_time % timestep:
        raise ValueError(f"lead time {lead_time} is not a positive multiple of the timestep {timestep}")
    return [k * timestep for k in range(1, lead_time // timestep + 1)]


def variable_info(variable: object | None) -> dict:
    """`param` and `level` of an anemoi typed variable (anemoi-transform `Variable`); None when unknown."""
    if variable is None:
        return {"param": None, "level": None}
    return {"param": getattr(variable, "param", None), "level": getattr(variable, "level", None)}


def select_variables(requested: list[str], available: list[str]) -> list[int]:
    """Indices of `requested` in `available`; raises on names that cannot be delivered."""
    unknown = [v for v in requested if v not in available]
    if unknown:
        raise ValueError(f"source cannot deliver variables {unknown}")
    index = {name: i for i, name in enumerate(available)}
    return [index[v] for v in requested]
