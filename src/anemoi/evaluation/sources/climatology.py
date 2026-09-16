"""Climatologies for the anomaly statistics: one (V, N) field per key (hour of day, month, day of year or constant)."""

from __future__ import annotations

import datetime
import json
import logging
from collections.abc import Callable
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from anemoi.utils.humanize import compress_dates

from anemoi.evaluation.frame import Grid
from anemoi.evaluation.sources.base import ClimatologySourceBase
from anemoi.evaluation.sources.base import MissingTargetError
from anemoi.evaluation.sources.base import select_variables

LOG = logging.getLogger(__name__)
KEYS: dict[str, Callable[[datetime.datetime], int]] = {
    "hour_of_day": lambda date: date.hour,
    "month": lambda date: date.month,
    "day_of_year": lambda date: date.timetuple().tm_yday,
    "constant": lambda date: 0,
}


class ArrayClimatology(ClimatologySourceBase):
    """In-memory climatology: `fields` maps a key of kind `key` to a float32 (V, N) array on `grid`.

    `frame()` hands out a device tensor per (key, variables) that is cached and shared: read it, do not write to it.
    Built from a target source with `from_targets`, stored and loaded with `to_netcdf` / `from_netcdf`.
    """

    def __init__(
        self,
        grid: Grid,
        variables: list[str],
        fields: dict[int, np.ndarray],
        key: str = "hour_of_day",
        counts: dict[int, int] | None = None,
        attrs: dict | None = None,
        path: str | None = None,
    ) -> None:
        if key not in KEYS:
            raise ValueError(f"climatology key must be one of {sorted(KEYS)}, got {key!r}")
        self.grid = grid
        self.variables = list(variables)
        self.key = key
        self.fields = {}
        for index, values in fields.items():
            values = np.ascontiguousarray(values, dtype=np.float32)
            if values.shape != (len(self.variables), grid.n):
                raise ValueError(
                    f"climatology field {index} has shape {values.shape}, expected {(len(variables), grid.n)}"
                )
            self.fields[int(index)] = values
        self.counts = {int(index): int(count) for index, count in (counts or {}).items()}
        self.attrs = dict(attrs or {})
        self.path = path
        self._device_cache: dict[tuple, torch.Tensor] = {}

    def key_of(self, valid_time: datetime.datetime) -> int:
        """The key (hour, month, day of year or 0) a valid time maps to."""
        return KEYS[self.key](valid_time)

    def frame(self, valid_time: datetime.datetime, variables: list[str], device: torch.device) -> torch.Tensor:
        """Climatology for `valid_time` as float32 (V, N) on `device`; raises when the key is missing."""
        index = self.key_of(valid_time)
        if index not in self.fields:
            raise ValueError(
                f"climatology has no {self.key} {index} (valid time {valid_time}); keys {sorted(self.fields)}"
            )
        cache_key = (index, tuple(variables), str(device))
        tensor = self._device_cache.get(cache_key)
        if tensor is None:
            columns = select_variables(list(variables), self.variables)
            tensor = torch.from_numpy(np.ascontiguousarray(self.fields[index][columns])).to(device)
            self._device_cache[cache_key] = tensor
        return tensor

    def nonfinite_nodes(self, variables: list[str]) -> dict[str, int]:
        """Per variable, the number of nodes that are non-finite in at least one key."""
        columns = select_variables(list(variables), self.variables)
        nonfinite = np.zeros((len(columns), self.grid.n), dtype=bool)
        for values in self.fields.values():
            nonfinite |= ~np.isfinite(values[columns])
        return {name: int(count) for name, count in zip(variables, nonfinite.sum(axis=1))}

    @classmethod
    def from_targets(
        cls,
        targets: object,
        dates: Iterable[datetime.datetime],
        variables: list[str],
        key: str = "hour_of_day",
        device: torch.device | str = "cpu",
    ) -> ArrayClimatology:
        """Mean of the target fields at `dates` per key, accumulated in float64; missing dates are skipped."""
        dates, variables, device = list(dates), list(variables), torch.device(device)
        targets.prefetch(dates, variables)
        sums: dict[int, torch.Tensor] = {}
        counts: dict[int, int] = {}
        for date in dates:
            try:
                values = targets.frame(date, variables, device)[0].double()
            except MissingTargetError as error:
                LOG.warning("%s: skipped in the climatology", error)
                continue
            index = KEYS[key](date)
            sums[index] = values if index not in sums else sums[index] + values
            counts[index] = counts.get(index, 0) + 1
        fields = {index: (total / counts[index]).float().cpu().numpy() for index, total in sums.items()}
        attrs = {"dates": ", ".join(compress_dates(dates)), "source": json.dumps(targets.describe(), default=str)}
        return cls(targets.grid, variables, fields, key, counts, attrs)

    def to_netcdf(self, path: str | Path) -> None:
        """Write `climatology(key, variable, values)` float32 with the grid, the key kind, the counts and the attrs."""
        import xarray as xr

        keys = sorted(self.fields)
        dataset = xr.Dataset(
            {"climatology": (("key", "variable", "values"), np.stack([self.fields[index] for index in keys]))},
            coords={
                "key": keys,
                "variable": list(self.variables),
                "latitude": ("values", self.grid.latitudes),
                "longitude": ("values", self.grid.longitudes),
            },
            attrs={
                "key_kind": self.key,
                "counts": json.dumps({str(index): count for index, count in sorted(self.counts.items())}),
                **{name: str(value) for name, value in self.attrs.items()},
            },
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        dataset.to_netcdf(path, engine="netcdf4")
        self.path = str(path)

    @classmethod
    def from_netcdf(cls, path: str | Path) -> ArrayClimatology:
        """Read a file written by `to_netcdf`."""
        import xarray as xr

        with xr.open_dataset(path, engine="netcdf4") as dataset:
            dataset = dataset.load()
        values = dataset["climatology"].values
        fields = {int(index): values[i] for i, index in enumerate(dataset["key"].values)}
        counts = {int(index): count for index, count in json.loads(dataset.attrs.get("counts", "{}")).items()}
        attrs = {name: value for name, value in dataset.attrs.items() if name not in ("key_kind", "counts")}
        grid = Grid(dataset["latitude"].values, dataset["longitude"].values)
        return cls(
            grid, dataset["variable"].values.tolist(), fields, dataset.attrs["key_kind"], counts, attrs, str(path)
        )

    def describe(self) -> dict:
        """Key kind and keys, grid and variable counts, bytes per (key, variables) set, the file and the build facts."""
        info = {
            "source": "climatology",
            "key": self.key,
            "keys": sorted(self.fields),
            "nodes": self.grid.n,
            "variables": len(self.variables),
            "bytes": len(self.fields) * len(self.variables) * self.grid.n * 4,
            "counts": dict(sorted(self.counts.items())),
        }
        if self.path is not None:
            info["path"] = self.path
        info.update({name: value for name, value in self.attrs.items() if name != "source"})
        return info

    def to_config(self) -> dict:
        """The `climatology:` block that rebuilds this source; raises when it was not loaded from a file."""
        if self.path is None:
            raise ValueError("an in-memory climatology cannot be written to a config, save it with to_netcdf first")
        return {"file": self.path}

    def close(self) -> None:
        """Drop the device copies."""
        self._device_cache.clear()
