"""Point targets from a gridded station dataset: an anemoi dataset whose grid is a fixed set of stations."""

from __future__ import annotations

import csv
import datetime
import hashlib
import io
import math
import os
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from anemoi.evaluation.points import PointSet
from anemoi.evaluation.sources.base import MissingTargetError
from anemoi.evaluation.sources.base import Observations
from anemoi.evaluation.sources.base import PointTargetSourceBase
from anemoi.evaluation.sources.base import select_variables
from anemoi.evaluation.variables import Conversion
from anemoi.evaluation.variables import linear_conversion


def _conversions(variables: Mapping[str, Mapping[str, Any] | str | None]) -> dict[str, Conversion]:
    """Result name -> conversion of a dataset variable (`value = raw * scale + offset`, in float64, then float32),
    from dataset name -> `{name, scale, offset}` (or the result name alone)."""
    conversions: dict[str, Conversion] = {}
    for raw, spec in variables.items():
        spec = {} if spec is None else {"name": spec} if isinstance(spec, str) else spec
        scale, offset = linear_conversion(spec, ("name",), f"variable {raw!r}")
        name = str(spec.get("name", raw))
        if name in conversions:
            raise ValueError(f"the result name {name!r} is given to {conversions[name].variable!r} and {raw!r}")
        conversions[name] = Conversion(str(raw), scale, offset)
    if not conversions:
        raise ValueError("no variables to read from the station dataset")
    return conversions


def read_station_file(
    path: str | os.PathLike,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    attributes: Mapping[str, str],
    tolerance: float,
    delimiter: str = ",",
) -> tuple[dict[str, np.ndarray], str, float]:
    """The per-station attributes (P,) float64 of a CSV with one row per station in dataset order, its sha256 and the
    largest coordinate difference in degrees.

    The file is `delimiter`-separated (comma by default; quoted fields as the `csv` module reads them), UTF-8, with a
    header row. The `latitude` and `longitude` columns must equal the dataset's coordinates within `tolerance` degrees
    (longitudes compared modulo 360; an empty coordinate is refused); `attributes` maps an attribute name to its
    column; other columns are ignored. An empty attribute cell is NaN (unknown), any other value must be a number.
    Blank lines are skipped; errors name the line of the file."""
    with open(path, "rb") as file:
        content = file.read()
    sha256 = hashlib.sha256(content).hexdigest()
    reader = csv.reader(io.StringIO(content.decode("utf-8-sig"), newline=""), delimiter=delimiter)
    rows, lines = [], []
    for row in reader:
        if any(cell.strip() for cell in row):
            rows.append(row)
            lines.append(reader.line_num)
    if not rows:
        raise ValueError(f"station file {path} is empty")
    header, rows, lines = [column.strip() for column in rows[0]], rows[1:], lines[1:]
    columns = ["latitude", "longitude", *attributes.values()]
    missing = [column for column in dict.fromkeys(columns) if column not in header]
    if missing:
        hint = f" (one column: not {delimiter!r}-separated? pass delimiter=)" if len(header) == 1 else ""
        raise ValueError(f"station file {path} has no columns {missing}, it has {header}{hint}")
    if len(rows) != len(latitudes):
        raise ValueError(f"station file {path} has {len(rows)} stations, the dataset {len(latitudes)} grid points")
    index = {column: header.index(column) for column in columns}

    def column(name: str) -> np.ndarray:
        values = np.empty(len(rows))
        for i, row in enumerate(rows):
            cell = row[index[name]].strip() if index[name] < len(row) else ""
            try:
                values[i] = float(cell) if cell else np.nan
            except ValueError:
                raise ValueError(
                    f"station file {path}, line {lines[i]}: column {name!r} must be a number, got {cell!r}"
                ) from None
        return values

    lats, lons = column("latitude"), column("longitude")
    dlat = np.abs(lats - np.asarray(latitudes, dtype=np.float64))
    dlon = np.abs((lons - np.asarray(longitudes, dtype=np.float64) + 180.0) % 360.0 - 180.0)
    offset = np.maximum(dlat, dlon)  # NaN, so refused, where a coordinate is empty
    bad = np.flatnonzero(~(offset <= tolerance))
    if len(bad):
        i = bad[0]
        raise ValueError(
            f"station file {path}: {len(bad)} stations differ from the dataset's coordinates by more than {tolerance} "
            f"degrees or have none, e.g. line {lines[i]} ({lats[i]}, {lons[i]}) vs ({latitudes[i]}, {longitudes[i]}); "
            "the rows must be the dataset's stations in dataset order (pass tolerance= for coordinates written with "
            "fewer decimals)"
        )
    return {name: column(source) for name, source in attributes.items()}, sha256, float(offset.max(initial=0.0))


class StationDatasetTargets(PointTargetSourceBase):
    """Observations from a gridded station dataset: an anemoi dataset whose grid is a fixed set of stations, NaN
    where a station did not report (e.g. SwissMetNet stations under their DWH names).

    `dataset` is the `DatasetTargets` that reads it (one read per valid time, its prefetch and cache); its grid is
    handed out as a fixed `PointSet` (timedeltas zero). `variables` maps a dataset variable to its result name and
    unit conversion, `{"tre200s0": {"name": "2t", "offset": 273.15}}` (name defaults to the dataset name, scale 1,
    offset 0): the values are `raw * scale + offset` in float64, then float32, and the targets offer the result names,
    so the scores are in the model's units under the model's names. Nothing checks the units: the dataset carries
    none, so the conversion is the caller's statement, recorded in `describe()`.

    `stations` is an optional CSV (`delimiter`-separated, comma by default) of one row per station in dataset order,
    with `latitude` and `longitude` columns checked against the dataset's within `tolerance` degrees; `attributes`
    maps a point attribute to a numeric column (`{"elevation": "elevation"}` for `LapseRate`), an empty cell being
    unknown (NaN). Its sha256 is in `merge_identity()`, so results made with another file do not merge.

    A valid time at which every requested value is NaN raises `MissingTargetError` once read (it is skipped, not
    scored with zero weight); `available()` cannot know that before reading and stays the dataset's. `variable_info`
    knows nothing of the result names: the evaluation takes their `param` and `level` from the forecast."""

    def __init__(
        self,
        dataset: Any,
        variables: Mapping[str, Mapping[str, Any] | str | None],
        stations: str | os.PathLike | None = None,
        attributes: Mapping[str, str] | None = None,
        tolerance: float = 1e-6,
        delimiter: str = ",",
    ) -> None:
        attributes = dict(attributes or {})
        if attributes and stations is None:
            raise ValueError(f"point attributes {sorted(attributes)} need a station file")
        if not (tolerance >= 0 and math.isfinite(tolerance)):
            raise ValueError(f"tolerance must be a non-negative number of degrees, got {tolerance!r}")
        self.dataset = dataset
        self.conversions = _conversions(variables)
        select_variables([c.variable for c in self.conversions.values()], list(dataset.variables))
        self.variables = list(self.conversions)
        grid = dataset.grid
        self.stations = None if stations is None else os.fspath(stations)
        self.attribute_columns = attributes
        self.tolerance = float(tolerance)
        values, self.stations_sha256, self.coordinate_offset = {}, None, None
        if stations is not None:
            values, self.stations_sha256, self.coordinate_offset = read_station_file(
                stations, grid.latitudes, grid.longitudes, attributes, self.tolerance, delimiter
            )
        key = hashlib.sha256(grid.latitudes.tobytes() + grid.longitudes.tobytes())
        key.update(f"\0{self.stations_sha256}\0{sorted(attributes.items())}".encode())
        self.fixed_points = PointSet(
            grid.latitudes, grid.longitudes, attributes=values, key=(type(self).__name__, key.hexdigest())
        )
        self.point_attributes = tuple(values)
        self._scale = np.array([c.scale for c in self.conversions.values()])
        self._offset = np.array([c.offset for c in self.conversions.values()])

    def _raw(self, variables: list[str]) -> list[str]:
        select_variables(list(variables), self.variables)
        return [self.conversions[name].variable for name in variables]

    def observations(self, valid_time: datetime.datetime, variables: list[str], device: torch.device) -> Observations:
        """The station set and the converted values (1, V, P) at `valid_time`, one dataset read; MissingTargetError
        when the date is absent or missing, or when every value read is NaN."""
        raw = self.dataset.frame(valid_time, self._raw(variables), torch.device("cpu")).numpy()
        if np.isnan(raw).all():
            raise MissingTargetError(f"every station value of {list(variables)} is NaN at {valid_time}")
        index = [self.variables.index(name) for name in variables]
        values = raw.astype(np.float64) * self._scale[index][None, :, None] + self._offset[index][None, :, None]
        return Observations(self.fixed_points, torch.from_numpy(values.astype(np.float32)).to(device))

    def available(self, valid_time: datetime.datetime) -> bool:
        """The dataset's: an all-NaN valid time is only found out by reading it."""
        return self.dataset.available(valid_time)

    def prefetch(self, valid_times: list[datetime.datetime], variables: list[str]) -> None:
        """Read ahead in the dataset, under the dataset's names."""
        self.dataset.prefetch(valid_times, self._raw(variables))

    def prefetch_hint(self, frames_per_pass: int) -> None:
        """The dataset's read-ahead."""
        self.dataset.prefetch_hint(frames_per_pass)

    @property
    def stats(self) -> dict:
        """The dataset's read counters."""
        return self.dataset.stats

    def close(self) -> None:
        """Release the dataset."""
        self.dataset.close()

    def merge_identity(self) -> dict:
        """What changes the values: the dataset's open arguments, the conversions, the station count and attributes,
        the station file's sha256 and column map (not its path, the tolerance or the dataset's prefetch and cache)."""
        dataset = self.dataset.describe()
        return {
            "source": type(self).__name__,
            "dataset": {key: dataset[key] for key in ("source", "args", "kwargs") if key in dataset},
            "variables": {name: conversion.describe() for name, conversion in self.conversions.items()},
            "points": len(self.fixed_points),
            "point_attributes": list(self.point_attributes),
            "stations": None
            if self.stations is None
            else {"sha256": self.stations_sha256, "attributes": dict(self.attribute_columns)},
        }

    def describe(self) -> dict:
        """The dataset, the conversions, the station file (path, sha256, columns, coordinate check), for the dry run;
        `merge_identity` is the part that changes the values."""
        dataset = self.dataset.describe()
        facts = {
            **super().describe(),
            "dataset": dataset,
            "variables": {name: conversion.describe() for name, conversion in self.conversions.items()},
            "stations": None
            if self.stations is None
            else {
                "file": self.stations,
                "sha256": self.stations_sha256,
                "attributes": dict(self.attribute_columns),
                "tolerance_deg": self.tolerance,
                "max_coordinate_offset_deg": self.coordinate_offset,
            },
        }
        facts.update({key: dataset[key] for key in ("row_bytes", "prefetch") if key in dataset})
        return facts
