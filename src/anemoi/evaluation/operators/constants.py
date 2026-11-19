"""Constants providers that read their fields: `ForecastConstants` asks the forecast source (the default when an
operator needs constants and none are given), `DatasetConstants` reads a variable of an anemoi dataset.

Both map a constant name to a field of the source with a unit conversion, `{variable, scale, offset, units}`: the
constant is `field * scale + offset` in float64, in `units` (the units the operator checks; geopotential in m**2 s**-2
is not a height in metres, hence the default `scale` of 1 / g for `orography`). The units are a declaration: nothing
checks them against the source, which does not carry units.

A constant is read once, on first use (the chain's `bind`); `describe()` then carries a sha256 of the names, the
conversions and the values served, so results made with other constants do not merge.
"""

from __future__ import annotations

import copy
import datetime
from collections.abc import Mapping
from typing import Any

import torch

from anemoi.evaluation.frame import Grid
from anemoi.evaluation.operators.base import Constant
from anemoi.evaluation.operators.base import constants_digest
from anemoi.evaluation.variables import Conversion
from anemoi.evaluation.variables import linear_conversion

G = 9.80665
DEFAULT_CONSTANTS = {"orography": {"variable": "z", "scale": 1 / G, "units": "m"}}
"""The default constants: the model orography in metres from the geopotential `z` (m**2 s**-2)."""


def _spec(name: str, spec: Mapping[str, Any]) -> Conversion:
    """How the constant `name` is made from a field of its source, from a mapping with `variable` (defaults to
    `name`) and optional `scale`, `offset` and `units`."""
    scale, offset = linear_conversion(spec, ("variable", "units"), f"constant {name!r}")
    units = spec.get("units")
    return Conversion(str(spec.get("variable", name)), scale, offset, None if units is None else str(units))


def _constant(name: str, spec: Conversion, values: Any) -> Constant:
    """The constant `name` from the raw field, converted in float64."""
    return Constant(name, spec.apply(values), spec.units)


def _specs(constants: Mapping[str, Mapping[str, Any]]) -> dict[str, Conversion]:
    specs = {str(name): _spec(str(name), spec) for name, spec in constants.items()}
    if not specs:
        raise ValueError("no constants to serve")
    return specs


class _ReadConstants:
    """The shared half: specs, the cache of served constants, the lookup and the digest."""

    def __init__(self, constants: Mapping[str, Mapping[str, Any]]) -> None:
        self.specs = _specs(constants)
        self._served: dict[str, Constant] = {}

    def constant(self, name: str) -> Constant:
        """The constant `name`, read and converted on first use."""
        if name not in self.specs:
            raise KeyError(f"no model constant {name!r}, only {sorted(self.specs)}")
        if name not in self._served:
            self._served.update(self._read([name]))
        return self._served[name]

    def _read(self, names: list[str]) -> dict[str, Constant]:
        raise NotImplementedError

    @property
    def names(self) -> tuple[str, ...]:
        """The constants it serves."""
        return tuple(self.specs)

    def merge_identity(self) -> dict:
        """The conversions of the constants served so far and a sha256 of their names, conversions and values: not
        where they were read from, nor the date (the values say it)."""
        served = {name: self.specs[name].describe() for name in sorted(self._served)}
        return {"constants": served, "sha256": constants_digest(self._served, served)}

    def describe(self) -> dict:
        """Every conversion, the constants served so far and the digest."""
        identity = self.merge_identity()
        return {
            "source": type(self).__name__,
            "constants": {name: spec.describe() for name, spec in self.specs.items()},
            "served": list(identity["constants"]),
            "sha256": identity["sha256"],
        }


class ForecastConstants(_ReadConstants):
    """Constants from the forecast source's `constant_field`, read at `date`: by default the orography in metres
    from the geopotential `z` (`DEFAULT_CONSTANTS`). Without a `date`, an `Evaluation` it is given to reads them at
    its first init time (the whole run's, so every shard reads the same date) through `with_date`, a copy: the
    provider given is not changed.

    An inference checkpoint serves the variables it takes as retrieved constant forcings (`z`, `lsm`, ... when the
    model reads them from its input), from its constant-forcings input without loading the model. A source that
    serves none (persistence, a baseline) is refused at construction: use `DatasetConstants`. `Evaluation` builds the
    default one, at its first init time, when an operator needs constants and none are given."""

    def __init__(
        self,
        forecast: Any,
        constants: Mapping[str, Mapping[str, Any]] | None = None,
        date: datetime.datetime | None = None,
    ) -> None:
        super().__init__(DEFAULT_CONSTANTS if constants is None else constants)
        self.forecast = forecast
        self.date = date
        available = list(getattr(forecast, "constant_variables", None) or [])
        missing = {name: spec.variable for name, spec in self.specs.items() if spec.variable not in available}
        if missing:
            serves = f"it serves {available}" if available else "it serves none"
            raise ValueError(
                f"{type(forecast).__name__} cannot serve the model constants {missing} ({serves}); pass "
                "constants=operators.DatasetConstants(...) to read them from a dataset"
            )

    def with_date(self, date: datetime.datetime) -> ForecastConstants:
        """A copy reading at `date`, with the same forecast and conversions and nothing served yet."""
        other = copy.copy(self)
        other.date, other._served = date, {}
        return other

    def _read(self, names: list[str]) -> dict[str, Constant]:
        if self.date is None:
            raise ValueError(
                "ForecastConstants has no date to read the constants at: pass date=, or give it to an Evaluation"
            )
        return {
            name: _constant(name, self.specs[name], self.forecast.constant_field(self.specs[name].variable, self.date))
            for name in names
        }

    def describe(self) -> dict:
        """The date on top of the conversions and the digest."""
        return {**super().describe(), "date": None if self.date is None else self.date.isoformat()}


class DatasetConstants(_ReadConstants):
    """Constants from variables of an anemoi dataset, read at one `date`, e.g. the orography of a persistence
    baseline, which has none of its own: `DatasetConstants(DatasetTargets(...), {"orography": {"variable": "z",
    "scale": 1 / 9.80665, "units": "m"}})`.

    `dataset` is a gridded target source (`DatasetTargets`: one read of every variable needed, its `grid_indices`);
    `date` defaults to its first available date. Its grid must be the forecast's (checked at bind)."""

    def __init__(
        self,
        dataset: Any,
        constants: Mapping[str, Mapping[str, Any]] | None = None,
        date: datetime.datetime | None = None,
    ) -> None:
        super().__init__(DEFAULT_CONSTANTS if constants is None else constants)
        self.dataset = dataset
        unknown = sorted({spec.variable for spec in self.specs.values()} - set(dataset.variables))
        if unknown:
            raise ValueError(f"the constants' dataset has no variables {unknown}")
        if date is None:
            dates = [d for d in getattr(dataset, "dates", []) if dataset.available(d)]
            if not dates:
                raise ValueError("the constants' dataset has no available date; pass date=")
            date = dates[0]
        elif not dataset.available(date):
            raise ValueError(f"the constants' dataset has no date {date}, or it is missing")
        self.date = date

    def check_grid(self, grid: Grid) -> None:
        """Raise unless the dataset is on the forecast's grid."""
        try:
            grid.check_compatible(self.dataset.grid)
        except ValueError as error:
            raise ValueError(f"the constants' dataset is not on the forecast grid: {error}") from None

    def _read(self, names: list[str]) -> dict[str, Constant]:
        # one read of every variable of every spec, so a second constant needs no second read
        names = [name for name in self.specs if name not in self._served]
        variables = list(dict.fromkeys(self.specs[name].variable for name in names))
        rows = self.dataset.frame(self.date, variables, torch.device("cpu"))[0].numpy()
        return {
            name: _constant(name, self.specs[name], rows[variables.index(self.specs[name].variable)]) for name in names
        }

    def describe(self) -> dict:
        """The dataset's arguments and the date on top of the conversions and the digest."""
        facts = self.dataset.describe()
        dataset = {key: facts[key] for key in ("source", "args", "kwargs") if key in facts}
        return {**super().describe(), "dataset": dataset, "date": self.date.isoformat()}
