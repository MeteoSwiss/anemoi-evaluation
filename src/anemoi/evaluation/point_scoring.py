"""Scoring against point targets: the half of `Evaluation` that maps frames to observation points.

`PointScoring` owns the observation operator (a `Chain`) and its bind, the cache of the operator's plan, `W` and
out-of-domain count per point set, the weights and regions on points and the nearest-node search behind
graph-attribute regions. `Evaluation` builds one for point targets and delegates to it; the gridded path never
reaches this module.
"""

from __future__ import annotations

import copy
import datetime
import json
from collections import OrderedDict
from collections.abc import Hashable
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
import torch
from pydantic import TypeAdapter

from anemoi.evaluation import regions as regions_module
from anemoi.evaluation import weights as weights_module
from anemoi.evaluation.aggregation import weight_matrix
from anemoi.evaluation.config import FileSpec
from anemoi.evaluation.config import GraphAttributeSpec
from anemoi.evaluation.config import GridSpec
from anemoi.evaluation.config import RegionConfig
from anemoi.evaluation.config import UniformSpec
from anemoi.evaluation.config import WeightsConfig
from anemoi.evaluation.config import region_entries
from anemoi.evaluation.frame import Frame
from anemoi.evaluation.frame import Grid
from anemoi.evaluation.operators.base import GRID
from anemoi.evaluation.operators.base import POINTS
from anemoi.evaluation.operators.base import VALID_TIME
from anemoi.evaluation.operators.base import Chain
from anemoi.evaluation.operators.base import ConstantsProvider
from anemoi.evaluation.operators.base import Identity
from anemoi.evaluation.operators.base import ModelSide
from anemoi.evaluation.operators.base import ObservationOperator
from anemoi.evaluation.operators.nearest import NearestNodes
from anemoi.evaluation.output import identity_json
from anemoi.evaluation.points import PointSet
from anemoi.evaluation.sources.base import MissingTargetError
from anemoi.evaluation.sources.base import Observations


@dataclass(frozen=True)
class PointPlan:
    """What is kept per point set: the points, the operator's plan, `W` (P, R) and the number of points out of the
    operator's domain."""

    points: PointSet
    plan: object
    W: torch.Tensor
    out_of_domain: int


def located(points: PointSet) -> Grid:
    """The point coordinates as a `Grid`, for the weight and region specs that only need coordinates."""
    return Grid(points.latitudes, points.longitudes)


def observation_chain(
    operator: ObservationOperator | None, constants: ConstantsProvider | None, on_points: bool
) -> Chain | None:
    """The operator as an unbound `Chain` of a deep copy, after the checks that need no variables; None for the
    gridded path (no operator, or `Identity` only)."""
    identity = operator is None or all(
        isinstance(step, Identity) for step in (operator.steps if isinstance(operator, Chain) else [operator])
    )
    if identity:
        if on_points:
            raise ValueError(
                "point targets need an observation_operator that maps the forecast to the points, e.g. "
                "operators.Nearest(max_distance_km=...)"
            )
        if constants is not None:
            raise ValueError("constants are only read by an observation operator, and this evaluation has none")
        return None
    chain = Chain([copy.deepcopy(operator)])
    if chain.bound:
        raise ValueError(
            "the observation operator was bound to a model before; give an unbound one (copy.deepcopy it before "
            "it is bound, or build a fresh one)"
        )
    if not on_points:
        if chain.output_space == POINTS:
            raise ValueError(
                "the observation operator maps the forecast to points, but the targets are on the grid; give point "
                "targets, or no operator"
            )
        raise ValueError("gridded targets take no observation operator (or an Identity) for now")
    if chain.input_space != GRID:
        raise ValueError(
            f"the observation operator takes {chain.input_space} frames, the forecast source gives grid frames"
        )
    if chain.output_space != POINTS:
        raise ValueError("the observation operator must map the forecast to points for point targets")
    if chain.stencil != VALID_TIME:
        raise ValueError(
            f"the observation operator has the time stencil {chain.stencil!r}, which is not supported yet: only "
            f"{VALID_TIME!r} operators are driven"
        )
    return chain


def check_timedeltas(points: PointSet, where: str) -> None:
    """Refuse non-zero `timedeltas`, any of them, float noise included: scoring a time offset needs a time stencil,
    which the driver does not run yet."""
    offsets = np.abs(points.timedeltas)
    if offsets.any():
        raise ValueError(
            f"{where} has points with time offsets (non-zero timedeltas, up to {float(offsets.max()):g} s), which are "
            "not supported yet: they need a time stencil, which the driver does not run yet; leave timedeltas out, or "
            "zero, to score the points at the valid time"
        )


def resolve_constants(
    chain: Chain, constants: ConstantsProvider | None, forecast_variables: list[str]
) -> ConstantsProvider | None:
    """The constants provider of the run: `constants`, required when the steps that run need constants; refused when
    no step of the operator needs any.
    The steps that run are those on frames holding `forecast_variables`: a step passed through needs nothing."""
    if not chain.constants:
        if constants is not None:
            raise ValueError(
                "constants are only read by an operator step that needs model constants, and this observation "
                "operator has none"
            )
        return None
    _, needed = chain.needs(forecast_variables)
    if not needed:
        return constants
    if constants is None:
        raise ValueError(
            f"the observation operator needs the model constants {[need.name for need in needed]}; pass "
            "constants= with a provider that serves them, e.g. operators.ArrayConstants"
        )
    served = getattr(constants, "names", None)
    missing = [need.name for need in needed if served is not None and need.name not in served]
    if missing:
        raise ValueError(
            f"the observation operator needs the model constants {missing}, the constants provider serves "
            f"{list(served)}; pass constants= with them"
        )
    return constants


def same_points(first: PointSet, second: PointSet) -> bool:
    """Whether two point sets hold the same coordinates, timedeltas and attributes, byte for byte as
    `PointSet.identity` digests them (so NaN and NaT compare equal to themselves)."""
    if first is second:
        return True
    if len(first) != len(second) or set(first.attributes) != set(second.attributes):
        return False
    pairs = [
        (first.latitudes, second.latitudes),
        (first.longitudes, second.longitudes),
        (first.timedeltas, second.timedeltas),
        *((first.attributes[name], second.attributes[name]) for name in first.attributes),
    ]
    return all(_content(a) == _content(b) for a, b in pairs)


def _content(values: np.ndarray) -> tuple[str, bytes]:
    """The dtype and bytes of an array, object arrays as strings, as `PointSet.identity` reads them."""
    return values.dtype.str, (values.astype(str) if values.dtype == object else values).tobytes()


class PointScoring:
    """The point-target side of an evaluation: weights and regions parsed once against the targets, the operator
    bound by `bind()` (idempotent, before the first frame), one plan, `W` and out-of-domain count per point set.

    `diagnostics` is the operator's `diagnostics()` of the fixed point set once it is prepared (`begin`), None for a
    varying set."""

    def __init__(
        self,
        chain: Chain,
        forecast: Any,
        targets: Any,
        variables: list[str],
        forecast_variables: list[str],
        constants: ConstantsProvider | None,
        weights: np.ndarray | dict | None,
        regions: dict | None,
        device: torch.device,
        cache_size: int,
    ) -> None:
        self.chain = chain
        self._unbound = copy.deepcopy(chain)  # the dry run binds copies of this one, never the run's chain
        self.forecast = forecast
        self.targets = targets
        self.variables = list(variables)
        self.forecast_variables = list(forecast_variables)
        self.constants = constants
        self.device = device
        self.cache_size = cache_size
        if targets.fixed_points is not None:
            check_timedeltas(targets.fixed_points, f"the fixed point set of {type(targets).__name__}")
        self._parse(weights, regions)
        self.diagnostics: dict | None = None
        self._identities: dict[str, str] = {}  # encoded at bind, written by attrs()
        self._cache: OrderedDict[Hashable, PointPlan] = OrderedDict()
        self._node_attributes: dict[str, np.ndarray] = {}

    def _parse(self, weights: np.ndarray | dict | None, regions: dict | None) -> None:
        """Parse and check the weights and regions once, at construction: per-model-node specs are refused, files and
        arrays need a fixed point set (arrays are shape-checked against it here, files when read)."""
        fixed = self.targets.fixed_points
        grid = None if fixed is None else located(fixed)

        def per_point(array: np.ndarray, what: str) -> np.ndarray:
            if grid is None:
                raise ValueError(f"{what} given as an array needs a fixed point set, the targets' point set varies")
            return grid.check_shape(array, what)

        if weights is None:
            weights = {"uniform": {}}
        if isinstance(weights, dict):
            spec = TypeAdapter(WeightsConfig).validate_python(weights)
            if not isinstance(spec, (UniformSpec, FileSpec)):
                raise ValueError(
                    f"the weights {weights} are per model node; point targets take uniform weights or a file of one "
                    "weight per point of a fixed point set"
                )
            if isinstance(spec, FileSpec) and fixed is None:
                raise ValueError(f"the weights file {spec.file} needs a fixed point set, the targets' point set varies")
            self.weights_spec, self._weights = weights, spec
        else:
            self.weights_spec = None
            self._weights = per_point(np.asarray(weights, dtype=np.float64), "weights")
        self.region_specs, entries = region_entries(regions, per_point)
        self._regions = {}
        for name, spec in entries.items():
            if isinstance(spec, np.ndarray):
                self._regions[name] = spec
                continue
            parsed = spec if spec == "all" else TypeAdapter(RegionConfig).validate_python(spec)
            if isinstance(parsed, GridSpec):
                raise ValueError(f"region {name!r}: sub-grid regions are per model node, not for point targets")
            if isinstance(parsed, FileSpec) and fixed is None:
                raise ValueError(
                    f"region {name!r}: the file {parsed.file} needs a fixed point set, the targets' point set varies"
                )
            self._regions[name] = parsed

    def model_side(self, device: torch.device | str) -> ModelSide:
        """What the operator is bound to: the forecast grid, the forecast variables, the constants, on `device`."""
        return ModelSide(
            GRID,
            self.forecast_variables,
            grid=self.forecast.grid,
            constants=self.constants,
            variable_info=self.forecast.variable_info,
            device=device,
        )

    def bind(self) -> Chain:
        """The run's chain, bound to the forecast on the run's device on the first call (so a constant of the wrong
        units, shape or grid, or a merge identity `identity_json` cannot encode, fails there); a no-op afterwards.
        When the bind or the identities fail, the chain is replaced by a fresh unbound copy, so the next call fails
        the same way. The identities encoded here are the ones `attrs()` writes."""
        if not self.chain.bound:
            try:
                self.chain.bind(self.model_side(self.device))
                self._identities = self.identities(self.chain)
            except Exception:
                self.chain = copy.deepcopy(self._unbound)
                raise
        return self.chain

    def identities(self, operator: Chain) -> dict[str, str]:
        """The `observation_operator` and `point_targets` attrs: the bound `operator`'s and the targets'
        `merge_identity()` (`describe()` when the targets have none) as `identity_json`, which raises TypeError on a
        value it cannot encode exactly."""
        identity = getattr(self.targets, "merge_identity", self.targets.describe)
        return {
            "observation_operator": identity_json(
                operator.merge_identity(), "the observation operator's merge_identity()"
            ),
            "point_targets": identity_json(identity(), f"{type(self.targets).__name__}.merge_identity()"),
        }

    def fixed(self, what: str) -> PointSet:
        """The fixed point set; raises for a varying one, naming `what` has no fixed value."""
        fixed = self.targets.fixed_points
        if fixed is None:
            raise ValueError(
                f"the targets' point set varies, so there are no fixed {what}; use point_{what}(points) for a point set"
            )
        return fixed

    def weights(self, points: PointSet) -> np.ndarray:
        """Weights (P,) float64 of `points`: uniform by default, or the fixed set's file or array."""
        if isinstance(self._weights, np.ndarray):
            return located(points).check_shape(self._weights, "weights")
        return self._weights.build(located(points)).astype(np.float64)

    def regions(self, points: PointSet) -> dict[str, np.ndarray]:
        """Region masks (P,) bool of `points` by name: `all`, a bbox on the point coordinates, the fixed set's file or
        array, and a graph attribute read at each point's nearest model node (never interpolated)."""
        grid = located(points)
        nodes = None
        masks = {}
        for name, spec in self._regions.items():
            if isinstance(spec, np.ndarray):
                mask = grid.check_shape(spec, f"region {name!r}")
            elif spec == "all":
                mask = regions_module.all(grid)
            elif isinstance(spec, GraphAttributeSpec):
                if nodes is None:
                    nodes, _, _ = self.nearest_nodes.query(points, None)
                mask = self._node_attribute(spec.graph_attribute)[nodes]
            else:
                mask = spec.build(grid, self.forecast, self.targets)
            masks[name] = np.asarray(mask, dtype=bool)
        return masks

    @cached_property
    def nearest_nodes(self) -> NearestNodes:
        """A nearest-node search on the forecast grid for graph-attribute regions, built on first use. It is a second
        KD-tree next to the operator's own (a `Nearest` step), kept apart because the operator may have none."""
        return NearestNodes(self.forecast.grid)

    def _node_attribute(self, name: str) -> np.ndarray:
        """A graph node attribute (N,) of the forecast, read once (a varying point set needs it every valid time)."""
        if name not in self._node_attributes:
            self._node_attributes[name] = weights_module.graph_node_attribute(self.forecast, name, self.forecast.grid)
        return self._node_attributes[name]

    def begin(self) -> None:
        """Start a run: bind the operator (before any model call), clear the cache and prepare the fixed set."""
        self.bind()
        self._cache.clear()
        self.diagnostics = None
        fixed = self.targets.fixed_points
        if fixed is not None and len(fixed):
            self.entry(fixed)

    def entry(self, points: PointSet) -> PointPlan:
        """The operator plan, `W` and out-of-domain count of a point set, from the cache while its identity is known.

        One LRU cache keyed by `PointSet.identity`, cleared at `begin()`. Frames come init time by init time in lead
        order, so a valid time scored at one lead comes back at a shorter lead of the next init time with fewer than
        `len(lead_times)` other valid times in between: `cache_size`, `len(lead_times) + 1`, entries keep every point
        set until that reuse (a fixed set has one entry). A reused key must come with the same coordinates,
        timedeltas and attributes."""
        key = points.identity
        entry = self._cache.get(key)
        if entry is not None:
            if not same_points(points, entry.points):
                raise ValueError(
                    f"point sets with the key {key!r} differ in their points (coordinates, timedeltas or attributes)"
                )
            self._cache.move_to_end(key)
            return entry
        operator = self.bind()
        plan = operator.prepare(points=points)
        outside = operator.out_of_domain(plan)
        masks = self.regions(points)
        matrix = weight_matrix(
            self.weights(points), np.stack([masks[name] for name in self._regions], axis=1), self.device
        )
        entry = PointPlan(points, plan, matrix, 0 if outside is None else int(np.count_nonzero(outside)))
        self._cache[key] = entry
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        fixed = self.targets.fixed_points
        if fixed is not None and key == fixed.identity:
            self.diagnostics = operator.diagnostics(plan)
        return entry

    def observations(self, valid_time: datetime.datetime) -> Observations:
        """One read of the point targets at `valid_time`, checked: an `Observations`, values float32 (1, V, P), the
        declared fixed point set when there is one and no time offsets; an empty point set is a missing target."""
        observations = self.targets.observations(valid_time, self.variables, self.device)
        if not isinstance(observations, Observations):
            raise TypeError(
                f"{type(self.targets).__name__}.observations returned {type(observations).__name__}, expected "
                "Observations(points, values)"
            )
        points, values = observations.points, observations.values
        fixed = self.targets.fixed_points
        if fixed is not None and points is not fixed and points.identity != fixed.identity:
            raise ValueError(
                f"{type(self.targets).__name__} declares a fixed point set but returned another one at {valid_time}"
            )
        if len(points) == 0:
            raise MissingTargetError(f"no observations at {valid_time}")
        expected = (1, len(self.variables), len(points))
        if tuple(values.shape) != expected or values.dtype != torch.float32:
            raise ValueError(
                f"{type(self.targets).__name__} returned values {values.dtype} {tuple(values.shape)} at {valid_time}, "
                f"expected float32 {expected}"
            )
        check_timedeltas(points, f"{type(self.targets).__name__} at {valid_time}")
        return observations

    def map(self, frame: Frame, points: PointSet) -> tuple[Frame, PointPlan]:
        """`frame` mapped by the operator to `points` with the point set's cached plan, reduced to the scored
        variables, and that plan; points out of the operator's domain have a NaN forecast, so they carry no weight."""
        entry = self.entry(points)
        out = self.chain.apply(frames=[frame], plan=entry.plan, valid_time=frame.valid_time)
        index = [out.variables.index(name) for name in self.variables]
        data = out.data if index == list(range(len(out.variables))) else out.data[:, index]
        mapped = Frame(frame.init_time, frame.lead_time, frame.valid_time, list(self.variables), data, points)
        return mapped, entry

    def plan(self, plan: dict, valid_times: list[datetime.datetime]) -> None:
        """The point-target part of the dry run, added to `plan`: the operator, the point set and the byte sizes in
        points.

        The operator described is a throwaway deep copy bound on the CPU (so a constant of the wrong units, shape or
        grid, or a merge identity `identity_json` cannot encode, fails here, a run's CUDA device need not be present,
        and the evaluation stays unbound; binding the copy builds its neighbour tree again on every call). A fixed set
        is prepared on it for its out-of-domain count and the operator's diagnostics; weights and regions are not
        resolved. Sizes that scale with the number of points P use the fixed set's size, else the source's
        `max_points`; with neither they are left out. The forecast frame stays on the grid (the operator's inputs,
        N nodes)."""
        operator = copy.deepcopy(self._unbound)
        operator.bind(self.model_side("cpu"))
        self.identities(operator)
        fixed, p = self.targets.fixed_points, self.targets.max_points
        v, r = len(self.variables), len(self.region_specs)
        sizes = plan["bytes"]
        sizes["frame"] = self.forecast.members * len(self.forecast_variables) * self.forecast.grid.n * 4
        points: dict = {"fixed": fixed is not None, "max_points": p}
        if p is None:
            for key in ("target_to_device", "cache_for_all_rows", "aggregator_weights", "statistics_temporaries"):
                sizes.pop(key)
            if "row_bytes" not in plan["targets"]:
                sizes.pop("target_rows_read")
            points["note"] = "the number of points is not known before reading, so the sizes in P are left out"
        else:
            row = v * p * 4
            sizes["target_to_device"] = len(valid_times) * row
            sizes["target_rows_read"] = len(valid_times) * plan["targets"].get("row_bytes", row)
            sizes["cache_for_all_rows"] = len(set(valid_times)) * row
            sizes["aggregator_weights"] = p * r * 8
            sizes["statistics_temporaries"] = 3 * v * p * 8
        if fixed is not None and len(fixed):
            prepared = operator.prepare(points=fixed)
            outside = operator.out_of_domain(prepared)
            points["out_of_domain"] = 0 if outside is None else int(np.count_nonzero(outside))
            points["diagnostics"] = operator.diagnostics(prepared)
        plan["points"] = points
        plan["observation_operator"] = operator.describe()
        plan["forecast_variables"] = list(self.forecast_variables)

    def attrs(self) -> dict:
        """The result attrs of point targets, outside `config`: `observation_operator` and `point_targets`
        (`identities`), which `merge` compares, and, for information, `point_targets_description` and, once a fixed
        set is prepared, `observation_diagnostics`. The identities are those encoded at bind, so a value changed
        during the run neither fails nor changes what is recorded."""
        self.bind()
        attrs = dict(self._identities)
        attrs["point_targets_description"] = json.dumps(self.targets.describe(), default=str)
        if self.diagnostics is not None:
            attrs["observation_diagnostics"] = json.dumps(self.diagnostics, default=str)
        return attrs
