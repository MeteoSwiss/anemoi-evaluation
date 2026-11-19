"""Observation operator contract: what maps forecast frames to what an observation measured.

An operator has three phases, one per cache lifetime: `bind(model)` once per run, from the model side (grid,
variables, constants); `prepare(points, constants)` once per point set, returning a plan the caller reuses while the
`PointSet` key stays the same; `apply(frames, plan, valid_time)` per valid time, from forecast frames (M, V, N) to
(M, V_out, P), deterministic, float64 inside and float32 out, never mixing members.

Each operator declares, as attributes checked by `Chain` at construction:

* `input_space` and `output_space`: `grid -> grid` (`Identity`, a derived gridded variable), `grid -> points` (the
  spatial step, `Nearest`) or `points -> points` (`LapseRate`, anything acting on point values);
* `reads`, `produces` and `touches`: the variables it needs unchanged, adds, and modifies in place; every other
  variable passes through (a spatial step maps it). A step that produces or touches something is only needed when
  one of those variables is: otherwise its reads are not asked for and it is passed through;
* `stencil`: `"valid_time"` (default, the frame at the valid time only) or `"bracketing"` (the frames bracketing each
  point's `valid_time + timedelta`, as many as that takes). Provisional: the driver runs `valid_time` operators only,
  so the stencil names and `time_window` may change when a driver buffers frames;
* `point_attributes` and `constants`: the per-point attributes and model constants (`ConstantNeed`) it needs.

The space and stencil names are the constants `GRID`, `POINTS`, `VALID_TIME` and `BRACKETING`.

The forecast is asked for `input_variables(scored)`; the output frame holds `output_variables` of those, a superset
of the scored variables in another order, from which the caller selects the scored ones by name.

`parameters()` holds the constructor parameters (by default the public scalar attributes); the result's
`observation_operator` attr is `merge_identity()`, the declarations and the parameters, which `merge` compares.

`Chain` composes steps and is itself an operator; it is also the runtime wrapper that runs the generic checks, so a
user operator only implements its own logic. `ObservationOperatorBase` gives the defaults. `Chain` and the driver call
`prepare`, `apply` and `map_constant` by keyword, so the contract grows by keyword arguments with defaults.

Bound state (a tree, constants, plans) lives on the instances, so an operator instance belongs to one evaluation:
`Chain.bind` refuses a chain or step bound before. To score two forecasts through one operator, `copy.deepcopy` it
before binding.
"""

from __future__ import annotations

import datetime
import hashlib
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Protocol

import numpy as np
import torch

from anemoi.evaluation.frame import Frame
from anemoi.evaluation.frame import Grid
from anemoi.evaluation.points import PointSet
from anemoi.evaluation.variables import variable_info as _variable_info

GRID = "grid"
POINTS = "points"
SPACES = ((GRID, GRID), (GRID, POINTS), (POINTS, POINTS))
VALID_TIME = "valid_time"
BRACKETING = "bracketing"
STENCILS = (VALID_TIME, BRACKETING)
_DECLARATIONS = (
    "input_space",
    "output_space",
    "reads",
    "produces",
    "touches",
    "stencil",
    "point_attributes",
    "constants",
)
_BOUND = "_anemoi_evaluation_bound"
_PARAMETERS = "_anemoi_evaluation_parameters"
_SCALARS = (bool, int, float, str, type(None))


_NOT_A_PARAMETER = object()


def _scalar(value: Any) -> Any:
    """`value` as a parameter scalar (int, float, str, bool, None; a numpy number or a 0-d tensor through `.item()`),
    or `_NOT_A_PARAMETER`."""
    if isinstance(value, (np.number, np.bool_)) or (isinstance(value, torch.Tensor) and value.ndim == 0):
        value = value.item()
    return value if isinstance(value, _SCALARS) else _NOT_A_PARAMETER


def _parameter_value(value: Any) -> Any:
    """`value` as a parameter value: a scalar, or a list for a tuple or list of scalars; `_NOT_A_PARAMETER` when it
    is neither."""
    if isinstance(value, (tuple, list)):
        items = [_scalar(item) for item in value]
        return _NOT_A_PARAMETER if any(item is _NOT_A_PARAMETER for item in items) else items
    return _scalar(value)


def _public_attributes(operator: Any) -> Iterable[tuple[str, Any]]:
    """The public instance attributes of `operator`, without the declarations and the model side `bind` keeps."""
    for name, value in vars(operator).items():
        if not name.startswith("_") and name not in _DECLARATIONS and not isinstance(value, ModelSide):
            yield name, value


def _public_parameters(operator: Any) -> dict:
    """The public instance attributes of `operator` that are parameter values, as `_parameter_value` gives them."""
    values = {name: _parameter_value(value) for name, value in _public_attributes(operator)}
    return {name: value for name, value in values.items() if value is not _NOT_A_PARAMETER}


def _check_parameters(operator: Any) -> None:
    """Refuse a public attribute that the default `parameters()` cannot represent, since `merge` would not see it;
    an operator that overrides `parameters()` is not checked."""
    if getattr(type(operator), "parameters", None) is not ObservationOperatorBase.parameters:
        return
    for name, value in _public_attributes(operator):
        if _parameter_value(value) is _NOT_A_PARAMETER:
            raise TypeError(
                f"{type(operator).__name__}.{name} ({type(value).__name__}) is not a parameter value (int, float, "
                f"str, bool, None, or a tuple or list of them), so merge would not see it: override parameters(), "
                f"or rename it _{name} if it is not a parameter"
            )


@dataclass(frozen=True)
class Constant:
    """A named model constant in `units`, float64 `values`: (N,) on the model grid, (P,) once mapped to points."""

    name: str
    values: np.ndarray
    units: str | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError(f"constant {self.name!r} must be one-dimensional, got shape {values.shape}")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class ConstantNeed:
    """A model constant an operator needs, in `units` (checked against the provider's: geopotential is not a
    height)."""

    name: str
    units: str | None = None


class ConstantsProvider(Protocol):
    """Serves model constants on the model grid, each with its units.

    Optional: `check_grid(grid)` (raise unless the constants are on the model grid; called at bind before any
    constant is served, on top of the shape check every constant gets), `names` (the constants it serves, checked
    against the operator's needs at construction) and `merge_identity()` (the part of `describe()` that changes the
    values, for the result's `observation_operator` attr; `describe()` when absent). Serving a constant must not need
    the model loaded: the dry run binds the operator."""

    def constant(self, name: str) -> Constant:
        """The constant `name`, (N,) on the model grid; raises KeyError when it cannot be served."""
        ...

    def describe(self) -> dict:
        """Where the constants come from, for the dry run and the provenance; it identifies the values (a content
        digest, a file hash), since results made with other constants must not merge."""
        ...


def constants_digest(constants: Mapping[str, Constant]) -> str:
    """sha256 of the names, units and float64 values of `constants`."""
    digest = hashlib.sha256()
    for name in sorted(constants):
        constant = constants[name]
        digest.update(f"\0{name}\0{constant.units}\0".encode())
        digest.update(np.ascontiguousarray(constant.values, dtype=np.float64).tobytes())
    return digest.hexdigest()


class ArrayConstants:
    """Constants from in-memory arrays: `{name: Constant}` or `{name: (values, units)}`."""

    def __init__(self, constants: Mapping[str, Constant | tuple[Any, str | None]]) -> None:
        self.constants = {
            name: value if isinstance(value, Constant) else Constant(name, *value) for name, value in constants.items()
        }

    def constant(self, name: str) -> Constant:
        """The array named `name`."""
        try:
            return self.constants[name]
        except KeyError:
            raise KeyError(f"no model constant {name!r}, only {sorted(self.constants)}") from None

    @property
    def names(self) -> tuple[str, ...]:
        """The constants it serves."""
        return tuple(self.constants)

    def describe(self) -> dict:
        """Names, units and a sha256 of names, units and values, so results made with other constants differ."""
        return {"source": type(self).__name__, **self.merge_identity()}

    def merge_identity(self) -> dict:
        """Names, units and the sha256."""
        return {
            "constants": {n: c.units for n, c in self.constants.items()},
            "sha256": constants_digest(self.constants),
        }


@dataclass(frozen=True)
class ModelSide:
    """What an operator binds to: the space the forecast frames are in (`grid` or `points`), the variables they can
    hold, the grid (None in points space), a constants provider, typed-variable lookup and the device."""

    space: str
    variables: list[str]
    grid: Grid | None = None
    constants: ConstantsProvider | None = None
    variable_info: Callable[[str], dict] = field(default=lambda name: _variable_info(None))
    device: torch.device | str = "cpu"

    def __post_init__(self) -> None:
        if self.space not in (GRID, POINTS):
            raise ValueError(f"space must be {GRID!r} or {POINTS!r}, got {self.space!r}")
        if self.space == GRID and self.grid is None:
            raise ValueError("a model side in grid space needs its grid")
        object.__setattr__(self, "variables", list(self.variables))


class ObservationOperator(Protocol):
    """The full operator contract; see the module docstring."""

    input_space: str
    output_space: str
    reads: tuple[str, ...]
    produces: tuple[str, ...]
    touches: tuple[str, ...]
    stencil: str
    point_attributes: tuple[str, ...]
    constants: tuple[ConstantNeed, ...]

    def bind(self, model: ModelSide) -> None:
        """Build what does not depend on the points, once per run."""
        ...

    def prepare(self, points: PointSet, constants: Mapping[str, Constant]) -> Any:
        """The plan for one point set: the observation points, whatever the step's own input space (a `grid -> grid`
        step ignores them). `constants` holds the declared constants mapped to the points (points space only)."""
        ...

    def apply(self, frames: Sequence[Frame], plan: Any, valid_time: datetime.datetime) -> Frame:
        """Map the stencil's frames (one, the valid time's, by default) to one frame at `valid_time` in the output
        space with `output_variables(frame.variables)`."""
        ...

    def map_constant(self, values: np.ndarray, plan: Any) -> np.ndarray:
        """A spatial step maps a constant (N,) on the grid to the points (P,) as it maps a field."""
        ...

    def out_of_domain(self, plan: Any) -> np.ndarray | None:
        """Boolean (P,) of the points the operator cannot map (NaN forecast); None when all are mapped. A `Chain`
        reads it from its spatial step (`grid -> points`) only, the result's `out_of_domain` counter."""
        ...

    def diagnostics(self, plan: Any) -> dict:
        """Facts about a plan for the dry run and the provenance, e.g. the largest nearest-node distance of the
        points in the domain; may be empty."""
        ...

    def input_variables(self, outputs: Iterable[str]) -> list[str]:
        """Variables the input frames must hold to produce `outputs`."""
        ...

    def output_variables(self, inputs: Iterable[str]) -> list[str]:
        """Variables of the output frame for input frames holding `inputs`."""
        ...

    def time_window(self, points: PointSet) -> tuple[float, float]:
        """Seconds relative to the valid time the frames handed to `apply` must cover. Provisional, as `stencil`."""
        ...

    def parameters(self) -> dict:
        """The constructor parameters, JSON scalars or lists of them; never state built at bind."""
        ...

    def describe(self) -> dict:
        """Declarations and parameters, for the dry run."""
        ...

    def merge_identity(self) -> dict:
        """The result's `observation_operator` attr, which `merge` compares: everything that changes the mapped
        numbers."""
        ...

    def to_config(self) -> dict | list[dict]:
        """The `observation_operator:` config that rebuilds this operator; raises when not serialisable. Provisional:
        nothing reads it yet (no YAML for operators), so the shapes may change when that schema is designed."""
        ...


def runs(operator: ObservationOperator, variables: Iterable[str]) -> bool:
    """Whether `operator` acts on frames holding `variables`: False when it produces or touches something and a read is
    missing (it is not needed and passes through); raises when it acts on every variable and a read is missing."""
    variables = list(variables)
    missing = [name for name in (*operator.reads, *operator.touches) if name not in variables]
    if not missing:
        return True
    if operator.produces or operator.touches:
        return False
    raise ValueError(f"{type(operator).__name__} needs the variables {missing}, the frames hold {variables}")


class ObservationOperatorBase:
    """Defaults: `points -> points`, nothing read or needed, the valid-time frame only."""

    input_space: str = POINTS
    output_space: str = POINTS
    reads: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    touches: tuple[str, ...] = ()
    stencil: str = VALID_TIME
    point_attributes: tuple[str, ...] = ()
    constants: tuple[ConstantNeed, ...] = ()

    def bind(self, model: ModelSide) -> None:
        """Keep the model side."""
        self.model = model

    def prepare(self, points: PointSet, constants: Mapping[str, Constant]) -> Any:
        """No plan."""
        return None

    def apply(self, frames: Sequence[Frame], plan: Any, valid_time: datetime.datetime) -> Frame:
        """To implement."""
        raise NotImplementedError(f"{type(self).__name__} does not implement apply")

    def map_constant(self, values: np.ndarray, plan: Any) -> np.ndarray:
        """A `grid -> grid` step keeps the values; a `grid -> points` step must map them; points have none."""
        if (self.input_space, self.output_space) == (GRID, GRID):
            return values
        raise ValueError(f"{type(self).__name__} does not map model constants")

    def out_of_domain(self, plan: Any) -> np.ndarray | None:
        """Every point is mapped."""
        return None

    def diagnostics(self, plan: Any) -> dict:
        """Nothing to report."""
        return {}

    def input_variables(self, outputs: Iterable[str]) -> list[str]:
        """`outputs` without the produced variables, then the reads and touches; `outputs` alone when none of the
        produced or touched variables is among them."""
        outputs = list(outputs)
        products = {*self.produces, *self.touches}
        if products and not products & set(outputs):
            return outputs
        needed = [name for name in outputs if name not in self.produces]
        return needed + [name for name in dict.fromkeys((*self.reads, *self.touches)) if name not in needed]

    def output_variables(self, inputs: Iterable[str]) -> list[str]:
        """`inputs`, then the produced variables when the step runs; raises on a produced name already in `inputs`."""
        inputs = list(inputs)
        clash = [name for name in self.produces if name in inputs]
        if clash:
            raise ValueError(f"{type(self).__name__} produces {clash}, which the frames already hold")
        return inputs + list(self.produces) if runs(self, inputs) else inputs

    def time_window(self, points: PointSet) -> tuple[float, float]:
        """(0, 0) for the valid-time stencil; the range of the timedeltas for the bracketing one (provisional)."""
        if self.stencil == VALID_TIME or len(points) == 0:
            return 0.0, 0.0
        return float(points.timedeltas.min()), float(points.timedeltas.max())

    def parameters(self) -> dict:
        """The public instance attributes that are scalars (int, float, str, bool, None; numpy numbers and 0-d tensors
        as Python scalars) or tuples or lists of them, lists in the result, without the declarations; once bound by a
        `Chain`, their values just before the bind, so state built in `bind` never counts. Public scalars set in
        `__init__` that are not parameters (a device string, a path) enter the identity too, so prefix them with `_`.
        `Chain.bind` refuses a public attribute that is none of these; override it when a parameter is kept
        otherwise."""
        before = getattr(self, _PARAMETERS, None)
        return dict(before) if before is not None else _public_parameters(self)

    def describe(self) -> dict:
        """Class, spaces, stencil, declared needs and the parameters."""
        facts = {"operator": type(self).__name__, "space": f"{self.input_space} -> {self.output_space}"}
        for name in ("reads", "produces", "touches", "point_attributes"):
            if getattr(self, name):
                facts[name] = list(getattr(self, name))
        if self.constants:
            facts["constants"] = {need.name: need.units for need in self.constants}
        if self.stencil != VALID_TIME:
            facts["stencil"] = self.stencil
        parameters = self.parameters()
        if parameters:
            facts["parameters"] = parameters
        return facts

    def merge_identity(self) -> dict:
        """The description, with the parameters."""
        return {**self.describe(), "parameters": self.parameters()}

    def to_config(self) -> dict | list[dict]:
        """Not serialisable."""
        raise ValueError(f"{type(self).__name__} cannot be written to a config")

    @staticmethod
    def output_frame(
        frames: Sequence[Frame], valid_time: datetime.datetime, variables: list[str], data: torch.Tensor
    ) -> Frame:
        """A frame at `valid_time` of the forecast that `frames` come from."""
        init_time = frames[0].init_time
        return Frame(init_time, valid_time - init_time, valid_time, list(variables), data)


def check_declarations(operator: ObservationOperator) -> None:
    """Raise unless the declarations of `operator` are well formed."""
    name = type(operator).__name__
    spaces = (operator.input_space, operator.output_space)
    if spaces not in SPACES:
        raise ValueError(f"{name}: {spaces[0]} -> {spaces[1]} is not an operator space, use one of {SPACES}")
    for attr in ("reads", "produces", "touches", "point_attributes"):
        values = getattr(operator, attr)
        if isinstance(values, str) or not all(isinstance(value, str) for value in values):
            raise ValueError(f"{name}: {attr} must be a sequence of names, got {values!r}")
    overlap = set(operator.produces) & set((*operator.reads, *operator.touches))
    if overlap:
        raise ValueError(f"{name}: {sorted(overlap)} are both produced and read or touched")
    if operator.stencil not in STENCILS:
        raise ValueError(f"{name}: stencil must be one of {STENCILS}, got {operator.stencil!r}")
    if not all(isinstance(need, ConstantNeed) for need in operator.constants):
        raise ValueError(f"{name}: constants must be ConstantNeed entries, got {operator.constants!r}")


@dataclass(frozen=True)
class ChainPlan:
    """The plans of the steps for one point set, the constants mapped to its points and the points out of the domain
    of its spatial step."""

    points: PointSet
    plans: list[Any]
    constants: dict[str, Constant]
    out_of_domain: np.ndarray | None


class Chain(ObservationOperatorBase):
    """Operators applied in order; a chain is itself an operator, and nested chains are flattened.

    Rules, checked at construction and at `bind`:

    * spaces connect, step after step; at most one `grid -> points` step (there is no way back), so a chain on a
      gridded forecast that ends on points has exactly one, first among the steps that change space, and a chain
      starting in points space is refused on a gridded forecast;
    * variables not declared by a step pass through it; `input_variables` resolves the reads backwards, skipping the
      steps whose products are not needed, and those steps pass through at `apply`. A step passed through for the
      variables the chain is bound to imposes nothing: its constants are not read, its point attributes not asked
      for, and it is not prepared (`needs(variables)`);
    * constants of a `points -> points` step are the ones the spatial step before it mapped to the points, never the
      raw (N,) field; a points-space step needing constants with no spatial step before it is refused;
    * at most one step has a wider stencil than the valid-time frame; the steps before it are applied to every frame.

    The caller keeps the plans (`prepare` once per point set, reused while `PointSet.identity` is the same).
    """

    def __init__(self, steps: Iterable[ObservationOperator]) -> None:
        flat: list[ObservationOperator] = []
        for step in steps:
            flat.extend(step.steps if isinstance(step, Chain) else [step])
        if not flat:
            raise ValueError("a chain needs at least one step")
        for step in flat:
            check_declarations(step)
        for before, after in zip(flat, flat[1:]):
            if before.output_space != after.input_space:
                raise ValueError(
                    f"{type(after).__name__} takes {after.input_space}, but {type(before).__name__} before it gives "
                    f"{before.output_space}"
                )
        wide = [i for i, step in enumerate(flat) if step.stencil != VALID_TIME]
        if len(wide) > 1:
            raise ValueError(f"at most one step may have a time stencil, got {[type(flat[i]).__name__ for i in wide]}")
        produced: list[str] = []
        external: list[str] = []
        for step in flat:
            again = [name for name in step.produces if name in produced]
            if again:
                raise ValueError(f"{type(step).__name__} produces {again}, already produced earlier in the chain")
            external.extend(n for n in (*step.reads, *step.touches) if n not in produced and n not in external)
            produced.extend(step.produces)
        needs: dict[str, ConstantNeed] = {}
        for need in (need for step in flat for need in step.constants):
            if needs.setdefault(need.name, need).units != need.units:
                raise ValueError(
                    f"constant {need.name!r} is needed in units {needs[need.name].units!r} and {need.units!r}"
                )
        touched = {name for step in flat for name in step.touches}
        self.steps = flat
        self.input_space = flat[0].input_space
        self.output_space = flat[-1].output_space
        self.stencil = flat[wide[0]].stencil if wide else VALID_TIME
        self._wide = wide[0] if wide else None
        self.produces = tuple(produced)
        self.touches = tuple(name for name in external if name in touched)
        self.reads = tuple(name for name in external if name not in touched)
        self.point_attributes = tuple(dict.fromkeys(a for step in flat for a in step.point_attributes))
        self.constants = tuple(needs.values())
        self._bound = False
        self._running: list[bool] | None = None
        self._nodes: int | None = None
        self._raw: dict[str, Constant] = {}
        self._provider: ConstantsProvider | None = None

    def input_variables(self, outputs: Iterable[str]) -> list[str]:
        """Resolved backwards through the steps."""
        needed = list(outputs)
        for step in reversed(self.steps):
            needed = step.input_variables(needed)
        return needed

    def output_variables(self, inputs: Iterable[str]) -> list[str]:
        """Resolved forwards through the steps."""
        variables = list(inputs)
        for step in self.steps:
            variables = step.output_variables(variables)
        return variables

    def running(self, variables: Iterable[str] | None = None) -> list[bool]:
        """Per step, whether it runs on frames holding `variables` (False: it is passed through); without
        `variables`, the steps of the bind, or every step before it."""
        if variables is None:
            return [True] * len(self.steps) if self._running is None else list(self._running)
        variables = list(variables)
        flags = []
        for step in self.steps:
            flags.append(runs(step, variables))
            variables = step.output_variables(variables)
        return flags

    def needs(self, variables: Iterable[str] | None = None) -> tuple[tuple[str, ...], tuple[ConstantNeed, ...]]:
        """The point attributes and model constants of the steps that run on frames holding `variables` (see
        `running`); a step passed through needs neither."""
        steps = [step for step, run in zip(self.steps, self.running(variables)) if run]
        attributes = tuple(dict.fromkeys(name for step in steps for name in step.point_attributes))
        constants = tuple({need.name: need for step in steps for need in step.constants}.values())
        return attributes, constants

    def bind(self, model: ModelSide) -> None:
        """Check the model side, fetch and check the constants of the steps that run on `model.variables`, and bind
        every step to the space it sees; refuses a chain or a step bound before (use a fresh operator, or
        `copy.deepcopy` an unbound one)."""
        if self.bound:
            raise RuntimeError(
                "the operator is already bound to a model; bind a fresh one, or copy.deepcopy it before binding"
            )
        for step in self.steps:
            _check_parameters(step)
        if model.space != self.input_space:
            raise ValueError(f"the chain takes {self.input_space} frames, the forecast gives {model.space} frames")
        running = self.running(model.variables)
        _, needed = self.needs(model.variables)
        if needed and model.space != GRID:
            raise ValueError(
                f"model constants {[need.name for need in needed]} are only available on points once a "
                "grid -> points step has mapped them, and this chain has none"
            )
        if needed and model.constants is None:
            raise ValueError(f"the operator needs the model constants {[n.name for n in needed]}, no provider")
        check_grid = getattr(model.constants, "check_grid", None) if needed else None
        if check_grid is not None:
            check_grid(model.grid)
        raw = {}
        for need in needed:
            constant = model.constants.constant(need.name)
            if constant.units != need.units:
                raise ValueError(
                    f"model constant {need.name!r} is in {constant.units!r}, the operator needs {need.units!r}"
                )
            model.grid.check_shape(constant.values, f"model constant {need.name!r}")
            raw[need.name] = constant
        space, variables = model.space, list(model.variables)
        for step, run in zip(self.steps, running):
            grid_space = space == GRID
            if run:
                setattr(step, _PARAMETERS, _public_parameters(step))
                step.bind(
                    ModelSide(
                        space=space,
                        variables=variables,
                        grid=model.grid if grid_space else None,
                        constants=ArrayConstants({n.name: raw[n.name] for n in step.constants}) if grid_space else None,
                        variable_info=model.variable_info,
                        device=model.device,
                    )
                )
            variables = step.output_variables(variables)
            space = step.output_space
        for step in self.steps:
            setattr(step, _BOUND, True)
        self._bound = True
        self._running = running
        self._nodes = model.grid.n if model.grid is not None else None
        self._raw, self._provider = raw, model.constants if needed else None

    @property
    def bound(self) -> bool:
        """Whether the chain or one of its steps was bound before (a deep copy of a bound operator counts)."""
        return self._bound or any(getattr(step, _BOUND, False) for step in self.steps)

    def _check_bound(self) -> None:
        if not self._bound:
            raise RuntimeError("bind the operator to a model first")

    def check_attributes(self, names: Iterable[str], variables: Iterable[str] | None = None) -> None:
        """Raise unless a point target offering the attributes `names` serves every attribute the steps need: the
        steps that run on frames holding `variables` (see `running`)."""
        names = set(names)
        attributes, _ = self.needs(variables)
        missing = [name for name in attributes if name not in names]
        if missing:
            raise ValueError(f"the operator needs the point attributes {missing}, the targets offer {sorted(names)}")

    def prepare(self, points: PointSet, constants: Mapping[str, Constant] | None = None) -> ChainPlan:
        """Prepare every step on `points`; a spatial step maps the constants the steps after it need. `constants` is
        there for the operator signature only: a chain maps its constants itself."""
        self._check_bound()
        if constants:
            raise ValueError("a chain maps its model constants itself and takes none at prepare")
        self.check_attributes(points.attributes)
        mapped: dict[str, Constant] = {}
        plans = []
        outside = None
        running = self.running()
        for i, step in enumerate(self.steps):
            if not running[i]:
                plans.append(None)
                continue
            step_constants = (
                {need.name: mapped[need.name] for need in step.constants} if step.input_space == POINTS else {}
            )
            plan = step.prepare(points=points, constants=step_constants)
            plans.append(plan)
            if (step.input_space, step.output_space) == (GRID, POINTS):
                later = [s for s, run in zip(self.steps[i + 1 :], running[i + 1 :]) if run]
                for need in {n.name: n for s in later for n in s.constants}.values():
                    values = np.asarray(
                        step.map_constant(values=self._raw[need.name].values, plan=plan), dtype=np.float64
                    )
                    if values.shape != (len(points),):
                        raise ValueError(f"{type(step).__name__} mapped {need.name!r} to shape {values.shape}")
                    mapped[need.name] = Constant(need.name, values, self._raw[need.name].units)
                outside = self._outside(step, plan, len(points))
        return ChainPlan(points, plans, mapped, outside)

    @staticmethod
    def _outside(step: ObservationOperator, plan: Any, p: int) -> np.ndarray | None:
        """The spatial step's `out_of_domain(plan)`, checked: None, or boolean of shape (P,)."""
        mask = step.out_of_domain(plan)
        if mask is None:
            return None
        array = np.asarray(mask.detach().cpu() if isinstance(mask, torch.Tensor) else mask)
        if array.dtype != np.bool_ or array.shape != (p,):
            raise ValueError(
                f"{type(step).__name__}.out_of_domain returned {array.dtype} {array.shape}, expected bool ({p},), "
                "one flag per point"
            )
        return array

    def out_of_domain(self, plan: ChainPlan) -> np.ndarray | None:
        """The points out of the domain of the spatial step (`grid -> points`): what it cannot map. Later steps give
        NaN where they cannot compute and are not counted; their `out_of_domain` is not called."""
        return plan.out_of_domain

    def diagnostics(self, plan: ChainPlan) -> dict:
        """The steps' diagnostics, merged."""
        facts: dict = {}
        for step, step_plan, run in zip(self.steps, plan.plans, self.running()):
            if run:
                facts.update(step.diagnostics(step_plan))
        return facts

    def map_constant(self, values: np.ndarray, plan: ChainPlan) -> np.ndarray:
        """Through the steps in grid space (the spatial step maps, `grid -> grid` steps keep the values)."""
        for step, step_plan, run in zip(self.steps, plan.plans, self.running()):
            if run and step.input_space == GRID:
                values = step.map_constant(values=values, plan=step_plan)
        return values

    def time_window(self, points: PointSet) -> tuple[float, float]:
        """The window of the stencil step."""
        return (0.0, 0.0) if self._wide is None else self.steps[self._wide].time_window(points)

    def apply(self, frames: Sequence[Frame], plan: ChainPlan, valid_time: datetime.datetime) -> Frame:
        """Steps before the stencil step on every frame, the stencil step on all of them, the rest on its result. The
        output holds `output_variables` of the frames' variables; the caller selects the scored ones by name."""
        self._check_bound()
        frames = list(frames)
        if self._wide is None and (len(frames) != 1 or frames[0].valid_time != valid_time):
            raise ValueError(f"the chain needs exactly the frame at {valid_time}, got {[f.valid_time for f in frames]}")
        wide = len(self.steps) if self._wide is None else self._wide
        steps = list(zip(self.steps, plan.plans, self._running))
        for step, step_plan, run in steps[:wide]:
            frames = [self._step(step, [frame], step_plan, run, frame.valid_time, len(plan.points)) for frame in frames]
        for step, step_plan, run in steps[wide:]:
            frames = [self._step(step, frames, step_plan, run, valid_time, len(plan.points))]
        return frames[0]

    def _step(
        self,
        step: ObservationOperator,
        frames: list[Frame],
        plan: Any,
        prepared: bool,
        valid_time: datetime.datetime,
        p: int,
    ) -> Frame:
        variables = frames[0].variables
        if runs(step, variables) and not prepared:
            raise ValueError(
                f"{type(step).__name__} runs on frames holding {variables}, but the chain was bound to frames on "
                "which it is passed through, so it was not prepared; bind the chain to the variables of its frames"
            )
        if not runs(step, variables):
            same = [frame for frame in frames if frame.valid_time == valid_time]
            if not same:
                raise ValueError(f"{type(step).__name__} is passed through but no frame is at {valid_time}")
            return same[0]
        out = step.apply(frames=frames, plan=plan, valid_time=valid_time)
        expected = step.output_variables(variables)
        size = p if step.output_space == POINTS else self._nodes
        if (
            out.variables != expected
            or out.members != frames[0].members
            or out.valid_time != valid_time
            or out.data.shape[-1] != size
        ):
            raise ValueError(
                f"{type(step).__name__} returned {out.members} members of {out.variables} on {out.data.shape[-1]} "
                f"locations at {out.valid_time}, expected {frames[0].members} of {expected} on {size} at {valid_time}"
            )
        return out

    def describe(self) -> dict:
        """The steps and the constants provider."""
        facts = {"operator": "chain", "space": f"{self.input_space} -> {self.output_space}"}
        facts["steps"] = [step.describe() for step in self.steps]
        if self._provider is not None:
            facts["constants"] = self._provider.describe()
        return facts

    def merge_identity(self) -> dict:
        """The steps' merge identities and the constants provider's."""
        facts: dict = {"steps": [step.merge_identity() for step in self.steps]}
        if self._provider is not None:
            identity = getattr(self._provider, "merge_identity", None)
            facts["constants"] = identity() if callable(identity) else self._provider.describe()
        return facts

    def to_config(self) -> list[dict]:
        """The list of step configs (provisional, see `ObservationOperator.to_config`); raises when a step is not
        serialisable."""
        configs = []
        for step in self.steps:
            config = step.to_config()
            configs.extend(config if isinstance(config, list) else [config])
        return configs


class Identity(ObservationOperatorBase):
    """Today's gridded path: targets on the model grid, frames returned as they are."""

    input_space = GRID
    output_space = GRID

    def apply(self, frames: Sequence[Frame], plan: Any, valid_time: datetime.datetime) -> Frame:
        """The frame itself, unchanged."""
        return frames[0]

    def to_config(self) -> dict:
        """`identity` (provisional, see `ObservationOperator.to_config`)."""
        return {"identity": self.parameters()}
