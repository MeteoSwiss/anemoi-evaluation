"""Forecasts produced in memory by an anemoi-inference runner."""

from __future__ import annotations

import contextlib
import copy
import datetime
import gc
import hashlib
import inspect
import itertools
import logging
import math
import os
from collections import OrderedDict
from collections.abc import Iterator
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import torch
from anemoi.utils.dates import frequency_to_string

from anemoi.evaluation.frame import Frame
from anemoi.evaluation.frame import Grid
from anemoi.evaluation.sources.base import ForecastSourceBase
from anemoi.evaluation.sources.base import MissingGraphError
from anemoi.evaluation.sources.base import accumulation_periods
from anemoi.evaluation.sources.base import lead_time_steps
from anemoi.evaluation.sources.base import select_variables
from anemoi.evaluation.sources.base import variable_info
from anemoi.evaluation.sources.temporal_downscaler import downscaler_window
from anemoi.evaluation.sources.temporal_downscaler import trajectory_trained
from anemoi.evaluation.sources.temporal_downscaler import window_input_state

if TYPE_CHECKING:
    from anemoi.evaluation.distributed import MemberGroup

LOG = logging.getLogger(__name__)
QUIET_LOGGERS = ("anemoi.inference", "anemoi.utils.timer")


def member_seed(seed: int, init_time: datetime.datetime, member: int = 0, step: int = 0) -> int:
    """Seed of one member's forward pass `step` of an init time: a 63-bit hash of the four values, so that a member
    does not depend on the ensemble size, on the other members or on how the rollouts are interleaved."""
    key = f"{seed}|{init_time.isoformat()}|{member}|{step}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") >> 1


def create_input_state(source: object, init_time: datetime.datetime, lagged: list[datetime.timedelta]) -> dict:
    """Input state at `init_time`: anemoi-inference >= 0.13 takes `dates=`, older releases `date=`."""
    if "dates" in inspect.signature(source.create_input_state).parameters:
        return source.create_input_state(dates=[init_time + lag for lag in lagged])
    return source.create_input_state(date=init_time)


def required_dates(
    init_time: datetime.datetime,
    input_offsets: list[datetime.timedelta],
    output_offsets: list[datetime.timedelta],
    rollout_shift: datetime.timedelta,
    lead_time: datetime.timedelta | None = None,
) -> list[datetime.datetime]:
    """Dates an input reads, sorted: `init_time` plus every input offset, plus, for forcings read during the rollout
    (`lead_time` given), every output date of each model call but the last, at which the runner loads the next call's
    forcings."""
    dates = {init_time + offset for offset in input_offsets}
    if lead_time is not None:
        calls = math.ceil(lead_time / rollout_shift)
        dates.update(
            init_time + call * rollout_shift + offset for call in range(calls - 1) for offset in output_offsets
        )
    return sorted(dates)


def missing_dates(required: list[datetime.datetime], available: set, missing: set) -> list[datetime.datetime]:
    """Dates of `required` that are absent from `available` or listed in `missing`."""
    return [date for date in required if date not in available or date in missing]


@contextlib.contextmanager
def quiet_inference(level: int = logging.WARNING) -> Iterator[None]:
    """Raise the anemoi-inference loggers, and the timer it logs every step with, to `level` for the block."""
    loggers = [logging.getLogger(name) for name in QUIET_LOGGERS]
    previous = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(level)
    try:
        yield
    finally:
        for logger, before in zip(loggers, previous):
            logger.setLevel(before)


CHUNKS_ENV = "ANEMOI_INFERENCE_NUM_CHUNKS"
PROCESSOR_CHUNKS_ENV = "ANEMOI_INFERENCE_NUM_CHUNKS_PROCESSOR"
MAPPER_CHUNKS_ENV = "ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER"
DEFAULT_FORCINGS_CACHE_BYTES = 2**30


def inference_env(run_config: Mapping, environ: Mapping[str, str] = os.environ) -> dict:
    """The run configuration's `env` block with the package default `ANEMOI_INFERENCE_NUM_CHUNKS_PROCESSOR=1` added when
    `ANEMOI_INFERENCE_NUM_CHUNKS` is set, in the block or in `environ`, and the processor count is set in neither.

    anemoi-models reads both at import time: the mapper count bounds the decoder's per-edge memory, the processor count
    only costs time (see docs/benchmarks.md). A processor count given anywhere is left alone."""
    env = dict(run_config.get("env") or {})
    chunked = CHUNKS_ENV in env or CHUNKS_ENV in environ
    if chunked and PROCESSOR_CHUNKS_ENV not in env and PROCESSOR_CHUNKS_ENV not in environ:
        env[PROCESSOR_CHUNKS_ENV] = 1
    return env


def _chunk_constants() -> dict[str, int] | None:
    """anemoi-models' import-time chunk counts, None when anemoi-models is not installed."""
    try:
        from anemoi.models.layers import block
        from anemoi.models.layers import mapper
    except ImportError:  # pragma: no cover
        return None
    return {"processor": int(block.NUM_CHUNKS_INFERENCE_PROCESSOR), "mapper": int(mapper.NUM_CHUNKS_INFERENCE_MAPPER)}


def inference_chunks(model: object) -> dict[str, int]:
    """The chunk counts a loaded model runs with: `processor` from anemoi-models' import-time constant, `mapper` the
    larger of that constant and the checkpoint's own mapper `num_chunks` (what the mapper uses). Empty when unknown."""
    chunks = _chunk_constants()
    if chunks is None:
        return {}
    inner = getattr(model, "model", model)
    configured = [
        int(m.num_chunks)
        for store in (getattr(inner, "encoder", None), getattr(inner, "decoder", None))
        if store is not None
        for m in (store.values() if hasattr(store, "values") else [store])
        if hasattr(m, "num_chunks")
    ]
    if configured:
        chunks["mapper"] = max(chunks["mapper"], *configured)
    return chunks


class ForcingsCache:
    """Host LRU of forcings arrays keyed by provider and dates, `cache_bytes` at most, for one grid.

    The arrays are what anemoi-inference's forcings providers return for a list of dates; they depend on the dates and on
    the state's latitudes and longitudes only, and the runner never writes them, so one array serves every lockstep
    member and every later init time that reaches the same dates. The grid is checked: the first latitudes and
    longitudes seen are remembered, a new pair of arrays is compared once, and a request on a different grid bypasses
    the cache."""

    def __init__(self, cache_bytes: int) -> None:
        if cache_bytes < 0:
            raise ValueError(f"cache_bytes must not be negative, got {cache_bytes}")
        self.cache_bytes = cache_bytes
        self._entries: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._size = 0
        self._grid: tuple[np.ndarray, np.ndarray] | None = None
        self.stats = {"computed": 0, "hits": 0, "bypassed": 0}

    def same_grid(self, state: Mapping) -> bool:
        """Whether `state` is on the cache's grid; the first grid seen defines it."""
        latitudes, longitudes = state["latitudes"], state["longitudes"]
        if self._grid is None:
            self._grid = (latitudes, longitudes)
            return True
        if latitudes is self._grid[0] and longitudes is self._grid[1]:
            return True
        if np.array_equal(latitudes, self._grid[0]) and np.array_equal(longitudes, self._grid[1]):
            self._grid = (latitudes, longitudes)  # remember the new objects, so the next check is an identity test
            return True
        return False

    def get(self, key: tuple, compute, state: Mapping) -> np.ndarray:
        """The cached array for `key`, else `compute()`, stored when it fits and the state is on the cache's grid."""
        if not self.same_grid(state):
            self.stats["bypassed"] += 1
            return compute()
        if key in self._entries:
            self._entries.move_to_end(key)
            self.stats["hits"] += 1
            return self._entries[key]
        array = compute()
        self.stats["computed"] += 1
        if array.nbytes <= self.cache_bytes:
            while self._entries and self._size + array.nbytes > self.cache_bytes:
                _, evicted = self._entries.popitem(last=False)
                self._size -= evicted.nbytes
            self._entries[key] = array
            self._size += array.nbytes
        return array

    def clear(self) -> None:
        """Drop every entry."""
        self._entries.clear()
        self._size = 0


class SharedForcings:
    """A forcings provider whose arrays come through a `ForcingsCache`; exposes what the tensor handler reads."""

    def __init__(self, provider: object, cache: ForcingsCache) -> None:
        self.provider = provider
        self.cache = cache

    @property
    def variables(self) -> list[str]:
        """The wrapped provider's variables."""
        return self.provider.variables

    @property
    def mask(self) -> object:
        """The wrapped provider's input-tensor mask."""
        return self.provider.mask

    @property
    def kinds(self) -> dict:
        """The wrapped provider's kinds (debugging labels)."""
        return self.provider.kinds

    def load_forcings_array(self, dates: list, current_state: Mapping) -> np.ndarray:
        """The provider's array for `dates`, computed once per dates on the cache's grid."""
        dates = list(dates) if isinstance(dates, (list, tuple)) else [dates]
        key = (id(self.provider), tuple(dates))
        return self.cache.get(key, lambda: self.provider.load_forcings_array(dates, current_state), current_state)

    def __repr__(self) -> str:
        return f"Shared({self.provider!r})"

    def __getattr__(self, name: str) -> object:
        return getattr(self.provider, name)


SUPPORTED_TASKS = ("forecaster", "offset-forecaster", "temporal-downscaler")
# Tasks blocked by anemoi-inference, with the reason the refusal gives.
INFERENCE_BLOCKERS = {
    "autoencoder": "anemoi-inference has no runner for it",
}
# Class names of the task classes (`config.task._target_`) and of the older lightning modules
# (`config.training.model_task`). Ensemble and diffusion modules are forecasters.
TASK_CLASSES = {
    "Forecaster": "forecaster",
    "OffsetForecaster": "offset-forecaster",
    "TemporalDownscaler": "temporal-downscaler",
    "Autoencoder": "autoencoder",
    "GraphForecaster": "forecaster",
    "GraphEnsForecaster": "forecaster",
    "GraphDiffusionForecaster": "forecaster",
    "GraphDiffusionTendForecaster": "forecaster",
    "GraphInterpolator": "temporal-downscaler",
    "GraphMultiOutInterpolator": "temporal-downscaler",
    "GraphAutoEncoder": "autoencoder",
}
# Older name stored in `metadata_inference.task`.
TASK_ALIASES = {"time-interpolator": "temporal-downscaler"}


def task_name(value: str) -> str:
    """Map a stored task name or class path to a task name (`_` becomes `-`); an unknown class path is kept as is."""
    cls = value.rsplit(".", 1)[-1]
    if cls in TASK_CLASSES:
        return TASK_CLASSES[cls]
    if "." in value:
        return value
    name = value.replace("_", "-")
    return TASK_ALIASES.get(name, name)


def checkpoint_task(metadata: Mapping) -> tuple[str, str | None]:
    """The checkpoint's task and the field it was read from (`None` for the default, `forecaster`). Lookup order:
    `metadata_inference.task`, top-level `task`, `config.task._target_`, `config.training.model_task`."""
    config = metadata.get("config") or {}
    task = config.get("task")
    candidates = (
        ("metadata_inference.task", (metadata.get("metadata_inference") or {}).get("task")),
        ("task", metadata.get("task")),
        ("config.task._target_", task.get("_target_") if isinstance(task, Mapping) else None),
        ("config.training.model_task", (config.get("training") or {}).get("model_task")),
    )
    for origin, value in candidates:
        if value:
            return task_name(str(value)), f"{origin}: {value}"
    return "forecaster", None


def recorded_variables_metadata(metadata: Mapping, name: str) -> dict:
    """The `variables_metadata` a checkpoint's metadata records for dataset `name` (either layout of the `dataset`
    block), read before anemoi-inference fills in assumed entries."""
    block = metadata.get("dataset") or {}
    if "variables_metadata" not in block:
        block = block.get(name) or {}
    return dict(block.get("variables_metadata") or {})


def patched_metadata(metadata: Mapping, patch: Mapping | None) -> Mapping:
    """`metadata` with a `patch_metadata` block applied to the first dataset, as anemoi-inference does."""
    if not patch:
        return metadata
    from anemoi.inference.config.utils import multi_datasets_config

    names = list((metadata.get("metadata_inference") or {}).get("dataset_names") or ["data"])

    def merge(main: dict, update: Mapping) -> None:
        for key, value in update.items():
            if isinstance(value, Mapping) and isinstance(main.get(key, {}), Mapping):
                merge(main.setdefault(key, {}), value)
            else:
                main[key] = value

    result = copy.deepcopy(dict(metadata))
    merge(result, multi_datasets_config(patch, names[0], names, strict=False))
    return result


def check_task(metadata: Mapping) -> str:
    """The checkpoint's task; refuse one that is not supported yet. An unknown class path only warns: anemoi-inference
    runs it as a forecaster. A temporal downscaler trained on forecast trajectories is refused too: its bounds are
    forecasts and its truth a trajectory, which the evaluation cannot read yet."""
    task, origin = checkpoint_task(metadata)
    if task not in SUPPORTED_TASKS and "." in task:
        LOG.warning("unknown task class (%s), the checkpoint is evaluated as a forecaster", origin)
    elif task not in SUPPORTED_TASKS:
        raise ValueError(
            f"the checkpoint's task is {task!r} ({origin}): not supported yet"
            + (f" ({INFERENCE_BLOCKERS[task]})" if task in INFERENCE_BLOCKERS else "")
            + f", only {', '.join(SUPPORTED_TASKS)} so far, see MeteoSwiss/anemoi-evaluation#5"
        )
    elif task == "temporal-downscaler" and trajectory_trained(metadata):
        raise ValueError(
            f"the checkpoint is a temporal downscaler ({origin}) trained on forecast trajectories: not supported yet "
            "(only downscalers between analysis bounds so far; forecast bounds need trajectory targets, see "
            "MeteoSwiss/anemoi-evaluation#4), see MeteoSwiss/anemoi-evaluation#5"
        )
    return task


def check_runner(task: str, runner: object, configured: object, stored: object) -> None:
    """Refuse a runner that does not fit the task, since the evaluation drives a temporal downscaler and a forecaster
    differently; `configured` is the run configuration's `runner`, `stored` the `metadata_inference.task`, the only
    field anemoi-inference's `auto` runner reads."""
    from anemoi.inference.runners.temporal_downscaler import TemporalDownscalerMultiOutRunner

    downscaler = task == "temporal-downscaler"
    if downscaler == isinstance(runner, TemporalDownscalerMultiOutRunner):
        return
    cause = ""
    if configured == "auto" and task_name(str(stored or "forecaster")) != task:
        cause = f" (anemoi-inference's `runner: auto` reads only `metadata_inference.task`, which is {stored!r})"
    raise ValueError(
        f"the checkpoint's task is {task!r} but anemoi-inference built a {type(runner).__name__}{cause}: set "
        f"`runner: {'temporal_downscaler' if downscaler else 'default'}` in the anemoi_inference block, or fix the "
        "stored task with `patch_metadata: {metadata_inference: {task: ...}}`"
    )


def check_rollout_timing(
    input_offsets: list[datetime.timedelta],
    output_offsets: list[datetime.timedelta],
    rollout_shift: datetime.timedelta,
) -> None:
    """Refuse a timing anemoi-core's `OffsetForecaster` does not produce (a hand-edited or patched checkpoint): the
    last input offset `0h`, positive increasing output offsets, a rollout shift that is one of them and longer than
    their span. The lead times rely on it: counted from the init time, one model call each, increasing."""
    increasing = all(a < b for a, b in zip(output_offsets, output_offsets[1:]))
    if not (
        input_offsets
        and input_offsets[-1] == datetime.timedelta(0)
        and output_offsets
        and output_offsets[0] > datetime.timedelta(0)
        and increasing
        and rollout_shift in output_offsets
        and output_offsets[-1] - output_offsets[0] < rollout_shift
    ):
        raise ValueError(
            f"invalid checkpoint timing: input offsets {[frequency_to_string(o) for o in input_offsets]}, output "
            f"offsets {[frequency_to_string(o) for o in output_offsets]}, rollout shift "
            f"{frequency_to_string(rollout_shift)}; the last input offset must be 0h and the output offsets positive, "
            "increasing and spanning less than the rollout shift, which must be one of them"
        )


def routed_datasets(metadata: Mapping) -> tuple[set[str], set[str]] | None:
    """The datasets a checkpoint's model encodes and decodes, from the training config it stores, or None when the
    config does not say (legacy single-dataset checkpoints have no `encoders`/`decoders` routing)."""
    model = metadata.get("config", {}).get("model") or {}
    encoders, decoders = model.get("encoders"), model.get("decoders")
    if not isinstance(encoders, dict) or not isinstance(decoders, dict):
        return None
    encoded = {name for encoder in encoders.values() for name in (encoder or {}).get("source_datasets", [])}
    decoded = {name for decoder in decoders.values() for name in (decoder or {}).get("target_datasets", [])}
    if not encoded or not decoded:
        return None
    return encoded, decoded


class DatasetForecast:
    """One decoded dataset of an `InferenceForecastSource`, as an ordinary single-dataset forecast source.

    A multi-dataset run is N of these sharing one runner and one rollout, so a view yields no frames of its own:
    the parent drives the shared rollout and hands each view's frames to its evaluation. Everything the view does
    not define itself (members, device, lead times, provenance, ...) is the parent's and is delegated.

    Deliberately not a `ForecastSourceBase`: the base's defaults (`has_graph`, `supports_lead_zero`, `device`,
    `frames_per_pass`, `provenance`, `to_config`, ...) would be found by normal attribute lookup and shadow the
    delegation, so a view would silently answer for itself instead of for the parent. Without a base class every
    name the view does not define reaches `__getattr__`, including names added to the protocol later."""

    def __init__(self, source: InferenceForecastSource, name: str) -> None:
        self.source = source
        self.name = name
        # Set when this view is the only dataset the run scores (a subset of one, `datasets: [name]`): the
        # evaluation is then an ordinary single-dataset one, so the view drives the rollout and owns the runner.
        self.solo = False
        metadata = source.runner.checkpoint.multi_dataset_metadata[name]
        names = metadata.output_tensor_index_to_variable
        self.variables = [names[i] for i in range(len(names))]
        self.typed_variables = dict(metadata.typed_variables)
        # anemoi-inference fills in assumed periods where the checkpoint records no `variables_metadata` (its legacy
        # path), so a period counts as known only where the recorded metadata gives one
        recorded = recorded_variables_metadata(source._recorded_metadata, name)
        self._accumulation_periods = {
            name: period if "period" in (recorded.get(name) or {}) else None
            for name, period in accumulation_periods(self.typed_variables).items()
        }
        arrays = metadata.supporting_arrays
        self.grid_indices = arrays.get("grid_indices")
        index = slice(None) if self.grid_indices is None else self.grid_indices
        self.grid = Grid(arrays["latitudes"][index], arrays["longitudes"][index])

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name})"

    def __getattr__(self, name: str) -> object:
        if name.startswith("_") or name in ("source", "name"):
            raise AttributeError(name)
        return getattr(self.source, name)

    @property
    def stats(self) -> dict:
        """This dataset's forcings counters, not the run-wide totals, so the N result files add up."""
        cache = self.source._forcings.get(self.name)
        return {} if cache is None else {f"forcings_{key}": value for key, value in cache.stats.items()}

    def frames(
        self,
        init_time: datetime.datetime,
        lead_time: datetime.timedelta,
        variables: list[str],
        device: torch.device,
    ) -> Iterator[Frame]:
        """The frames of this dataset, when it is the only one the run scores (`solo`); otherwise not available,
        because the parent drives the one rollout the datasets share.

        A solo rollout still predicts every dataset of the checkpoint — anemoi-inference runs every decoder — only
        the frames of the others are not built."""
        if not self.solo:
            raise ValueError(
                f"the frames of {self.name!r} come from the shared rollout: run the evaluation through "
                "MultiEvaluation, or call the source's multi_frames()"
            )
        for frames in self.source.multi_frames(init_time, lead_time, {self.name: variables}, device):
            yield frames[self.name]

    def initial_frame(self, init_time: datetime.datetime, variables: list[str], device: torch.device) -> Frame:
        """This dataset's lead-0 frame."""
        return self.source.initial_frame(init_time, variables, device, dataset=self.name)

    def check_init_times(self, init_times: list[datetime.datetime], lead_time: datetime.timedelta) -> None:
        """Check this dataset's inputs only."""
        self.source.check_init_times(init_times, lead_time, dataset=self.name)

    def dataset_args_kwargs(self) -> tuple[tuple, dict]:
        """`open_dataset` arguments of this dataset's prognostics input."""
        return self.source.dataset_args_kwargs(dataset=self.name)

    def graph_node_attribute(self, name: str, nodes: str = "data") -> np.ndarray:
        """Per-node attribute of this dataset's node set of the checkpoint's graph."""
        return self.source.graph_node_attribute(name, nodes, dataset=self.name)

    def variable_info(self, name: str) -> dict:
        """`param` and `level` of a variable of this dataset."""
        return variable_info(self.typed_variables.get(name))

    def accumulation_periods(self) -> dict:
        """The period of each accumulated variable of this dataset, as the checkpoint records it; None when not
        recorded."""
        return dict(self._accumulation_periods)

    def describe(self) -> dict:
        """What the parent reports for this dataset."""
        return self.source.describe(dataset=self.name)

    def close(self) -> None:
        """Nothing to release unless this view is the run's only dataset, in which case it owns the parent."""
        if self.solo:
            self.source.close()


class InferenceForecastSource(ForecastSourceBase):
    """Runs an anemoi-inference checkpoint; `members` lockstep rollouts per init time (a rank's block of them in a
    member group, see `split_members`), `quiet` silences the runner's per-step logging while it runs.

    A checkpoint trained on multiple datasets gives one `DatasetForecast` view per dataset (`datasets`), all sharing
    this runner and one rollout; `multi_frames()` yields one frame per dataset per step. A single-dataset
    checkpoint keeps the flat single-dataset API (`grid`, `variables`, `frames()`) unchanged."""

    has_graph = True  # may carry one: trimmed checkpoints store none, found out when the model loads
    supports_lead_zero = True

    def __init__(
        self,
        members: int = 1,
        seed: int = 0,
        quiet: bool = True,
        forcings_cache_bytes: int = DEFAULT_FORCINGS_CACHE_BYTES,
        **run_config: object,
    ) -> None:
        from anemoi.inference.checkpoint import Checkpoint
        from anemoi.inference.config.run import RunConfiguration
        from anemoi.inference.runners import create_runner
        from anemoi.utils.checkpoints import load_metadata

        if members < 1:
            raise ValueError(f"members must be at least 1, got {members}")
        if "date" in run_config:
            raise ValueError("do not set 'date' in the inference config, init times come from the evaluation")
        if run_config.get("output", "none") != "none":
            raise ValueError("the inference config's output must be 'none', forecasts stay in memory")
        env = inference_env(run_config)
        self.run_config = {**run_config, **({"env": env} if env else {})}
        self.members = members
        self.seed = seed
        self.quiet = quiet
        self.forcings_cache_bytes = forcings_cache_bytes
        self.member_group: MemberGroup | None = None
        self._initial: dict[str, tuple[datetime.datetime, dict]] = {}
        with self._quiet():
            config = RunConfiguration.load({"verbosity": 0, **self.run_config, "output": "none"})
            # before the runner is built: the `auto` runner raises on a task it has no runner for
            metadata = load_metadata(Checkpoint(config.checkpoint).path)
            patched = patched_metadata(metadata, config.patch_metadata)
            self._recorded_metadata = patched  # as stored (and patched), before the runner fills anything in
            self.task = check_task(patched)
            self.runner = create_runner(config)
            check_runner(self.task, self.runner, config.runner, (patched.get("metadata_inference") or {}).get("task"))
        self.dataset_names = list(self.runner.dataset_names)
        # One cache per dataset: the arrays depend on the state's latitudes and longitudes, so a single cache
        # would serve the first grid and bypass every other one for ever. `forcings_cache_bytes` is the budget
        # of each cache.
        self._forcings = (
            {name: ForcingsCache(forcings_cache_bytes) for name in self.dataset_names} if forcings_cache_bytes else {}
        )
        for name, handler in self.runner.tensor_handlers.items():
            cache = self._forcings.get(name)
            if cache is None:
                continue
            handler.constant_forcings_providers = [
                SharedForcings(p, cache) for p in handler.constant_forcings_providers
            ]
            handler.dynamic_forcings_providers = [SharedForcings(p, cache) for p in handler.dynamic_forcings_providers]
        self.checkpoint = self.runner.checkpoint.path
        self.timestep = self.runner.checkpoint.timestep
        self.multi_step_input = self.runner.checkpoint.multi_step_input
        self.multi_step_output = self.runner.checkpoint.multi_step_output
        self._check_shared_timing()
        self.input_offsets = list(self.runner.checkpoint.lagged)
        self.output_offsets = list(self.runner.checkpoint.output_offsets)
        if self.task == "temporal-downscaler":
            # one window per init time: its outputs come from one model call, placed by `downscaler_window`
            self.temporal_downscaling = True
            self.window, self.window_leads = downscaler_window(self.runner.checkpoint)
            self.output_horizon = self.window
            self.frames_per_pass = len(self.window_leads)
            self.supports_lead_zero = False  # lead 0 is the left bound, an input
            self._window_constants: dict[str, dict] = {}
        else:
            self.output_horizon = self.runner.checkpoint.rollout_shift
            check_rollout_timing(self.input_offsets, self.output_offsets, self.output_horizon)
            self.frames_per_pass = len(self.output_offsets)
        self._check_every_dataset_is_decoded(metadata)
        # Which datasets the run scores; every one of them until an evaluation narrows it (the `datasets:` config
        # key). The model predicts them all whatever this says: the runner needs an input state for each one.
        self.scored_datasets: list[str] | None = None
        self.datasets = {name: DatasetForecast(self, name) for name in self.dataset_names}
        if len(self.dataset_names) == 1:
            self.dataset_name = self.dataset_names[0]
            single = self.datasets[self.dataset_name]
            self.variables, self.typed_variables = single.variables, single.typed_variables
            self.grid_indices, self.grid = single.grid_indices, single.grid
        self.checkpoint_uuid, self.checkpoint_run_id = metadata.get("uuid"), metadata.get("run_id")
        if members > 1 and metadata["config"]["model"].get("noise_injector") is None:
            LOG.warning("%s has no noise injector, its %d members will be identical", self.checkpoint, members)

    def __getattr__(self, name: str) -> object:
        if name in ("grid", "variables", "typed_variables", "dataset_name"):
            names = ", ".join(self.__dict__.get("dataset_names", ()))
            raise AttributeError(
                f"{name!r} is per dataset on a checkpoint with multiple datasets ({names}): use datasets[name].{name}"
            )
        raise AttributeError(name)

    @property
    def multi_dataset(self) -> bool:
        """Whether the checkpoint was trained on more than one dataset."""
        return len(self.dataset_names) > 1

    def _resolve_dataset(self, dataset: str | None) -> str:
        """`dataset`, or the only dataset of a single-dataset checkpoint."""
        if dataset is None:
            if self.multi_dataset:
                raise ValueError(f"the checkpoint has multiple datasets ({', '.join(self.dataset_names)}), name one")
            return self.dataset_names[0]
        if dataset not in self.dataset_names:
            raise ValueError(f"unknown dataset {dataset!r}, the checkpoint has {', '.join(self.dataset_names)}")
        return dataset

    def _check_shared_timing(self) -> None:
        """Refuse a checkpoint whose datasets disagree on the timing.

        anemoi-core writes one and the same `timesteps` block to every dataset and anemoi-inference forwards the
        first dataset's timing as the checkpoint's, silently; a hand-edited checkpoint could differ, and the
        evaluation would score every dataset on the first one's lead times."""
        metadata = self.runner.checkpoint.multi_dataset_metadata
        fields = (
            "timestep",
            "multi_step_input",
            "multi_step_output",
            "lagged",
            "output_offsets",
            "rollout_shift",
            "advance_map",
        )
        found = {name: tuple(str(getattr(metadata[name], field)) for field in fields) for name in self.dataset_names}
        if len(set(found.values())) > 1:
            reported = {name: dict(zip(fields, values)) for name, values in found.items()}
            raise ValueError(
                f"the datasets of a checkpoint must share the same timestep, input and output steps, got {reported}"
            )

    def _check_every_dataset_is_decoded(self, metadata: Mapping) -> None:
        """Refuse a checkpoint whose model does not decode every dataset (a spatial downscaler).

        anemoi-inference runs one since ecmwf/anemoi-inference#605, but the evaluation does not score it yet. The
        routing is only readable from the stored training config, so a checkpoint that does not carry it is let
        through."""
        routed = routed_datasets(metadata)
        if routed is None:
            return
        _, decoded = routed
        missing = [name for name in self.dataset_names if name not in decoded]
        if missing:
            raise ValueError(
                f"the checkpoint's model does not decode {missing}, it only decodes {sorted(decoded)}: "
                "spatial downscalers are not supported yet, see MeteoSwiss/anemoi-evaluation#5"
            )

    @property
    def device(self) -> torch.device:
        """Device the model runs on."""
        return self.runner.device

    def split_members(self, group: MemberGroup) -> None:
        """Run this rank's block of the members of a member group, reseeded by their index in the whole ensemble, and
        yield frames of every member and this rank's block of the requested variables, exchanged with the other
        ranks once per frame per dataset. `members` stays the ensemble size; the lead-0 frame is built locally."""
        if self.members < 2:
            raise ValueError("a member-parallel run needs an ensemble, the forecast has one member")
        if self.members < group.size:
            raise ValueError(
                f"{self.members} members cannot be split over {group.size} ranks: every rank needs a member"
            )
        device = self.run_config.get("device")
        configured = None if device is None else torch.device(str(device))
        if configured is not None and configured.type == "cuda" and configured.index is not None:
            # the same answer on every rank: only a task that sees just its own GPU may name it, as cuda:0
            visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if visible != 1 or configured.index != 0:
                raise ValueError(
                    f"device {device!r} names a GPU, and this task sees {visible}: set device: cuda, each rank then "
                    "takes its own"
                )
        self.member_group = group

    def _member_indices(self) -> range:
        """The members this process runs, by their index in the ensemble."""
        block = slice(0, self.members) if self.member_group is None else self.member_group.block(self.members)
        return range(block.start, block.stop)

    def _frames(
        self,
        init_time: datetime.datetime,
        step: datetime.timedelta,
        date: datetime.datetime,
        variables: Mapping[str, list[str]],
        outputs: list[dict],
        device: torch.device,
    ) -> dict[str, Frame]:
        """One frame per dataset of `variables` from the members' `outputs` (tensors, or numpy arrays from a
        temporal downscaler), copied into a new tensor. In a member group it is every member's frame of this rank's
        variables, and the rank's frame of every variable is freed here, before the frames are scored."""
        frames = {}
        for name, columns in variables.items():
            columns = list(columns)
            data = torch.stack(
                [torch.stack([torch.as_tensor(output[name]["fields"][v]) for v in columns]) for output in outputs]
            )
            data = data.float().to(device)
            if self.member_group is not None:
                from anemoi.evaluation.distributed import swap

                data = swap(data, self.members)
                columns = columns[self.member_group.block(len(columns))]
            frames[name] = Frame(init_time, step, date, columns, data)
        return frames

    def lead_times(self, lead_time: datetime.timedelta) -> list[datetime.timedelta]:
        """Lead times of the model's outputs up to `lead_time` (see `rollout_schedule`); warns when the model's last
        call computes beyond `lead_time`. A forecaster's `lead_time` must be a multiple of its timestep, a temporal
        downscaler's must be its window."""
        if self.temporal_downscaling:
            if lead_time != self.window:
                raise ValueError(
                    "a temporal downscaler is scored one window per init time: the lead time must be its window "
                    f"{frequency_to_string(self.window)}, got {frequency_to_string(lead_time)}"
                )
            return list(self.window_leads)
        n = len(self.output_offsets)
        regular = self.output_offsets == [(i + 1) * self.timestep for i in range(n)]
        if regular and self.output_horizon == n * self.timestep:
            lead_time_steps(lead_time, self.timestep)  # raises unless a positive multiple of the timestep
        elif lead_time <= datetime.timedelta(0):
            raise ValueError(f"lead time {lead_time} must be positive")
        leads = [lead for _, lead in self.rollout_schedule(lead_time)]
        if not leads:
            raise ValueError(
                f"lead time {frequency_to_string(lead_time)} is shorter than the model's first output offset "
                f"{frequency_to_string(self.output_offsets[0])}"
            )
        calls, computed = self.model_calls(lead_time)
        if computed > lead_time:
            LOG.warning(
                "lead time %s does not end the model's last call: the model computes %s (%d calls of %s) and the "
                "outputs beyond %s are discarded",
                frequency_to_string(lead_time),
                frequency_to_string(computed),
                calls,
                frequency_to_string(self.output_horizon),
                frequency_to_string(lead_time),
            )
        return leads

    def rollout_schedule(self, lead_time: datetime.timedelta) -> list[tuple[int, datetime.timedelta]]:
        """`(model call, lead time)` of every frame a rollout up to `lead_time` yields, in order: as anemoi-inference's
        runner, `ceil(lead_time / output_horizon)` calls, call `k` at `k * output_horizon + offset` for every output
        offset, the outputs beyond `lead_time` dropped."""
        shift = self.output_horizon
        return [
            (call, call * shift + offset)
            for call in range(math.ceil(lead_time / shift))
            for offset in self.output_offsets
            if call * shift + offset <= lead_time
        ]

    def model_calls(self, lead_time: datetime.timedelta) -> tuple[int, datetime.timedelta]:
        """Model calls a rollout up to `lead_time` makes and the lead time the last one computes."""
        if self.temporal_downscaling:
            return 1, self.window
        calls = math.ceil(lead_time / self.output_horizon)
        return calls, (calls - 1) * self.output_horizon + self.output_offsets[-1]

    def frames(
        self,
        init_time: datetime.datetime,
        lead_time: datetime.timedelta,
        variables: list[str],
        device: torch.device,
    ) -> Iterator[Frame]:
        """Yield one frame per output time with all members, advanced in lockstep on one runner; a model call yields
        `frames_per_pass` frames and the global RNG is reseeded with `member_seed` before every member's model call.

        Single-dataset checkpoints only; a multi-dataset checkpoint's rollout is `multi_frames()`."""
        name = self._resolve_dataset(None)
        for frames in self.multi_frames(init_time, lead_time, {name: variables}, device):
            yield frames[name]

    def multi_frames(
        self,
        init_time: datetime.datetime,
        lead_time: datetime.timedelta,
        variables: Mapping[str, list[str]],
        device: torch.device,
    ) -> Iterator[dict[str, Frame]]:
        """Yield, per output time, one frame per dataset of `variables`, all from one shared rollout.

        The runner reuses the dict it yields and the per-dataset `fields` dicts, replacing only the arrays, so every
        dataset's frame is built (which copies the fields into a new tensor) before the dict is yielded."""
        if self.temporal_downscaling:
            yield from self._window_frames(init_time, lead_time, variables, device)
            return
        names = list(variables)
        for name in names:
            select_variables(list(variables[name]), self.datasets[self._resolve_dataset(name)].variables)
        states = {name: self._initial_state(name, init_time) for name in self.dataset_names}
        self.runner.reference_date = None
        # The runner enters inference mode inside each generator; interleaved generators exit out of
        # order, so this outer guard is what restores the caller's mode.
        with self._quiet(), torch.inference_mode():
            members = self._member_indices()
            generators = [
                self.runner.run(input_states=states, lead_time=lead_time, return_numpy=False) for _ in members
            ]
            # reseeded before the `next()` that makes a call; every yielded step must be the scheduled one
            schedule = self.rollout_schedule(lead_time)
            try:
                for step_index in itertools.count():
                    outputs = []
                    call, lead = schedule[step_index] if step_index < len(schedule) else (None, None)
                    new_call = call is not None and (step_index == 0 or schedule[step_index - 1][0] != call)
                    for member, generator in zip(members, generators):
                        if new_call:
                            torch.manual_seed(member_seed(self.seed, init_time, member, call))
                        outputs.append(next(generator, None))
                    if all(output is None for output in outputs) and lead is None:
                        return
                    if any(output is None for output in outputs):
                        raise RuntimeError(f"members out of step, or fewer outputs than scheduled, at {init_time}")
                    step, date = outputs[0][names[0]]["step"], outputs[0][names[0]]["date"]
                    if any(output[names[0]]["step"] != step for output in outputs):
                        raise RuntimeError(f"members out of step at {init_time} + {step}")
                    if lead is None:
                        raise RuntimeError(f"the runner yielded more outputs than scheduled at {init_time} + {step}")
                    if step != lead:
                        raise RuntimeError(f"the runner yielded step {step} at {init_time}, lead time {lead} was due")
                    yield self._frames(init_time, step, date, variables, outputs, device)
            finally:
                for generator in generators:
                    generator.close()
                self._initial = {}

    def initial_frame(
        self,
        init_time: datetime.datetime,
        variables: list[str],
        device: torch.device,
        dataset: str | None = None,
    ) -> Frame:
        """Lead-0 frame of `dataset`: the init-time slice of every requested variable present in the input state
        (prognostics), NaN for the others (diagnostics have no analysis), identical for every member."""
        if self.temporal_downscaling:
            raise ValueError("a temporal downscaler has no lead-0 frame: lead 0 is its left bound, an input")
        name = self._resolve_dataset(dataset)
        view = self.datasets[name]
        select_variables(list(variables), view.variables)
        fields = self._initial_state(name, init_time)["fields"]
        data = np.full((len(variables), view.grid.n), np.nan, dtype=np.float32)
        for row, v in zip(data, variables):
            if v in fields:
                row[:] = fields[v][-1]
        data = torch.from_numpy(data).to(device)[None].repeat(self.members, 1, 1)
        return Frame(init_time, datetime.timedelta(0), init_time, list(variables), data)

    def _quiet(self) -> contextlib.AbstractContextManager:
        return quiet_inference() if self.quiet else contextlib.nullcontext()

    def _initial_state(self, dataset: str, init_time: datetime.datetime) -> dict:
        """Prognostic, constant and dynamic forcing inputs of `dataset` at `init_time` combined as `Runner.execute`
        does; kept until `multi_frames()` finishes so that `initial_frame()` and the rollout of one init time build it
        once (the runner copies the state and its fields dict before touching them, so the memo is never mutated)."""
        memo = self._initial.get(dataset)
        if memo is None or memo[0] != init_time:
            runner = self.runner
            inputs = (runner.prognostics_inputs, runner.constant_forcings_inputs, runner.dynamic_forcings_inputs)
            lagged = runner.checkpoint.multi_dataset_metadata[dataset].lagged
            state = runner._combine_states(
                *(create_input_state(source[dataset], init_time, lagged) for source in inputs)
            )
            self._initial[dataset] = (init_time, state)
        return self._initial[dataset][1]

    def _window_frames(
        self,
        init_time: datetime.datetime,
        lead_time: datetime.timedelta,
        variables: Mapping[str, list[str]],
        device: torch.device,
    ) -> Iterator[dict[str, Frame]]:
        """A temporal downscaler's `multi_frames`: one model call per member over the window starting at
        `init_time`, one frame per output offset, the left bound skipped.

        The members advance in lockstep as a forecaster's do and each member's model call is preceded by
        `member_seed(seed, init_time, member, 0)`, `member` its index in the ensemble. The runner yields the left bound
        first when it writes the initial state (no model call), so a member is reseeded before every `next()` until its
        first output; its fields are numpy arrays whatever `return_numpy` says, and are copied into tensors before the
        next `next()`. Any other step than the window's leads, or a missing one, raises."""
        names = list(variables)
        for name in names:
            select_variables(list(variables[name]), self.datasets[self._resolve_dataset(name)].variables)
        leads = self.lead_times(lead_time)
        states = {
            name: window_input_state(self.runner, name, init_time, self.window, self._window_constants)
            for name in self.dataset_names
        }
        self.runner.reference_date = None
        with self._quiet(), torch.inference_mode():
            members = self._member_indices()
            generators = [
                self.runner.run(input_states=states, lead_time=self.window, return_numpy=False) for _ in members
            ]
            called, yielded = [False] * len(members), []
            try:
                while True:
                    outputs = []
                    for local, (member, generator) in enumerate(zip(members, generators)):
                        if not called[local]:
                            torch.manual_seed(member_seed(self.seed, init_time, member, 0))
                        output = next(generator, None)
                        if output is not None and output[names[0]]["step"] > datetime.timedelta(0):
                            called[local] = True
                        outputs.append(output)
                    if all(output is None for output in outputs):
                        break
                    if any(output is None for output in outputs):
                        raise RuntimeError(f"members out of step at {init_time}")
                    first = outputs[0][names[0]]
                    step, date = first["step"], first["date"]
                    if any(output[names[0]]["step"] != step for output in outputs):
                        raise RuntimeError(f"members out of step at {init_time} + {step}")
                    if step == datetime.timedelta(0) or first.get("temporally_downscaled") is False:
                        continue  # the left bound, an input
                    if step not in leads:
                        raise RuntimeError(f"the runner yielded {step} at {init_time}, not a lead of the window")
                    yielded.append(step)
                    yield self._frames(init_time, step, date, variables, outputs, device)
            finally:
                for generator in generators:
                    generator.close()
        if yielded != leads:
            raise RuntimeError(f"the runner yielded the steps {yielded} at {init_time}, the window's leads are {leads}")

    def check_init_times(
        self, init_times: list[datetime.datetime], lead_time: datetime.timedelta, dataset: str | None = None
    ) -> None:
        """Raise unless every date a rollout reads exists in the runner's datasets and is not missing; every dataset
        of the checkpoint unless one is named. A temporal downscaler reads its inputs at the two bounds of the window
        only, and no boundary forcings."""
        from anemoi.inference.inputs.dataset import DatasetInput
        from anemoi.inference.inputs.empty import EmptyInput

        runner = self.runner
        names = self.dataset_names if dataset is None else [self._resolve_dataset(dataset)]
        for name in names:
            for kind, sources, rollout in (
                ("prognostics", runner.prognostics_inputs, False),
                ("dynamic forcings", runner.dynamic_forcings_inputs, True),
                ("boundary forcings", runner.boundary_forcings_inputs, True),
            ):
                source = sources[name]
                if isinstance(source, EmptyInput) or (self.temporal_downscaling and kind == "boundary forcings"):
                    continue
                if not isinstance(source, DatasetInput):
                    LOG.warning("%s input %s is not a dataset, init times are not checked against it", kind, source)
                    continue
                dates = source.ds_dates.astype("datetime64[us]").tolist()
                available, missing = set(dates), {dates[i] for i in source.ds.missing}
                for init_time in init_times:
                    if self.temporal_downscaling:
                        required = [init_time, init_time + self.window]
                    else:
                        required = required_dates(
                            init_time,
                            self.input_offsets,
                            self.output_offsets,
                            self.output_horizon,
                            lead_time if rollout else None,
                        )
                    absent = missing_dates(required, available, missing)
                    if absent:
                        raise ValueError(
                            f"init time {init_time}: dates {absent[:3]} are not in the {kind} dataset "
                            f"of {name!r} ({dates[0]} to {dates[-1]}, {len(missing)} missing)"
                        )

    def dataset_args_kwargs(self, dataset: str | None = None) -> tuple[tuple, dict]:
        """`open_dataset` arguments of the runner's prognostics input, for targets from the same dataset."""
        from anemoi.inference.inputs.dataset import DatasetInput

        source = self.runner.prognostics_inputs[self._resolve_dataset(dataset)]
        if not isinstance(source, DatasetInput):
            raise ValueError(f"prognostics input {source} is not an anemoi dataset, configure the targets explicitly")
        return tuple(source.open_dataset_args), dict(source.open_dataset_kwargs)

    def graph_node_attribute(self, name: str, nodes: str = "data", dataset: str | None = None) -> np.ndarray:
        """Per-node attribute of the checkpoint's training graph as a flat numpy array."""
        chosen = self._resolve_dataset(dataset)
        graph = getattr(self.runner.model, "graph_data", None)
        # Some anemoi-models builds keep one HeteroData per input dataset in a dict, others
        # expose the HeteroData itself; a HeteroData indexed by an unknown name silently returns
        # an empty NodeStorage, so the dict has to be unwrapped first. The single entry of a
        # one-dataset checkpoint is that dataset whatever it is keyed by; with several of them an
        # unknown key would serve another dataset's graph, on a grid of the wrong size, silently.
        if isinstance(graph, dict):
            if chosen not in graph and (self.multi_dataset or len(graph) > 1):
                raise ValueError(
                    f"the model's graph_data is keyed by {sorted(graph)}, which does not name the dataset {chosen!r}"
                )
            graph = graph[chosen] if chosen in graph else next(iter(graph.values()))
        # anemoi-core >= 42260df6a ("trim checkpoint") saves inference checkpoints without the raw graph.
        if graph is None:
            raise MissingGraphError(
                f"the checkpoint stores no training graph, so there is no {name!r} to read; "
                f"replace `graph_attribute: {name}` in the evaluation config"
            )
        # A multi-dataset graph names the node set after the dataset rather than 'data'; the fallback
        # is only from the default 'data', so an explicitly named node set is never substituted.
        types = tuple(getattr(graph, "node_types", ()))
        if nodes == "data" and nodes not in types and chosen in types:
            nodes = chosen
        if types and nodes not in types:
            raise ValueError(f"the graph of {chosen!r} has no node set {nodes!r}, it has {sorted(types)}")
        store = graph[nodes]
        if name not in store:
            raise ValueError(f"graph nodes {nodes!r} have no attribute {name!r}, available: {sorted(store.keys())}")
        return store[name].detach().cpu().numpy().reshape(-1)

    def variable_info(self, name: str) -> dict:
        """`param` and `level` of a variable from the checkpoint's typed variables; None when unknown."""
        return variable_info(self.typed_variables.get(name))

    def accumulation_periods(self) -> dict:
        """The period of each accumulated variable, as the checkpoint records it (its training dataset's)."""
        return self.datasets[self.dataset_name].accumulation_periods()

    def provenance(self) -> dict:
        """Result-file attrs identifying the checkpoint: its path, uuid and training run id; once the model is loaded,
        also the GPU model and the inference chunk counts, which are part of a run's numerical identity."""
        attrs = {"checkpoint": str(self.checkpoint)}
        attrs.update(
            {
                key: str(value)
                for key, value in (
                    ("checkpoint_uuid", self.checkpoint_uuid),
                    ("checkpoint_run_id", self.checkpoint_run_id),
                )
                if value is not None
            }
        )
        if self.model_loaded:
            if self.device.type == "cuda":
                attrs["gpu"] = torch.cuda.get_device_name(self.device)
            for kind, count in inference_chunks(self.runner.model).items():
                attrs[f"inference_chunks_{kind}"] = count
        return attrs

    @property
    def model_loaded(self) -> bool:
        """Whether the runner has loaded the model."""
        return self.runner is not None and "model" in self.runner.__dict__

    @property
    def stats(self) -> dict:
        """Forcings arrays computed, served from the cache, and computed off-grid without caching, over every
        dataset's cache."""
        if not self._forcings:
            return {}
        totals: dict[str, int] = {}
        for cache in self._forcings.values():
            for key, value in cache.stats.items():
                totals[key] = totals.get(key, 0) + value
        return {f"forcings_{key}": value for key, value in totals.items()}

    def to_config(self) -> dict:
        """The `forecast:` block that rebuilds this source."""
        return {
            "anemoi_inference": {
                **self.run_config,
                "members": self.members,
                "seed": self.seed,
                "quiet": self.quiet,
                "forcings_cache_bytes": self.forcings_cache_bytes,
            }
        }

    def describe(self, dataset: str | None = None) -> dict:
        """Checkpoint, model shape, the runner's dataset range and whether the model is loaded; loads nothing.

        Without `dataset`, a multi-dataset checkpoint reports the shared facts plus a `datasets` mapping of the
        per-dataset ones."""
        shared = self._describe_shared()
        if dataset is None and self.multi_dataset:
            return {**shared, "datasets": {name: self._describe_dataset(name) for name in self.dataset_names}}
        name = self._resolve_dataset(dataset)
        return {**shared, **self._describe_dataset(name)}

    def _describe_shared(self) -> dict:
        """The half of `describe` that is the same for every dataset of the checkpoint."""
        info = {
            "source": "anemoi_inference",
            **self.provenance(),
            "task": self.task,
            **({"window": frequency_to_string(self.window)} if self.temporal_downscaling else {}),
            "timestep": frequency_to_string(self.timestep),
            "multi_step_input": self.multi_step_input,
            "multi_step_output": self.multi_step_output,
            **(
                {}
                if self.temporal_downscaling
                else {
                    "input_offsets": [frequency_to_string(offset) for offset in self.input_offsets],
                    "output_offsets": [frequency_to_string(offset) for offset in self.output_offsets],
                }
            ),
            "output_horizon": frequency_to_string(self.output_horizon),
            "members": self.members,
            "seed": self.seed,
            "device": str(self.device),
            "model_loaded": self.model_loaded,
            "forcings_cache_bytes": self.forcings_cache_bytes,
        }
        if self.model_loaded:
            info["inference_chunks"] = inference_chunks(self.runner.model)
        skipped = [name for name in self.dataset_names if name not in (self.scored_datasets or self.dataset_names)]
        if skipped:
            # the run scores a subset: the others are still predicted, and the runner still holds their state
            info["datasets_scored"] = list(self.scored_datasets)
            info["datasets_predicted_only"] = skipped
            info["predicted_only_state_bytes"] = sum(
                self._describe_dataset(name)["per_member_state_bytes"] for name in skipped
            )
        with contextlib.suppress(OSError):
            info["checkpoint_bytes"] = os.path.getsize(self.checkpoint)
        return info

    def _describe_dataset(self, name: str) -> dict:
        """The half of `describe` that is this dataset's own."""
        from anemoi.inference.inputs.dataset import DatasetInput

        view = self.datasets[name]
        inputs = len(self.runner.checkpoint.multi_dataset_metadata[name].variable_to_input_tensor_index)
        outputs = self.multi_step_output * len(view.variables)
        info = {
            "nodes": view.grid.n,
            "variables": len(view.variables),
            "per_member_state_bytes": (self.multi_step_input * inputs + outputs) * view.grid.n * 4,
        }
        source = self.runner.prognostics_inputs[name]
        if isinstance(source, DatasetInput):
            dates = source.ds_dates.astype("datetime64[us]").tolist()
            info["dataset"] = {
                "first": dates[0].isoformat(),
                "last": dates[-1].isoformat(),
                "count": len(dates),
                "missing": len(source.ds.missing),
            }
        return info

    def close(self) -> None:
        """Release the runner and the model."""
        self._initial = {}
        if self.temporal_downscaling:
            self._window_constants = {}
        self.runner = None
        for cache in self._forcings.values():
            cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
