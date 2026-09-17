"""The evaluation driver: forecast and target sources in, an aggregation state out."""

from __future__ import annotations

import contextlib
import datetime
import itertools
import json
import logging
import math
import time
from collections.abc import Iterator
from functools import cached_property
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import yaml
from anemoi.utils.dates import as_datetime
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta
from anemoi.utils.humanize import compress_dates

from anemoi.evaluation import metrics as metrics_module
from anemoi.evaluation import regions as regions_module
from anemoi.evaluation.aggregation import AggregationState
from anemoi.evaluation.aggregation import Aggregator
from anemoi.evaluation.binning import Binning
from anemoi.evaluation.config import AnemoiDatasetConfig
from anemoi.evaluation.config import BinsConfig
from anemoi.evaluation.config import EvaluationConfig
from anemoi.evaluation.config import InitTimesRange
from anemoi.evaluation.config import build_region
from anemoi.evaluation.config import build_weights
from anemoi.evaluation.config import config_dataset_names
from anemoi.evaluation.config import load_config
from anemoi.evaluation.config import per_dataset
from anemoi.evaluation.frame import Frame
from anemoi.evaluation.metrics import Metric
from anemoi.evaluation.sources.anemoi_dataset import DatasetTargets
from anemoi.evaluation.sources.anemoi_inference import DatasetForecast
from anemoi.evaluation.sources.anemoi_inference import InferenceForecastSource
from anemoi.evaluation.sources.base import ClimatologySource
from anemoi.evaluation.sources.base import ForecastSource
from anemoi.evaluation.sources.base import MissingTargetError
from anemoi.evaluation.sources.base import TargetSource
from anemoi.evaluation.sources.base import select_variables
from anemoi.evaluation.sources.climatology import ArrayClimatology
from anemoi.evaluation.sources.persistence import PersistenceForecastSource

LOG = logging.getLogger(__name__)
PHASES = ("model", "target", "statistics", "total")
DEFAULT_DATASET = "data"
PACKAGES = ("anemoi-evaluation", "anemoi-inference", "anemoi-models", "anemoi-graphs", "anemoi-datasets", "torch")
# benchmarks.md sections 7 and 9: about 90 s of cold-node startup plus about 15 s of first-call overheads.
STARTUP_SECONDS = 105.0


def init_times(
    start: str | datetime.datetime, end: str | datetime.datetime, frequency: str | datetime.timedelta
) -> list:
    """Dates from `start` to `end` inclusive, every `frequency`."""
    return InitTimesRange(start=start, end=end, frequency=frequency).resolve()


class PhaseTimer:
    """Wall time per phase between successive `mark` calls; synchronises CUDA first so GPU work is counted."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device
        self.totals: dict[str, float] = {}
        self._last = self._now()

    def mark(self, phase: str) -> None:
        """Charge the time since the previous mark to `phase`."""
        now = self._now()
        self.totals[phase] = self.totals.get(phase, 0.0) + now - self._last
        self._last = now

    def skip(self) -> None:
        """Discard the time since the previous mark: it belongs to another timer (the shared rollout, or another
        dataset of the same rollout)."""
        self._last = self._now()

    def _now(self) -> float:
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()


class Evaluation:
    """Scores a forecast source against a target source over init times; the API behind the YAML config.

    Construction validates the arguments and checks the init times against the sources; weights and regions given
    as specs are resolved on first use (`weights`, `regions`, `aggregator`), so a model graph is only loaded by `run()`.

    `dataset` names the dataset of a multi-dataset checkpoint this evaluation scores; it is written to the result as
    the `dataset` attr and is None for a checkpoint that has only one dataset, whose results are unchanged.
    """

    def __init__(
        self,
        forecast: ForecastSource,
        targets: TargetSource,
        init_times: list[datetime.datetime],
        lead_time: datetime.timedelta | str,
        metrics: list[Metric | str | dict],
        variables: list[str] | None = None,
        weights: np.ndarray | dict | None = None,
        regions: dict[str, np.ndarray | str | dict] | None = None,
        bins: Binning | dict | str | None = None,
        device: torch.device | str | None = None,
        on_missing_target: str = "skip",
        include_lead_zero: bool = False,
        climatology: ClimatologySource | None = None,
        dataset: str | None = None,
    ) -> None:
        self.dataset = dataset
        self.forecast = forecast
        self.targets = targets
        self.init_times = [as_datetime(date) for date in init_times]
        self.lead_time = frequency_to_timedelta(lead_time)
        self.lead_times = forecast.lead_times(self.lead_time)
        if include_lead_zero and not forecast.supports_lead_zero:
            raise ValueError(f"{type(forecast).__name__} cannot produce lead-0 frames")
        self.include_lead_zero = include_lead_zero
        if include_lead_zero:
            self.lead_times = [datetime.timedelta(0), *self.lead_times]
        forecast.grid.check_compatible(targets.grid)
        targets.prefetch_hint(forecast.frames_per_pass)
        self.variables = self._resolve_variables(variables)
        self.metrics = metrics_module.build(metrics, self.variables, members=forecast.members)
        needed = metrics_module.min_members(self.metrics)
        if needed > forecast.members:
            raise ValueError(f"the metrics need at least {needed} members, the forecast source has {forecast.members}")
        self.climatology = climatology
        if "climatology" in metrics_module.required_aux(self.metrics) and climatology is None:
            raise ValueError("the anomaly metrics need a climatology source")
        if climatology is not None:
            forecast.grid.check_compatible(climatology.grid)
            select_variables(self.variables, climatology.variables)
        if on_missing_target not in ("skip", "raise"):
            raise ValueError(f"on_missing_target must be 'skip' or 'raise', got {on_missing_target!r}")
        self.on_missing_target = on_missing_target
        if isinstance(bins, str):
            bins = {"time": bins}
        if bins is None or isinstance(bins, dict):
            bins = BinsConfig.model_validate(bins or {}).build(self.init_times, self.lead_times)
        self.binning = bins
        self.device = torch.device(device or forecast.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        forecast.check_init_times(self.init_times, self.lead_time)

        grid = forecast.grid
        if weights is None:
            weights = {"graph_attribute": "area_weight"} if forecast.has_graph else {"uniform": {}}
        self._weights_spec = weights if isinstance(weights, dict) else None
        self._weights = (
            weights if isinstance(weights, dict) else grid.check_shape(np.asarray(weights, dtype=np.float64), "weights")
        )
        regions = dict(regions or {"global": "all"})
        if not regions:
            raise ValueError("at least one region is required")
        self._region_specs = {name: spec if isinstance(spec, (str, dict)) else None for name, spec in regions.items()}
        self._regions = {
            name: spec
            if isinstance(spec, (str, dict))
            else grid.check_shape(np.asarray(spec, dtype=bool), f"region {name!r}")
            for name, spec in regions.items()
        }
        self.timing: dict[str, float] = {}
        self.model_calls = 0
        self.climatology_nonfinite: dict[str, int] = {}
        self.peak_memory_bytes: int | None = None
        self._config: EvaluationConfig | None = None

    @cached_property
    def weights(self) -> np.ndarray:
        """Node weights (N,) float64, resolved from the spec on first use."""
        if isinstance(self._weights, dict):
            return build_weights(self._weights, self.forecast.grid, self.forecast)
        return self._weights

    @cached_property
    def regions(self) -> dict[str, np.ndarray]:
        """Region masks (N,) bool by name, resolved from the specs on first use."""
        grid = self.forecast.grid
        return {
            name: build_region(spec, grid, self.forecast, self.targets) if isinstance(spec, (str, dict)) else spec
            for name, spec in self._regions.items()
        }

    @cached_property
    def aggregator(self) -> Aggregator:
        """The aggregator over the resolved weights and regions."""
        _, masks = regions_module.stack(self.regions, self.forecast.grid.n)
        statistics = metrics_module.unique_statistics(self.metrics)
        return Aggregator(self.weights, masks, statistics, self.binning, self.device)

    def _resolve_variables(self, variables: list[str] | None) -> list[str]:
        if variables is None:
            variables = [name for name in self.forecast.variables if name in self.targets.variables]
        else:
            select_variables(list(variables), self.forecast.variables)
            select_variables(list(variables), self.targets.variables)
        if not variables:
            raise ValueError("no variables to evaluate")
        return list(variables)

    def variable_coords(self) -> dict[str, list]:
        """`param` and `level` of each variable from the sources' typed variables; '' and NaN when unknown."""
        params, levels = [], []
        for name in self.variables:
            info = {}
            for source in (self.targets, self.forecast):
                info.update({key: value for key, value in source.variable_info(name).items() if value is not None})
            params.append(str(info.get("param", "")))
            try:
                levels.append(float(info["level"]))
            except (KeyError, TypeError, ValueError):
                levels.append(float("nan"))
        return {"param": params, "level": levels}

    def model_calls_per_init(self) -> tuple[int, datetime.timedelta]:
        """Model calls one init time needs and the lead time they compute, from the source's frames per call."""
        steps = [lead for lead in self.lead_times if lead > datetime.timedelta(0)]
        calls = math.ceil(len(steps) / self.forecast.frames_per_pass)
        return calls, calls * self.forecast.frames_per_pass * (steps[0] if steps else self.lead_time)

    def time_estimate(self, step_time: float, n_init: int, shards: int = 1) -> dict:
        """Model and wall seconds of a shard of `n_init` init times and of the whole run over `shards` shards, from
        `step_time` seconds of model time per model call per member; wall adds `STARTUP_SECONDS` per shard."""
        if not (step_time > 0 and math.isfinite(step_time)):
            raise ValueError(f"step_time must be a positive number of seconds, got {step_time!r}")
        if shards < 1:
            raise ValueError(f"shards must be at least 1, got {shards}")
        calls, _ = self.model_calls_per_init()
        per_init = calls * self.forecast.members * step_time
        largest = math.ceil(len(self.init_times) / shards)
        return {
            "step_time": step_time,
            "model_per_init_s": per_init,
            "shard": {"model_s": n_init * per_init, "wall_s": n_init * per_init + STARTUP_SECONDS},
            "run": {
                "gpu_hours": round(len(self.init_times) * per_init / 3600, 2),
                "wall_per_shard_s": largest * per_init + STARTUP_SECONDS,
            },
            "startup_s": STARTUP_SECONDS,
            "note": "model calls x members x step time, plus one process startup per shard; gpu_hours is model "
            "time and the reserved budget is about shards x wall_per_shard_s. Target reads (assumed hidden by "
            "prefetch), statistics, the lead-0 frame and writing the result are not counted.",
        }

    def shard(self, index: int, count: int) -> list[datetime.datetime]:
        """Init times of shard `index` of `count`: every `count`-th init time from `index`, for `run(init_times=...)`."""
        if not 0 <= index < count:
            raise ValueError(f"shard index {index} is out of range for {count} shards")
        return self.init_times[index::count]

    def plan(
        self, init_times: list[datetime.datetime] | None = None, step_time: float | None = None, shards: int = 1
    ) -> dict:
        """What a run over `init_times` would do, without loading a model or resolving weights and regions;
        `step_time` adds the `time` estimate of `time_estimate`, for a run split over `shards` shards."""
        init_times = list(self.init_times if init_times is None else init_times)
        valid_times = [init + lead for init in init_times for lead in self.lead_times]
        v, n, r = len(self.variables), self.forecast.grid.n, len(self._regions)
        row = v * n * 4
        forecast, targets = self.forecast.describe(), self.targets.describe()
        calls, computed = self.model_calls_per_init()
        plan = {
            "forecast": forecast,
            "targets": targets,
            "device": str(self.device),
            "init_times": {"count": len(init_times), "dates": ", ".join(compress_dates(init_times))},
            "lead_times": {
                "count": len(self.lead_times),
                "first": frequency_to_string(self.lead_times[0]),
                "last": frequency_to_string(self.lead_times[-1]),
            },
            "model_calls": calls * len(init_times),
            "computed_lead_time": frequency_to_string(computed),
            "include_lead_zero": self.include_lead_zero,
            "variables": {"names": list(self.variables), **self.variable_coords()},
            "metrics": [metric.spec for metric in self.metrics],
            "statistics": sorted(metrics_module.unique_statistics(self.metrics)),
            "weights": self._weights_spec if self._weights_spec is not None else "array",
            "regions": {name: "array" if spec is None else spec for name, spec in self._region_specs.items()},
            "bins": {"time": self.binning.kind, "by": self.binning.by, "count": len(self.binning.coords)},
            "on_missing_target": self.on_missing_target,
            **({"climatology": self.climatology.describe()} if self.climatology is not None else {}),
            "frames": {
                "total": len(valid_times),
                "with_targets": sum(self.targets.available(t) for t in valid_times),
                "distinct_valid_times": len(set(valid_times)),
            },
            "bytes": {
                "target_to_device": len(valid_times) * row,
                "target_rows_read": len(valid_times) * targets.get("row_bytes", row),
                "cache_for_all_rows": len(set(valid_times)) * row,
                "frame": self.forecast.members * row,
                "aggregator_weights": n * r * 8,
                "statistics_temporaries": 3 * v * n * 8,
                # `predicted_only_state_bytes` is the state of the datasets the run does not score: the runner
                # holds one for every dataset of the checkpoint, scored or not
                **{
                    key: forecast[key]
                    for key in ("checkpoint_bytes", "per_member_state_bytes", "predicted_only_state_bytes")
                    if key in forecast
                },
                **(
                    {"climatology": plan_climatology_bytes}
                    if (plan_climatology_bytes := self._climatology_bytes())
                    else {}
                ),
            },
            **({"time": self.time_estimate(step_time, len(init_times), shards)} if step_time is not None else {}),
        }
        if computed > self.lead_time:
            plan["note"] = (
                f"the last of the {calls} model calls per init time computes up to {plan['computed_lead_time']}; "
                f"the outputs beyond {frequency_to_string(self.lead_time)} are computed and discarded"
            )
        return plan

    def _climatology_bytes(self) -> int:
        if self.climatology is None:
            return 0
        return int(self.climatology.describe().get("bytes", 0))

    def _aux(self, frame: Frame) -> dict[str, torch.Tensor] | None:
        if self.climatology is None:
            return None
        return {"climatology": self.climatology.frame(frame.valid_time, self.variables, self.device)}

    def run(self, init_times: list[datetime.datetime] | None = None) -> AggregationState:
        """Evaluate `init_times` (default: all) and return the state, with result attrs and timing totals."""
        init_times = list(self.init_times if init_times is None else init_times)
        state = self.begin()
        calls_per_init, _ = self.model_calls_per_init()
        cuda = self.device.type == "cuda"
        if cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        with torch.inference_mode():
            for init_time in init_times:
                start = time.perf_counter()
                timer = PhaseTimer(self.device)
                for frame in self.frames(init_time):
                    self.add_frame(state, frame, timer)
                timer.mark("model")
                self.charge(timer, start, init_time, calls_per_init)
        self.finish(state, init_times, cuda)
        return state

    def begin(self) -> AggregationState:
        """A fresh aggregation state, with the timing counters reset; the first half of `run`, for a driver that
        feeds the frames itself (see `MultiEvaluation`)."""
        if self.climatology is not None:
            self.climatology_nonfinite = self.climatology.nonfinite_nodes(self.variables)
            if any(self.climatology_nonfinite.values()):
                LOG.warning(
                    "climatology has non-finite nodes, excluded from the anomaly statistics only: %s",
                    {name: count for name, count in self.climatology_nonfinite.items() if count},
                )
        self.timing = {}
        self.model_calls = 0
        return self.aggregator.new_state(
            self.lead_times, self.variables, list(self.regions), self.forecast.members, self.variable_coords()
        )

    def frames(self, init_time: datetime.datetime) -> Iterator[Frame]:
        """The forecast frames of one init time, the lead-0 one first when it is scored; announces the valid times
        to the target source first."""
        self.targets.prefetch([init_time + lead for lead in self.lead_times], self.variables)
        frames = self.forecast.frames(init_time, self.lead_time, self.variables, self.device)
        if self.include_lead_zero:
            frames = itertools.chain(self._initial_frames(init_time), frames)
        return frames

    def add_frame(self, state: AggregationState, frame: Frame, timer: PhaseTimer, charge_model: bool = True) -> None:
        """Score one frame into `state`; a frame whose target is missing is skipped or raises.

        `charge_model` False leaves the time before the target read uncharged: it is another timer's, which is how
        `MultiEvaluation` keeps one dataset's target reads out of another's `model` (see `_run_shared`)."""
        target = self._target(frame, timer, charge_model)
        if target is None:
            return
        self.aggregator.add(state, frame, target, self._aux(frame))
        timer.mark("statistics")

    def charge(
        self,
        timer: PhaseTimer,
        start: float,
        init_time: datetime.datetime,
        calls_per_init: int,
        shared: dict[str, float] | None = None,
    ) -> None:
        """Add one init time's phase times to the totals and log them.

        `shared` are the phases of a rollout this evaluation shares with others: they are added to the timer as
        they are (not divided), and `total` is then this dataset's own phases plus them rather than the wall clock
        of the whole init time, which would be the same number in every dataset's result file."""
        if shared is None:
            timer.totals["total"] = time.perf_counter() - start
        else:
            for phase, seconds in shared.items():
                timer.totals[phase] = timer.totals.get(phase, 0.0) + seconds
            timer.totals["total"] = sum(timer.totals.values())
        for phase, seconds in timer.totals.items():
            self.timing[phase] = self.timing.get(phase, 0.0) + seconds
        self.model_calls += calls_per_init
        LOG.info(
            "%s: %s (%d model calls)",
            init_time.isoformat(),
            ", ".join(f"{p} {timer.totals.get(p, 0.0):.2f}s" for p in PHASES),
            calls_per_init,
        )

    def finish(self, state: AggregationState, init_times: list[datetime.datetime], cuda: bool = False) -> None:
        """Log the totals and put the result attrs on `state`; the second half of `run`."""
        self.peak_memory_bytes = torch.cuda.max_memory_allocated(self.device) if cuda else None
        LOG.info(
            "%d init times: %s%s",
            len(init_times),
            ", ".join(f"{p} {self.timing.get(p, 0.0):.2f}s" for p in PHASES),
            f", peak GPU memory {self.peak_memory_bytes / 2**30:.2f} GiB" if cuda else "",
        )
        stats = self.targets.stats
        if stats:
            LOG.info(
                "targets: %d rows read in %.2fs%s, %d cache hits",
                stats.get("reads", 0),
                stats.get("read_seconds", 0.0),
                " on the worker"
                if stats.get("reads", 0) and self.timing.get("target", 0.0) < stats.get("read_seconds", 0.0)
                else "",
                stats.get("cache_hits", 0),
            )
        state.attrs.update(self.attrs(init_times, self.timing))

    def pairs(self, init_times: list[datetime.datetime] | None = None) -> Iterator[tuple[Frame, torch.Tensor]]:
        """Yield `(frame, target)` pairs for a custom loop, without aggregation; the loop runs in inference mode."""
        with torch.inference_mode():
            for init_time in self.init_times if init_times is None else init_times:
                yield from self._pairs_one(init_time, PhaseTimer())

    def _initial_frames(self, init_time: datetime.datetime) -> Iterator[Frame]:
        try:
            yield self.forecast.initial_frame(init_time, self.variables, self.device)
        except MissingTargetError as error:
            LOG.warning("%s: no lead-0 frame for init %s", error, init_time)

    def _target(self, frame: Frame, timer: PhaseTimer, charge_model: bool = True) -> torch.Tensor | None:
        """The target of `frame`, None when it is missing and missing targets are skipped."""
        timer.mark("model") if charge_model else timer.skip()
        try:
            target = self.targets.frame(frame.valid_time, self.variables, self.device)
        except MissingTargetError as error:
            if self.on_missing_target == "raise":
                raise
            LOG.warning("%s: skipping init %s lead %s", error, frame.init_time, frequency_to_string(frame.lead_time))
            timer.mark("target")
            return None
        timer.mark("target")
        return target

    def _pairs_one(self, init_time: datetime.datetime, timer: PhaseTimer) -> Iterator[tuple[Frame, torch.Tensor]]:
        for frame in self.frames(init_time):
            target = self._target(frame, timer)
            if target is None:
                continue
            yield frame, target
            timer.mark("statistics")
        timer.mark("model")

    def close(self) -> None:
        """Release the sources."""
        self.forecast.close()
        self.targets.close()
        if self.climatology is not None:
            self.climatology.close()

    def attrs(self, init_times: list[datetime.datetime] | None = None, timing: dict | None = None) -> dict:
        """Result-file attributes: provenance of the run, timing totals in seconds (`time_target_s` is the main thread's
        wait for targets, `time_target_read_s` the seconds the target source's worker spent reading) and read counters."""
        timing = self.timing if timing is None else timing
        attrs = dict(self.forecast.provenance())
        if self.dataset is not None:
            attrs["dataset"] = self.dataset
        attrs["init_times"] = ", ".join(compress_dates(list(self.init_times if init_times is None else init_times)))
        attrs["lead_time"] = frequency_to_string(self.lead_time)
        try:
            attrs["config"] = yaml.safe_dump(self.to_config(), sort_keys=False)
        except ValueError:
            pass
        attrs["package_versions"] = json.dumps(package_versions())
        attrs.update({f"time_{phase}_s": float(timing.get(phase, 0.0)) for phase in PHASES})
        attrs["model_calls"] = int(self.model_calls)
        if self.climatology is not None:
            attrs["climatology"] = json.dumps(self.climatology.describe(), default=str)
            attrs["climatology_nonfinite_nodes"] = json.dumps(self.climatology_nonfinite)
        prefetch = self.targets.describe().get("prefetch")
        if prefetch is not None:
            attrs["target_prefetch"] = int(prefetch)
        if self.peak_memory_bytes is not None:
            attrs["peak_gpu_memory_bytes"] = int(self.peak_memory_bytes)
        stats = self.targets.stats
        if stats:
            attrs["time_target_read_s"] = float(stats.get("read_seconds", 0.0))
            attrs["target_reads"] = int(stats.get("reads", 0))
            attrs["target_cache_hits"] = int(stats.get("cache_hits", 0))
        for key, value in getattr(self.forecast, "stats", {}).items():
            attrs[key] = int(value)
        return attrs

    @classmethod
    def from_config(
        cls,
        config: str | Path | dict | EvaluationConfig,
        forecast: ForecastSource | None = None,
        targets: TargetSource | dict[str, TargetSource] | None = None,
        climatology: ClimatologySource | dict[str, ClimatologySource] | None = None,
    ) -> Evaluation | MultiEvaluation:
        """Build from a YAML path, a dict or a config; the source arguments override the configured sources.

        A checkpoint trained on multiple datasets gives a `MultiEvaluation`: one evaluation per dataset, sharing one
        runner and one rollout. `targets` and `climatology` may then be given per dataset as `{name: source}`."""
        return build(config, forecast, targets, climatology)

    def to_config(self) -> dict:
        """YAML-ready dict; raises when sources, weights or regions were given as objects without a spec."""
        if self._config is not None:
            return self._config.model_dump(mode="json", exclude_none=True)
        if self._weights_spec is None or None in self._region_specs.values():
            raise ValueError("weights and regions given as arrays cannot be serialised, give specs instead")
        config = EvaluationConfig.model_validate(
            {
                "forecast": self.forecast.to_config(),
                "targets": self.targets.to_config(),
                "lead_time": frequency_to_string(self.lead_time),
                "init_times": {"dates": [date.isoformat() for date in self.init_times]},
                "variables": self.variables,
                "metrics": [metric.spec for metric in self.metrics],
                "weights": self._weights_spec,
                "regions": self._region_specs,
                "bins": {"time": self.binning.kind, "by": self.binning.by},
                "on_missing_target": self.on_missing_target,
                "include_lead_zero": self.include_lead_zero,
                "climatology": self.climatology.to_config() if self.climatology is not None else None,
            }
        )
        return config.model_dump(mode="json", exclude_none=True)


class MultiEvaluation:
    """A multi-dataset run: one `Evaluation` per decoded dataset, sharing one runner and one rollout.

    Every dataset has its own targets, variables, weights, regions, climatology, aggregator and result; the datasets
    only share the model call, the init times and the lead times. `run()` returns one aggregation state per dataset.
    """

    def __init__(self, evaluations: dict[str, Evaluation], forecast: InferenceForecastSource | None = None) -> None:
        if not evaluations:
            raise ValueError("a multi-dataset evaluation needs at least one dataset")
        self.evaluations = dict(evaluations)
        self.forecast = forecast
        first = next(iter(self.evaluations.values()))
        self.init_times = first.init_times
        self.lead_time = first.lead_time
        self.lead_times = first.lead_times
        self.include_lead_zero = first.include_lead_zero
        self.device = first.device
        for name, evaluation in self.evaluations.items():
            if evaluation.init_times != self.init_times or evaluation.lead_times != self.lead_times:
                raise ValueError(f"dataset {name!r} does not share the init times and lead times of the others")
        self.timing: dict[str, float] = {}
        self.model_calls = 0
        self.peak_memory_bytes: int | None = None
        self._config: EvaluationConfig | None = None

    @property
    def dataset_names(self) -> list[str]:
        """The datasets scored, in the checkpoint's order."""
        return list(self.evaluations)

    @property
    def metrics(self) -> dict[str, list[Metric]]:
        """The metrics of each dataset, for `AggregationState.to_xarray`."""
        return {name: evaluation.metrics for name, evaluation in self.evaluations.items()}

    @property
    def variables(self) -> dict[str, list[str]]:
        """The variables scored in each dataset."""
        return {name: evaluation.variables for name, evaluation in self.evaluations.items()}

    def shard(self, index: int, count: int) -> list[datetime.datetime]:
        """Init times of shard `index` of `count`, the same for every dataset."""
        return next(iter(self.evaluations.values())).shard(index, count)

    def plan(
        self, init_times: list[datetime.datetime] | None = None, step_time: float | None = None, shards: int = 1
    ) -> dict:
        """One plan per dataset under `datasets`, plus the shared init times and device.

        The datasets share the model call, so every dataset's `time` estimate is the whole run's, not a share
        of it. `predicted_only` names the checkpoint's datasets the run does not score: the model still predicts
        them, they simply have no targets, no aggregator and no result."""
        init_times = list(self.init_times if init_times is None else init_times)
        predicted_only = [name for name in getattr(self.forecast, "dataset_names", ()) if name not in self.evaluations]
        return {
            "datasets": {
                name: evaluation.plan(init_times, step_time, shards) for name, evaluation in self.evaluations.items()
            },
            "scored": self.dataset_names,
            **({"predicted_only": predicted_only} if predicted_only else {}),
            "device": str(self.device),
            "init_times": {"count": len(init_times), "dates": ", ".join(compress_dates(init_times))},
        }

    def run(self, init_times: list[datetime.datetime] | None = None) -> dict[str, AggregationState]:
        """Evaluate `init_times` (default: all) and return one aggregation state per dataset."""
        init_times = list(self.init_times if init_times is None else init_times)
        if self.forecast is None:  # no shared rollout (a persistence baseline per dataset)
            states = {name: evaluation.run(init_times) for name, evaluation in self.evaluations.items()}
            self.timing = {}
            self._sum_timing()
        else:
            states = self._run_shared(init_times)  # sets self.timing, whose model phase is shared
            self._sum_timing(shared=("model", "total"))
        for name, state in states.items():
            state.attrs["dataset"] = name
        self.model_calls = next(iter(self.evaluations.values())).model_calls
        return states

    def _sum_timing(self, shared: tuple[str, ...] = ()) -> None:
        """The run's timing: the per-dataset phases summed, the shared ones (already in `self.timing`) left alone."""
        for evaluation in self.evaluations.values():
            for phase, seconds in evaluation.timing.items():
                if phase in shared:
                    continue
                self.timing[phase] = self.timing.get(phase, 0.0) + seconds

    def _run_shared(self, init_times: list[datetime.datetime]) -> dict[str, AggregationState]:
        """One rollout per init time, its per-dataset frames scored by the dataset's own evaluation.

        The rollout is timed once, by `shared`, and charged whole to every dataset's `model`; each dataset's own
        timer sees only its target reads and its statistics, so no dataset is charged for another's work. The
        run's `model` is the rollout's, counted once."""
        names = self.dataset_names
        states = {name: evaluation.begin() for name, evaluation in self.evaluations.items()}
        calls_per_init, _ = next(iter(self.evaluations.values())).model_calls_per_init()
        cuda = self.device.type == "cuda"
        if cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        self.timing = {}
        wall = time.perf_counter()
        with torch.inference_mode():
            for init_time in init_times:
                start = time.perf_counter()
                shared = PhaseTimer(self.device)
                timers = {name: PhaseTimer(self.device) for name in names}
                for name in names:
                    evaluation = self.evaluations[name]
                    evaluation.targets.prefetch(
                        [init_time + lead for lead in evaluation.lead_times], evaluation.variables
                    )
                    if evaluation.include_lead_zero:
                        for frame in evaluation._initial_frames(init_time):
                            evaluation.add_frame(states[name], frame, timers[name], charge_model=False)
                shared.skip()  # the prefetch calls and the lead-0 frames are not the rollout
                stream = self.forecast.multi_frames(
                    init_time, self.lead_time, {name: self.evaluations[name].variables for name in names}, self.device
                )
                with contextlib.closing(stream):
                    for frames in stream:
                        shared.mark("model")
                        for name in names:
                            self.evaluations[name].add_frame(states[name], frames[name], timers[name], False)
                        shared.skip()
                shared.mark("model")
                for name in names:
                    self.evaluations[name].charge(timers[name], start, init_time, calls_per_init, shared.totals)
                for phase, seconds in shared.totals.items():
                    self.timing[phase] = self.timing.get(phase, 0.0) + seconds
        for name in names:
            self.evaluations[name].finish(states[name], init_times, cuda)
        self.timing["total"] = time.perf_counter() - wall
        self.peak_memory_bytes = torch.cuda.max_memory_allocated(self.device) if cuda else None
        return states

    def pairs(self, init_times: list[datetime.datetime] | None = None) -> Iterator[tuple[str, Frame, torch.Tensor]]:
        """Yield `(dataset, frame, target)` triples for a custom loop, without aggregation."""
        if self.forecast is None:
            for name, evaluation in self.evaluations.items():
                for frame, target in evaluation.pairs(init_times):
                    yield name, frame, target
            return
        names = self.dataset_names
        with torch.inference_mode():
            for init_time in self.init_times if init_times is None else init_times:
                timer = PhaseTimer()
                for name in names:
                    evaluation = self.evaluations[name]
                    if evaluation.include_lead_zero:
                        for frame in evaluation._initial_frames(init_time):
                            target = evaluation._target(frame, timer)
                            if target is not None:
                                yield name, frame, target
                stream = self.forecast.multi_frames(
                    init_time, self.lead_time, {name: self.evaluations[name].variables for name in names}, self.device
                )
                with contextlib.closing(stream):  # a caller that breaks out closes the runner generators
                    for frames in stream:
                        for name in names:
                            target = self.evaluations[name]._target(frames[name], timer)
                            if target is not None:
                                yield name, frames[name], target

    def attrs(self, init_times: list[datetime.datetime] | None = None) -> dict[str, dict]:
        """The result attrs of each dataset."""
        return {name: evaluation.attrs(init_times) for name, evaluation in self.evaluations.items()}

    def to_config(self) -> dict:
        """The YAML-ready config of the run."""
        if self._config is not None:
            return self._config.model_dump(mode="json", exclude_none=True)
        raise ValueError("a multi-dataset evaluation built by hand cannot be written to a config")

    def close(self) -> None:
        """Release every dataset's sources and the shared runner."""
        for evaluation in self.evaluations.values():
            evaluation.close()
        if self.forecast is not None:
            self.forecast.close()


def _forecast_views(
    config: EvaluationConfig,
    forecast: ForecastSource | None,
    names: list[str],
    targets: dict[str, TargetSource],
) -> tuple[dict[str, ForecastSource], InferenceForecastSource | None]:
    """One forecast source per dataset and, when they share a runner, that runner's source."""
    if isinstance(forecast, InferenceForecastSource) and forecast.multi_dataset:
        return {name: forecast.datasets[name] for name in names}, forecast
    if forecast is not None:
        if len(names) > 1 and not isinstance(forecast, PersistenceForecastSource):
            LOG.warning(
                "%s is not a multi-dataset source: its rollout is run once per dataset (%s), %d times in all",
                type(forecast).__name__,
                ", ".join(names),
                len(names),
            )
        return dict.fromkeys(names, forecast), None
    spec = config.forecast.persistence
    return {name: PersistenceForecastSource(targets[name], spec.timestep, spec.members) for name in names}, None


def _dataset_names(config: EvaluationConfig, forecast: ForecastSource | None, targets: object) -> list[str]:
    """The datasets a run could score: the checkpoint's when there is one, else the names the `datasets:` blocks or
    the given target sources agree on, else the one unnamed dataset."""
    if isinstance(forecast, InferenceForecastSource):
        names = list(forecast.dataset_names)
    elif isinstance(targets, dict):
        names = list(targets)
    else:
        names = config_dataset_names(config)
    return names or [DEFAULT_DATASET]


def _selected_datasets(config: EvaluationConfig, available: list[str]) -> list[str]:
    """The datasets the run scores: every one of `available` unless the top-level `datasets:` key names a subset.

    The model still predicts every dataset — anemoi-inference runs every decoder and needs an input state for each
    one — so the skipped datasets only lose their targets, their weights, regions and climatology, their aggregator
    and their result file."""
    chosen = config.datasets
    if chosen is None:
        return list(available)
    unknown = [name for name in chosen if name not in available]
    if unknown:
        raise ValueError(f"datasets: names the unknown datasets {unknown}, the checkpoint has {available}")
    return [name for name in available if name in chosen]  # the checkpoint's order, duplicates dropped


def build(
    config: str | Path | dict | EvaluationConfig,
    forecast: ForecastSource | None = None,
    targets: TargetSource | dict[str, TargetSource] | None = None,
    climatology: ClimatologySource | dict[str, ClimatologySource] | None = None,
) -> Evaluation | MultiEvaluation:
    """An `Evaluation`, or a `MultiEvaluation` when the run scores multiple datasets; see `Evaluation.from_config`."""
    config = load_config(config)
    if forecast is None and config.forecast.anemoi_inference is not None:
        forecast = InferenceForecastSource(**config.forecast.anemoi_inference.model_dump())
    available = _dataset_names(config, forecast, targets)
    names = _selected_datasets(config, available)
    skipped = [name for name in available if name not in names]
    if isinstance(forecast, InferenceForecastSource):
        forecast.scored_datasets = names
    blocks = {
        key: per_dataset(getattr(config, key), names, key)
        for key in ("targets", "variables", "weights", "regions", "climatology")
    }
    given_targets = targets if isinstance(targets, dict) else {names[0]: targets} if targets is not None else {}
    unknown = [name for name in given_targets if name not in names]
    if unknown:
        raise ValueError(f"targets were given for the unknown datasets {unknown}, the run scores {names}")
    if given_targets and set(given_targets) != set(names):  # the all-or-nothing rule of the `datasets:` blocks
        raise ValueError(f"targets must be given for every dataset or for none, got {sorted(given_targets)} of {names}")
    sources = {
        name: given_targets.get(name)
        or _targets_from_config(
            blocks["targets"][name],
            forecast.datasets[name]
            if isinstance(forecast, InferenceForecastSource) and forecast.multi_dataset
            else forecast,
        )
        for name in names
    }
    views, shared = _forecast_views(config, forecast, names, sources)
    given_climatology = (
        climatology if isinstance(climatology, dict) else {names[0]: climatology} if climatology is not None else {}
    )
    climatologies = {
        name: given_climatology.get(name)
        or (ArrayClimatology.from_netcdf(blocks["climatology"][name].file) if blocks["climatology"][name] else None)
        for name in names
    }
    evaluations = {}
    for name in names:
        regions = {
            region: spec if isinstance(spec, str) else spec.spec() for region, spec in blocks["regions"][name].items()
        }
        weights = blocks["weights"][name]
        try:
            evaluations[name] = Evaluation(
                views[name],
                sources[name],
                config.init_times.resolve(),
                config.lead_time,
                config.metrics,
                blocks["variables"][name],
                weights.spec() if weights is not None else None,
                regions,
                config.bins.model_dump(),
                on_missing_target=config.on_missing_target,
                include_lead_zero=config.include_lead_zero,
                climatology=climatologies[name],
                dataset=name if len(available) > 1 else None,
            )
        except ValueError as error:  # a shared block that fits one dataset's grid but not another's
            if len(available) == 1:
                raise
            raise ValueError(f"dataset {name!r}: {error}") from error
        evaluations[name]._config = config
    if skipped and isinstance(forecast, InferenceForecastSource):
        # the runner reads every dataset's inputs whether or not the dataset is scored (FR-7)
        for name in skipped:
            forecast.check_init_times(config.init_times.resolve(), config.lead_time, dataset=name)
    if len(names) == 1:
        view = views[names[0]]
        if isinstance(view, DatasetForecast):  # one dataset of a multi-dataset checkpoint: it owns the runner
            view.solo = True
        return evaluations[names[0]]
    evaluation = MultiEvaluation(evaluations, shared)
    evaluation._config = config
    return evaluation


def _targets_from_config(block: object, forecast: ForecastSource | None) -> TargetSource:
    """The configured target source; without an `anemoi_dataset` mapping, the forecast source's own dataset."""
    options = block.anemoi_dataset if block is not None else AnemoiDatasetConfig()
    settings = {"prefetch": options.prefetch, "cache_bytes": options.cache_bytes}
    kwargs = options.open_dataset_kwargs()
    if kwargs:
        return DatasetTargets(**kwargs, **settings)
    if forecast is None:
        raise ValueError("the targets block needs a dataset when the forecast source has none to share")
    return DatasetTargets.from_forecast(forecast, **settings)


def package_versions(packages: tuple[str, ...] = PACKAGES) -> dict[str, str | None]:
    """Installed versions of the packages behind a result, None when a package is not installed."""
    from anemoi.evaluation import __version__

    versions = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:  # a source tree on the path: known for this package, unknown for the others
            versions[package] = __version__ if package == "anemoi-evaluation" else None
    return versions
