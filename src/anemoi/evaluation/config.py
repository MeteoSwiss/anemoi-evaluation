"""YAML config model: a serialisation of the `Evaluation` constructor arguments."""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any
from typing import Literal

import numpy as np
import yaml
from anemoi.utils.dates import as_datetime
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta
from anemoi.utils.humanize import human_to_bytes
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import TypeAdapter
from pydantic import field_serializer
from pydantic import field_validator
from pydantic import model_validator

from anemoi.evaluation import binning
from anemoi.evaluation import metrics as metrics_module
from anemoi.evaluation import regions
from anemoi.evaluation import weights
from anemoi.evaluation.frame import Grid

LOG = logging.getLogger(__name__)
FROM_CHECKPOINT = "from_checkpoint"


def _positive_timedelta(value: int | str | datetime.timedelta) -> datetime.timedelta:
    value = frequency_to_timedelta(value)
    if value <= datetime.timedelta(0):
        raise ValueError(f"duration must be positive, got {value}")
    return value


def _exactly_one(model: BaseModel, what: str) -> None:
    given = [name for name, value in model if value is not None]
    if len(given) != 1:
        raise ValueError(f"{what} needs exactly one source, got {given or 'none'}")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Spec(_Model):
    """Single-key mapping that builds a per-node array from the grid and the sources."""

    def spec(self) -> dict:
        """The YAML form."""
        return self.model_dump(mode="json")

    def build(self, grid: Grid, forecast: Any = None, targets: Any = None) -> np.ndarray:
        """The (N,) array the spec describes."""
        raise NotImplementedError


class InitTimesRange(_Model):
    """Init times from `start` to `end` inclusive, every `frequency`."""

    start: datetime.datetime
    end: datetime.datetime
    frequency: datetime.timedelta

    @field_validator("start", "end", mode="before")
    @classmethod
    def _parse_date(cls, value: object) -> datetime.datetime:
        return as_datetime(value)

    @field_validator("frequency", mode="before")
    @classmethod
    def _parse_frequency(cls, value: object) -> datetime.timedelta:
        return _positive_timedelta(value)

    @field_serializer("frequency", when_used="json")
    def _frequency_string(self, value: datetime.timedelta) -> str:
        return frequency_to_string(value)

    @model_validator(mode="after")
    def _check_order(self) -> InitTimesRange:
        if self.end < self.start:
            raise ValueError(f"end {self.end} is before start {self.start}")
        return self

    def resolve(self) -> list[datetime.datetime]:
        """The dates in the range."""
        dates, date = [], self.start
        while date <= self.end:
            dates.append(date)
            date += self.frequency
        return dates


class InitTimesList(_Model):
    """An explicit list of init times."""

    dates: list[datetime.datetime] = Field(min_length=1)

    @field_validator("dates", mode="before")
    @classmethod
    def _parse_dates(cls, value: object) -> object:
        return [as_datetime(date) for date in value] if isinstance(value, list) else value

    def resolve(self) -> list[datetime.datetime]:
        """The dates."""
        return list(self.dates)


InitTimesConfig = InitTimesRange | InitTimesList


class AnemoiInferenceConfig(BaseModel):
    """The `anemoi_inference` block: `members`, `seed`, `quiet` and `forcings_cache_bytes`, the rest goes to the runner
    configuration."""

    model_config = ConfigDict(extra="allow")

    members: int = Field(1, ge=1)
    seed: int = 0
    quiet: bool = True
    forcings_cache_bytes: int = Field(2**30, ge=0)


class PersistenceConfig(_Model):
    """Persistence of the target at init time; needs an explicit `targets` block."""

    timestep: datetime.timedelta
    members: int = Field(1, ge=1)

    @field_validator("timestep", mode="before")
    @classmethod
    def _parse_timestep(cls, value: object) -> datetime.timedelta:
        return _positive_timedelta(value)

    @field_serializer("timestep", when_used="json")
    def _timestep_string(self, value: datetime.timedelta) -> str:
        return frequency_to_string(value)


class ForecastConfig(_Model):
    """Forecast source, exactly one key."""

    anemoi_inference: AnemoiInferenceConfig | None = None
    persistence: PersistenceConfig | None = None

    @model_validator(mode="after")
    def _one_source(self) -> ForecastConfig:
        _exactly_one(self, "forecast")
        return self


class AnemoiDatasetConfig(BaseModel):
    """The `anemoi_dataset` block: `prefetch` (rows read ahead; null means 2 or the forecast's frames per model call,
    whichever is larger) and `cache_bytes` (host cache, `2GiB` style strings accepted) are consumed by the source, the
    rest goes to `open_dataset`; nothing else means the forecast's dataset."""

    model_config = ConfigDict(extra="allow")

    prefetch: int | None = Field(None, ge=0)
    cache_bytes: int = Field(0, ge=0)

    @field_validator("cache_bytes", mode="before")
    @classmethod
    def _parse_bytes(cls, value: object) -> object:
        return human_to_bytes(value) if isinstance(value, str) else value

    def open_dataset_kwargs(self) -> dict[str, Any]:
        """The keys that go to `open_dataset`."""
        return dict(self.model_extra or {})


class TargetsConfig(_Model):
    """Target source, exactly one key."""

    anemoi_dataset: AnemoiDatasetConfig | None = None

    @model_validator(mode="after")
    def _one_source(self) -> TargetsConfig:
        _exactly_one(self, "targets")
        return self


class GraphAttributeSpec(_Spec):
    """Node attribute of the forecast model's graph, e.g. `area_weight` or `cutout_mask`."""

    graph_attribute: str

    def build(self, grid: Grid, forecast: Any = None, targets: Any = None) -> np.ndarray:
        return weights.graph_node_attribute(forecast, self.graph_attribute, grid)


class FileSpec(_Spec):
    """Precomputed (N,) array in a .npy file."""

    file: str

    def build(self, grid: Grid, forecast: Any = None, targets: Any = None) -> np.ndarray:
        return grid.check_shape(np.load(self.file), self.file)


class UniformSpec(_Spec):
    """Equal weights."""

    uniform: dict[str, Any]

    @field_validator("uniform", mode="before")
    @classmethod
    def _none_to_empty(cls, value: object) -> object:
        return {} if value is None else value

    def build(self, grid: Grid, forecast: Any = None, targets: Any = None) -> np.ndarray:
        return weights.uniform(grid)


class SphericalVoronoiSpec(_Spec):
    """Spherical Voronoi cell areas of the grid."""

    spherical_voronoi: dict[str, Any]

    @field_validator("spherical_voronoi", mode="before")
    @classmethod
    def _none_to_empty(cls, value: object) -> object:
        return {} if value is None else value

    def build(self, grid: Grid, forecast: Any = None, targets: Any = None) -> np.ndarray:
        return weights.spherical_voronoi(grid)


class Bbox(_Model):
    north: float
    west: float
    south: float
    east: float


class BboxSpec(_Spec):
    """Nodes inside a latitude/longitude box."""

    bbox: Bbox

    def build(self, grid: Grid, forecast: Any = None, targets: Any = None) -> np.ndarray:
        return regions.bbox(grid, **self.bbox.model_dump())


class GridSpec(_Spec):
    """Nodes of sub-grid `grid` of the target dataset (a cutout or join)."""

    grid: int

    def build(self, grid: Grid, forecast: Any = None, targets: Any = None) -> np.ndarray:
        if targets is None:
            raise ValueError(f"a target source with sub-grids is needed for the region 'grid: {self.grid}'")
        return grid.check_shape(np.asarray(targets.grid_mask(self.grid), dtype=bool), f"grid {self.grid}")


WeightsConfig = GraphAttributeSpec | SphericalVoronoiSpec | UniformSpec | FileSpec
RegionConfig = Literal["all"] | BboxSpec | GraphAttributeSpec | GridSpec | FileSpec


def build_weights(spec: dict, grid: Grid, forecast: Any = None) -> np.ndarray:
    """(N,) float64 weights from a YAML spec (see `WeightsConfig`)."""
    return TypeAdapter(WeightsConfig).validate_python(spec).build(grid, forecast).astype(np.float64)


def build_region(spec: str | dict, grid: Grid, forecast: Any = None, targets: Any = None) -> np.ndarray:
    """(N,) boolean mask from a YAML spec: `all` or one of `RegionConfig`."""
    if spec == "all":
        return regions.all(grid)
    return TypeAdapter(RegionConfig).validate_python(spec).build(grid, forecast, targets).astype(bool)


class BinsConfig(_Model):
    """Time binning: one bin per season, month or date (`time`), of the init or valid time (`by`)."""

    time: Literal["season", "month", "init_time", "none"] = "season"
    by: Literal["init_time", "valid_time"] = "init_time"

    def build(
        self, init_times: list[datetime.datetime] = (), lead_times: list[datetime.timedelta] = ()
    ) -> binning.Binning:
        """The binning object."""
        return binning.build(self.time, self.by, init_times, lead_times)


class ClimatologyConfig(_Model):
    """Climatology for the anomaly statistics: a netcdf written by `ArrayClimatology.to_netcdf`."""

    file: str


class OutputConfig(_Model):
    path: str


class EvaluationConfig(_Model):
    """Everything `anemoi-evaluation run` needs; see `Evaluation.from_config`."""

    forecast: ForecastConfig
    targets: TargetsConfig | None = None
    lead_time: datetime.timedelta
    init_times: InitTimesConfig
    variables: list[str] | None = None
    metrics: list[str | dict[str, dict[str, Any] | None]] = ["rmse", "mae", "bias"]
    weights: WeightsConfig | None = None
    regions: dict[str, RegionConfig] = {"global": "all"}
    bins: BinsConfig = BinsConfig()
    climatology: ClimatologyConfig | None = None
    output: OutputConfig | None = None
    on_missing_target: Literal["skip", "raise"] = "skip"
    include_lead_zero: bool = False
    log_level: str = "INFO"

    @field_validator("lead_time", mode="before")
    @classmethod
    def _parse_lead_time(cls, value: object) -> datetime.timedelta:
        return _positive_timedelta(value)

    @field_serializer("lead_time", when_used="json")
    def _lead_time_string(self, value: datetime.timedelta) -> str:
        return frequency_to_string(value)

    @field_validator("metrics")
    @classmethod
    def _check_metrics(cls, value: list) -> list:
        for spec in value:
            metrics_module.from_spec(spec)
        return value


def _load_yaml(path: Path) -> dict:
    with open(path) as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"config {path} must be a mapping, got {type(config).__name__}")
    return config


def _merge(base: dict, override: dict) -> dict:
    """`override` on top of `base`: mappings merge key by key in `base`'s order, anything else replaces."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_base(config: dict, directory: Path, source: str, seen: tuple[Path, ...] = ()) -> dict:
    """`config` merged on top of the `base:` file it names, recursively; a `base` is relative to the file naming it."""
    config = dict(config)
    base = config.pop("base", None)
    if base is None:
        return config
    path = Path(base)
    path = path if path.is_absolute() else directory / path
    if not path.is_file():
        raise ValueError(f"base config {path} of {source} not found")
    path = path.resolve()
    if path in seen:
        raise ValueError("base config cycle: " + " -> ".join(str(seen_path) for seen_path in (*seen, path)))
    LOG.info("%s extends %s", source, path)
    return _merge(resolve_base(_load_yaml(path), path.parent, str(path), (*seen, path)), config)


def checkpoint_dataset_arguments(checkpoint: str) -> tuple[tuple, dict]:
    """The `open_dataset` arguments a checkpoint records, with the dataset paths it was trained on."""
    from anemoi.inference.checkpoint import Checkpoint

    metadata = Checkpoint(str(checkpoint)).multi_dataset_metadata
    if len(metadata) != 1:
        raise ValueError(f"only single-dataset checkpoints are supported, got {sorted(metadata)}")
    args, kwargs = next(iter(metadata.values())).open_dataset_args_kwargs(use_original_paths=True)
    return tuple(args), dict(kwargs)


def _dataset_from_checkpoint(block: dict, checkpoint: str | None, where: str) -> dict:
    """`block` with its `from_checkpoint` replaced by the checkpoint's recorded arguments, the rest of `block` on top."""
    block = dict(block)
    source = block.pop(FROM_CHECKPOINT)
    if source is False:
        return block
    if source is True:
        if checkpoint is None:
            raise ValueError(
                f"{where}.from_checkpoint: true needs forecast.anemoi_inference.checkpoint, "
                "give a checkpoint path instead"
            )
        source = checkpoint
    if not isinstance(source, str):
        raise ValueError(f"{where}.from_checkpoint must be true, false or a checkpoint path, got {source!r}")
    args, kwargs = checkpoint_dataset_arguments(source)
    if len(args) != 1 or not isinstance(args[0], dict):
        raise ValueError(
            f"checkpoint {source} records the open_dataset arguments {args}, only a single mapping can be reused; "
            "spell the dataset out instead"
        )
    dataset = {**args[0], **kwargs, **block}
    select = dataset.get("select")
    LOG.info(
        "%s: dataset recorded in %s, %s variables, start %s, end %s",
        where,
        source,
        len(select) if isinstance(select, (list, tuple, dict)) else "all",
        dataset.get("start"),
        dataset.get("end"),
    )
    return dataset


def _check_resolved(config: Any, where: str = "") -> None:
    if isinstance(config, dict):
        if FROM_CHECKPOINT in config:
            raise ValueError(
                f"from_checkpoint at {where or 'the top level'} is only resolved in "
                "forecast.anemoi_inference.input.dataset and targets.anemoi_dataset"
            )
        for key, value in config.items():
            _check_resolved(value, f"{where}.{key}" if where else str(key))
    elif isinstance(config, list):
        for index, value in enumerate(config):
            _check_resolved(value, f"{where}[{index}]")


def resolve_from_checkpoint(config: dict) -> dict:
    """`config` with the dataset blocks that ask for it replaced by the checkpoint's recorded `open_dataset`
    arguments; `from_checkpoint: true` means the checkpoint of the `anemoi_inference` forecast."""
    config = dict(config)
    forecast = config.get("forecast") if isinstance(config.get("forecast"), dict) else {}
    inference = forecast.get("anemoi_inference") if isinstance(forecast.get("anemoi_inference"), dict) else None
    checkpoint = inference.get("checkpoint") if inference is not None else None
    if inference is not None:
        block = inference.get("input") if isinstance(inference.get("input"), dict) else {}
        dataset = block.get("dataset") if isinstance(block.get("dataset"), dict) else {}
        if FROM_CHECKPOINT in dataset:
            where = "forecast.anemoi_inference.input.dataset"
            dataset = _dataset_from_checkpoint(dataset, checkpoint, where)
            inference = {**inference, "input": {**block, "dataset": dataset}}
            config["forecast"] = {**forecast, "anemoi_inference": inference}
    targets = config.get("targets") if isinstance(config.get("targets"), dict) else {}
    dataset = targets.get("anemoi_dataset") if isinstance(targets.get("anemoi_dataset"), dict) else {}
    if FROM_CHECKPOINT in dataset:
        dataset = _dataset_from_checkpoint(dataset, checkpoint, "targets.anemoi_dataset")
        config["targets"] = {**targets, "anemoi_dataset": dataset}
    _check_resolved(config)
    return config


def load_config(source: str | Path | dict | EvaluationConfig) -> EvaluationConfig:
    """Validated config from a YAML path, a dict, or an existing config; a `base:` chain and the `from_checkpoint:`
    dataset blocks are resolved first, so the returned config is the resolved one."""
    if isinstance(source, EvaluationConfig):
        return source
    if isinstance(source, (str, Path)):
        path = Path(source)
        source = resolve_base(_load_yaml(path), path.parent, str(path), (path.resolve(),))
    else:
        source = resolve_base(source, Path.cwd(), "config")
    return EvaluationConfig.model_validate(resolve_from_checkpoint(source))
