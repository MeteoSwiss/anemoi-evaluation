"""A fake multi-dataset anemoi-inference runner, built from the committed fixture in
`tests/fixtures/multi-dataset`.

The fixture is the checkpoint metadata of a two-dataset forecaster over the global ERA5 n320
analysis (`era5`) and the MeteoSwiss 1 km LAM (`realch1`), both at a 6 h timestep, with a few
hundred real grid points and the real per-variable statistics; see its README for provenance.
The two variable namespaces collide on purpose (`2t`, `msl`, `tp`, ... are in both).

Everything here is real anemoi-inference machinery: `save_fake_checkpoint` writes a checkpoint
carrying the fixture metadata and a `SimpleMockModel`, and a real `SimpleRunner` drives it, so the
yielded `dict[name, State]` has exactly the structure, dtypes and object-reuse behaviour of a real
run (the outer dict and each `fields` dict are the same objects at every step). The runner's
per-dataset inputs are replaced by `FixtureInput`, which reads the same `FixtureDataset` the
targets read, so forecast and targets come from the same (fake) zarr, as FR-9 wants.
"""

from __future__ import annotations

import copy
import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
from anemoi.utils.dates import frequency_to_timedelta

from anemoi.evaluation.sources.anemoi_dataset import DatasetTargets

FIXTURE = Path(__file__).parent / "fixtures" / "multi-dataset"
DATASET_NAMES = ("era5", "realch1")
COLLIDING_VARIABLES = ("2t", "10u", "10v", "msl", "t_850", "q_850", "tp")
HOUR = datetime.timedelta(hours=1)


def load_metadata() -> dict:
    """The fixture's checkpoint metadata, for every dataset."""
    with open(FIXTURE / "metadata.json") as file:
        return json.load(file)


def load_arrays() -> dict[str, np.ndarray]:
    """The fixture's per-dataset latitudes, longitudes and statistics."""
    with np.load(FIXTURE / "arrays.npz") as arrays:
        return {key: arrays[key] for key in arrays.files}


def checkpoint_metadata(names: tuple[str, ...] = DATASET_NAMES, routing: str = "forecaster") -> dict:
    """The fixture metadata restricted to `names`: one dataset gives a single-dataset checkpoint.

    `routing="downscaler"` rewrites `config.model.encoders/decoders` so that every dataset is
    encoded but only the last one is decoded, which is the model kind the evaluation refuses.
    """
    metadata = copy.deepcopy(load_metadata())  # the nested dicts below are mutated in place
    names = tuple(names)
    unknown = [name for name in names if name not in metadata["metadata_inference"]["dataset_names"]]
    if unknown:
        raise ValueError(f"unknown dataset names {unknown}")
    metadata["dataset"] = {name: metadata["dataset"][name] for name in names}
    metadata["data_indices"] = {name: metadata["data_indices"][name] for name in names}
    config = metadata["config"]
    config["data"]["datasets"] = {name: config["data"]["datasets"][name] for name in names}
    config["data"]["num_features"] = {name: config["data"]["num_features"][name] for name in names}
    for stage in ("training", "validation"):
        stage_config = config["dataloader"][stage]["datasets"]
        config["dataloader"][stage]["datasets"] = {name: stage_config[name] for name in names}
    if routing == "downscaler":
        config["model"]["encoders"] = {"0": {"source_datasets": list(names)}}
        config["model"]["decoders"] = {"0": {"target_datasets": [names[-1]]}}
    elif routing == "forecaster":
        config["model"]["encoders"] = {str(i): {"source_datasets": [name]} for i, name in enumerate(names)}
        config["model"]["decoders"] = {str(i): {"target_datasets": [name]} for i, name in enumerate(names)}
    else:
        raise ValueError(f"unknown routing {routing!r}")
    inference = metadata["metadata_inference"]
    metadata["metadata_inference"] = {
        **{key: value for key, value in inference.items() if key not in DATASET_NAMES},
        "dataset_names": list(names),
        **{name: inference[name] for name in names},
    }
    return metadata


def supporting_arrays(names: tuple[str, ...] = DATASET_NAMES) -> dict:
    """The per-dataset supporting arrays a multi-dataset checkpoint stores."""
    arrays = load_arrays()
    return {
        name: {"latitudes": arrays[f"{name}.latitudes"], "longitudes": arrays[f"{name}.longitudes"]} for name in names
    }


def save_checkpoint(
    path: str | Path,
    names: tuple[str, ...] = DATASET_NAMES,
    routing: str = "forecaster",
    patch: Any = None,
) -> Path:
    """Write a fake checkpoint over `names` (a `SimpleMockModel` plus the fixture metadata).

    `patch` is called with the metadata dict before it is written, for the hand-edited checkpoints
    the refusals are tested against.
    """
    from anemoi.inference.testing import save_fake_checkpoint

    path = Path(path)
    metadata = checkpoint_metadata(names, routing)
    if patch is not None:
        patch(metadata)
    save_fake_checkpoint(metadata, path, supporting_arrays=supporting_arrays(names))
    return path


def dates(metadata: dict, name: str) -> list[datetime.datetime]:
    """The fixture dates of one dataset."""
    dataset = metadata["dataset"][name]
    step = frequency_to_timedelta(dataset["frequency"])
    first = datetime.datetime.fromisoformat(dataset["start_date"])
    last = datetime.datetime.fromisoformat(dataset["end_date"])
    out, date = [], first
    while date <= last:
        out.append(date)
        date += step
    return out


class FixtureDataset:
    """An anemoi dataset over one of the fixture's datasets, as `DatasetTargets` reads it.

    Values are the real per-variable mean and standard deviation modulated by a smooth,
    deterministic function of the date and the node, so a forecast that carries the initial state
    forward (what `SimpleMockModel` does) has an error that grows with lead time.
    """

    def __init__(self, name: str, metadata: dict | None = None, arrays: dict | None = None) -> None:
        metadata = load_metadata() if metadata is None else metadata
        arrays = load_arrays() if arrays is None else arrays
        self.name = name
        self.variables = list(metadata["dataset"][name]["variables"])
        self.name_to_index = {variable: i for i, variable in enumerate(self.variables)}
        self.latitudes = arrays[f"{name}.latitudes"]
        self.longitudes = arrays[f"{name}.longitudes"]
        self.mean = arrays[f"{name}.mean"]
        self.stdev = arrays[f"{name}.stdev"]
        self._dates = dates(metadata, name)
        self.dates = np.array(self._dates, dtype="datetime64[s]")
        self.missing: set[int] = set()
        self.grids = (len(self.latitudes),)
        self.dtype = np.dtype(np.float32)
        self.shape = (len(self._dates), len(self.variables), 1, len(self.latitudes))
        self.reads = 0
        self.typed_variables = _typed_variables(metadata["dataset"][name]["variables_metadata"])

    def index(self, date: datetime.datetime) -> int:
        """Row of `date`; raises when the fixture does not cover it."""
        return self._dates.index(date)

    def row(self, i: int) -> np.ndarray:
        """Row `i` as (variables, nodes) float32."""
        nodes = np.arange(self.shape[3], dtype=np.float64)
        phase = 0.31 * i + 0.017 * nodes[None, :] + np.arange(self.shape[1], dtype=np.float64)[:, None]
        return (self.mean[:, None] + self.stdev[:, None] * np.sin(phase)).astype(np.float32)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, i: int) -> np.ndarray:
        self.reads += 1
        return self.row(i)[:, None, :]


def _typed_variables(variables_metadata: dict) -> dict:
    from anemoi.transform.variables import Variable

    return {name: Variable.from_dict(name, value) for name, value in variables_metadata.items()}


def _fixture_input_class() -> type:
    """`FixtureInput`, built lazily so that importing this module does not import anemoi-inference."""
    from anemoi.inference.inputs.dataset import DatasetInput

    class FixtureInput(DatasetInput):
        """A real `DatasetInput` whose dataset is a `FixtureDataset`.

        Subclassing the real input (rather than duck-typing one) is what lets
        `InferenceForecastSource.check_init_times` and `DatasetTargets.from_forecast` run against
        the fixture: both test `isinstance(source, DatasetInput)` and read `ds`, `ds_dates` and the
        recorded `open_dataset` arguments.
        """

        def __init__(self, context: Any, dataset: FixtureDataset, metadata: Any) -> None:
            super().__init__(
                context,
                metadata,
                open_dataset_args=({"dataset": f"fixture-{dataset.name}"},),
                open_dataset_kwargs={},
            )
            self.dataset = dataset

        @property
        def ds(self) -> FixtureDataset:  # overrides DatasetInput's cached_property
            """The fixture dataset."""
            return self.dataset

        def __repr__(self) -> str:
            return f"FixtureInput({self.dataset.name})"

        def create_input_state(self, *, date: datetime.datetime, **kwargs: Any) -> dict:
            """The multi-step input state at `date`: every input variable the runner does not compute."""
            typed = self.metadata.typed_variables
            variables = [
                variable
                for variable in self.metadata.variable_to_input_tensor_index
                if not typed[variable].is_computed_forcing
            ]
            rows = [self.dataset.row(self.dataset.index(date + lag)) for lag in self.metadata.lagged]
            index = self.dataset.name_to_index
            fields = {
                variable: np.stack([row[index[variable]] for row in rows]).astype(np.float32) for variable in variables
            }
            return {
                "date": date,
                "latitudes": self.dataset.latitudes,
                "longitudes": self.dataset.longitudes,
                "fields": fields,
            }

    return FixtureInput


def make_runner(checkpoint: str | Path, names: tuple[str, ...] = DATASET_NAMES, device: str = "cpu") -> Any:
    """A real `SimpleRunner` over the fake checkpoint, its per-dataset inputs served by the fixture."""
    from anemoi.inference.runners.simple import SimpleRunner

    runner = SimpleRunner(str(checkpoint), device=device)
    metadata = load_metadata()
    arrays = load_arrays()
    fixture_input = _fixture_input_class()
    for name in names:
        dataset = FixtureDataset(name, metadata, arrays)
        runner.prognostics_inputs[name] = fixture_input(runner, dataset, runner.checkpoint.multi_dataset_metadata[name])
    return runner


def make_targets(name: str, **kwargs: Any) -> DatasetTargets:
    """Targets over the same fixture dataset the forecast is initialised from."""
    return DatasetTargets.from_dataset(FixtureDataset(name), kwargs={"dataset": f"fixture-{name}"}, **kwargs)


def patch_create_runner(monkeypatch: Any, checkpoint: str | Path, names: tuple[str, ...] = DATASET_NAMES) -> list:
    """Make `InferenceForecastSource` build the fixture runner instead of a real one.

    The run configuration the source builds is kept (and its `device` honoured), so a test can
    assert what reached anemoi-inference; the returned list collects one entry per call.
    """
    import anemoi.inference.runners as runners

    configs: list = []

    def create_runner(config: Any) -> Any:
        configs.append(config)
        return make_runner(checkpoint, names, device=str(getattr(config, "device", None) or "cpu"))

    monkeypatch.setattr(runners, "create_runner", create_runner)
    return configs


def forecast_source(
    monkeypatch: Any,
    tmp_path: Path,
    names: tuple[str, ...] = DATASET_NAMES,
    patch: Any = None,
    **options: Any,
) -> Any:
    """An `InferenceForecastSource` over a fake checkpoint of `names`; `patch` is `save_checkpoint`'s."""
    from anemoi.evaluation.sources.anemoi_inference import InferenceForecastSource

    checkpoint = save_checkpoint(tmp_path / f"{'-'.join(names)}.ckpt", names, patch=patch)
    patch_create_runner(monkeypatch, checkpoint, names)
    return InferenceForecastSource(checkpoint=str(checkpoint), device="cpu", **options)


def two_output_steps(metadata: dict) -> None:
    """A `save_checkpoint` patch making the fake a two-output-step model (one model call per 12 h)."""
    metadata["config"]["training"]["multistep_output"] = 2
    for name in metadata["metadata_inference"]["dataset_names"]:
        timesteps = metadata["metadata_inference"][name]["timesteps"]
        timesteps["relative_date_indices_training"] = [0, 1, 2, 3]
        timesteps["output_relative_date_indices"] = [2, 3]
        timesteps["output_offsets"] = ["6h", "12h"]
        timesteps["rollout_shift"] = "12h"
        timesteps["advance_map"] = {"inin": [], "outin": [[0, 0], [1, 1]]}


def init_times(name: str = DATASET_NAMES[0], count: int = 2) -> list[datetime.datetime]:
    """Init times the fixture can serve a multi-step input and a rollout from."""
    available = dates(load_metadata(), name)
    return available[1 : 1 + count]


def patch_targets_from_forecast(monkeypatch: Any) -> None:
    """Make `DatasetTargets.from_forecast` serve the fixture dataset of the forecast source it is given.

    The fixture's inputs record the `open_dataset` arguments of a dataset that does not exist on disk, so a run
    driven by the command line (which builds its targets from the config) needs this stand-in.
    """

    def from_forecast(forecast: Any, prefetch: int | None = None, cache_bytes: int = 0) -> DatasetTargets:
        name = getattr(forecast, "name", None) or forecast.dataset_names[0]
        return make_targets(name, prefetch=prefetch, cache_bytes=cache_bytes)

    monkeypatch.setattr(DatasetTargets, "from_forecast", staticmethod(from_forecast))


def attach_graph(source: Any, keyed: bool = False) -> Any:
    """Give the source's model a graph whose node sets are named after the datasets.

    `area_weight` is `1 .. n` on each dataset's own grid, so a run that reads the wrong dataset's node set
    (or the wrong graph of a `graph_data` mapping) gets weights of the wrong length or the wrong values.
    `keyed` wraps the graphs in the per-dataset mapping form some anemoi-models builds expose instead.
    """
    import torch
    from torch_geometric.data import HeteroData

    graphs = {}
    for name, view in source.datasets.items():
        graph = HeteroData()
        graph[name].area_weight = torch.arange(1, view.grid.n + 1, dtype=torch.float32).reshape(-1, 1)
        graphs[name] = graph
    if keyed:
        source.runner.model.graph_data = graphs
    else:
        merged = HeteroData()
        for name, graph in graphs.items():
            merged[name].area_weight = graph[name].area_weight
        source.runner.model.graph_data = merged
    return source
