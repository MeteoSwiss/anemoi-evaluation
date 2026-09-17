"""Checkpoints trained on multiple datasets: the fixture runner, today's refusals, and the
end-to-end run the work is aiming at.

`_multi_dataset.py` builds a real `SimpleRunner` over a fake two-dataset checkpoint derived from
the ERA5 n320 and MeteoSwiss `realch1` zarrs, so the tests see the real yielded structure.
"""

import _multi_dataset as fixture
import numpy as np
import pytest
import torch
import xarray as xr
import yaml

from anemoi.evaluation.__main__ import format_output_path
from anemoi.evaluation.__main__ import main
from anemoi.evaluation.config import checkpoint_dataset_arguments
from anemoi.evaluation.config import load_config
from anemoi.evaluation.evaluate import Evaluation
from anemoi.evaluation.output import write
from anemoi.evaluation.sources.anemoi_inference import InferenceForecastSource

CPU = torch.device("cpu")
METRICS = ["rmse", "bias", "mae"]
BINS = {"time": "none"}


def test_fixture_runner_yields_one_state_per_dataset(tmp_path):
    """The helper reproduces what a real multi-dataset runner yields (step1-verification.md A.1)."""
    checkpoint = fixture.save_checkpoint(tmp_path / "two.ckpt")
    runner = fixture.make_runner(checkpoint)
    assert runner.dataset_names == list(fixture.DATASET_NAMES)
    metadata = runner.checkpoint.multi_dataset_metadata
    grids = {name: metadata[name].number_of_grid_points for name in fixture.DATASET_NAMES}
    assert grids == {"era5": 288, "realch1": 176}  # two grids of different sizes
    assert runner.checkpoint.timestep == 6 * fixture.HOUR and runner.checkpoint.multi_step_input == 2
    for name in fixture.DATASET_NAMES:
        # the forecaster keys are present, so the faithful branch is the one exercised, not
        # anemoi-inference's legacy derivation (MultiDatasetMetadata.lagged and friends)
        timesteps = fixture.load_metadata()["metadata_inference"][name]["timesteps"]
        assert set(timesteps) >= {"input_offsets", "output_offsets", "rollout_shift", "advance_map"}
        assert metadata[name].lagged == [-6 * fixture.HOUR, 0 * fixture.HOUR]
        assert metadata[name].output_offsets == [6 * fixture.HOUR]
        assert metadata[name].rollout_shift == 6 * fixture.HOUR

    init_time = fixture.init_times()[0]
    states = {name: runner.prognostics_inputs[name].create_input_state(date=init_time) for name in grids}
    yields = []
    for step, output in enumerate(runner.run(input_states=states, lead_time="12h"), start=1):
        yields.append(output)
        assert list(output) == list(fixture.DATASET_NAMES)
        for name, state in output.items():
            assert set(state) == {"date", "step", "previous_step", "latitudes", "longitudes", "fields"}
            assert state["date"] == init_time + step * 6 * fixture.HOUR
            assert state["step"] == step * 6 * fixture.HOUR
            for axis in ("latitudes", "longitudes"):
                assert state[axis].shape == (grids[name],) and state[axis].dtype == np.float64
            for variable, values in state["fields"].items():
                assert values.shape == (grids[name],) and values.dtype == np.float32, variable
    assert len(yields) == 2
    # the runner reuses the containers and only replaces the arrays: a consumer must not buffer
    assert yields[0] is yields[1]
    assert yields[0]["era5"]["fields"] is yields[1]["era5"]["fields"]

    outputs = {name: set(yields[0][name]["fields"]) for name in grids}
    assert set(fixture.COLLIDING_VARIABLES) <= outputs["era5"] & outputs["realch1"]
    assert "tp" in outputs["era5"]  # the diagnostic is an output variable but not an input one
    assert "z" not in outputs["era5"] and "lsm" not in outputs["realch1"]  # forcings are not decoded


def test_fixture_runner_honours_return_numpy(tmp_path):
    """`frames()` asks for tensors; the fields are then non-contiguous float32 torch tensors."""
    runner = fixture.make_runner(fixture.save_checkpoint(tmp_path / "two.ckpt"))
    init_time = fixture.init_times()[0]
    states = {
        name: runner.prognostics_inputs[name].create_input_state(date=init_time) for name in fixture.DATASET_NAMES
    }
    output = next(runner.run(input_states=states, lead_time="6h", return_numpy=False))
    for name in fixture.DATASET_NAMES:
        values = output[name]["fields"]["2t"]
        assert isinstance(values, torch.Tensor) and values.dtype == torch.float32
        assert values.shape == (runner.checkpoint.multi_dataset_metadata[name].number_of_grid_points,)
        assert isinstance(output[name]["latitudes"], np.ndarray)  # never converted to torch


def test_single_dataset_fixture_runs_end_to_end(tmp_path, monkeypatch):
    """The fixture drives the whole framework, so the multi-dataset test below differs in one thing only."""
    forecast = fixture.forecast_source(monkeypatch, tmp_path, ("era5",))
    targets = fixture.make_targets("era5")
    assert forecast.grid.n == targets.grid.n == 288
    assert forecast.variables == ["2t", "10u", "10v", "msl", "t_850", "q_850", "tp"]
    evaluation = Evaluation(forecast, targets, fixture.init_times(), "12h", METRICS, weights={"uniform": {}}, bins=BINS)
    dataset = evaluation.run().to_xarray(evaluation.metrics).sel(bin="all", region="global")
    assert dataset["rmse"].shape == (2, 7) and np.isfinite(dataset["rmse"].values).all()
    # the mock model carries the initial state forward, so the error grows with lead time
    assert (dataset["rmse"].isel(lead_time=1) >= dataset["rmse"].isel(lead_time=0)).all()


def test_multi_dataset_source_exposes_one_view_per_dataset(tmp_path, monkeypatch):
    """The source of a two-dataset checkpoint has no single grid or variable list: both are per dataset."""
    checkpoint = fixture.save_checkpoint(tmp_path / "two.ckpt")
    fixture.patch_create_runner(monkeypatch, checkpoint)
    source = InferenceForecastSource(checkpoint=str(checkpoint), device="cpu")
    assert source.multi_dataset and list(source.datasets) == list(fixture.DATASET_NAMES)
    assert [view.grid.n for view in source.datasets.values()] == [288, 176]
    assert source.datasets["era5"].variables == source.datasets["realch1"].variables  # colliding namespaces
    assert source.datasets["era5"].members == source.members == 1  # delegated to the parent
    for name in ("grid", "variables", "dataset_name"):
        with pytest.raises(AttributeError, match="per dataset"):
            getattr(source, name)
    with pytest.raises(ValueError, match="shared rollout"):
        next(source.datasets["era5"].frames(fixture.init_times()[0], 6 * fixture.HOUR, ["2t"], CPU))
    # the per-dataset target resolution and init-time check address that dataset's own input
    for name, view in source.datasets.items():
        args, kwargs = view.dataset_args_kwargs()
        assert args == ({"dataset": f"fixture-{name}"},) and kwargs == {}
        view.check_init_times(fixture.init_times(), 12 * fixture.HOUR)
    with pytest.raises(ValueError, match="'era5'"):
        source.datasets["era5"].check_init_times([fixture.init_times()[0] - 365 * 24 * fixture.HOUR], 6 * fixture.HOUR)


def _single(tmp_path, monkeypatch, name, members=1, patch=None):
    """The single-dataset run of the same fake, restricted to `name`."""
    directory = tmp_path / name
    directory.mkdir(exist_ok=True)
    forecast = fixture.forecast_source(monkeypatch, directory, (name,), patch=patch, members=members)
    evaluation = Evaluation(
        forecast,
        fixture.make_targets(name),
        fixture.init_times(),
        "12h",
        METRICS,
        weights={"uniform": {}},
        bins=BINS,  # the multi-dataset config below binds the same binning on both sides
    )
    return evaluation, evaluation.run()


# (members, checkpoint patch): one member and one output step, then the two paths the shared rollout
# has of its own — lockstep members stacked per dataset, and a model call yielding two output steps.
@pytest.mark.parametrize(
    "members, patch",
    [(1, None), (2, None), (1, fixture.two_output_steps)],
    ids=["one-member", "two-members", "two-output-steps"],
)
def test_multi_dataset_matches_the_per_dataset_runs(tmp_path, monkeypatch, members, patch):
    """A two-dataset run is N per-dataset evaluations sharing one runner and one rollout: it writes one
    result per dataset, each identical to the single-dataset run of the same fake.

    `output.path` takes a `{dataset}` placeholder and the per-dataset config blocks follow
    anemoi-inference's `datasets: {name: block}` convention.
    """
    names = fixture.DATASET_NAMES
    references = {name: _single(tmp_path, monkeypatch, name, members, patch) for name in names}

    checkpoint = fixture.save_checkpoint(tmp_path / "two.ckpt", patch=patch)
    fixture.patch_create_runner(monkeypatch, checkpoint)
    path = str(tmp_path / "result-{dataset}.nc")
    config = {
        "forecast": {
            "anemoi_inference": {"checkpoint": str(checkpoint), "device": "cpu", "members": members},
        },
        "lead_time": "12h",
        "init_times": {"dates": [date.isoformat() for date in fixture.init_times()]},
        "metrics": METRICS,
        "weights": {"datasets": {name: {"uniform": {}} for name in names}},
        "bins": BINS,
        "output": {"path": path},
    }
    evaluation = Evaluation.from_config(config, targets={name: fixture.make_targets(name) for name in names})
    # the views report the shared rollout's shape: a two-output-step model yields two frames per call
    frames_per_pass = 2 if patch is not None else 1
    assert evaluation.forecast.frames_per_pass == frames_per_pass
    assert all(view.frames_per_pass == frames_per_pass for view in evaluation.forecast.datasets.values())
    assert all(view.members == members for view in evaluation.forecast.datasets.values())
    states = evaluation.run()
    assert set(states) == set(names)

    for name in names:
        reference_evaluation, reference = references[name]
        state = states[name]
        assert state.coords == reference.coords
        assert set(state.sums) == set(reference.sums)
        assert all(torch.equal(state.sums[key], reference.sums[key]) for key in reference.sums)
        assert torch.equal(state.weights, reference.weights) and torch.equal(state.n_init, reference.n_init)
        written = path.format(dataset=name)
        write(state.to_xarray(evaluation.metrics[name]), written)
        expected = tmp_path / f"reference-{name}.nc"
        write(reference.to_xarray(reference_evaluation.metrics), expected)
        with xr.open_dataset(written, decode_timedelta=True) as result:
            with xr.open_dataset(expected, decode_timedelta=True) as wanted:
                result, wanted = result.load(), wanted.load()
                xr.testing.assert_equal(result, wanted)
                _assert_attrs_match(result.attrs, wanted.attrs, name)

    # the two datasets really are scored on their own grid and their own data: uniform weights sum
    # to the node count (288 vs 176) and the errors of one are not the errors of the other
    first, second = (states[name] for name in names)
    assert not torch.equal(first.weights, second.weights)
    assert not torch.equal(first.sums["squared_error"], second.sums["squared_error"])
    # one forcings cache per grid: with two grids a single cache would serve the first and bypass
    # the second for ever, which no numeric assertion above can see
    assert evaluation.forecast.stats["forcings_bypassed"] == 0
    assert evaluation.forecast.stats["forcings_computed"] > 0


# Attrs that cannot match: the two runs are driven by two different checkpoint files, and a shared
# rollout's timings are not a single-dataset run's. Everything else must be the same number.
DIFFERING_ATTRS = ("checkpoint", "checkpoint_uuid", "checkpoint_run_id", "gpu", "config", "dataset")


def _kept(attrs: dict) -> dict:
    skip = ("time_", "inference_chunks")
    return {k: v for k, v in attrs.items() if k not in DIFFERING_ATTRS and not k.startswith(skip)}


def _assert_attrs_match(result: dict, wanted: dict, name: str) -> None:
    """The result file of a dataset carries the same attrs as its single-dataset run, plus `dataset`."""
    assert _kept(result) == _kept(wanted)
    assert result["dataset"] == name and "dataset" not in wanted
    assert result["checkpoint"] and result["checkpoint"] != wanted["checkpoint"]
    for key in ("forcings_computed", "forcings_hits", "forcings_bypassed", "model_calls", "init_times"):
        assert key in result, key
    # the shared rollout is charged whole to every dataset, the rest is the dataset's own
    assert result["time_total_s"] == pytest.approx(
        result["time_model_s"] + result["time_target_s"] + result["time_statistics_s"]
    )


def test_multi_dataset_default_weights_come_from_each_dataset_graph(tmp_path, monkeypatch):
    """With no `weights:` block the run must use the model's area weights, per dataset, and a view must
    answer for the parent rather than for `ForecastSourceBase`'s defaults (which would silently give
    uniform weights, refuse lead 0 and strip the provenance and the forcings counters)."""
    names = fixture.DATASET_NAMES
    checkpoint = fixture.save_checkpoint(tmp_path / "two.ckpt")
    fixture.patch_create_runner(monkeypatch, checkpoint)
    source = InferenceForecastSource(checkpoint=str(checkpoint), device="cpu")
    fixture.attach_graph(source)

    for name, view in source.datasets.items():
        assert view.has_graph and view.supports_lead_zero
        assert view.device == source.device and view.frames_per_pass == source.frames_per_pass
        assert view.members == source.members and view.to_config() == source.to_config()
        assert view.provenance() == source.provenance() and view.provenance()["checkpoint"] == str(checkpoint)
        assert set(view.stats) == {"forcings_computed", "forcings_hits", "forcings_bypassed"}
        np.testing.assert_array_equal(
            view.graph_node_attribute("area_weight"), np.arange(1, view.grid.n + 1, dtype=np.float32)
        )

    config = {
        "forecast": {"anemoi_inference": {"checkpoint": str(checkpoint), "device": "cpu"}},
        "lead_time": "12h",
        "init_times": {"dates": [date.isoformat() for date in fixture.init_times()]},
        "metrics": METRICS,
        "bins": BINS,
        "include_lead_zero": True,  # the view supports it; the base class default would refuse it
    }
    evaluation = Evaluation.from_config(
        config, forecast=source, targets={name: fixture.make_targets(name) for name in names}
    )
    for name in names:
        one = evaluation.evaluations[name]
        assert one._weights_spec == {"graph_attribute": "area_weight"}  # not {'uniform': {}}
        np.testing.assert_array_equal(one.weights, np.arange(1, one.forecast.grid.n + 1))
        assert one.lead_times[0] == 0 * fixture.HOUR

    states = evaluation.run()
    per_dataset = {}
    for name, state in states.items():
        attrs = state.attrs
        assert attrs["dataset"] == name
        assert attrs["checkpoint"] == str(checkpoint) and attrs["checkpoint_uuid"]
        assert attrs["forcings_bypassed"] == 0 and attrs["forcings_computed"] > 0
        per_dataset[name] = attrs["forcings_computed"]
        assert state.lead_times[0] == 0 * fixture.HOUR  # lead-0 frames were scored
    # the counters in the N files are each dataset's own, so they add up to the run's
    assert sum(per_dataset.values()) == source.stats["forcings_computed"]
    # the shared rollout is charged once, not once per dataset
    model = {states[name].attrs["time_model_s"] for name in names}
    assert len(model) == 1 and evaluation.timing["model"] == pytest.approx(model.pop())


def test_graph_refuses_a_foreign_key_and_an_unknown_node_set(tmp_path, monkeypatch):
    """With multiple datasets an unrecognised `graph_data` key or node set must raise, naming what is
    available, rather than serving the first dataset's graph on a grid of the wrong size."""
    from torch_geometric.data import HeteroData

    checkpoint = fixture.save_checkpoint(tmp_path / "two.ckpt")
    fixture.patch_create_runner(monkeypatch, checkpoint)
    source = InferenceForecastSource(checkpoint=str(checkpoint), device="cpu")
    fixture.attach_graph(source, keyed=True)
    for name, view in source.datasets.items():  # the per-dataset mapping form resolves by name
        np.testing.assert_array_equal(
            view.graph_node_attribute("area_weight"), np.arange(1, view.grid.n + 1, dtype=np.float32)
        )

    source.runner.model.graph_data = {"other": source.runner.model.graph_data["era5"]}
    with pytest.raises(ValueError, match=r"keyed by \['other'\].*does not name the dataset 'era5'"):
        source.datasets["era5"].graph_node_attribute("area_weight")

    hidden = HeteroData()
    hidden["hidden"].area_weight = torch.ones(3, 1)
    source.runner.model.graph_data = hidden
    with pytest.raises(ValueError, match=r"no node set 'data', it has \['hidden'\]"):
        source.datasets["era5"].graph_node_attribute("area_weight")


def test_output_path_needs_the_dataset_placeholder():
    """One netcdf per dataset: `{dataset}` is required with multiple datasets and refused with one."""
    assert format_output_path("out/run-{shard}.nc", 0, 1) == "out/run-0.nc"
    assert format_output_path("out/run-{dataset}-{shard}.nc", 1, 2, "era5") == "out/run-era5-1.nc"
    with pytest.raises(ValueError, match="{dataset}"):
        format_output_path("out/run-{shard}.nc", 0, 2, "era5")
    with pytest.raises(ValueError, match="{dataset}"):
        format_output_path("out/run-{dataset}.nc", 0, 1)


def test_downscaler_checkpoints_are_refused(tmp_path, monkeypatch):
    """A model that encodes both datasets but decodes only one cannot be run by anemoi-inference at
    all (step1-verification.md A.2), so the evaluation refuses it where the routing is readable."""
    checkpoint = fixture.save_checkpoint(tmp_path / "down.ckpt", routing="downscaler")
    fixture.patch_create_runner(monkeypatch, checkpoint)
    with pytest.raises(ValueError, match="does not decode"):
        InferenceForecastSource(checkpoint=str(checkpoint), device="cpu")


def test_datasets_must_share_the_timing(tmp_path, monkeypatch):
    """`Checkpoint.timestep` silently reports the first dataset's, so the evaluation asserts equality
    across `multi_dataset_metadata` and refuses a checkpoint whose datasets disagree."""

    def patch(metadata):
        metadata["metadata_inference"]["realch1"]["timesteps"]["timestep"] = "12h"

    checkpoint = fixture.save_checkpoint(tmp_path / "skewed.ckpt", patch=patch)
    fixture.patch_create_runner(monkeypatch, checkpoint)
    with pytest.raises(ValueError, match="same timestep"):
        InferenceForecastSource(checkpoint=str(checkpoint), device="cpu")


def test_from_checkpoint_resolves_per_dataset(tmp_path):
    """`from_checkpoint:` has to resolve to one `open_dataset` argument set per dataset."""
    arguments = checkpoint_dataset_arguments(str(fixture.save_checkpoint(tmp_path / "two.ckpt")))
    assert set(arguments) == set(fixture.DATASET_NAMES)
    for name, (args, kwargs) in arguments.items():
        assert len(args) == 1 and not kwargs
        assert name in args[0]["dataset"] or args[0]["dataset"].endswith(".zarr")
    era5, realch1 = (arguments[name][0][0] for name in fixture.DATASET_NAMES)
    assert era5["dataset"] != realch1["dataset"] and realch1["frequency"] == "6h"


def test_regions_and_variables_are_per_dataset(tmp_path, monkeypatch):
    """Every per-dataset block takes either one value for all datasets or `datasets: {name: ...}`."""
    names = fixture.DATASET_NAMES
    checkpoint = fixture.save_checkpoint(tmp_path / "two.ckpt")
    fixture.patch_create_runner(monkeypatch, checkpoint)
    config = {
        "forecast": {"anemoi_inference": {"checkpoint": str(checkpoint), "device": "cpu"}},
        "lead_time": "6h",
        "init_times": {"dates": [fixture.init_times()[0].isoformat()]},
        "metrics": ["rmse"],
        "weights": {"uniform": {}},  # one block, applied to every dataset
        "variables": {"datasets": {names[0]: ["2t", "msl"], names[1]: ["2t"]}},
        "regions": {
            "datasets": {
                names[0]: {"global": "all"},
                names[1]: {"alps": {"bbox": {"north": 48, "west": 5, "south": 45, "east": 11}}},
            }
        },
        "bins": BINS,
    }
    evaluation = Evaluation.from_config(config, targets={name: fixture.make_targets(name) for name in names})
    states = evaluation.run()
    assert states[names[0]].variables == ["2t", "msl"] and states[names[1]].variables == ["2t"]
    assert states[names[0]].regions == ["global"] and states[names[1]].regions == ["alps"]
    with pytest.raises(ValueError, match="unknown dataset"):
        Evaluation.from_config({**config, "weights": {"datasets": {"nope": {"uniform": {}}}}})


def test_from_checkpoint_in_a_config_resolves_both_blocks(tmp_path, monkeypatch):
    """A multi-dataset `from_checkpoint:` fans out: the runner's `input` is keyed by dataset name (anemoi-inference's
    own convention) and the targets become the evaluation's `datasets:` form."""
    checkpoint = fixture.save_checkpoint(tmp_path / "two.ckpt")
    config = load_config(
        {
            "forecast": {
                "anemoi_inference": {
                    "checkpoint": str(checkpoint),
                    "input": {"dataset": {"from_checkpoint": True, "start": 2020}},
                }
            },
            "lead_time": "6h",
            "init_times": {"dates": [fixture.init_times()[0].isoformat()]},
            "targets": {"anemoi_dataset": {"from_checkpoint": True, "cache_bytes": "1GiB"}},
        }
    )
    resolved = config.forecast.anemoi_inference.model_extra["input"]
    assert set(resolved) == set(fixture.DATASET_NAMES)
    assert resolved["era5"]["dataset"]["start"] == 2020
    assert resolved["realch1"]["dataset"]["frequency"] == "6h"
    assert set(config.targets.datasets) == set(fixture.DATASET_NAMES)
    targets = config.targets.datasets["realch1"].anemoi_dataset
    assert targets.cache_bytes == 2**30 and targets.open_dataset_kwargs()["rename"]["T_2M"] == "2t"

    single = load_config(
        {
            "forecast": {
                "anemoi_inference": {"checkpoint": str(fixture.save_checkpoint(tmp_path / "one.ckpt", ("era5",)))}
            },
            "lead_time": "6h",
            "init_times": {"dates": [fixture.init_times()[0].isoformat()]},
            "targets": {"anemoi_dataset": {"from_checkpoint": True}},
        }
    )
    assert "dataset" in single.targets.anemoi_dataset.open_dataset_kwargs()  # unchanged single-dataset shape


def test_sharded_multi_dataset_run_writes_and_merges_one_file_per_dataset(tmp_path, monkeypatch):
    """Sharding is per dataset: each shard writes N files and each dataset's shards merge on their own."""
    names = fixture.DATASET_NAMES
    checkpoint = fixture.save_checkpoint(tmp_path / "two.ckpt")
    fixture.patch_create_runner(monkeypatch, checkpoint)
    fixture.patch_targets_from_forecast(monkeypatch)
    config = {
        "forecast": {"anemoi_inference": {"checkpoint": str(checkpoint), "device": "cpu"}},
        "lead_time": "6h",
        "init_times": {"dates": [date.isoformat() for date in fixture.init_times()]},
        "metrics": ["rmse"],
        "weights": {"uniform": {}},
        "bins": BINS,
        "output": {"path": str(tmp_path / "run-{dataset}-{shard}.nc")},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    main(["run", str(tmp_path / "config.yaml"), "--dry-run", "--step-time", "1.0"])
    for shard in range(2):
        main(["run", str(tmp_path / "config.yaml"), "--shard", f"{shard}/2"])
    for name in names:
        parts = [str(tmp_path / f"run-{name}-{shard}.nc") for shard in range(2)]
        main(["merge", *parts, "-o", str(tmp_path / f"run-{name}.nc")])
        with xr.open_dataset(tmp_path / f"run-{name}.nc", decode_timedelta=True) as merged:
            assert merged.attrs["dataset"] == name
            assert merged.sizes["variable"] == 7 and "shard" not in merged.attrs
    with pytest.raises(ValueError, match="different datasets"):
        main(
            ["merge", str(tmp_path / "run-era5-0.nc"), str(tmp_path / "run-realch1-1.nc"), "-o", str(tmp_path / "x.nc")]
        )


def _two_dataset_config(checkpoint, **overrides) -> dict:
    """A two-dataset config over the fixture checkpoint, without an output block."""
    return {
        "forecast": {"anemoi_inference": {"checkpoint": str(checkpoint), "device": "cpu"}},
        "lead_time": "12h",
        "init_times": {"dates": [date.isoformat() for date in fixture.init_times()]},
        "metrics": METRICS,
        "weights": {"uniform": {}},
        "bins": BINS,
        **overrides,
    }


def test_scoring_a_subset_is_the_single_dataset_run_of_that_dataset(tmp_path, monkeypatch):
    """`datasets: [name]` scores one dataset of a two-dataset checkpoint and nothing else.

    The model still predicts both — the runner runs every decoder and needs an input state for each dataset, so
    the init times of the skipped one are still checked — but only the selected dataset gets targets, an
    aggregator and a result, which is the single-dataset run of that dataset, bit for bit.
    """
    scored, skipped = fixture.DATASET_NAMES[1], fixture.DATASET_NAMES[0]
    reference_evaluation, reference = _single(tmp_path, monkeypatch, scored)

    checkpoint = fixture.save_checkpoint(tmp_path / "two.ckpt")
    fixture.patch_create_runner(monkeypatch, checkpoint)
    source = InferenceForecastSource(checkpoint=str(checkpoint), device="cpu")
    checked, original = [], source.check_init_times
    monkeypatch.setattr(
        source,
        "check_init_times",
        lambda init_times, lead_time, dataset=None: (checked.append(dataset), original(init_times, lead_time, dataset)),
    )
    evaluation = Evaluation.from_config(
        _two_dataset_config(checkpoint, datasets=[scored]),
        forecast=source,
        targets={scored: fixture.make_targets(scored)},
    )
    # one dataset selected: an ordinary single-dataset evaluation, not a MultiEvaluation
    assert isinstance(evaluation, Evaluation) and not hasattr(evaluation, "evaluations")
    assert evaluation.dataset == scored and evaluation.forecast.solo
    assert sorted(checked) == sorted(fixture.DATASET_NAMES)  # every dataset's inputs are checked (FR-7)

    plan = evaluation.plan()
    assert plan["forecast"]["datasets_scored"] == [scored]
    assert plan["forecast"]["datasets_predicted_only"] == [skipped]
    # the runner holds the skipped dataset's state too, so the memory estimate keeps it
    assert plan["bytes"]["predicted_only_state_bytes"] > 0

    state = evaluation.run()
    assert set(state.sums) == set(reference.sums)
    assert all(torch.equal(state.sums[key], reference.sums[key]) for key in reference.sums)
    assert torch.equal(state.weights, reference.weights) and torch.equal(state.n_init, reference.n_init)

    written, expected = str(tmp_path / "subset.nc"), str(tmp_path / "whole.nc")
    write(state.to_xarray(evaluation.metrics), written)
    write(reference.to_xarray(reference_evaluation.metrics), expected)
    with xr.open_dataset(written, decode_timedelta=True) as result:
        with xr.open_dataset(expected, decode_timedelta=True) as wanted:
            result, wanted = result.load(), wanted.load()
            xr.testing.assert_equal(result, wanted)
            assert _kept(result.attrs) == _kept(wanted.attrs)
            # the dataset is named even though the run wrote one file: cheap, and unambiguous
            assert result.attrs["dataset"] == scored and "dataset" not in wanted.attrs
    evaluation.close()
    assert source.runner is None  # the sole view owns the runner


def test_the_selected_datasets_are_refused_when_they_are_not_the_checkpoints(tmp_path, monkeypatch):
    """An unknown name, an empty list, and a per-dataset block naming a dataset the run does not score."""
    names = fixture.DATASET_NAMES
    checkpoint = fixture.save_checkpoint(tmp_path / "two.ckpt")
    fixture.patch_create_runner(monkeypatch, checkpoint)
    targets = {name: fixture.make_targets(name) for name in names}

    with pytest.raises(ValueError, match=r"unknown datasets \['cerra'\], the checkpoint has \['era5', 'realch1'\]"):
        Evaluation.from_config(_two_dataset_config(checkpoint, datasets=["cerra"]), targets=targets)
    with pytest.raises(ValueError, match="at least one dataset"):
        Evaluation.from_config(_two_dataset_config(checkpoint, datasets=[]), targets=targets)
    with pytest.raises(ValueError, match=r"unknown datasets \['realch1'\], the run scores \['era5'\]"):
        Evaluation.from_config(
            _two_dataset_config(
                checkpoint,
                datasets=["era5"],
                variables={"datasets": {name: ["2t"] for name in names}},
            ),
            targets={"era5": targets["era5"]},
        )


def test_a_single_dataset_checkpoint_takes_the_key_only_for_its_own_dataset(tmp_path, monkeypatch):
    """`datasets:` names the checkpoint's datasets, so a single-dataset checkpoint accepts its one name and
    nothing else; the result is unchanged, `dataset` attr included."""
    checkpoint = fixture.save_checkpoint(tmp_path / "one.ckpt", ("era5",))
    fixture.patch_create_runner(monkeypatch, checkpoint, ("era5",))
    config = {
        **_two_dataset_config(checkpoint, datasets=["era5"]),
        "lead_time": "6h",
        "init_times": {"dates": [fixture.init_times()[0].isoformat()]},
    }
    evaluation = Evaluation.from_config(config, targets=fixture.make_targets("era5"))
    assert isinstance(evaluation, Evaluation) and evaluation.dataset is None
    state = evaluation.run()
    assert "dataset" not in state.attrs  # a genuinely single-dataset run must not gain the attr
    assert "datasets_scored" not in evaluation.plan()["forecast"]
    with pytest.raises(ValueError, match=r"unknown datasets \['realch1'\], the checkpoint has \['era5'\]"):
        Evaluation.from_config({**config, "datasets": ["realch1"]}, targets=fixture.make_targets("era5"))
