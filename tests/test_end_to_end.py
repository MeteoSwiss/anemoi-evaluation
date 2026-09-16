import datetime
import json
import logging

import numpy as np
import pytest
import torch
import xarray as xr

from anemoi.evaluation import metrics
from anemoi.evaluation.__main__ import _humanise
from anemoi.evaluation.__main__ import format_output_path
from anemoi.evaluation.__main__ import main
from anemoi.evaluation.__main__ import resolve_shard
from anemoi.evaluation.aggregation import AggregationState
from anemoi.evaluation.evaluate import STARTUP_SECONDS
from anemoi.evaluation.evaluate import Evaluation
from anemoi.evaluation.frame import Frame
from anemoi.evaluation.frame import Grid
from anemoi.evaluation.output import load_state
from anemoi.evaluation.output import merge
from anemoi.evaluation.output import to_xarray
from anemoi.evaluation.output import write
from anemoi.evaluation.sources.base import MissingTargetError
from anemoi.evaluation.sources.fake import ArrayTargets
from anemoi.evaluation.sources.fake import FakeForecastSource
from anemoi.evaluation.sources.persistence import PersistenceForecastSource

HOUR = datetime.timedelta(hours=1)
T0 = datetime.datetime(2024, 1, 1)
INITS = [T0, T0 + 12 * HOUR]
METRICS = [
    "rmse",
    "mae",
    "bias",
    "crps",
    "fair_crps",
    "spread",
    "crps_0p5",
    "spread_skill",
    "member_rmse",
    "member_mae",
]


def evaluate(forecast, targets, **kwargs):
    metrics = [{"crps": {"alpha": 0.5}} if m == "crps_0p5" else m for m in METRICS]
    return Evaluation(forecast, targets, INITS, "24h", metrics, **kwargs)


def test_metrics_match_closed_form(fake_sources):
    drift, spread = 0.5, 0.2
    forecast, targets = fake_sources(members=3, drift=drift, spread=spread)
    evaluation = evaluate(forecast, targets)
    full = evaluation.run().to_xarray(evaluation.metrics)
    dataset = full.sel(bin="all", region="global")

    d = np.arange(1, 5) * drift
    skill = (np.abs(d - spread) + np.abs(d) + np.abs(d + spread)) / 3
    pairs = 4 * spread
    expected = {
        "bias": d,
        "rmse": np.abs(d),
        "mae": np.abs(d),
        "spread": np.full(4, spread),
        "crps": skill - pairs / 9,
        "fair_crps": skill - pairs / 6,
        "crps_0p5": skill - (0.5 / 6 + 0.5 / 9) * pairs,
        "spread_skill": spread / np.abs(d),
        "member_rmse": np.sqrt(d**2 + 2 * spread**2 / 3),
        "member_mae": skill,
    }
    for name, values in expected.items():
        np.testing.assert_allclose(dataset[name].values, np.repeat(values[:, None], 2, axis=1), rtol=1e-5)
    assert dataset["n_init"].values.tolist() == [2, 2, 2, 2]
    assert np.isnan(full["rmse"].sel(bin="JJA")).all()
    xr.testing.assert_allclose(full.sel(bin="DJF", drop=True), full.sel(bin="all", drop=True))
    assert full["param"].values.tolist() == ["", ""] and np.isnan(full["level"].values).all()

    forecast, targets = fake_sources(members=3, drift=drift, spread=spread, frames_per_pass=2)
    evaluation = evaluate(forecast, targets)
    xr.testing.assert_allclose(evaluation.run().to_xarray(evaluation.metrics), full)


def test_lead_zero(fake_sources):
    forecast, targets = fake_sources(members=3, drift=0.5, spread=0.2)
    evaluation = evaluate(forecast, targets, include_lead_zero=True)
    assert evaluation.lead_times[0] == datetime.timedelta(0)
    plan = evaluation.plan()
    assert plan["lead_times"] == {"count": 5, "first": "0h", "last": "1d"} and plan["model_calls"] == 8
    dataset = evaluation.run().to_xarray(evaluation.metrics).sel(bin="all", region="global")
    zero = dataset.isel(lead_time=0)
    assert float(zero["rmse"].max()) == 0.0 and float(zero["spread"].max()) == 0.0 and float(zero["crps"].max()) == 0.0
    assert np.isnan(zero["spread_skill"].values).all() and dataset["n_init"].values.tolist() == [2] * 5
    np.testing.assert_allclose(
        dataset["rmse"].values[1:], np.repeat((np.arange(1, 5) * 0.5)[:, None], 2, axis=1), rtol=1e-5
    )
    forecast.supports_lead_zero = False
    with pytest.raises(ValueError, match="lead-0"):
        evaluate(forecast, targets, include_lead_zero=True)


class LockstepSource(FakeForecastSource):
    """Interleaved per-member generators that each enter inference mode, as the anemoi-inference runner does."""

    def frames(self, *args):
        def member(m):
            with torch.inference_mode():
                for frame in FakeForecastSource.frames(self, *args):
                    yield frame, frame.data[m]

        generators = [member(m) for m in range(self.members)]
        try:
            for outputs in zip(*generators):
                frame = outputs[0][0]
                yield Frame(
                    frame.init_time,
                    frame.lead_time,
                    frame.valid_time,
                    frame.variables,
                    torch.stack([d for _, d in outputs]),
                )
        finally:
            for generator in generators:
                generator.close()


def test_pairs_restores_inference_mode(fake_sources):
    forecast, targets = fake_sources(members=3)
    forecast.__class__ = LockstepSource
    evaluation = evaluate(forecast, targets)
    assert not torch.is_inference_mode_enabled()
    pairs = list(evaluation.pairs())
    assert not torch.is_inference_mode_enabled()
    assert len(pairs) == 8 and all(frame.members == 3 for frame, _ in pairs)
    evaluation.run()
    assert not torch.is_inference_mode_enabled()


def test_missing_target_and_member_requirement(fake_sources):
    forecast, targets = fake_sources(missing=(T0 + 18 * HOUR,))
    evaluation = evaluate(forecast, targets)
    dataset = evaluation.run().to_xarray(evaluation.metrics).sel(bin="all", region="global")
    assert dataset["n_init"].values.tolist() == [1, 2, 1, 2]
    np.testing.assert_allclose(dataset["rmse"].values, np.repeat(np.arange(1, 5)[:, None] * 0.5, 2, axis=1), rtol=1e-6)
    with pytest.raises(MissingTargetError):
        evaluate(forecast, targets, on_missing_target="raise").run()

    forecast, targets = fake_sources(members=1)
    with pytest.raises(ValueError):
        evaluate(forecast, targets)

    forecast, targets = fake_sources()
    evaluation = evaluate(forecast, targets, weights={"graph_attribute": "area_weight"})
    assert len(list(evaluation.pairs([T0]))) == 4
    with pytest.raises(ValueError, match="graph"):
        evaluation.run()


def test_netcdf_round_trip_and_cli_merge(fake_sources, tmp_path):
    forecast, targets = fake_sources()
    evaluation = evaluate(forecast, targets)
    state = evaluation.run()
    full = state.to_xarray(evaluation.metrics)
    write(full, tmp_path / "full.nc")
    loaded = load_state(tmp_path / "full.nc")
    assert loaded.coords == state.coords and loaded.attrs["init_times"] == state.attrs["init_times"]
    assert loaded.init_times == state.init_times == INITS and loaded.members == 3
    assert all(torch.equal(loaded.sums[name], state.sums[name]) for name in state.sums)
    assert torch.equal(loaded.weights, state.weights) and torch.equal(loaded.n_init, state.n_init)
    assert full["init_time"].values.tolist() == [np.datetime64(t, "ns").astype(int) for t in INITS]
    assert json.loads(full.attrs["package_versions"])["anemoi-evaluation"]

    parts = [str(tmp_path / f"part{i}.nc") for i in range(len(INITS))]
    for init_time, part in zip(INITS, parts):
        write(evaluation.run([init_time]).to_xarray(evaluation.metrics), part)
    main(["merge", *parts, "-o", str(tmp_path / "merged.nc")])
    with xr.open_dataset(tmp_path / "merged.nc", decode_timedelta=True) as merged:
        merged = merged.load()
    xr.testing.assert_equal(merged.drop_attrs(), full.drop_attrs())
    assert json.loads(merged.attrs["merged_from"]) == parts
    assert merged.attrs["metrics"] == full.attrs["metrics"]
    assert merged.attrs["init_times"] == full.attrs["init_times"]
    assert len(json.loads(merged.attrs["config"])["merge"]) == 2
    with pytest.raises(ValueError):
        main(["merge", parts[0], parts[0], "-o", str(tmp_path / "twice.nc")])


def test_merge_validates_shard_sets(tmp_path, caplog):
    def shard(index, count, day, peak, config="c", tag=None, name=None):
        state = AggregationState.zeros([6 * HOUR], ["all"], ["a"], ["global"], ["squared_error"], 1)
        state.init_times = [T0 + day * 24 * HOUR]
        state.attrs.update(
            shard=f"{index}/{count}" if tag is None else tag,
            config=config,
            time_total_s=1.5,
            model_calls=2,
            peak_gpu_memory_bytes=peak,
        )
        path = tmp_path / (name or f"{index}of{count}.nc")
        write(to_xarray(state, [metrics.RMSE()]), path)
        return str(path)

    parts = [shard(index, 3, index, peak) for index, peak in enumerate((10, 30, 20))]
    merged = merge(parts)
    assert "shard" not in merged.attrs and len(merged.init_times) == 3
    assert merged.attrs["peak_gpu_memory_bytes"] == 30 and merged.attrs["model_calls"] == 6
    assert merged.attrs["time_total_s"] == 4.5

    untagged = str(tmp_path / "untagged.nc")
    write(to_xarray(merged, [metrics.RMSE()]), untagged)
    for match, items in (
        ("missing 1", [parts[0], parts[2]]),
        ("duplicate shards", [*parts, shard(2, 3, 2, 20, name="again.nc")]),
        ("number of shards", [parts[0], parts[1], shard(0, 2, 5, 10)]),
        ("no shard attr", [*parts, untagged]),
        ("expected 'i/n'", [shard(0, 3, 6, 10, tag="3 of 3", name="bad.nc")]),
    ):
        with pytest.raises(ValueError, match=match):
            merge(items)

    with caplog.at_level(logging.WARNING):
        merge([parts[0], parts[1], shard(2, 3, 2, 20, config="other", name="other.nc")])
    assert "different configs" in caplog.text

    output = tmp_path / "partial.nc"
    main(["merge", parts[0], parts[2], "-o", str(output), "--partial"])
    with xr.open_dataset(output, decode_timedelta=True) as dataset:
        assert dataset.attrs["shard"] == "0,2/3" and len(dataset["init_time"]) == 2
    complete = merge([str(output), parts[1]])
    assert "shard" not in complete.attrs and len(complete.init_times) == 3


def test_sharding_and_dry_run(fake_sources):
    step = {"SLURM_STEP_NUM_TASKS": "4", "SLURM_PROCID": "2"}
    array = {"SLURM_ARRAY_TASK_COUNT": "3", "SLURM_ARRAY_TASK_ID": "5", "SLURM_ARRAY_TASK_MIN": "4"}
    assert resolve_shard("1/4", {}) == (1, 4)
    assert resolve_shard(None, {}) == resolve_shard(None, {"SLURM_NTASKS": "4", "SLURM_PROCID": "0"}) == (0, 1)
    assert resolve_shard(None, step) == (2, 4) and resolve_shard(None, array) == (1, 3)
    assert resolve_shard(None, {**step, **array}) == (6, 12) and resolve_shard("0/1", step) == (0, 1)
    for bad in ("4/4", "-1/2", "x"):
        with pytest.raises(ValueError):
            resolve_shard(bad, {})
    assert format_output_path("a-{shard}of{shards}.nc", 1, 4) == "a-1of4.nc"
    assert _humanise({"bytes": {"frame": 2048}, "counts": {0: 31}}) == {
        "bytes": {"frame": "2 KiB (2048)"},
        "counts": {0: 31},
    }
    assert format_output_path("a.nc", 0, 1) == "a.nc"
    with pytest.raises(ValueError):
        format_output_path("a.nc", 0, 2)

    forecast, targets = fake_sources()
    evaluation = evaluate(forecast, targets, bins="init_time")
    plan = evaluation.plan(evaluation.shard(1, 2))
    assert plan["init_times"]["count"] == 1 and plan["lead_times"] == {"count": 4, "first": "6h", "last": "1d"}
    assert plan["model_calls"] == 4 and plan["computed_lead_time"] == "1d" and "note" not in plan
    assert plan["frames"] == {"total": 4, "with_targets": 4, "distinct_valid_times": 4}
    assert plan["weights"] == {"uniform": {}} and plan["regions"] == {"global": "all"} and plan["bins"]["count"] == 2
    assert plan["bytes"]["frame"] == 3 * 2 * 30 * 4 and plan["forecast"]["members"] == 3
    assert "weights" not in evaluation.__dict__ and "regions" not in evaluation.__dict__
    full = evaluation.run()
    parts = [evaluation.run(evaluation.shard(i, 2)) for i in range(2)]
    assert [part.init_times for part in parts] == [[INITS[0]], [INITS[1]]]
    merged = parts[0].merge(parts[1])
    assert all(torch.equal(merged.sums[name], full.sums[name]) for name in full.sums)
    assert torch.equal(merged.weights, full.weights) and torch.equal(merged.n_init, full.n_init)
    with pytest.raises(ValueError):
        evaluation.shard(2, 2)

    class HintTargets(ArrayTargets):
        def prefetch_hint(self, frames_per_pass):
            self.hint = frames_per_pass

    forecast, targets = fake_sources(frames_per_pass=3)
    targets.__class__ = HintTargets
    plan = evaluate(forecast, targets).plan()
    assert targets.hint == 3 and plan["model_calls"] == 4 and plan["computed_lead_time"] == "36h" and "note" in plan


def test_dry_run_time_estimate(fake_sources):
    forecast, targets = fake_sources(members=3)  # 2 init times, 24 h of lead in 6 h steps: 4 model calls per init
    evaluation = evaluate(forecast, targets)
    assert "time" not in evaluation.plan()
    estimate = evaluation.plan(step_time=2.0)["time"]
    assert estimate["step_time"] == 2.0 and estimate["model_per_init_s"] == 4 * 3 * 2.0
    assert estimate["shard"] == {"model_s": 48.0, "wall_s": 48.0 + STARTUP_SECONDS}
    assert estimate["run"] == {"gpu_hours": round(48.0 / 3600, 2), "wall_per_shard_s": 48.0 + STARTUP_SECONDS}
    assert estimate["startup_s"] == STARTUP_SECONDS and "prefetch" in estimate["note"]
    with pytest.raises(ValueError, match="step_time"):
        evaluation.plan(step_time=0)

    inits = [T0 + k * 6 * HOUR for k in range(3)]
    evaluation = Evaluation(forecast, targets, inits, "24h", ["rmse"])
    estimate = evaluation.plan(evaluation.shard(1, 2), step_time=2.0, shards=2)["time"]
    assert estimate["shard"]["model_s"] == 24.0  # one of the three init times
    assert estimate["run"] == {"gpu_hours": round(72.0 / 3600, 2), "wall_per_shard_s": 48.0 + STARTUP_SECONDS}

    forecast, targets = fake_sources(members=1)
    assert Evaluation(forecast, targets, INITS, "24h", ["rmse"]).plan(step_time=2.0)["time"]["shard"] == {
        "model_s": 16.0,
        "wall_s": 16.0 + STARTUP_SECONDS,
    }

    forecast, targets = fake_sources(members=3, frames_per_pass=3)  # 2 calls per init instead of 4
    assert evaluate(forecast, targets).plan(step_time=2.0)["time"]["model_per_init_s"] == 12.0

    assert _humanise({"time": {"shard": {"wall_s": 90.0}}, "step_time": 2.5}) == {
        "time": {"shard": {"wall_s": "1 minute 30 seconds (90.0)"}},
        "step_time": 2.5,
    }
    with pytest.raises(SystemExit, match="--step-time"):
        main(["run", "no-such-config.yaml", "--step-time", "3.0"])


def test_persistence_source():
    rng = np.random.default_rng(3)
    grid = Grid(rng.uniform(-90, 90, 20), rng.uniform(0, 360, 20))
    base = rng.standard_normal((2, 20)).astype(np.float32)
    fields = {T0 + 6 * k * HOUR: base + 0.25 * k for k in range(7)}
    targets = ArrayTargets(grid, ["t", "q"], fields)
    config = {
        "forecast": {"persistence": {"timestep": "6h", "members": 2}},
        "lead_time": "24h",
        "init_times": {"dates": [t.isoformat() for t in INITS]},
        "metrics": ["rmse", "bias", "crps", "mae"],
        "weights": {"uniform": {}},
        "bins": {"time": "none"},
    }
    evaluation = Evaluation.from_config(config, targets=targets)
    assert isinstance(evaluation.forecast, PersistenceForecastSource) and evaluation.to_config()["forecast"] == {
        "persistence": {"timestep": "6h", "members": 2}
    }
    dataset = evaluation.run().to_xarray(evaluation.metrics).sel(bin="all", region="global")
    expected = np.repeat(np.arange(1, 5)[:, None] * 0.25, 2, axis=1)
    np.testing.assert_allclose(dataset["rmse"].values, expected, rtol=1e-6)
    np.testing.assert_allclose(dataset["bias"].values, -expected, rtol=1e-6)
    np.testing.assert_allclose(dataset["crps"].values, dataset["mae"].values, rtol=1e-6)  # identical members
    assert dataset["n_init"].values.tolist() == [2, 2, 2, 2]

    partial = ArrayTargets(grid, ["t", "q"], {t: f for t, f in fields.items() if t != INITS[1]})
    evaluation = Evaluation.from_config(config, targets=partial)
    dataset = evaluation.run().to_xarray(evaluation.metrics).sel(bin="all", region="global")
    assert dataset["n_init"].values.tolist() == [1, 0, 1, 1]
    with pytest.raises(ValueError, match="targets"):
        Evaluation.from_config(config)
