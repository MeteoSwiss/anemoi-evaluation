import datetime
import json

import numpy as np
import pytest
import torch

from anemoi.evaluation.evaluate import Evaluation
from anemoi.evaluation.frame import Grid
from anemoi.evaluation.sources.climatology import ArrayClimatology
from anemoi.evaluation.sources.fake import ArrayTargets

HOUR = datetime.timedelta(hours=1)
T0 = datetime.datetime(2024, 1, 1)
CPU = torch.device("cpu")


def test_array_climatology(tmp_path):
    rng = np.random.default_rng(5)
    grid = Grid(rng.uniform(-90, 90, 12), rng.uniform(0, 360, 12))
    fields = {hour: rng.standard_normal((2, 12)).astype(np.float32) for hour in (0, 6)}
    fields[0][0, 2] = np.nan
    climatology = ArrayClimatology(grid, ["t", "q"], fields, "hour_of_day", counts={0: 3, 6: 3})
    frame = climatology.frame(T0 + 6 * HOUR, ["q"], CPU)
    np.testing.assert_array_equal(frame.numpy(), fields[6][1:])
    assert climatology.frame(T0 + 30 * HOUR, ["q"], CPU) is frame
    with pytest.raises(ValueError, match="hour_of_day 3"):
        climatology.frame(T0 + 3 * HOUR, ["q"], CPU)
    with pytest.raises(ValueError):
        climatology.frame(T0, ["nope"], CPU)
    with pytest.raises(ValueError):
        ArrayClimatology(grid, ["t"], fields, "hour_of_day")
    assert climatology.nonfinite_nodes(["t", "q"]) == {"t": 1, "q": 0}
    assert climatology.describe()["keys"] == [0, 6] and climatology.describe()["bytes"] == 2 * 2 * 12 * 4
    with pytest.raises(ValueError):
        climatology.to_config()
    keys = {"month": 3, "day_of_year": 61, "constant": 0}
    assert {
        k: ArrayClimatology(grid, ["t", "q"], {v: fields[0]}, k).key_of(datetime.datetime(2024, 3, 1))
        for k, v in keys.items()
    } == keys

    path = tmp_path / "climatology.nc"
    climatology.to_netcdf(path)
    loaded = ArrayClimatology.from_netcdf(path)
    assert loaded.key == "hour_of_day" and loaded.variables == ["t", "q"] and loaded.counts == {0: 3, 6: 3}
    for hour in (0, 6):
        np.testing.assert_array_equal(loaded.fields[hour], fields[hour])
    assert loaded.to_config() == {"file": str(path)} and loaded.grid.n == 12

    dates = [T0 + 6 * k * HOUR for k in range(8)]
    values = {date: rng.standard_normal((2, 12)).astype(np.float32) for date in dates}
    targets = ArrayTargets(grid, ["t", "q"], {date: field for date, field in values.items() if date != dates[5]})
    built = ArrayClimatology.from_targets(targets, dates, ["q", "t"], "hour_of_day")
    assert built.counts == {0: 2, 6: 1, 12: 2, 18: 2} and built.variables == ["q", "t"]
    expected = np.mean([values[dates[0]][::-1], values[dates[4]][::-1]], axis=0, dtype=np.float64).astype(np.float32)
    np.testing.assert_allclose(built.fields[0], expected, rtol=1e-6)
    np.testing.assert_array_equal(built.fields[6], values[dates[1]][::-1])


def numpy_acc(evaluation, climatology, nan_node=None):
    """Pooled ACC per (lead, variable) over the finite climatology nodes, uniform weights, global region."""
    sums = {}
    for frame, target in evaluation.pairs():
        c = climatology.frame(frame.valid_time, evaluation.variables, CPU).numpy().astype(np.float64)
        mean = frame.data.numpy().astype(np.float64).mean(0)
        fa, ta = mean - c, target[0].numpy().astype(np.float64) - c
        finite = np.isfinite(c)
        totals = sums.setdefault(frame.lead_time, np.zeros((3, len(evaluation.variables))))
        totals[0] += np.where(finite, fa * ta, 0.0).sum(1)
        totals[1] += np.where(finite, fa * fa, 0.0).sum(1)
        totals[2] += np.where(finite, ta * ta, 0.0).sum(1)
    return np.stack([sums[lead][0] / np.sqrt(sums[lead][1] * sums[lead][2]) for lead in evaluation.lead_times])


def test_acc_end_to_end(fake_sources, tmp_path):
    forecast, targets = fake_sources(members=3, drift=0.5, spread=0.2)
    rng = np.random.default_rng(6)
    fields = {hour: rng.standard_normal((2, forecast.grid.n)).astype(np.float32) for hour in (0, 6, 12, 18)}
    inits = [T0, T0 + 12 * HOUR]
    climatology = ArrayClimatology(forecast.grid, targets.variables, fields, "hour_of_day")
    evaluation = Evaluation(forecast, targets, inits, "24h", ["rmse", "acc"], climatology=climatology)
    reference = evaluation.run()
    dataset = reference.to_xarray(evaluation.metrics).sel(bin="all", region="global")
    np.testing.assert_allclose(dataset["acc"].values, numpy_acc(evaluation, climatology), rtol=1e-12)
    assert set(reference.sums) == {
        "squared_error",
        "anomaly_product",
        "forecast_anomaly_squared",
        "target_anomaly_squared",
    }

    fields[6][0, 3] = np.nan
    with_nan = ArrayClimatology(forecast.grid, targets.variables, fields, "hour_of_day")
    evaluation = Evaluation(forecast, targets, inits, "24h", ["rmse", "acc"], climatology=with_nan)
    state = evaluation.run()
    assert torch.equal(state.weights, reference.weights) and torch.equal(
        state.sums["squared_error"], reference.sums["squared_error"]
    )
    assert not torch.equal(state.sums["anomaly_product"], reference.sums["anomaly_product"])
    dataset = state.to_xarray(evaluation.metrics).sel(bin="all", region="global")
    np.testing.assert_allclose(dataset["acc"].values, numpy_acc(evaluation, with_nan), rtol=1e-12)
    assert json.loads(state.attrs["climatology_nonfinite_nodes"]) == {"t": 1, "q": 0}

    with pytest.raises(ValueError, match="climatology"):
        Evaluation(forecast, targets, inits, "24h", ["rmse", "acc"])
    path = tmp_path / "climatology.nc"
    climatology.to_netcdf(path)
    config = {
        "forecast": {"anemoi_inference": {"checkpoint": "/x.ckpt"}},
        "lead_time": "24h",
        "init_times": {"dates": [t.isoformat() for t in inits]},
        "metrics": ["rmse", "acc"],
        "weights": {"uniform": {}},
        "climatology": {"file": str(path)},
    }
    evaluation = Evaluation.from_config(config, forecast=forecast, targets=targets)
    assert evaluation.climatology.key == "hour_of_day" and evaluation.to_config()["climatology"] == {"file": str(path)}
    assert evaluation.plan()["climatology"]["keys"] == [0, 6, 12, 18]
