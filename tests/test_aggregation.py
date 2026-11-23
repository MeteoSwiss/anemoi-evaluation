import datetime

import numpy as np
import pytest
import torch

from anemoi.evaluation import metrics
from anemoi.evaluation.aggregation import Aggregator
from anemoi.evaluation.binning import SeasonBinning
from anemoi.evaluation.binning import build
from anemoi.evaluation.frame import Frame
from anemoi.evaluation.output import to_xarray

HOUR = datetime.timedelta(hours=1)


def test_state_matches_naive_reference_and_merges_exactly():
    rng = np.random.default_rng(0)
    n, variables, regions = 40, ["a", "b", "c"], ["all", "box"]
    weights = rng.random(n) + 0.1
    masks = np.stack([np.ones(n, dtype=bool), rng.random(n) < 0.5], axis=1)
    inits = [datetime.datetime(2024, 1, 1), datetime.datetime(2024, 7, 1)]
    leads = [6 * HOUR, 12 * HOUR]
    frames = []
    for init in inits:
        for lead in leads:
            pred = (rng.integers(-8, 9, (2, 3, n)) / 4).astype(np.float32)
            target = (rng.integers(-8, 9, (1, 3, n)) / 4).astype(np.float32)
            pred[0, 0, :3] = np.nan
            target[0, 1, 5:8] = np.nan
            frames.append((Frame(init, lead, init + lead, variables, torch.from_numpy(pred)), torch.from_numpy(target)))

    ms = [metrics.RMSE(), metrics.MAE(), metrics.Bias()]
    aggregator = Aggregator(weights, masks, metrics.unique_statistics(ms), SeasonBinning("init_time"))
    state = aggregator.new_state(leads, variables, regions, 2)
    for frame, target in frames:
        aggregator.add(state, frame, target)

    shape = (2, 4, 3, 2)
    sums = {name: np.zeros(shape) for name in ("error", "squared_error", "absolute_median_error")}
    weight_sum, n_init = np.zeros(shape), np.zeros(shape[:2], dtype=np.int64)
    W = weights[:, None] * masks
    for frame, target in frames:
        pred, y = frame.data.numpy().astype(np.float64), target.numpy()[0].astype(np.float64)
        valid = np.isfinite(pred).all(0) & np.isfinite(y)
        error, median_error = pred.mean(0) - y, np.median(pred, 0) - y
        lead, season = leads.index(frame.lead_time), {1: 0, 7: 2}[frame.init_time.month]
        for name, values in (
            ("error", error),
            ("squared_error", error**2),
            ("absolute_median_error", np.abs(median_error)),
        ):
            sums[name][lead, season] += np.where(valid, values, 0.0) @ W
        weight_sum[lead, season] += valid.astype(np.float64) @ W
        n_init[lead, season] += 1

    def pooled(a):
        return np.concatenate([a, a.sum(1, keepdims=True)], axis=1)

    with np.errstate(invalid="ignore"):
        for name, mean in state.means().items():
            np.testing.assert_allclose(mean.numpy(), sums[name] / weight_sum, rtol=1e-10)
        expected = {
            "rmse": np.sqrt(pooled(sums["squared_error"]) / pooled(weight_sum)),
            "mae": pooled(sums["absolute_median_error"]) / pooled(weight_sum),
            "bias": pooled(sums["error"]) / pooled(weight_sum),
        }
    dataset = to_xarray(state, ms)
    assert dataset["bin"].values.tolist() == ["DJF", "MAM", "JJA", "SON", "all"]
    assert dataset["state_bin"].values.tolist() == ["DJF", "MAM", "JJA", "SON"]
    assert (dataset.attrs["bin_kind"], dataset.attrs["bin_by"], dataset.attrs["members"]) == ("season", "init_time", 2)
    for name, values in expected.items():
        np.testing.assert_allclose(dataset[name].values, values, rtol=1e-10)
    np.testing.assert_array_equal(dataset["n_init"].values, pooled(n_init))
    np.testing.assert_allclose(dataset["weight_sum"].values, pooled(weight_sum), rtol=1e-12)

    winter, summer = (aggregator.new_state(leads, variables, regions, 2) for _ in range(2))
    for frame, target in frames:
        aggregator.add(winter if frame.init_time.month == 1 else summer, frame, target)
    merged = winter.merge(summer)
    assert all(torch.equal(merged.sums[name], state.sums[name]) for name in state.sums)
    assert torch.equal(merged.weights, state.weights) and torch.equal(merged.n_init, state.n_init)
    with pytest.raises(ValueError):
        winter.merge(aggregator.new_state(leads, variables, regions, 3))
    with pytest.raises(ValueError):
        aggregator.add(winter, frames[0][0], frames[0][1][:, :1])


def test_variable_subset_aggregates_like_the_matching_slice():
    """A frame holding only some of the variables gives, bit for bit, those variables' sums of the full frame, which is
    what lets members be scored on variable blocks; three regions and an odd node count, where one product over all
    the variables groups the terms differently with the variable count."""
    rng = np.random.default_rng(3)
    members, n, variables, leads = 5, 1001, ["a", "b", "c", "d", "e"], [6 * HOUR]
    weights = rng.random(n) + 0.1
    masks = np.stack([np.ones(n, dtype=bool), rng.random(n) < 0.5, rng.random(n) < 0.3], axis=1)
    init = datetime.datetime(2024, 1, 1)
    pred = torch.from_numpy(rng.normal(280, 2, (members, 5, n)).astype(np.float32))
    target = torch.from_numpy(rng.normal(280, 2, (1, 5, n)).astype(np.float32))
    climatology = torch.from_numpy(rng.normal(280, 1, (5, n)))
    pred[1, 2, :4], target[0, 1, 10:13], climatology[3, 20:25] = float("nan"), float("nan"), float("nan")
    warm = {"label": "warm", "thresholds": {"b": 280.5}}
    specs = ["rmse", "mae", "bias", "fair_crps", "spread", "member_rmse", "acc", "rank_histogram"]
    specs += [{"csi": warm}, {"brier": warm}, {"reliability": warm}]

    def aggregate(columns):
        statistics = metrics.unique_statistics(metrics.build(specs, [variables[j] for j in columns], members))
        aggregator = Aggregator(weights, masks, statistics, SeasonBinning("init_time"))
        state = aggregator.new_state(leads, [variables[j] for j in columns], ["all", "half", "third"], members)
        frame = Frame(init, leads[0], init + leads[0], state.variables, pred[:, columns].contiguous())
        aggregator.add(state, frame, target[:, columns].contiguous(), {"climatology": climatology[columns]})
        return state

    full = aggregate(list(range(5)))
    # each subset binds its own variables, so the map names only `b`, which every subset holds
    for columns in ([1, 2], [1, 3, 4], [1]):
        part = aggregate(columns)
        assert torch.equal(part.weights, full.weights[:, :, columns])
        assert part.sums.keys() == full.sums.keys()
        for name, total in part.sums.items():
            torch.testing.assert_close(total, full.sums[name][:, :, columns], rtol=0, atol=0, equal_nan=True, msg=name)


def test_binnings():
    init, lead = datetime.datetime(2024, 11, 30, 12), 18 * HOUR
    frame = Frame(init, lead, init + lead, ["a"], torch.zeros(1, 1, 3))
    inits = [init, init + 12 * HOUR]
    cases = {
        ("season", "init_time"): (["DJF", "MAM", "JJA", "SON"], 3),
        ("season", "valid_time"): (["DJF", "MAM", "JJA", "SON"], 0),
        ("month", "init_time"): ([f"{m:02d}" for m in range(1, 13)], 10),
        ("month", "valid_time"): ([f"{m:02d}" for m in range(1, 13)], 11),
        ("hour", "init_time"): ([f"{h:02d}" for h in range(24)], 12),
        ("hour", "valid_time"): ([f"{h:02d}" for h in range(24)], 6),
        ("init_time", "init_time"): (["2024-11-30T12:00:00", "2024-12-01T00:00:00"], 0),
        ("init_time", "valid_time"): (
            [
                "2024-11-30T18:00:00",
                "2024-12-01T00:00:00",
                "2024-12-01T06:00:00",
                "2024-12-01T12:00:00",
                "2024-12-01T18:00:00",
            ],
            2,
        ),
        ("none", "init_time"): (["all"], 0),
    }
    for (kind, by), (coords, index) in cases.items():
        binning = build(kind, by, inits, [6 * HOUR, 12 * HOUR, 18 * HOUR])
        assert (binning.kind, binning.by, binning.coords, binning.index(frame)) == (kind, by, coords, index), (kind, by)
    with pytest.raises(ValueError):
        build("init_time", "init_time", inits).index(
            Frame(init + 6 * HOUR, lead, init + 6 * HOUR + lead, ["a"], frame.data)
        )
    lead = lead + datetime.timedelta(minutes=50)
    sub_hourly = Frame(init, lead, init + lead, ["a"], frame.data)
    assert build("hour", "valid_time").index(sub_hourly) == 6
    with pytest.raises(ValueError):
        build("week")
