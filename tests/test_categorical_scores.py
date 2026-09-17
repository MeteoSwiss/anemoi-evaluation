import datetime
import json

import numpy as np
import pytest
import torch
import xarray as xr

from anemoi.evaluation import metrics
from anemoi.evaluation import statistics as st
from anemoi.evaluation.aggregation import Aggregator
from anemoi.evaluation.binning import NoBinning
from anemoi.evaluation.evaluate import Evaluation
from anemoi.evaluation.frame import Frame
from anemoi.evaluation.output import load_state
from anemoi.evaluation.output import write

HOUR = datetime.timedelta(hours=1)
T0 = datetime.datetime(2024, 1, 1)
INITS = [T0, T0 + 12 * HOUR]
LABEL = "warm"
THRESHOLD = 0.6
BASE_METRICS = ["rmse", "mae", "bias", "crps", "spread"]
THRESHOLD_METRICS = ["pod", "far", "csi", "ets", "frequency_bias", "hss", "pss", "brier", "bss", "event_frequency"]


def spec(kind, label=LABEL, thresholds=None):
    """The YAML spec of one threshold metric."""
    return {kind: {"label": label, "thresholds": dict(thresholds or {"t": THRESHOLD})}}


def test_threshold_statistics_match_numpy():
    rng = np.random.default_rng(0)
    variables = ["a", "b", "c"]
    thresholds = {"a": 0.0, "c": 0.5}
    pred = rng.standard_normal((4, 3, 60)).astype(np.float32)
    target = rng.standard_normal((1, 3, 60)).astype(np.float32)
    values = {}
    for name, statistic in (
        ("hit", st.Hit),
        ("miss", st.Miss),
        ("false_alarm", st.FalseAlarm),
        ("brier", st.Brier),
        ("event_frequency", st.EventFrequency),
    ):
        instance = statistic(label=LABEL, thresholds=thresholds)
        instance.bind_variables(variables)
        values[name] = instance.compute(torch.from_numpy(pred), torch.from_numpy(target)).numpy()

    expected = {name: np.full((3, 60), np.nan) for name in values}
    for j, variable in enumerate(variables):
        if variable not in thresholds:
            continue
        threshold = thresholds[variable]
        observed = (target[0, j].astype(np.float64) > threshold).astype(np.float64)
        forecast = (pred[:, j].astype(np.float64).mean(0) > threshold).astype(np.float64)
        fraction = (pred[:, j].astype(np.float64) > threshold).astype(np.float64).mean(0)
        expected["hit"][j] = forecast * observed
        expected["miss"][j] = (1 - forecast) * observed
        expected["false_alarm"][j] = forecast * (1 - observed)
        expected["brier"][j] = (fraction - observed) ** 2
        expected["event_frequency"][j] = observed
        if variable == "a":
            assert ((fraction > 0) & (fraction < 1)).any()  # the ensemble path is genuinely exercised

    for name, value in values.items():
        np.testing.assert_array_equal(value, expected[name])
        assert np.isnan(value[1]).all()


def test_deterministic_brier_equals_miss_plus_false_alarm():
    rng = np.random.default_rng(2)  # data where the two sums are not bit-identical, see below
    n, variables, leads = 50, ["a", "b"], [6 * HOUR, 12 * HOUR]
    weights, masks = rng.random(n) + 0.1, np.ones((n, 1), dtype=bool)
    frames = []
    for lead in leads:
        pred = rng.standard_normal((1, 2, n)).astype(np.float32)
        target = rng.standard_normal((1, 2, n)).astype(np.float32)
        frames.append((Frame(T0, lead, T0 + lead, variables, torch.from_numpy(pred)), torch.from_numpy(target)))

    built = metrics.build([spec("csi", thresholds={"a": 0.0}), spec("brier", thresholds={"a": 0.0})], variables)
    statistics = metrics.unique_statistics(built)
    for frame, target in frames:
        values = {name: statistic.compute(frame.data, target) for name, statistic in statistics.items()}
        misses, alarms = values[f"miss_{LABEL}"], values[f"false_alarm_{LABEL}"]
        assert (misses[0] > 0).any() and (alarms[0] > 0).any()  # the identity is not trivially true here
        assert torch.equal(values[f"brier_{LABEL}"][0], (misses + alarms)[0])  # elementwise, exactly
        assert torch.isnan(values[f"brier_{LABEL}"][1]).all()  # 'b' has no threshold

    aggregator = Aggregator(weights, masks, statistics, NoBinning())
    state = aggregator.new_state(leads, variables, ["global"], 1)
    for frame, target in frames:
        aggregator.add(state, frame, target)
    # The weighted sums group the same terms differently ((m + fa) @ W against m @ W + fa @ W), and float64
    # addition is not associative, so on the sums and the means they agree only up to rounding.
    sums, means = state.sums, state.means()
    for values in (sums, means):
        both = (values[f"miss_{LABEL}"] + values[f"false_alarm_{LABEL}"])[:, :, 0].numpy()
        np.testing.assert_allclose(values[f"brier_{LABEL}"][:, :, 0].numpy(), both, rtol=1e-12)
    assert torch.isnan(sums[f"brier_{LABEL}"][:, :, 1]).all()


def test_contingency_cells_sum_to_one():
    rng = np.random.default_rng(2)
    n, variables, regions = 40, ["a", "b"], ["all", "box"]
    weights = rng.random(n) + 0.1
    masks = np.stack([np.ones(n, dtype=bool), rng.random(n) < 0.5], axis=1)
    init, leads = T0, [6 * HOUR, 12 * HOUR]
    frames = []
    for lead in leads:
        pred = (rng.integers(-8, 9, (2, 2, n)) / 4).astype(np.float32)
        target = (rng.integers(-8, 9, (1, 2, n)) / 4).astype(np.float32)
        pred[0, 0, :3] = np.nan
        target[0, 0, 5:8] = np.nan
        frames.append((Frame(init, lead, init + lead, variables, torch.from_numpy(pred)), torch.from_numpy(target)))

    built = metrics.build([spec("csi", thresholds={"a": 0.25}), spec("pod", thresholds={"a": 0.25})], variables)
    aggregator = Aggregator(weights, masks, metrics.unique_statistics(built), NoBinning())
    state = aggregator.new_state(leads, variables, regions, 2)
    for frame, target in frames:
        aggregator.add(state, frame, target)

    W = weights[:, None] * masks
    for index, (frame, target) in enumerate(frames):
        pred = frame.data.numpy().astype(np.float64)
        truth = target.numpy()[0].astype(np.float64)
        valid = np.isfinite(pred).all(0) & np.isfinite(truth)
        observed = (truth[0] > 0.25).astype(np.float64)
        forecast = (pred[:, 0].mean(0) > 0.25).astype(np.float64)
        cells = [forecast * observed, (1 - forecast) * observed, forecast * (1 - observed)]
        negative = np.where(valid[0], (1 - forecast) * (1 - observed), 0.0) @ W / (valid[0].astype(np.float64) @ W)
        means = state.means()
        got = [means[f"{name}_{LABEL}"][index, 0, 0].numpy() for name in ("hit", "miss", "false_alarm")]
        for value, cell in zip(got, cells):
            np.testing.assert_allclose(value, np.where(valid[0], cell, 0.0) @ W / (valid[0].astype(np.float64) @ W))
        assert np.all(np.abs(sum(got) + negative - 1.0) < 1e-12)
        assert np.all(sum(got) <= 1 + 1e-12)
        assert np.isnan([means[f"{name}_{LABEL}"][index, 0, 1].numpy() for name in ("hit", "miss")]).all()


def test_threshold_metrics_from_means():
    h, m, fa, b, obar = 0.2, 0.1, 0.1, 0.15, 0.3
    cn = 1.0 - h - m - fa
    means = {
        f"hit_{LABEL}": torch.tensor(h, dtype=torch.float64),
        f"miss_{LABEL}": torch.tensor(m, dtype=torch.float64),
        f"false_alarm_{LABEL}": torch.tensor(fa, dtype=torch.float64),
        f"brier_{LABEL}": torch.tensor(b, dtype=torch.float64),
        f"event_frequency_{LABEL}": torch.tensor(obar, dtype=torch.float64),
    }
    random_hits = (h + fa) * (h + m)
    expected = {
        "pod": h / (h + m),
        "far": fa / (h + fa),
        "csi": h / (h + m + fa),
        "ets": (h - random_hits) / (h + m + fa - random_hits),
        "frequency_bias": (h + fa) / (h + m),
        "hss": 2 * (h * cn - m * fa) / ((h + m) * (m + cn) + (h + fa) * (fa + cn)),
        "pss": h / (h + m) - fa / (fa + cn),
        "brier": b,
        "bss": 1 - b / (obar * (1 - obar)),
        "event_frequency": obar,
    }
    values = {}
    for kind, value in expected.items():
        metric = metrics.from_spec(spec(kind))
        assert metric.name == f"{kind}_{LABEL}"
        values[kind] = float(metric.from_means(means, 3))
        assert values[kind] == pytest.approx(value, rel=1e-12)
    assert values["csi"] <= values["pod"] and values["csi"] <= 1 - values["far"]
    assert values["ets"] <= values["csi"]
    assert values["frequency_bias"] == pytest.approx((h + fa) / (h + m), rel=1e-12)

    empty = dict.fromkeys(means, torch.tensor(0.0, dtype=torch.float64))
    for kind in ("pod", "far", "csi", "ets", "frequency_bias", "hss", "pss", "bss"):
        assert np.isnan(float(metrics.from_spec(spec(kind)).from_means(empty, 3)))


def test_binding_errors(fake_sources):
    pred, target = torch.zeros(2, 3, 5), torch.zeros(1, 3, 5)
    statistic = st.Hit(label=LABEL, thresholds={"a": 1.0})
    with pytest.raises(ValueError, match="was not bound"):
        statistic.compute(pred, target)
    with pytest.raises(ValueError, match=r"the run does not have: \['a'\]"):
        statistic.bind_variables(["b", "c"])
    statistic.bind_variables(["a", "b", "c"])
    with pytest.raises(ValueError, match="bound to 3 variables but the frame has 2"):
        statistic.compute(pred[:, :2], target[:, :2])

    statistic.bind_variables(["a", "b", "c"])  # the same variables again is a no-op
    with pytest.raises(ValueError, match=r"already bound to \['a', 'b', 'c'\]"):
        statistic.bind_variables(["c", "b", "a"])

    forecast, targets = fake_sources()
    with pytest.raises(ValueError, match="the run does not have"):
        Evaluation(forecast, targets, INITS, "24h", [spec("csi", thresholds={"nope": 1.0})])
    with pytest.raises(ValueError, match="must be a finite number"):
        metrics.from_spec(spec("csi", thresholds={"t": float("nan")}))

    metric = metrics.from_spec(spec("event_frequency"))
    Evaluation(forecast, targets, INITS, "24h", [metric])  # binds the instance to ('t', 'q')
    with pytest.raises(ValueError, match="already bound"):
        Evaluation(forecast, targets, INITS, "24h", [metric], variables=["q", "t"])


def test_statistic_identity_and_duplicate_names():
    # One label names one event, so `build` refuses two threshold maps behind it before it looks at the statistics.
    conflicting = [spec("csi", thresholds={"t": 1.0}), spec("pod", thresholds={"t": 2.0})]
    with pytest.raises(ValueError, match="one label, one threshold map"):
        metrics.build(conflicting)
    # The identity guard is the second line of defence for the same fault, and still names the statistic.
    with pytest.raises(ValueError, match="both need a statistic named 'hit_warm'") as error:
        metrics.unique_statistics([metrics.from_spec(spec_) for spec_ in conflicting])
    assert "1.0" in str(error.value) and "2.0" in str(error.value)

    shared = metrics.unique_statistics([metrics.from_spec(spec("csi")), metrics.from_spec(spec("pod"))])
    assert sorted(shared) == [f"false_alarm_{LABEL}", f"hit_{LABEL}", f"miss_{LABEL}"]
    assert shared[f"hit_{LABEL}"] == st.Hit(label=LABEL, thresholds={"t": THRESHOLD})
    assert hash(shared[f"hit_{LABEL}"]) == hash(st.Hit(label=LABEL, thresholds={"t": THRESHOLD}))
    assert shared[f"hit_{LABEL}"] != st.Miss(label=LABEL, thresholds={"t": THRESHOLD})
    assert shared[f"hit_{LABEL}"] != st.Hit(label="other", thresholds={"t": THRESHOLD})

    with pytest.raises(ValueError, match="duplicate metric names: 'csi_warm' appears 2 times"):
        metrics.build([spec("csi"), spec("csi")])
    with pytest.raises(ValueError, match="duplicate metric names"):
        metrics.build(["rmse", "rmse"])


def test_threshold_metrics_end_to_end(fake_sources, tmp_path):
    forecast, targets = fake_sources(members=3, drift=0.5, spread=0.2)
    evaluation = Evaluation(forecast, targets, INITS, "24h", ["rmse", *(spec(kind) for kind in THRESHOLD_METRICS)])
    weights = evaluation.weights
    leads = evaluation.lead_times
    sums = {name: np.zeros(len(leads)) for name in ("hit", "miss", "false_alarm", "brier", "event_frequency")}
    weight_sum = np.zeros(len(leads))
    for frame, target in evaluation.pairs():
        pred = frame.data.numpy().astype(np.float64)[:, 0]
        truth = target.numpy()[0, 0].astype(np.float64)
        observed = (truth > THRESHOLD).astype(np.float64)
        binary = (pred.mean(0) > THRESHOLD).astype(np.float64)
        fraction = (pred > THRESHOLD).astype(np.float64).mean(0)
        lead = leads.index(frame.lead_time)
        for name, value in (
            ("hit", binary * observed),
            ("miss", (1 - binary) * observed),
            ("false_alarm", binary * (1 - observed)),
            ("brier", (fraction - observed) ** 2),
            ("event_frequency", observed),
        ):
            sums[name][lead] += value @ weights
        weight_sum[lead] += weights.sum()
    h, m, fa = (sums[name] / weight_sum for name in ("hit", "miss", "false_alarm"))
    b, obar = sums["brier"] / weight_sum, sums["event_frequency"] / weight_sum
    cn = 1 - h - m - fa
    random_hits = (h + fa) * (h + m)
    expected = {
        "pod": h / (h + m),
        "far": fa / (h + fa),
        "csi": h / (h + m + fa),
        "ets": (h - random_hits) / (h + m + fa - random_hits),
        "frequency_bias": (h + fa) / (h + m),
        "hss": 2 * (h * cn - m * fa) / ((h + m) * (m + cn) + (h + fa) * (fa + cn)),
        "pss": h / (h + m) - fa / (fa + cn),
        "brier": b,
        "bss": 1 - b / (obar * (1 - obar)),
        "event_frequency": obar,
    }
    assert (obar > 0).all() and (obar < 1).all()  # the threshold is scorable everywhere

    state = evaluation.run()
    dataset = state.to_xarray(evaluation.metrics)
    scored = dataset.sel(bin="all", region="global", variable="t")
    for kind, values in expected.items():
        np.testing.assert_allclose(scored[f"{kind}_{LABEL}"].values, values, rtol=1e-12)
    missing = dataset.sel(variable="q")
    for kind in THRESHOLD_METRICS:
        assert np.isnan(missing[f"{kind}_{LABEL}"].values).all()
    for name in sums:
        populated = missing[f"state_sum_{name}_{LABEL}"].sel(state_bin="DJF")  # the only bin the init times fall in
        assert np.isnan(populated.values).all()
    assert np.isfinite(dataset.sel(variable="q", bin="all")["rmse"].values).all()

    path = tmp_path / "thresholds.nc"
    write(dataset, path)
    loaded = load_state(path)
    for name in sums:
        assert torch.isnan(loaded.sums[f"{name}_{LABEL}"][:, 0, 1]).all()
        np.testing.assert_array_equal(loaded.sums[f"{name}_{LABEL}"].numpy(), state.sums[f"{name}_{LABEL}"].numpy())
    with xr.open_dataset(path, decode_timedelta=True) as stored:
        specs = json.loads(stored.attrs["metrics"])
    assert spec("csi") in specs and specs[0] == "rmse"
    again = loaded.to_xarray(evaluation.metrics)
    for kind in THRESHOLD_METRICS:
        np.testing.assert_array_equal(again[f"{kind}_{LABEL}"].values, dataset[f"{kind}_{LABEL}"].values)


def test_thresholds_do_not_perturb_the_other_sums(fake_sources):
    forecast, targets = fake_sources(members=3, drift=0.5, spread=0.2)
    base = Evaluation(forecast, targets, INITS, "24h", BASE_METRICS).run()
    forecast, targets = fake_sources(members=3, drift=0.5, spread=0.2)
    extended = Evaluation(
        forecast, targets, INITS, "24h", [*BASE_METRICS, spec("csi"), spec("brier"), spec("bss")]
    ).run()
    assert list(extended.sums)[: len(base.sums)] == list(base.sums)
    assert all(torch.equal(extended.sums[name], total) for name, total in base.sums.items())
    assert torch.equal(extended.weights, base.weights) and torch.equal(extended.n_init, base.n_init)


def test_merge_with_threshold_sums(fake_sources):
    forecast, targets = fake_sources(members=3, drift=0.5, spread=0.2)
    evaluation = Evaluation(forecast, targets, INITS, "24h", ["rmse", spec("csi"), spec("bss")])
    full = evaluation.run()
    shards = [evaluation.run([init]) for init in INITS]
    merged = shards[0].merge(shards[1])
    for name, total in full.sums.items():
        other = merged.sums[name]
        assert bool(((other == total) | (other.isnan() & total.isnan())).all())
    assert torch.isnan(merged.sums[f"hit_{LABEL}"][:, 0, 1]).all()
    assert torch.equal(merged.weights, full.weights) and torch.equal(merged.n_init, full.n_init)
    without = Evaluation(forecast, targets, INITS, "24h", ["rmse"]).run([INITS[0]])
    with pytest.raises(ValueError, match="different coordinates or statistics"):
        shards[1].merge(without)
