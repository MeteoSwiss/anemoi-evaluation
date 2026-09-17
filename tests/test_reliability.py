import datetime
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from anemoi.evaluation import metrics
from anemoi.evaluation import statistics as st
from anemoi.evaluation.__main__ import main
from anemoi.evaluation.aggregation import Aggregator
from anemoi.evaluation.binning import NoBinning
from anemoi.evaluation.evaluate import Evaluation
from anemoi.evaluation.frame import Frame
from anemoi.evaluation.output import load_state
from anemoi.evaluation.output import write

HOUR = datetime.timedelta(hours=1)
T0 = datetime.datetime(2024, 1, 1)
INITS = [T0, T0 + 12 * HOUR]
LABEL = "heavy"
THRESHOLD = 0.6
BASE_METRICS = ["rmse", "mae", "bias", "crps", "spread"]
KINDS = ("reliability", "forecast_frequency", "brier_reliability", "brier_resolution", "brier_uncertainty")
M = 4  # the end-to-end ensemble size


def same(x, y):
    """NaN-aware elementwise equality of two tensors; a threshold sum is NaN for the variables its map omits."""
    return bool(((x == y) | (x.isnan() & y.isnan())).all())


def spec(kind, label=LABEL, thresholds=None):
    """The YAML spec of one labelled metric."""
    return {kind: {"label": label, "thresholds": dict(thresholds or {"t": THRESHOLD})}}


def level_ensemble(members, nodes, link, seed):
    """(M, 1, N) float32 members with a drawn number of them above 0 per node, and (1, 1, N) targets observed with
    probability `link(k / M)`.

    Members drawn from the target's own distribution are *not* calibrated in this sense: their exceedance fraction is a
    noisy estimate of the true probability, so the curve is pulled toward the base rate. Fixing the level first and
    drawing the observation from it is the only construction that puts the curve on the diagonal.
    """
    rng = np.random.default_rng(seed)
    k = rng.integers(0, members + 1, nodes)
    pred = np.where(np.arange(members)[:, None] < k[None, :], 1.0, -1.0).astype(np.float32)[:, None, :]
    observed = rng.random(nodes) < link(k / members)
    return pred, np.where(observed, 1.0, -1.0).astype(np.float32)[None, None, :], k


def single_frame_values(pred, target, members, specs, seed=7, variables=("a",)):
    """Metric values of one frame through a real `Aggregator` with random node weights, as `(V, R)` arrays."""
    variables = list(variables)
    nodes = pred.shape[-1]
    weights = np.random.default_rng(seed).uniform(0.1, 1.1, nodes)
    built = metrics.build(specs, variables, members=members)
    aggregator = Aggregator(weights, np.ones((nodes, 1), dtype=bool), metrics.unique_statistics(built), NoBinning())
    state = aggregator.new_state([6 * HOUR], variables, ["global"], members)
    frame = Frame(T0, 6 * HOUR, T0 + 6 * HOUR, variables, torch.from_numpy(pred))
    aggregator.add(state, frame, torch.from_numpy(target))
    means = state.means()
    return {metric.name: metric.from_means(means, members).numpy()[0, 0, 0, 0] for metric in built}, state


def test_level_statistics_match_numpy():
    members, nodes = 6, 3000
    rng = np.random.default_rng(0)
    variables = ["a", "b", "c"]
    thresholds = {"a": 0.0, "c": 0.5}
    pred = rng.standard_normal((members, 3, nodes)).astype(np.float32)
    target = rng.standard_normal((1, 3, nodes)).astype(np.float32)

    counts = np.full((3, nodes), np.nan)
    observed = np.full((3, nodes), np.nan)
    for j, variable in enumerate(variables):
        if variable not in thresholds:
            continue
        threshold = thresholds[variable]
        counts[j] = (pred[:, j].astype(np.float64) > threshold).sum(0)
        observed[j] = (target[0, j].astype(np.float64) > threshold).astype(np.float64)
        occurring = np.bincount(counts[j].astype(int), minlength=members + 1)
        assert (occurring > 0).all()  # every level is compared against the reference, and cannot silently narrow

    level_sum = np.zeros((3, nodes))
    for k in range(members + 1):
        values = {}
        for kind, statistic in (("count", st.ReliabilityCount), ("event", st.ReliabilityEvent)):
            instance = statistic(label=LABEL, thresholds=thresholds, k=k, members=members)
            assert instance.name == f"reliability_{kind}_{LABEL}_{k}"
            instance.bind_variables(variables)
            values[kind] = instance.compute(torch.from_numpy(pred), torch.from_numpy(target)).numpy()
        np.testing.assert_array_equal(values["count"], (counts == k).astype(np.float64) + 0.0 * counts)
        np.testing.assert_array_equal(values["event"], (counts == k) * observed)
        assert np.isnan(values["count"][1]).all() and np.isnan(values["event"][1]).all()
        assert ((values["event"] <= values["count"]) | np.isnan(values["count"])).all()
        level_sum += np.where(np.isnan(values["count"]), 0.0, values["count"])
    np.testing.assert_array_equal(level_sum[[0, 2]], np.ones((2, nodes)))  # a partition of 0..M, exactly


def test_exceedance_count_refactor_is_value_preserving():
    """The one phase 1 expression this family touched: `Brier` must be bit for bit what it was."""
    rng = np.random.default_rng(0)
    variables = ["a", "b", "c"]
    thresholds = {"a": 0.0, "c": 0.5}
    pred = rng.standard_normal((4, 3, 60)).astype(np.float32)
    target = rng.standard_normal((1, 3, 60)).astype(np.float32)
    per_variable = [thresholds.get(name) for name in variables]

    count = st.exceedance_count(torch.from_numpy(pred), per_variable)
    fraction = st.exceedance_fraction(torch.from_numpy(pred), per_variable)
    assert same(fraction, count / pred.shape[0])
    for j, variable in enumerate(variables):
        if variable not in thresholds:
            assert torch.isnan(count[j]).all() and torch.isnan(fraction[j]).all()
            continue
        # the pre-refactor expression, one variable at a time
        before = (torch.from_numpy(pred)[:, j].double() > thresholds[variable]).double().sum(0) / pred.shape[0]
        assert torch.equal(fraction[j], before)

    statistic = st.Brier(label=LABEL, thresholds=thresholds)
    statistic.bind_variables(variables)
    expected = np.full((3, 60), np.nan)
    for j, variable in enumerate(variables):
        if variable in thresholds:
            observed = (target[0, j].astype(np.float64) > thresholds[variable]).astype(np.float64)
            probability = (pred[:, j].astype(np.float64) > thresholds[variable]).astype(np.float64).mean(0)
            expected[j] = (probability - observed) ** 2
    got = statistic.compute(torch.from_numpy(pred), torch.from_numpy(target)).numpy()
    np.testing.assert_array_equal(got, expected)


def test_level_matches_the_fraction_for_every_k():
    """The design compares the integer count, but the fraction would have been exact too: one IEEE division of two
    exactly representable integers is the python `k / M` in every case up to 64 members."""
    for members in range(1, 65):
        for k in range(members + 1):
            pred = torch.cat([torch.ones(k, 1, 1), -torch.ones(members - k, 1, 1)])
            fraction = float(st.exceedance_fraction(pred, [0.0])[0, 0])
            assert fraction == k / members
            assert np.float64(k) / np.float64(members) == k / members


def test_metrics_from_means():
    members = 4
    counts = np.array([0.4, 0.1, 0.0, 0.2, 0.3])  # level 2 is empty
    events = np.array([0.04, 0.02, 0.0, 0.14, 0.3])
    means = {f"reliability_count_{LABEL}_{k}": torch.tensor(counts[k]) for k in range(members + 1)}
    means.update({f"reliability_event_{LABEL}_{k}": torch.tensor(events[k]) for k in range(members + 1)})
    means[f"event_frequency_{LABEL}"] = torch.tensor(events.sum() / counts.sum())

    built = {kind: metrics.build([spec(kind)], members=members)[0] for kind in KINDS}
    values = {kind: metric.from_means(means, members).numpy() for kind, metric in built.items()}

    probabilities = np.arange(members + 1) / members
    populated = counts > 0
    observed = np.where(populated, events / np.where(populated, counts, 1.0), np.nan)
    base_rate = events.sum() / counts.sum()
    np.testing.assert_allclose(values["reliability"][populated], observed[populated], rtol=1e-12)
    assert np.isnan(values["reliability"][2])
    np.testing.assert_allclose(values["forecast_frequency"], counts / counts.sum(), rtol=1e-12)
    assert values["forecast_frequency"][2] == 0.0
    assert values["forecast_frequency"].sum() == pytest.approx(1.0, rel=1e-12)

    reliability = (counts[populated] * (probabilities[populated] - observed[populated]) ** 2).sum() / counts.sum()
    resolution = (counts[populated] * (observed[populated] - base_rate) ** 2).sum() / counts.sum()
    assert values["brier_reliability"] == pytest.approx(reliability, rel=1e-12)
    assert values["brier_resolution"] == pytest.approx(resolution, rel=1e-12)
    assert values["brier_uncertainty"] == pytest.approx(base_rate * (1 - base_rate), rel=1e-12)
    brier = (counts * probabilities**2 - 2 * probabilities * events + events).sum() / counts.sum()
    identity = values["brier_reliability"] - values["brier_resolution"] + values["brier_uncertainty"]
    assert identity == pytest.approx(brier, rel=1e-12)

    nan_means = dict.fromkeys(means, torch.tensor(float("nan")))
    for kind, metric in built.items():
        assert np.isnan(metric.from_means(nan_means, members).numpy()).all()


def test_decomposition_identity_through_the_aggregator():
    members, nodes = 8, 20000
    variables, regions, leads = ["a", "b"], ["all", "box"], [6 * HOUR, 12 * HOUR]
    rng = np.random.default_rng(1)
    weights = rng.uniform(0.1, 1.1, nodes)
    masks = np.stack([np.ones(nodes, dtype=bool), rng.random(nodes) < 0.5], axis=1)
    thresholds = {"a": 0.0, "b": 0.0}
    specs = [spec(kind, thresholds=thresholds) for kind in (*KINDS, "brier", "event_frequency")]
    built = metrics.build(specs, variables, members=members)
    aggregator = Aggregator(weights, masks, metrics.unique_statistics(built), NoBinning())
    state = aggregator.new_state(leads, variables, regions, members)
    for seed, lead in enumerate(leads):
        pred, target, _ = level_ensemble(members, nodes, lambda p: p, seed)
        pred = np.concatenate([pred, pred[:, :, ::-1]], axis=1)
        target = np.concatenate([target, target[:, :, ::-1]], axis=1)
        pred, target = pred.copy(), target.copy()
        pred[0, 0, :5] = np.nan  # planted non-finite members and targets, excluded by the shared validity mask
        target[0, 1, 7:11] = np.nan
        aggregator.add(state, Frame(T0, lead, T0 + lead, variables, torch.from_numpy(pred)), torch.from_numpy(target))

    means = state.means()
    values = {metric.name: metric.from_means(means, members).numpy() for metric in built}
    identity = (
        values[f"brier_reliability_{LABEL}"]
        - values[f"brier_resolution_{LABEL}"]
        + values[f"brier_uncertainty_{LABEL}"]
    )
    np.testing.assert_allclose(identity, values[f"brier_{LABEL}"], rtol=1e-12)

    counts = np.stack([state.sums[f"reliability_count_{LABEL}_{k}"].numpy() for k in range(members + 1)])
    events = np.stack([state.sums[f"reliability_event_{LABEL}_{k}"].numpy() for k in range(members + 1)])
    np.testing.assert_allclose(counts.sum(0), state.weights.numpy(), rtol=1e-12)
    np.testing.assert_allclose(events.sum(0) / counts.sum(0), values[f"event_frequency_{LABEL}"], rtol=1e-12)
    np.testing.assert_allclose(values[f"forecast_frequency_{LABEL}"].sum(-1), 1.0, rtol=1e-12, atol=0.0)


def test_diagram_shapes():
    members, nodes = 8, 20000
    links = {
        "calibrated": lambda p: p,
        "over_confident": lambda p: 0.5 + 0.5 * (p - 0.5),
        "under_confident": lambda p: np.clip(0.5 + 2.0 * (p - 0.5), 0.0, 1.0),
    }
    specs = [spec(kind, thresholds={"a": 0.0}) for kind in KINDS]
    curves, components = {}, {}
    for seed, (name, link) in enumerate(links.items()):
        pred, target, _ = level_ensemble(members, nodes, link, seed + 10)
        values, _ = single_frame_values(pred, target, members, specs, seed=seed)
        curves[name] = values[f"reliability_{LABEL}"]
        components[name] = {kind: float(values[f"{kind}_{LABEL}"]) for kind in KINDS[2:]}

    diagonal = np.arange(members + 1) / members
    assert np.abs(curves["calibrated"] - diagonal).max() < 0.05  # about one binomial sd per level
    over = curves["over_confident"]
    assert np.abs(over - diagonal).max() > 0.15 and over[0] > 0.15 and over[-1] < 0.85
    under = curves["under_confident"]
    assert under[0] == 0.0 and under[-1] == 1.0
    inner = (diagonal > 0.25) & (diagonal < 0.75)  # the levels the link does not clip to 0 or 1
    assert np.diff(under[inner]).min() > np.diff(diagonal).max()  # steeper than the diagonal where it is not clipped

    reliability = {name: value["brier_reliability"] for name, value in components.items()}
    assert reliability["over_confident"] > 100 * reliability["calibrated"]
    resolution = {name: value["brier_resolution"] for name, value in components.items()}
    assert resolution["over_confident"] == min(resolution.values())


def test_deterministic_decomposition(fake_sources):
    """One member: the levels are `{0, 1}` and the diagram is the contingency table."""
    forecast, targets = fake_sources(members=1, drift=0.5, spread=0.8)
    evaluation = Evaluation(forecast, targets, INITS, "24h", [spec(kind) for kind in (*KINDS, "brier", "far")])
    diagram = next(metric for metric in evaluation.metrics if metric.name == f"reliability_{LABEL}")
    assert diagram.min_members == 1 and diagram.output_dim == ("probability", [0.0, 1.0])

    dataset = evaluation.run().to_xarray(evaluation.metrics).sel(bin="all", region="global", variable="t")
    assert dataset["probability"].values.tolist() == [0.0, 1.0]
    curve = dataset[f"reliability_{LABEL}"].values
    assert curve.shape[-1] == 2
    np.testing.assert_allclose(curve[:, 1], 1.0 - dataset[f"far_{LABEL}"].values, rtol=1e-12)
    identity = (
        dataset[f"brier_reliability_{LABEL}"].values
        - dataset[f"brier_resolution_{LABEL}"].values
        + dataset[f"brier_uncertainty_{LABEL}"].values
    )
    np.testing.assert_allclose(identity, dataset[f"brier_{LABEL}"].values, rtol=1e-12)

    # The fakes offset every forecast the same way, so they produce no misses; a random frame gives both off-diagonal
    # cells, which is what makes the two points of the diagram non-trivial.
    rng = np.random.default_rng(3)
    nodes = 4000
    pred = rng.standard_normal((1, 1, nodes)).astype(np.float32)
    target = rng.standard_normal((1, 1, nodes)).astype(np.float32)
    specs = [spec(kind, thresholds={"a": 0.0}) for kind in (*KINDS, "brier", "far", "csi")]
    values, state = single_frame_values(pred, target, 1, specs, seed=4)
    means = state.means()
    hit, miss = (float(means[f"{kind}_{LABEL}"]) for kind in ("hit", "miss"))
    false_alarm = float(means[f"false_alarm_{LABEL}"])
    correct_negative = 1.0 - hit - miss - false_alarm
    assert miss > 0 and false_alarm > 0  # both off-diagonal cells occur, so neither level is trivial
    curve = values[f"reliability_{LABEL}"]
    assert curve[0] == pytest.approx(miss / (miss + correct_negative), rel=1e-12)
    assert curve[1] == pytest.approx(1.0 - float(values[f"far_{LABEL}"]), rel=1e-12)
    identity = float(
        values[f"brier_reliability_{LABEL}"]
        - values[f"brier_resolution_{LABEL}"]
        + values[f"brier_uncertainty_{LABEL}"]
    )
    assert identity == pytest.approx(float(values[f"brier_{LABEL}"]), rel=1e-12)


def test_unmapped_variables_are_nan(fake_sources):
    forecast, targets = fake_sources(members=M, drift=0.5, spread=0.8)
    evaluation = Evaluation(forecast, targets, INITS, "24h", ["rmse", *(spec(kind) for kind in KINDS)])
    state = evaluation.run()
    dataset = state.to_xarray(evaluation.metrics)
    missing = dataset.sel(variable="q")
    for kind in KINDS:
        assert np.isnan(missing[f"{kind}_{LABEL}"].values).all()
    for k in range(M + 1):
        for kind in ("count", "event"):
            populated = missing[f"state_sum_reliability_{kind}_{LABEL}_{k}"].sel(state_bin="DJF")
            assert np.isnan(populated.values).all()
    assert np.isfinite(dataset.sel(variable="q", bin="all")["rmse"].values).all()


def test_binding_rules(fake_sources):
    metric = metrics.from_spec(spec("reliability"))
    assert metric.statistics == {} and metric.min_members == 1 and metric.output_dim is None
    assert metric.spec == {"reliability": {"label": LABEL, "thresholds": {"t": THRESHOLD}}}
    assert metrics.unique_statistics([metric]) == {} and metrics.required_aux([metric]) == set()
    with pytest.raises(ValueError, match="was not bound to the run's ensemble size"):
        metric.from_means({}, 4)

    metric.bind_members(4)
    metric.bind_members(4)  # the same ensemble size again is a no-op
    assert metric.output_dim == ("probability", [0.0, 0.25, 0.5, 0.75, 1.0])
    assert sorted(metric.statistics) == sorted(
        [f"reliability_{kind}_{LABEL}_{k}" for kind in ("count", "event") for k in range(5)]
    )
    with pytest.raises(ValueError, match="already bound to 4 members, cannot rebind to 8"):
        metric.bind_members(8)
    with pytest.raises(ValueError, match="is bound to 4 members but the state has 8"):
        metric.from_means({}, 8)
    counts_only = metrics.build([spec("forecast_frequency")], members=3)[0]
    assert sorted(counts_only.statistics) == sorted(f"reliability_count_{LABEL}_{k}" for k in range(4))

    statistic = st.ReliabilityCount(LABEL, {"t": THRESHOLD}, k=0, members=4)
    statistic.bind_variables(["t", "q"])
    statistic.bind_variables(["t", "q"])  # the same variables again is a no-op
    with pytest.raises(ValueError, match=r"already bound to \['t', 'q'\]"):
        statistic.bind_variables(["q", "t"])
    with pytest.raises(ValueError, match="built for 4 members but the frame has 3"):
        st.ReliabilityCount(LABEL, {"a": 0.0}, k=0, members=4).compute(torch.zeros(3, 1, 5), torch.zeros(1, 1, 5))
    with pytest.raises(ValueError, match="outside 0..4"):
        st.ReliabilityCount(LABEL, {"a": 0.0}, k=5, members=4)

    with pytest.raises(ValueError, match="does not accept"):
        metrics.from_spec({"reliability": {"label": LABEL, "thresholds": {"t": 1.0}, "members": 8}})
    forecast, targets = fake_sources()
    with pytest.raises(ValueError, match="the run does not have"):
        Evaluation(forecast, targets, INITS, "24h", [spec("reliability", thresholds={"nope": 1.0})])

    # one label, one threshold map, across the families and without any binding
    pair = [spec("reliability", label="x"), spec("brier_reliability", label="x", thresholds={"t": 2.0})]
    with pytest.raises(ValueError, match="one label, one threshold map"):
        metrics.build(pair)
    bound = [metrics.from_spec(item) for item in pair]
    for metric in bound:
        metric.bind_members(4)
    with pytest.raises(ValueError, match="both need a statistic named 'reliability_count_x_0'"):
        metrics.unique_statistics(bound)


def test_state_at_another_ensemble_size(fake_sources):
    forecast, targets = fake_sources(members=M, drift=0.5, spread=0.8)
    evaluation = Evaluation(forecast, targets, INITS, "24h", [spec("reliability")])
    state = evaluation.run()
    smaller = metrics.build([spec("reliability")], ["t", "q"], members=M - 1)
    with pytest.raises(ValueError, match="is bound to 3 members but the state has 4"):
        state.to_xarray(smaller)
    larger = metrics.build([spec("reliability")], ["t", "q"], members=M + 1)
    with pytest.raises(ValueError, match=r"no sums for statistics \['reliability_count_heavy_5'"):
        state.to_xarray(larger)


def test_merge_and_cli_rebuild(fake_sources, tmp_path):
    forecast, targets = fake_sources(members=M, drift=0.5, spread=0.8)
    evaluation = Evaluation(forecast, targets, INITS, "24h", ["rmse", *(spec(kind) for kind in KINDS)])
    full = evaluation.run()
    shards = [evaluation.run([init]) for init in INITS]
    merged = shards[0].merge(shards[1])
    for name, total in full.sums.items():
        other = merged.sums[name]
        assert bool(((other == total) | (other.isnan() & total.isnan())).all())

    parts = [str(tmp_path / f"part{i}.nc") for i in range(len(INITS))]
    for shard, part in zip(shards, parts):
        write(shard.to_xarray(evaluation.metrics), part)
    output = tmp_path / "merged.nc"
    main(["merge", *parts, "-o", str(output)])
    with xr.open_dataset(output, decode_timedelta=True) as stored:
        stored = stored.load()
    xr.testing.assert_equal(stored.drop_attrs(), full.to_xarray(evaluation.metrics).drop_attrs())

    other = Evaluation(*fake_sources(members=M + 1, drift=0.5, spread=0.8), INITS, "24h", ["rmse"]).run([INITS[0]])
    with pytest.raises(ValueError, match="cannot merge states with"):
        shards[0].merge(other)


def test_netcdf_round_trip_with_both_axes(fake_sources, tmp_path):
    forecast, targets = fake_sources(members=M, drift=0.5, spread=0.8)
    specs = ["rmse", "rank_histogram", *(spec(kind) for kind in KINDS)]
    evaluation = Evaluation(forecast, targets, INITS, "24h", specs)
    state = evaluation.run()
    dataset = state.to_xarray(evaluation.metrics)
    assert dataset["rank"].values.tolist() == list(range(M + 1))
    assert dataset["rank"].dtype == np.int64
    assert dataset["probability"].dtype == np.float64
    np.testing.assert_array_equal(dataset["probability"].values, np.arange(M + 1) / M)
    for kind in ("reliability", "forecast_frequency"):
        assert dataset[f"{kind}_{LABEL}"].dims == ("lead_time", "bin", "variable", "region", "probability")
    for kind in KINDS[2:]:
        assert dataset[f"{kind}_{LABEL}"].dims == ("lead_time", "bin", "variable", "region")

    path = tmp_path / "reliability.nc"
    write(dataset, path)
    with xr.open_dataset(path, decode_timedelta=True) as stored:
        assert stored["rank"].values.tolist() == list(range(M + 1))
        np.testing.assert_array_equal(stored["probability"].values, np.arange(M + 1) / M)
        assert stored[f"reliability_{LABEL}"].dims[-1] == "probability"
    loaded = load_state(path)
    for name, total in state.sums.items():
        assert loaded.sums[name].shape == total.shape and len(total.shape) == 4
        assert same(loaded.sums[name], total)
    again = loaded.to_xarray(evaluation.metrics)
    for metric in evaluation.metrics:
        np.testing.assert_array_equal(again[metric.name].values, dataset[metric.name].values)


def test_reliability_does_not_perturb_the_other_sums(fake_sources):
    extra = ["rank_histogram", {"csi": {"label": "warm", "thresholds": {"t": 0.4}}}]
    forecast, targets = fake_sources(members=M, drift=0.5, spread=0.2)
    base = Evaluation(forecast, targets, INITS, "24h", [*BASE_METRICS, *extra]).run()
    forecast, targets = fake_sources(members=M, drift=0.5, spread=0.2)
    extended = Evaluation(
        forecast, targets, INITS, "24h", [*BASE_METRICS, *extra, *(spec(kind) for kind in KINDS)]
    ).run()
    assert list(extended.sums)[: len(base.sums)] == list(base.sums)
    assert all(same(extended.sums[name], total) for name, total in base.sums.items())
    assert torch.equal(extended.weights, base.weights) and torch.equal(extended.n_init, base.n_init)


def test_cached_and_uncached_levels_agree():
    members, nodes, variables, leads = 5, 60, ["a", "b"], [6 * HOUR]
    rng = np.random.default_rng(5)
    weights, masks = rng.random(nodes) + 0.1, np.ones((nodes, 1), dtype=bool)
    pred = (rng.integers(-4, 5, (members, 2, nodes)) / 2).astype(np.float32)
    truth = (rng.integers(-4, 5, (1, 2, nodes)) / 2).astype(np.float32)
    frame = Frame(T0, leads[0], T0 + leads[0], variables, torch.from_numpy(pred))
    target = torch.from_numpy(truth)
    thresholds = {"a": 0.0}

    def level(kind, k):
        built = st.ReliabilityCount if kind == "count" else st.ReliabilityEvent
        statistic = built(LABEL, thresholds, k=k, members=members)
        statistic.bind_variables(variables)
        return statistic

    def accumulate(statistics, aux=None):
        aggregator = Aggregator(weights, masks, statistics, NoBinning())
        state = aggregator.new_state(leads, variables, ["global"], members)
        aggregator.add(state, frame, target, aux)
        return state.sums

    names = [(kind, k) for kind in ("count", "event") for k in range(members + 1)]
    shared = accumulate({f"reliability_{kind}_{LABEL}_{k}": level(kind, k) for kind, k in names})
    for kind, k in names:
        name = f"reliability_{kind}_{LABEL}_{k}"
        alone = accumulate({name: level(kind, k)})  # this one fills the cache itself
        assert same(shared[name], alone[name])
        cache = {}
        filling = level(kind, k).compute(frame.data, target, cache)
        # an empty mapping is the caller's own, and the helper fills it, so the second call reads the cache
        assert sorted(cache) == [f"exceedance_count_{LABEL}", f"exceedance_event_{LABEL}"]
        assert same(filling, level(kind, k).compute(frame.data, target, cache))
        assert same(filling, level(kind, k).compute(frame.data, target))  # recomputed, aux is empty

    caller = {"climatology": torch.zeros(2, nodes, dtype=torch.float64)}
    accumulate({f"reliability_count_{LABEL}_{k}": level("count", k) for k in range(members + 1)}, caller)
    assert list(caller) == ["climatology"]
    assert st.ReliabilityCount(LABEL, thresholds, k=0, members=members).aux == ()


def run_tool(name, *arguments):
    """One of the CPU tools under `-W default`, with its stderr checked for a numpy warning."""
    tools = Path(__file__).resolve().parents[1] / "tools"
    result = subprocess.run(
        [sys.executable, "-W", "default", str(tools / name), *arguments], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # the environment's optional xarray engines warn on import, which is not the tool's doing
    noise = [line for line in result.stderr.splitlines() if "cfgrib" not in line and "eccodes" not in line.lower()]
    assert not [line for line in noise if "Warning" in line], "\n".join(noise)
    return result.stdout


def test_tools_on_a_fake_result_file(fake_sources, tmp_path):
    """The two CPU tools on a file carrying both extra axes, the threshold metrics and the five new ones."""
    forecast, targets = fake_sources(members=M, drift=0.5, spread=0.8)
    specs = ["rmse", "mae", "rank_histogram", "outlier_fraction"]
    specs += [spec(kind) for kind in ("csi", "pod", "far", "brier", "bss", "event_frequency", *KINDS)]
    evaluation = Evaluation(forecast, targets, INITS, "24h", specs)
    path = tmp_path / "tools.nc"
    write(evaluation.run().to_xarray(evaluation.metrics), path)

    printed = run_tool("inspect_results.py", str(path))
    assert "MECHANICS OK" in printed
    assert printed.count(f"reliability label '{LABEL}'") == 1
    assert printed.count(f"threshold label '{LABEL}'") == 1
    # the label split is longest kind first, so `brier_reliability_heavy` is not a label `reliability_heavy`
    assert f"reliability_{LABEL}'" not in printed.replace(f"reliability label '{LABEL}'", "")

    printed = run_tool("compare_results.py", str(path), str(path))
    assert "STRUCTURE OK" in printed
    assert "max relative difference" not in printed
    for name in (f"sum reliability_count_{LABEL}_0", f"metric reliability_{LABEL}", f"metric brier_resolution_{LABEL}"):
        assert f"{name}: bit-exact" in printed


def test_inspect_results_with_the_uncertainty_alone(fake_sources, tmp_path):
    """`brier_uncertainty` stores no level sums, so the tool must find the scored variables from the base rate."""
    forecast, targets = fake_sources(members=M, drift=0.5, spread=0.8)
    evaluation = Evaluation(forecast, targets, INITS, "24h", ["rmse", spec("brier_uncertainty")])
    path = tmp_path / "uncertainty.nc"
    write(evaluation.run().to_xarray(evaluation.metrics), path)

    printed = run_tool("inspect_results.py", str(path))
    assert "MECHANICS OK" in printed
    assert f"reliability label '{LABEL}': scored variables ['t']" in printed
    assert "no threshold (all NaN): ['q']" in printed
