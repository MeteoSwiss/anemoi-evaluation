import datetime
import json

import numpy as np
import pytest
import torch
import xarray as xr

from anemoi.evaluation import metrics
from anemoi.evaluation import statistics as st
from anemoi.evaluation.__main__ import main
from anemoi.evaluation.aggregation import AggregationState
from anemoi.evaluation.aggregation import Aggregator
from anemoi.evaluation.binning import NoBinning
from anemoi.evaluation.evaluate import Evaluation
from anemoi.evaluation.frame import Frame
from anemoi.evaluation.output import load_state
from anemoi.evaluation.output import write

HOUR = datetime.timedelta(hours=1)
T0 = datetime.datetime(2024, 1, 1)
INITS = [T0, T0 + 12 * HOUR]
BASE_METRICS = ["rmse", "mae", "bias", "crps", "spread"]
M = 4  # the end-to-end ensemble size


def planted_ties(seed=0, members=5, variables=3, nodes=4000):
    """(M, V, N) and (1, V, N) float32 arrays with ties planted: variable 0 has 1 to M-1 tied members on half its
    nodes, scattered exact zeros on both sides on the other half and untied nodes in both; variable 1 has 50 nodes
    where every member equals the target, variable 2 has none. So `ties` covers 0 to M there, and `ties + 1` covers
    the powers of two and the other counts alike."""
    rng = np.random.default_rng(seed)
    pred = rng.standard_normal((members, variables, nodes)).astype(np.float32)
    target = rng.standard_normal((1, variables, nodes)).astype(np.float32)
    half = nodes // 2
    counts = rng.integers(1, members, half)  # 1 to M-1 members tied with the target, none of them all tied
    chosen = rng.random((members, half)).argsort(0) < counts
    partial = pred[:, 0, :half]
    pred[:, 0, :half] = np.where(chosen, np.broadcast_to(target[0, 0, :half], partial.shape), partial)
    pred[rng.integers(0, members, 500), 0, half + rng.choice(nodes - half, 500, replace=False)] = 0.0
    target[0, 0, half + rng.choice(nodes - half, 800, replace=False)] = 0.0
    tied = rng.choice(nodes, 50, replace=False)
    pred[:, 1, tied] = target[0, 1, tied]
    return pred, target, tied


def reference_bins(pred, target, members):
    """(M + 1, V, N) numpy float64 reference for the rank bins."""
    below = (pred < target[0]).sum(0).astype(np.float64)
    ties = (pred == target[0]).sum(0).astype(np.float64)
    return np.stack([((below <= k) & (k <= below + ties)) / (ties + 1.0) for k in range(members + 1)]), below, ties


def test_rank_bin_matches_numpy_reference():
    members = 5
    pred, target, tied = planted_ties(members=members)
    expected, below, ties = reference_bins(pred, target, members)
    values = np.stack(
        [st.RankBin(k, members).compute(torch.from_numpy(pred), torch.from_numpy(target)).numpy() for k in range(6)]
    )
    for k in range(members + 1):
        np.testing.assert_array_equal(values[k], expected[k])
        assert (values[k] > 0).any()  # every bin is exercised by this data

    np.testing.assert_array_equal(values[:, 1, tied], np.full((members + 1, len(tied)), 1.0 / (members + 1)))
    untied = ties[2] == 0
    assert untied.all()  # variable 2 was left alone, no tie can occur in random floats
    assert (values[:, 2][:, untied] == 1.0).sum() == untied.sum()
    occurring = np.bincount(ties[0].astype(int), minlength=members + 1)
    assert (occurring[:members] > 0).all()  # every tie count from 0 to M-1 is compared against the reference
    assert (below[0] > 0).any()


def test_rank_bins_sum_to_one():
    """Per element the bins sum to 1 exactly when `ties + 1` is a power of two and to a few ulp otherwise; on the
    weighted means the same terms are summed in different groupings, so the identity is a tolerance, not an equality."""
    members = 5
    pred, target, _ = planted_ties(members=members)
    expected, _, ties = reference_bins(pred, target, members)
    values = np.stack(
        [
            st.RankBin(k, members).compute(torch.from_numpy(pred), torch.from_numpy(target)).numpy()
            for k in range(members + 1)
        ]
    )
    total = values.sum(0)
    assert np.abs(total - 1.0).max() < 1e-15
    power_of_two = np.isin(ties + 1, [1, 2, 4])
    assert power_of_two.any()
    assert (total[power_of_two] == 1.0).all()

    rng = np.random.default_rng(1)
    n, variables, regions, leads = 40, ["a", "b"], ["all", "box"], [6 * HOUR, 12 * HOUR]
    weights = rng.random(n) + 0.1
    masks = np.stack([np.ones(n, dtype=bool), rng.random(n) < 0.5], axis=1)
    frames = []
    for lead in leads:
        frame = (rng.integers(-8, 9, (members, 2, n)) / 4).astype(np.float32)  # coarse values, so ties are common
        truth = (rng.integers(-8, 9, (1, 2, n)) / 4).astype(np.float32)
        frame[0, 0, :3] = np.nan
        truth[0, 1, 5:8] = np.nan
        frames.append((Frame(T0, lead, T0 + lead, variables, torch.from_numpy(frame)), torch.from_numpy(truth)))

    built = metrics.build(["rank_histogram"], variables, members=members)
    aggregator = Aggregator(weights, masks, metrics.unique_statistics(built), NoBinning())
    state = aggregator.new_state(leads, variables, regions, members)
    for frame, truth in frames:
        aggregator.add(state, frame, truth)
    means = state.means()
    total = sum(means[f"rank_bin_{k}"] for k in range(members + 1)).numpy()
    positive = state.weights.numpy() > 0
    assert positive.all()
    np.testing.assert_allclose(total[positive], 1.0, rtol=1e-12, atol=0.0)


def test_histogram_shapes():
    members, nodes = 8, 20000
    rng = np.random.default_rng(3)
    target = rng.standard_normal((1, 4, nodes)).astype(np.float32)
    pred = np.empty((members, 4, nodes), dtype=np.float32)
    pred[:, 0] = rng.standard_normal((members, nodes))  # calibrated
    pred[:, 1] = 0.5 * rng.standard_normal((members, nodes))  # under-dispersed
    pred[:, 2] = 0.5 + rng.standard_normal((members, nodes))  # biased high
    zeros = rng.random((members, nodes)) < 0.9  # 90 % exact zeros on both sides, calibrated otherwise
    pred[:, 3] = np.where(zeros, 0.0, rng.exponential(1.0, (members, nodes)))
    target[0, 3] = np.where(rng.random(nodes) < 0.9, 0.0, rng.exponential(1.0, nodes))

    histogram = np.stack(
        [
            st.RankBin(k, members).compute(torch.from_numpy(pred), torch.from_numpy(target)).numpy().mean(-1)
            for k in range(members + 1)
        ]
    )
    flat = 1.0 / (members + 1)
    calibrated, under, biased, tp = (histogram[:, j] for j in range(4))
    assert np.abs(calibrated - flat).max() < 0.02  # about 9 times the per-bin binomial sd at this sample size
    assert under[0] > 0.2 and under[-1] > 0.2 and (under[1:-1] < 0.1).all()
    assert np.abs(under - under[::-1]).max() < 0.02
    assert (np.diff(biased) < 0).all() and biased[0] - biased[-1] > 0.1
    assert np.abs(tp - flat).max() < 0.02  # the tie rule's headline property


def test_outlier_fraction():
    members = 4
    metric = metrics.build(["outlier_fraction"], members=members)[0]
    assert sorted(metric.statistics) == ["rank_bin_0", "rank_bin_4"]
    means = {f"rank_bin_{k}": torch.tensor(0.1, dtype=torch.float64) for k in range(members + 1)}
    means["rank_bin_0"] = torch.tensor(0.3, dtype=torch.float64)
    means[f"rank_bin_{members}"] = torch.tensor(0.2, dtype=torch.float64)
    assert float(metric.from_means(means, members)) == pytest.approx(0.5, rel=1e-12)

    big, nodes = 8, 20000
    rng = np.random.default_rng(4)
    pred = rng.standard_normal((big, 1, nodes)).astype(np.float32)
    target = rng.standard_normal((1, 1, nodes)).astype(np.float32)
    sample = {
        f"rank_bin_{k}": st.RankBin(k, big).compute(torch.from_numpy(pred), torch.from_numpy(target)).mean(-1)
        for k in (0, big)
    }
    calibrated = metrics.build(["outlier_fraction"], members=big)[0]
    assert float(calibrated.from_means(sample, big)) == pytest.approx(2 / (big + 1), abs=0.02)

    # identical members, the lead-0 reading: offset from the target every node is an outlier, equal to it the mass is
    # spread over all M + 1 bins and the fraction is the calibrated one.
    truth = torch.zeros(1, 1, 50, dtype=torch.float32)
    for offset, expected in ((1.0, 1.0), (0.0, 2 / (members + 1))):
        frozen = torch.full((members, 1, 50), offset, dtype=torch.float32)
        degenerate = {f"rank_bin_{k}": st.RankBin(k, members).compute(frozen, truth).mean(-1) for k in (0, members)}
        assert float(metric.from_means(degenerate, members)) == pytest.approx(expected, rel=1e-12)


def test_binding_rules(fake_sources):
    metric = metrics.from_spec("rank_histogram")
    assert metric.statistics == {} and metric.min_members == 2 and metric.spec == "rank_histogram"
    assert metric.output_dim is None
    assert metrics.unique_statistics([metric]) == {} and metrics.required_aux([metric]) == set()
    with pytest.raises(ValueError, match="was not bound"):
        metric.from_means({}, 4)

    metric.bind_members(4)
    metric.bind_members(4)  # the same ensemble size again is a no-op
    assert metric.output_dim == ("rank", [0, 1, 2, 3, 4])
    with pytest.raises(ValueError, match="already bound to 4 members, cannot rebind to 8"):
        metric.bind_members(8)
    with pytest.raises(ValueError, match="is bound to 4 members but the state has 5"):
        metric.from_means({}, 5)
    with pytest.raises(ValueError, match="needs at least 2 members, the run has 1"):
        metrics.from_spec("rank_histogram").bind_members(1)

    built = metrics.build(["rank_histogram"], members=4)[0]
    assert sorted(built.statistics) == [f"rank_bin_{k}" for k in range(5)]
    assert metrics.build(["rank_histogram"])[0].statistics == {}
    with pytest.raises(ValueError, match="built for 4 members but the frame has 3"):
        st.RankBin(0, 4).compute(torch.zeros(3, 2, 5), torch.zeros(1, 2, 5))

    forecast, targets = fake_sources(members=1)
    with pytest.raises(ValueError, match="at least 2 members"):
        Evaluation(forecast, targets, INITS, "24h", ["rank_histogram"])


def test_rank_histogram_end_to_end(fake_sources, tmp_path):
    forecast, targets = fake_sources(members=M, drift=0.5, spread=0.8)
    evaluation = Evaluation(forecast, targets, INITS, "24h", ["rmse", "rank_histogram", "outlier_fraction"])
    weights, leads = evaluation.weights, evaluation.lead_times
    reference = np.zeros((len(leads), M + 1))
    weight_sum = np.zeros(len(leads))
    for frame, target in evaluation.pairs():
        pred = frame.data.numpy()[:, 0]
        truth = target.numpy()[0, 0]
        below = (pred < truth).sum(0).astype(np.float64)
        ties = (pred == truth).sum(0).astype(np.float64)
        lead = leads.index(frame.lead_time)
        for k in range(M + 1):
            reference[lead, k] += (((below <= k) & (k <= below + ties)) / (ties + 1.0)) @ weights
        weight_sum[lead] += weights.sum()
    reference /= weight_sum[:, None]

    state = evaluation.run()
    dataset = state.to_xarray(evaluation.metrics)
    assert dataset["rank_histogram"].dims == ("lead_time", "bin", "variable", "region", "rank")
    assert dataset["rank_histogram"].shape == (len(leads), 5, 2, 1, M + 1)
    assert dataset["rank"].values.tolist() == [0, 1, 2, 3, 4]
    scored = dataset["rank_histogram"].sel(bin="all", region="global", variable="t")
    np.testing.assert_allclose(scored.values, reference, rtol=1e-12)
    histogram = dataset["rank_histogram"].sel(bin="all").values
    both_ends = histogram[..., 0] + histogram[..., -1]
    np.testing.assert_allclose(dataset["outlier_fraction"].sel(bin="all").values, both_ends, rtol=1e-12)
    np.testing.assert_allclose(histogram.sum(-1), 1.0, rtol=1e-12, atol=0.0)

    for k in range(M + 1):
        assert state.sums[f"rank_bin_{k}"].shape == (len(leads), 4, 2, 1)
    assert json.loads(dataset.attrs["metrics"]) == ["rmse", "rank_histogram", "outlier_fraction"]

    path = tmp_path / "ranks.nc"
    write(dataset, path)
    with xr.open_dataset(path, decode_timedelta=True) as stored:
        assert stored["rank"].values.tolist() == [0, 1, 2, 3, 4]
        assert stored["rank_histogram"].dims == ("lead_time", "bin", "variable", "region", "rank")
    loaded = load_state(path)
    for k in range(M + 1):
        assert torch.equal(loaded.sums[f"rank_bin_{k}"], state.sums[f"rank_bin_{k}"])
    again = loaded.to_xarray(evaluation.metrics)
    np.testing.assert_array_equal(again["rank_histogram"].values, dataset["rank_histogram"].values)


def test_rank_histogram_does_not_perturb_the_other_sums(fake_sources):
    threshold = {"csi": {"label": "warm", "thresholds": {"t": 0.6}}}
    forecast, targets = fake_sources(members=M, drift=0.5, spread=0.2)
    base = Evaluation(forecast, targets, INITS, "24h", BASE_METRICS).run()
    forecast, targets = fake_sources(members=M, drift=0.5, spread=0.2)
    extended = Evaluation(
        forecast, targets, INITS, "24h", [*BASE_METRICS, "rank_histogram", "outlier_fraction", threshold]
    ).run()
    assert list(extended.sums)[: len(base.sums)] == list(base.sums)
    assert all(torch.equal(extended.sums[name], total) for name, total in base.sums.items())
    assert torch.equal(extended.weights, base.weights) and torch.equal(extended.n_init, base.n_init)


def test_merge_and_cli_rebuild(fake_sources, tmp_path):
    forecast, targets = fake_sources(members=M, drift=0.5, spread=0.8)
    evaluation = Evaluation(forecast, targets, INITS, "24h", ["rmse", "rank_histogram", "outlier_fraction"])
    full = evaluation.run()
    shards = [evaluation.run([init]) for init in INITS]
    merged = shards[0].merge(shards[1])
    for k in range(M + 1):
        x, y = merged.sums[f"rank_bin_{k}"], full.sums[f"rank_bin_{k}"]
        assert bool(((x == y) | (x.isnan() & y.isnan())).all())

    parts = [str(tmp_path / f"part{i}.nc") for i in range(len(INITS))]
    for shard, part in zip(shards, parts):
        write(shard.to_xarray(evaluation.metrics), part)
    output = tmp_path / "merged.nc"
    main(["merge", *parts, "-o", str(output)])
    with xr.open_dataset(output, decode_timedelta=True) as stored:
        stored = stored.load()
    xr.testing.assert_equal(stored.drop_attrs(), full.to_xarray(evaluation.metrics).drop_attrs())

    other = AggregationState.zeros(
        evaluation.lead_times, list(full.bins), list(full.variables), list(full.regions), list(full.sums), 5
    )
    with pytest.raises(ValueError, match="cannot merge states with"):
        shards[0].merge(other)


def test_cached_and_uncached_counts_agree():
    members, n, variables, leads = 5, 60, ["a", "b"], [6 * HOUR]
    rng = np.random.default_rng(5)
    weights, masks = rng.random(n) + 0.1, np.ones((n, 1), dtype=bool)
    pred = (rng.integers(-4, 5, (members, 2, n)) / 2).astype(np.float32)  # coarse, so ties are common
    truth = (rng.integers(-4, 5, (1, 2, n)) / 2).astype(np.float32)
    frame = Frame(T0, leads[0], T0 + leads[0], variables, torch.from_numpy(pred))
    target = torch.from_numpy(truth)
    names = [f"rank_bin_{k}" for k in range(members + 1)]

    def accumulate(statistics, aux=None):
        aggregator = Aggregator(weights, masks, statistics, NoBinning())
        state = aggregator.new_state(leads, variables, ["global"], members)
        aggregator.add(state, frame, target, aux)
        return state.sums

    shared = accumulate({name: st.RankBin(k, members) for k, name in enumerate(names)})
    for k, name in enumerate(names):
        alone = accumulate({name: st.RankBin(k, members)})  # this one fills the cache itself
        assert torch.equal(shared[name], alone[name])
        cache = {}
        filling = st.RankBin(k, members).compute(frame.data, target, cache)
        assert sorted(cache) == ["rank_below", "rank_ties"]  # an empty mapping is the caller's, and it is filled
        assert torch.equal(filling, st.RankBin(k, members).compute(frame.data, target, cache))  # from the cache
        assert torch.equal(filling, st.RankBin(k, members).compute(frame.data, target))  # recomputed, aux is empty

    caller = {"climatology": torch.zeros(2, n, dtype=torch.float64)}
    accumulate({name: st.RankBin(k, members) for k, name in enumerate(names)}, caller)
    assert "rank_below" not in caller and "rank_ties" not in caller
    assert st.RankBin(0, members).aux == ()
