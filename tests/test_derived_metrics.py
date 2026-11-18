"""The metrics derived from statistics the other families already store: the error standard deviation, the activity
ratio, the contingency scores pofd, sedi, odds_ratio, orss and f1, the ROC curve and its area, and the relative
economic value. Each is checked against a naive numpy float64 computation from the frames themselves."""

import datetime
import json

import numpy as np
import pytest
import torch
import xarray as xr
from pydantic import ValidationError

from anemoi.evaluation import metrics
from anemoi.evaluation.__main__ import main
from anemoi.evaluation.config import load_config
from anemoi.evaluation.evaluate import Evaluation
from anemoi.evaluation.frame import Grid
from anemoi.evaluation.output import load_state
from anemoi.evaluation.output import write
from anemoi.evaluation.sources.climatology import ArrayClimatology
from anemoi.evaluation.sources.fake import ArrayTargets
from anemoi.evaluation.sources.fake import FakeForecastSource

HOUR = datetime.timedelta(hours=1)
T0 = datetime.datetime(2024, 1, 1)
INITS = [T0, T0 + 12 * HOUR]
CPU = torch.device("cpu")
LABEL = "warm"
THRESHOLD = 0.3
COST_LOSS = [0.05, 0.2, 0.5, 0.8]
CONTINGENCY = ("pofd", "sedi", "odds_ratio", "orss", "f1")
LEVELS = ("roc_hit_rate", "roc_false_alarm_rate", "roc_area")


def spec(kind, label=LABEL, thresholds=None, **extra):
    """The YAML spec of one labelled metric; `rev` takes the module's cost/loss ratios unless given others."""
    body = {"label": label, "thresholds": dict(thresholds or {"t": THRESHOLD})}
    if kind == "rev":
        body["cost_loss"] = list(extra.pop("cost_loss", COST_LOSS))
    return {kind: {**body, **extra}}


ALL_SPECS = ["rmse", "error_std", *(spec(kind) for kind in (*CONTINGENCY, *LEVELS, "rev"))]


def random_sources(members, n=400, seed=0, drift=0.1, spread=0.4):
    """Persistence forecasts of random, time-varying targets, so the errors vary over nodes and lead times and every
    cell of the contingency table and every exceedance level is populated, unlike the constant targets of `conftest`."""
    rng = np.random.default_rng(seed)
    grid = Grid(rng.uniform(-90, 90, n), rng.uniform(0, 360, n))
    dates = [T0 + k * 6 * HOUR for k in range(7)]
    fields = {date: rng.standard_normal((2, n)).astype(np.float32) for date in dates}
    targets = ArrayTargets(grid, ["t", "q"], fields)
    return FakeForecastSource(targets, 6 * HOUR, members, drift, spread, device="cpu"), targets


def climatology_for(forecast, seed=6):
    """A random hour-of-day climatology on the forecast's grid."""
    rng = np.random.default_rng(seed)
    fields = {hour: 0.3 * rng.standard_normal((2, forecast.grid.n)).astype(np.float32) for hour in (0, 6, 12, 18)}
    return ArrayClimatology(forecast.grid, forecast.variables, fields, "hour_of_day")


def numpy_reference(evaluation, members, climatology=None):
    """Every new metric for variable `t`, global region, per lead time, computed node by node in float64 from the
    frames: the error standard deviation in two passes, ROC and REV from the decisions `count >= k` rather than from
    level sums, and the REV from the expenses of the three strategies."""
    weights = evaluation.weights
    leads = evaluation.lead_times
    per_lead = {lead: {"e": [], "w": [], "count": [], "o": [], "binary": [], "fa": [], "ta": []} for lead in leads}
    for frame, target in evaluation.pairs():
        pred = frame.data.numpy().astype(np.float64)[:, 0]
        truth = target.numpy()[0, 0].astype(np.float64)
        mean = pred.mean(0)
        data = per_lead[frame.lead_time]
        data["e"].append(mean - truth)
        data["w"].append(weights)
        data["count"].append((pred > THRESHOLD).sum(0))
        data["o"].append((truth > THRESHOLD).astype(np.float64))
        data["binary"].append((mean > THRESHOLD).astype(np.float64))
        if climatology is not None:
            c = climatology.frame(frame.valid_time, evaluation.variables, CPU).numpy()[0].astype(np.float64)
            data["fa"].append(mean - c)
            data["ta"].append(truth - c)

    expected = {name: [] for name in ("error_std", "activity_ratio", *CONTINGENCY, *LEVELS, "rev")}
    for lead in leads:
        data = {key: np.concatenate(values) if values else None for key, values in per_lead[lead].items()}
        w, e, o, binary, count = data["w"], data["e"], data["o"], data["binary"], data["count"]
        W = w.sum()
        mean_error = (w * e).sum() / W
        expected["error_std"].append(np.sqrt((w * (e - mean_error) ** 2).sum() / W))
        if climatology is not None:
            expected["activity_ratio"].append(np.sqrt((w * data["fa"] ** 2).sum() / (w * data["ta"] ** 2).sum()))

        h = (w * binary * o).sum() / W
        m = (w * (1 - binary) * o).sum() / W
        fa = (w * binary * (1 - o)).sum() / W
        cn = (w * (1 - binary) * (1 - o)).sum() / W
        H, F = h / (h + m), fa / (fa + cn)
        expected["pofd"].append(F)
        expected["sedi"].append(
            (np.log(F) - np.log(H) - np.log(1 - F) + np.log(1 - H))
            / (np.log(F) + np.log(H) + np.log(1 - F) + np.log(1 - H))
        )
        expected["odds_ratio"].append(h * cn / (m * fa))
        expected["orss"].append((h * cn - m * fa) / (h * cn + m * fa))
        expected["f1"].append(2 * h / (2 * h + m + fa))

        decisions = [(count >= k).astype(np.float64) for k in range(members + 1)]
        hit_rate = np.array([(w * d * o).sum() / (w * o).sum() for d in decisions])
        false_alarm_rate = np.array([(w * d * (1 - o)).sum() / (w * (1 - o)).sum() for d in decisions])
        expected["roc_hit_rate"].append(hit_rate)
        expected["roc_false_alarm_rate"].append(false_alarm_rate)
        # np.trapezoid on the points in increasing F: (0, 0), the levels M..1, then (1, 1) which is level 0
        expected["roc_area"].append(np.trapezoid(np.r_[0.0, hit_rate[::-1]], np.r_[0.0, false_alarm_rate[::-1]]))

        s = (w * o).sum() / W
        value = []
        for a in COST_LOSS:
            climate, perfect = min(a, s), s * a
            expenses = [(w * (a * d + (1 - d) * o)).sum() / W for d in decisions[1:]]
            value.append(max((climate - expense) / (climate - perfect) for expense in expenses))
        expected["rev"].append(np.array(value))
    return {name: np.array(values) for name, values in expected.items() if values}


def test_metrics_from_means():
    h, m, fa = 0.2, 0.1, 0.05
    cn = 1.0 - h - m - fa
    means = {
        f"hit_{LABEL}": torch.tensor(h, dtype=torch.float64),
        f"miss_{LABEL}": torch.tensor(m, dtype=torch.float64),
        f"false_alarm_{LABEL}": torch.tensor(fa, dtype=torch.float64),
        "error": torch.tensor(0.3, dtype=torch.float64),
        "squared_error": torch.tensor(0.25, dtype=torch.float64),
        "forecast_anomaly_squared": torch.tensor(0.5, dtype=torch.float64),
        "target_anomaly_squared": torch.tensor(0.8, dtype=torch.float64),
    }
    H, F = h / (h + m), fa / (fa + cn)
    expected = {
        "pofd": F,
        "sedi": (np.log(F) - np.log(H) - np.log(1 - F) + np.log(1 - H))
        / (np.log(F) + np.log(H) + np.log(1 - F) + np.log(1 - H)),
        "odds_ratio": h * cn / (m * fa),
        "orss": (h * cn - m * fa) / (h * cn + m * fa),
        "f1": 2 * h / (2 * h + m + fa),
    }
    values = {}
    for kind, value in expected.items():
        metric = metrics.from_spec(spec(kind))
        assert metric.name == f"{kind}_{LABEL}" and metric.min_members == 1
        assert sorted(metric.statistics) == [f"false_alarm_{LABEL}", f"hit_{LABEL}", f"miss_{LABEL}"]
        values[kind] = float(metric.from_means(means, 3))
        assert values[kind] == pytest.approx(value, rel=1e-12)
    odds = values["odds_ratio"]
    assert values["orss"] == pytest.approx((odds - 1) / (odds + 1), rel=1e-12)
    pod, success = h / (h + m), h / (h + fa)
    assert values["f1"] == pytest.approx(2 * pod * success / (pod + success), rel=1e-12)

    error_std = metrics.from_spec("error_std")
    assert float(error_std.from_means(means, 1)) == pytest.approx(np.sqrt(0.25 - 0.3**2), rel=1e-12)
    rmse = float(metrics.from_spec("rmse").from_means(means, 1))
    assert rmse**2 == pytest.approx(0.3**2 + float(error_std.from_means(means, 1)) ** 2, rel=1e-12)
    constant = {
        "error": torch.tensor(0.1, dtype=torch.float64),
        "squared_error": torch.tensor(0.1, dtype=torch.float64),
    }
    constant["squared_error"] = constant["error"] ** 2 * (1 - 1e-16)  # a rounding-negative variance is clamped, not NaN
    assert float(error_std.from_means(constant, 1)) == 0.0
    assert np.isnan(float(error_std.from_means(dict.fromkeys(constant, torch.tensor(float("nan"))), 1)))
    activity = metrics.from_spec("activity_ratio")
    assert activity.statistics.keys() == {"forecast_anomaly_squared", "target_anomaly_squared"}
    assert metrics.required_aux([activity]) == {"climatology"}
    assert float(activity.from_means(means, 1)) == pytest.approx(np.sqrt(0.5 / 0.8), rel=1e-12)

    # Nothing observed and nothing forecast: the scores dividing by an empty margin are NaN, as the existing ones;
    # pofd is a defined 0 there (every non-event was a correct negative).
    empty = dict.fromkeys(
        [f"hit_{LABEL}", f"miss_{LABEL}", f"false_alarm_{LABEL}"], torch.tensor(0.0, dtype=torch.float64)
    )
    for kind in ("sedi", "odds_ratio", "orss", "f1"):
        assert np.isnan(float(metrics.from_spec(spec(kind)).from_means(empty, 3)))
    assert float(metrics.from_spec(spec("pofd")).from_means(empty, 3)) == 0.0
    # A perfect forecast: the odds ratio is infinite, its skill score and f1 are 1, and sedi is undefined (H = 1, F = 0)
    perfect = {**empty, f"hit_{LABEL}": torch.tensor(0.3, dtype=torch.float64)}
    perfect_values = {kind: float(metrics.from_spec(spec(kind)).from_means(perfect, 3)) for kind in CONTINGENCY}
    assert perfect_values["odds_ratio"] == np.inf and perfect_values["orss"] == 1.0 and perfect_values["f1"] == 1.0
    assert perfect_values["pofd"] == 0.0 and np.isnan(perfect_values["sedi"])


def test_level_metrics_from_means():
    members = 4
    counts = np.array([0.4, 0.1, 0.0, 0.2, 0.3])  # level 2 is empty
    events = np.array([0.04, 0.02, 0.0, 0.14, 0.27])
    means = {f"reliability_count_{LABEL}_{k}": torch.tensor(counts[k]) for k in range(members + 1)}
    means.update({f"reliability_event_{LABEL}_{k}": torch.tensor(events[k]) for k in range(members + 1)})
    built = {kind: metrics.build([spec(kind)], members=members)[0] for kind in (*LEVELS, "rev")}
    values = {kind: metric.from_means(means, members).numpy() for kind, metric in built.items()}

    non_events = counts - events
    hit_rate = np.array([events[k:].sum() / events.sum() for k in range(members + 1)])
    false_alarm_rate = np.array([non_events[k:].sum() / non_events.sum() for k in range(members + 1)])
    np.testing.assert_allclose(values["roc_hit_rate"], hit_rate, rtol=1e-12)
    np.testing.assert_allclose(values["roc_false_alarm_rate"], false_alarm_rate, rtol=1e-12)
    assert values["roc_hit_rate"][0] == 1.0 and values["roc_false_alarm_rate"][0] == 1.0  # exactly, at level 0
    assert values["roc_hit_rate"][2] == values["roc_hit_rate"][3]  # the empty level repeats the next point
    area = np.trapezoid(np.r_[0.0, hit_rate[::-1]], np.r_[0.0, false_alarm_rate[::-1]])
    assert float(values["roc_area"]) == pytest.approx(area, rel=1e-12)
    assert 0.5 < float(values["roc_area"]) <= 1.0

    s = events.sum() / counts.sum()
    expected = []
    for a in COST_LOSS:
        climate = min(a, s)
        level_values = (climate - false_alarm_rate[1:] * a * (1 - s) + hit_rate[1:] * s * (1 - a) - s) / (
            climate - s * a
        )
        expected.append(level_values.max())
    np.testing.assert_allclose(values["rev"], expected, rtol=1e-12)
    assert values["rev"].shape == (len(COST_LOSS),) and values["rev"].max() <= 1.0

    nan_means = dict.fromkeys(means, torch.tensor(float("nan")))
    no_event = {**means, **{f"reliability_event_{LABEL}_{k}": torch.tensor(0.0) for k in range(members + 1)}}
    for kind, metric in built.items():
        assert np.isnan(metric.from_means(nan_means, members).numpy()).all()
        result = metric.from_means(no_event, members).numpy()
        if kind == "roc_false_alarm_rate":
            assert np.isfinite(result).all()  # the non-events are all there
        else:
            assert np.isnan(result).all()  # the hit rate is 0/0, and with it the area and the value
    always = {**means, **{f"reliability_event_{LABEL}_{k}": torch.tensor(counts[k]) for k in range(members + 1)}}
    assert np.isnan(built["rev"].from_means(always, members).numpy()).all()  # no non-event: s = 1, F is 0/0


def test_new_metrics_end_to_end():
    members = 4
    forecast, targets = random_sources(members)
    climatology = climatology_for(forecast)
    specs = [*ALL_SPECS, "bias", "acc", "activity_ratio"]
    evaluation = Evaluation(forecast, targets, INITS, "24h", specs, climatology=climatology)
    expected = numpy_reference(evaluation, members, climatology)
    for name in CONTINGENCY:
        assert np.isfinite(expected[name]).all()  # every cell populated, so no comparison is NaN against NaN
    assert (expected["error_std"] > 0.1).all() and (expected["rev"] > -1).any()

    state = evaluation.run()
    dataset = state.to_xarray(evaluation.metrics)
    scored = dataset.sel(bin="all", region="global", variable="t")
    for name, values in expected.items():
        key = name if name in ("error_std", "activity_ratio") else f"{name}_{LABEL}"
        np.testing.assert_allclose(scored[key].values, values, rtol=1e-12, err_msg=key)
    identity = scored["bias"].values ** 2 + scored["error_std"].values ** 2
    np.testing.assert_allclose(scored["rmse"].values ** 2, identity, rtol=1e-12)
    assert set(state.sums) >= {"error", "squared_error", "forecast_anomaly_squared", "target_anomaly_squared"}

    missing = dataset.sel(variable="q")
    for kind in (*CONTINGENCY, *LEVELS, "rev"):
        assert np.isnan(missing[f"{kind}_{LABEL}"].values).all()
    assert np.isfinite(missing["error_std"].sel(bin="all").values).all()
    assert dataset[f"rev_{LABEL}"].dims == ("lead_time", "bin", "variable", "region", "cost_loss")
    assert dataset["cost_loss"].values.tolist() == COST_LOSS and dataset["cost_loss"].dtype == np.float64
    for kind in LEVELS[:2]:
        assert dataset[f"{kind}_{LABEL}"].dims[-1] == "probability"
    assert dataset[f"roc_area_{LABEL}"].dims == ("lead_time", "bin", "variable", "region")

    with pytest.raises(ValueError, match="climatology"):
        Evaluation(*random_sources(members), INITS, "24h", ["rmse", "activity_ratio"])


def test_one_member_reduces_to_the_contingency_table():
    """At one member the levels are `{0, 1}`, the ROC has one interior point at (pofd, pod) and the REV is the value
    of the deterministic forecast, from the contingency cells."""
    forecast, targets = random_sources(1, seed=3)
    specs = ["rmse", *(spec(kind) for kind in ("pod", "pofd", *LEVELS, "rev"))]
    evaluation = Evaluation(forecast, targets, INITS, "24h", specs)
    state = evaluation.run()
    dataset = state.to_xarray(evaluation.metrics).sel(bin="all", region="global", variable="t")
    assert dataset["probability"].values.tolist() == [0.0, 1.0]
    pod, pofd = dataset[f"pod_{LABEL}"].values, dataset[f"pofd_{LABEL}"].values
    np.testing.assert_allclose(dataset[f"roc_hit_rate_{LABEL}"].values[:, 1], pod, rtol=1e-12)
    np.testing.assert_allclose(dataset[f"roc_false_alarm_rate_{LABEL}"].values[:, 1], pofd, rtol=1e-12)
    np.testing.assert_allclose(dataset[f"roc_area_{LABEL}"].values, (1 + pod - pofd) / 2, rtol=1e-12)

    means = state.means()
    h, m, fa = (means[f"{kind}_{LABEL}"][:, 0, 0, 0].numpy() for kind in ("hit", "miss", "false_alarm"))
    assert (m > 0).all() and (fa > 0).all()  # neither off-diagonal cell is empty, so the point is not trivial
    s = h + m
    expected = np.stack(
        [(np.minimum(a, s) - a * (h + fa) - m) / (np.minimum(a, s) - s * a) for a in COST_LOSS], axis=-1
    )
    np.testing.assert_allclose(dataset[f"rev_{LABEL}"].values, expected, rtol=1e-12)
    assert (dataset[f"rev_{LABEL}"].values < 0).any()  # one level, no envelope: a negative value stays visible


def test_specs_and_config_validation():
    base = {
        "forecast": {"anemoi_inference": {"checkpoint": "/x/inference-last.ckpt", "members": 4}},
        "lead_time": "24h",
        "init_times": {"start": "2024-01-01", "end": "2024-01-01T12", "frequency": "12h"},
        "weights": {"uniform": {}},
    }
    accepted = [*ALL_SPECS, "activity_ratio", spec("rev", label="cold", thresholds={"t": -0.5})]
    assert load_config({**base, "metrics": accepted}).metrics == accepted
    for item in accepted:
        metric = metrics.from_spec(item)
        assert metric.spec == item
        assert metrics.from_spec(json.loads(json.dumps(metric.spec))).spec == item  # as `merge` rebuilds them
    rev = metrics.from_spec(spec("rev", cost_loss=[0.1, 1 / 3]))
    assert rev.output_dim == ("cost_loss", [0.1, 1 / 3]) and rev.statistics == {}  # the axis needs no binding
    assert metrics.from_spec(spec("rev", cost_loss=[0.25])).cost_loss == [0.25]

    invalid = {
        "no cost_loss": {"rev": {"label": LABEL, "thresholds": {"t": THRESHOLD}}},
        "empty": spec("rev", cost_loss=[]),
        "zero": spec("rev", cost_loss=[0.0, 0.5]),
        "one": spec("rev", cost_loss=[0.5, 1.0]),
        "decreasing": spec("rev", cost_loss=[0.5, 0.2]),
        "repeated": spec("rev", cost_loss=[0.2, 0.2]),
        "string": spec("rev", cost_loss=["0.1"]),
        "bool": spec("rev", cost_loss=[True]),
        "nan": spec("rev", cost_loss=[float("nan")]),
        "scalar": {"rev": {"label": LABEL, "thresholds": {"t": THRESHOLD}, "cost_loss": 0.1}},
        "no label": {"sedi": {"thresholds": {"t": THRESHOLD}}},
        "no thresholds": {"roc_area": {"label": LABEL}},
        "members": spec("roc_hit_rate", members=8),
        "cost_loss on roc": spec("roc_area", cost_loss=[0.1]),
        "kwargs on error_std": {"error_std": {"ddof": 1}},
    }
    for name, item in invalid.items():
        with pytest.raises(ValidationError):
            load_config({**base, "metrics": ["rmse", item]})
        with pytest.raises(ValueError):
            metrics.from_spec(item)
    with pytest.raises(ValidationError, match="cost_loss must be strictly increasing"):
        load_config({**base, "metrics": [spec("rev", cost_loss=[0.5, 0.2])]})
    with pytest.raises(ValidationError, match="rev needs cost_loss"):
        load_config({**base, "metrics": [invalid["no cost_loss"]]})

    # one `cost_loss` coordinate per file: two `rev` of different labels must agree on it, refused at config load
    pair = [spec("rev"), spec("rev", label="cold", thresholds={"t": -0.5}, cost_loss=[0.1, 0.3])]
    with pytest.raises(ValidationError, match="disagree on the 'cost_loss' axis"):
        load_config({**base, "metrics": pair})
    with pytest.raises(ValueError, match="rev_warm and rev_cold disagree"):
        metrics.build(pair)
    # one label, one threshold map, with the new families too
    with pytest.raises(ValidationError, match="one label, one threshold map"):
        load_config({**base, "metrics": [spec("sedi"), spec("roc_area", thresholds={"t": 1.0})]})
    with pytest.raises(ValueError, match="duplicate metric names"):
        metrics.build([spec("f1"), spec("f1")])
    # the level metrics share the reliability sums of their label
    shared = metrics.build([spec("reliability"), *(spec(kind) for kind in (*LEVELS, "rev"))], members=3)
    assert len(metrics.unique_statistics(shared)) == 2 * (3 + 1)


def test_round_trip_and_merge(tmp_path):
    members = 3
    forecast, targets = random_sources(members, seed=1)
    specs = [*ALL_SPECS, spec("reliability")]
    evaluation = Evaluation(forecast, targets, INITS, "24h", specs)
    full = evaluation.run()
    dataset = full.to_xarray(evaluation.metrics)

    path = tmp_path / "derived.nc"
    write(dataset, path)
    with xr.open_dataset(path, decode_timedelta=True) as stored:
        assert stored["cost_loss"].values.tolist() == COST_LOSS
        assert stored[f"rev_{LABEL}"].dims[-1] == "cost_loss"
        assert json.loads(stored.attrs["metrics"]) == specs
    again = load_state(path).to_xarray(evaluation.metrics)
    for metric in evaluation.metrics:
        np.testing.assert_array_equal(again[metric.name].values, dataset[metric.name].values)

    parts = [str(tmp_path / f"part{i}.nc") for i in range(len(INITS))]
    for init, part in zip(INITS, parts):
        write(evaluation.run([init]).to_xarray(evaluation.metrics), part)
    output = tmp_path / "merged.nc"
    main(["merge", *parts, "-o", str(output)])  # rebuilds every metric, `rev` with its ratios, from the stored specs
    with xr.open_dataset(output, decode_timedelta=True) as stored:
        stored = stored.load()
    xr.testing.assert_equal(stored.drop_attrs(), dataset.drop_attrs())
