"""Point targets through `Evaluation`: observation operators in the driver, the per-point-set plan and W, regions and
weights on points, out-of-domain counting, provenance and merging; against numpy float64 references."""

import datetime
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from anemoi.graphs import EARTH_RADIUS

from anemoi.evaluation import Grid
from anemoi.evaluation import PointSet
from anemoi.evaluation.aggregation import Aggregator
from anemoi.evaluation.evaluate import Evaluation
from anemoi.evaluation.evaluate import MultiEvaluation
from anemoi.evaluation.operators import ArrayConstants
from anemoi.evaluation.operators import Chain
from anemoi.evaluation.operators import Identity
from anemoi.evaluation.operators import LapseRate
from anemoi.evaluation.operators import ModelSide
from anemoi.evaluation.operators import Nearest
from anemoi.evaluation.operators import ObservationOperatorBase
from anemoi.evaluation.output import IDENTITY_ARRAY_VALUES
from anemoi.evaluation.output import identity_json
from anemoi.evaluation.output import load_state
from anemoi.evaluation.output import merge
from anemoi.evaluation.output import write
from anemoi.evaluation.point_scoring import same_points
from anemoi.evaluation.sources.base import MissingTargetError
from anemoi.evaluation.sources.base import Observations
from anemoi.evaluation.sources.fake import ArrayPointTargets
from anemoi.evaluation.sources.fake import ArrayTargets
from anemoi.evaluation.sources.fake import FakeForecastSource
from anemoi.evaluation.sources.persistence import PersistenceForecastSource

HOUR = datetime.timedelta(hours=1)
T0 = datetime.datetime(2024, 1, 1)
INITS = [T0, T0 + 6 * HOUR, T0 + 12 * HOUR]
LEAD = 12 * HOUR
DATES = [T0 + k * 6 * HOUR for k in range(6)]
VARIABLES = ["2t", "2d"]
METRICS = ["bias", "rmse", "mae"]
STATISTICS = {"error": lambda e: e, "squared_error": lambda e: e**2, "absolute_error": np.abs}
MAX_KM = 60.0
RATE = 0.0065


class WindSpeed(ObservationOperatorBase):
    """Toy derived variable on points: two reads, one product."""

    reads = ("10u", "10v")
    produces = ("ws",)

    def apply(self, frames, plan, valid_time):
        frame = frames[0]
        u, v = (frame.data[:, frame.variables.index(name)].double() for name in self.reads)
        data = torch.cat([frame.data, torch.sqrt(u * u + v * v).float()[:, None]], dim=1)
        return self.output_frame(frames, valid_time, self.output_variables(frame.variables), data)


class GridWindSpeed(WindSpeed):
    input_space = output_space = "grid"


class Scaled(ObservationOperatorBase):
    """Toy user step with a parameter and no merge identity of its own; the size it sets at bind is not recorded."""

    def __init__(self, factor):
        self.factor = factor
        self.size = None

    def bind(self, model):
        super().bind(model)
        self.size = len(model.variables)

    def apply(self, frames, plan, valid_time):
        frame = frames[0]
        return self.output_frame(frames, valid_time, list(frame.variables), frame.data * self.factor)


class Tabled(ObservationOperatorBase):
    """Toy user step whose parameter is not a scalar, recorded through an overridden `parameters()`."""

    def __init__(self, table):
        self._table = table

    def parameters(self):
        return {"table": self._table}

    def apply(self, frames, plan, valid_time):
        return frames[0]


class Calibrated(ArrayPointTargets):
    """Point targets whose merge identity carries an array."""

    offsets = None

    def merge_identity(self):
        return {**self.describe(), "offsets": self.offsets}


class Located(ArrayConstants):
    """A constants provider whose merge identity holds a set, which has no order to record."""

    def merge_identity(self):
        return {**super().merge_identity(), "files": {"orography.npy"}}


class Bracketing(ObservationOperatorBase):
    stencil = "bracketing"

    def apply(self, frames, plan, valid_time):
        return frames[0]


class GraphForecast(FakeForecastSource):
    """A fake forecast with a model graph holding one node attribute, `cutout_mask`."""

    has_graph = True

    def __init__(self, *args, mask, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = mask

    def graph_node_attribute(self, name, nodes="data"):
        if name != "cutout_mask":
            raise ValueError(f"no node attribute {name!r}")
        return self.mask


class NoModel(FakeForecastSource):
    """A forecast that must not be asked for frames."""

    def frames(self, *args):
        raise AssertionError("the model ran")


class OtherSource(ArrayPointTargets):
    """The same observations under another source description."""


class MultiFake:
    """A shared rollout over per-dataset fake forecasts, as `InferenceForecastSource.multi_frames` yields it."""

    def __init__(self, sources):
        self.sources = sources

    def multi_frames(self, init_time, lead_time, variables, device):
        streams = {name: self.sources[name].frames(init_time, lead_time, variables[name], device) for name in variables}
        for frames in zip(*streams.values()):
            yield dict(zip(streams, frames))

    def close(self):
        pass


def lam_grid(n=150, seed=0):
    rng = np.random.default_rng(seed)
    return Grid(rng.uniform(44, 49, n), rng.uniform(5, 12, n))


def truth(grid, variables=VARIABLES, seed=1, gaps=False, dates=DATES):
    rng = np.random.default_rng(seed)
    fields = {}
    for date in dates:
        field = rng.normal(280, 5, (len(variables), grid.n)).astype(np.float32)
        if gaps:
            field[:, rng.integers(0, grid.n, 5)] = np.nan
        fields[date] = field
    return ArrayTargets(grid, variables, fields)


def forecast_of(targets, cls=FakeForecastSource, **kwargs):
    return cls(targets, 6 * HOUR, members=3, drift=0.5, spread=0.2, device="cpu", **kwargs)


def random_points(rng, p, outside=2, key=None):
    """`p` points over the Alps and `outside` far outside the LAM (longitudes in [0, 360)), with elevations."""
    lats = np.concatenate([rng.uniform(44.5, 48.5, p), rng.uniform(25, 35, outside)])
    lons = np.concatenate([rng.uniform(5.5, 11.5, p), rng.uniform(340, 350, outside)])
    return PointSet(lats, lons, attributes={"elevation": rng.uniform(0, 3000, p + outside)}, key=key)


def varying_targets(variables=VARIABLES, seed=2, missing=(T0 + 18 * HOUR,), empty=(), dates=DATES, cls=None):
    """A different point set at every valid time, NaN gaps, `missing` valid times absent and `empty` ones with no
    point."""
    rng = np.random.default_rng(seed)
    observations = {}
    for date in dates:
        if date in missing:
            continue
        points = (
            PointSet([], [], attributes={"elevation": []}) if date in empty else random_points(rng, rng.integers(8, 25))
        )
        values = rng.normal(280, 5, (len(variables), len(points))).astype(np.float32)
        if len(points):
            values[0, 0] = np.nan
            values[-1, rng.integers(0, len(points))] = np.nan
        observations[date] = (points, values)
    return (cls or ArrayPointTargets)(variables, observations)


def fixed_targets(points, variables=VARIABLES, seed=3, dates=DATES):
    rng = np.random.default_rng(seed)
    fields = {}
    for date in dates:
        values = rng.normal(280, 5, (len(variables), len(points))).astype(np.float32)
        values[0, 1] = np.nan
        fields[date] = values
    return ArrayPointTargets.fixed(points, variables, fields)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS * np.arcsin(np.sqrt(a))


def brute_nearest(points, grid):
    distance = haversine_km(points.latitudes[:, None], points.longitudes[:, None], grid.latitudes, grid.longitudes)
    return distance.argmin(1), distance.min(1)


def reference(forecast, targets, orography=None, inits=INITS, regions=None, weights=None):
    """Summed statistics, weights and out-of-domain counts per (lead, variable, region) by brute force in numpy
    float64: nearest node by haversine argmin, the lapse rate on 2t, the ensemble mean error over finite pairs.
    `regions` and `weights` are functions of the point set (all points, weight one by default)."""
    leads = forecast.lead_times(LEAD)
    regions = regions or {"global": lambda points: np.ones(len(points), bool)}
    weights = weights or (lambda points: np.ones(len(points)))
    shape = (len(leads), len(targets.variables), len(regions))
    sums = {name: np.zeros(shape) for name in STATISTICS}
    total, outside = np.zeros(shape), np.zeros(len(leads), dtype=np.int64)
    for init in inits:
        for frame in forecast.frames(init, LEAD, list(targets.variables), "cpu"):
            if frame.valid_time not in targets.data or not len(targets.data[frame.valid_time][0]):
                continue
            points, values = targets.data[frame.valid_time]
            nodes, distance = brute_nearest(points, forecast.grid)
            inside = distance <= MAX_KM
            data = frame.data.numpy()[:, :, nodes].copy()
            data[:, :, ~inside] = np.nan
            if orography is not None:
                height = points.attribute("elevation") - orography[nodes]
                data[:, 0] = (data[:, 0].astype(np.float64) - RATE * height).astype(np.float32)
            error = data.astype(np.float64).mean(0) - values.astype(np.float64)
            valid = np.isfinite(data).all(0) & np.isfinite(values)
            lead = leads.index(frame.lead_time)
            outside[lead] += np.count_nonzero(~inside)
            w = weights(points)
            for r, mask in enumerate(regions.values()):
                select = valid & mask(points)[None]
                total[lead, :, r] += np.where(select, w, 0.0).sum(1)
                for name, function in STATISTICS.items():
                    sums[name][lead, :, r] += np.where(select, function(error) * w, 0.0).sum(1)
    return sums, total, outside


def point_evaluation(forecast, targets, operator=None, inits=INITS, lead=LEAD, **kwargs):
    operator = Nearest(max_distance_km=MAX_KM) if operator is None else operator
    kwargs.setdefault("bins", "none")
    return Evaluation(forecast, targets, inits, lead, METRICS, observation_operator=operator, **kwargs)


def counting(function, calls):
    def wrapper(*args, **kwargs):
        calls.append(args[0] if args else next(iter(kwargs.values())))
        return function(*args, **kwargs)

    return wrapper


@pytest.mark.parametrize("lapse", [False, True], ids=["nearest", "nearest+lapse_rate"])
def test_varying_points_match_numpy_reference(lapse):
    grid = lam_grid()
    forecast = forecast_of(truth(grid))
    targets = varying_targets()
    orography = np.random.default_rng(9).uniform(0, 2500, grid.n)
    if lapse:
        operator = Chain([Nearest(max_distance_km=MAX_KM), LapseRate("2t", rate=RATE)])
        constants = ArrayConstants({"orography": (orography, "m")})
        evaluation = point_evaluation(forecast, targets, operator, constants=constants)
    else:
        evaluation = point_evaluation(forecast, targets)
    assert evaluation.variables == VARIABLES and evaluation.forecast_variables == VARIABLES
    state = evaluation.run()
    sums, weights, outside = reference(forecast, targets, orography if lapse else None)
    assert torch.equal(state.weights[:, 0], torch.from_numpy(weights))
    for name in STATISTICS:
        np.testing.assert_allclose(state.sums[name][:, 0].numpy(), sums[name], rtol=1e-12, atol=1e-12)
    assert state.out_of_domain[:, 0].tolist() == outside.tolist() and outside.min() > 0
    # one read per frame with observations: 3 inits x 2 leads, two of them at the missing 18 h
    assert targets.reads == 4 and state.n_init[:, 0].tolist() == [2, 2]
    dataset = state.to_xarray(evaluation.metrics)
    assert np.array_equal(dataset["weight_sum"].sel(bin="all", region="global").values, weights[..., 0])
    assert dataset["out_of_domain"].sel(bin="all").values.tolist() == outside.tolist()
    attrs = state.attrs
    steps = json.loads(attrs["observation_operator"])["steps"]
    assert [step["operator"] for step in steps] == (["Nearest", "LapseRate"] if lapse else ["Nearest"])
    if lapse:
        assert json.loads(attrs["observation_operator"])["constants"]["sha256"]
    assert json.loads(attrs["point_targets"])["points"] == "varying" and "config" not in attrs
    assert "observation_diagnostics" not in attrs  # a fixed set's only

    frame, target = next(evaluation.pairs([INITS[0]]))
    points, values = targets.data[frame.valid_time]
    assert frame.variables == VARIABLES and tuple(frame.data.shape) == (3, 2, len(points)) and frame.points is points
    np.testing.assert_array_equal(target[0].numpy(), values)
    if not lapse:
        nodes, distance = brute_nearest(points, grid)
        gridded = next(iter(forecast.frames(INITS[0], LEAD, VARIABLES, "cpu"))).data[:, :, nodes]
        inside = torch.from_numpy(distance <= MAX_KM)
        assert torch.equal(frame.data[:, :, inside], gridded[:, :, inside])
        assert frame.data[:, :, ~inside].isnan().all()


def test_empty_and_missing_valid_times():
    grid = lam_grid()
    forecast = forecast_of(truth(grid))
    targets = varying_targets(missing=(T0 + 18 * HOUR,), empty=(T0 + 12 * HOUR,))
    state = point_evaluation(forecast, targets).run()
    _, weights, _ = reference(forecast, targets)
    assert torch.equal(state.weights[:, 0], torch.from_numpy(weights))
    assert state.n_init[:, 0].tolist() == [1, 1]  # 12 h is empty, 18 h missing: only 6 h and 24 h are scored
    with pytest.raises(MissingTargetError, match="no observations"):
        point_evaluation(forecast, targets, on_missing_target="raise").run()


def test_fixed_set_weights_and_regions_match_reference(tmp_path):
    grid = lam_grid()
    rng = np.random.default_rng(12)
    mask = rng.random(grid.n) < 0.5
    forecast = forecast_of(truth(grid), GraphForecast, mask=mask)
    points = random_points(rng, 20)
    targets = fixed_targets(points)
    weights = rng.uniform(0.5, 2.0, len(points))
    np.save(tmp_path / "even.npy", np.arange(len(points)) % 2 == 0)
    regions = {
        "all": "all",
        "box": {"bbox": {"north": 47, "west": 6, "south": 45, "east": 9}},
        "cutout": {"graph_attribute": "cutout_mask"},
        "even": {"file": str(tmp_path / "even.npy")},
    }
    evaluation = point_evaluation(forecast, targets, weights=weights, regions=regions)
    state = evaluation.run()
    expected = {
        "all": lambda p: np.ones(len(p), bool),
        "box": lambda p: (p.latitudes >= 45) & (p.latitudes <= 47) & (p.longitudes >= 6) & (p.longitudes <= 9),
        "cutout": lambda p: mask[brute_nearest(p, grid)[0]],
        "even": lambda p: np.arange(len(p)) % 2 == 0,
    }
    sums, total, _ = reference(forecast, targets, regions=expected, weights=lambda p: weights)
    assert all(m(points).any() and not m(points).all() for name, m in expected.items() if name != "all")
    np.testing.assert_allclose(state.weights[:, 0].numpy(), total, rtol=1e-12)
    for name in STATISTICS:
        np.testing.assert_allclose(state.sums[name][:, 0].numpy(), sums[name], rtol=1e-12, atol=1e-12)


def test_plans_and_matrices_built_once_per_point_set():
    dates = [T0 + k * 6 * HOUR for k in range(14)]
    inits, lead = dates[:6], 48 * HOUR
    grid = lam_grid()
    forecast = forecast_of(truth(grid, dates=dates))

    def counted(targets):
        evaluation = point_evaluation(forecast, targets, inits=inits, lead=lead)
        prepares, builds = [], []
        scoring = evaluation._points
        scoring.chain.prepare = counting(scoring.chain.prepare, prepares)
        scoring.weights = counting(scoring.weights, builds)
        evaluation.run()
        return evaluation, prepares, builds

    varying, prepares, builds = counted(varying_targets(missing=(), dates=dates))
    assert len(varying.lead_times) == 8
    assert len(prepares) == len(builds) == len({p.identity for p in prepares}) == 13  # 48 frames, 13 valid times
    assert len(varying._points._cache) == len(varying.lead_times) + 1  # bounded: 13 sets seen

    points = random_points(np.random.default_rng(4), 12, key="stations")
    fixed, prepares, builds = counted(fixed_targets(points, dates=dates))
    assert len(prepares) == len(builds) == 1
    with pytest.raises(ValueError, match="differ in their points"):
        fixed._points.entry(PointSet(points.latitudes + 0.1, points.longitudes, key="stations"))


def test_a_fixed_source_must_return_its_set():
    grid = lam_grid()
    forecast = forecast_of(truth(grid))
    points = random_points(np.random.default_rng(13), 10)
    targets = fixed_targets(points)
    other = random_points(np.random.default_rng(14), 10)
    targets.data[DATES[2]] = (other, targets.data[DATES[2]][1])
    with pytest.raises(ValueError, match="declares a fixed point set"):
        point_evaluation(forecast, targets).run()


def test_regions_on_points():
    rng = np.random.default_rng(5)
    grid = Grid(rng.uniform(-90, 90, 400), rng.uniform(0, 360, 400))
    mask = rng.random(grid.n) < 0.5
    forecast = forecast_of(truth(grid), GraphForecast, mask=mask)
    points = PointSet([10.0, 10.0, 10.0, 10.0, 10.0, 60.0], [355.0, -5.0, 5.0, 15.0, 180.0, 0.0])
    targets = fixed_targets(points)
    boxes = {
        "box": {"bbox": {"north": 20, "west": -10, "south": 0, "east": 10}},
        "box360": {"bbox": {"north": 20, "west": 350, "south": 0, "east": 370}},
    }
    regions = {"all": "all", **boxes, "cutout": {"graph_attribute": "cutout_mask"}}
    evaluation = point_evaluation(forecast, targets, Nearest(max_distance_km=None), regions=regions)
    masks = evaluation.regions  # the fixed set's
    assert masks["all"].all()
    assert masks["box"].tolist() == [True, True, True, False, False, False]
    assert masks["box360"].tolist() == masks["box"].tolist()
    nodes, _ = brute_nearest(points, grid)
    assert masks["cutout"].tolist() == mask[nodes].tolist()
    state = evaluation.run()
    assert state.regions == list(regions)
    assert state.weights[0, 0, 0, 0].item() == 3 * (len(points) - 1)  # 2t, lead 6 h: point 1 has a NaN target

    boxed = point_evaluation(forecast, targets, Nearest(max_distance_km=None), regions=boxes)
    boxed.run()
    assert "nearest_nodes" not in boxed._points.__dict__  # no graph-attribute region, no second tree
    varying = point_evaluation(forecast, varying_targets(), regions=regions)
    with pytest.raises(ValueError, match="point set varies"):
        varying.regions
    some = random_points(rng, 6, outside=0)
    assert set(varying.point_regions(some)) == set(regions)


def test_default_weights_uniform_per_observation():
    grid = lam_grid()
    forecast = forecast_of(truth(grid))
    points = random_points(np.random.default_rng(6), 10)
    fixed = point_evaluation(forecast, fixed_targets(points))
    assert np.array_equal(fixed.weights, np.ones(len(points)))
    varying = point_evaluation(forecast, varying_targets())
    with pytest.raises(ValueError, match="point set varies"):
        varying.weights
    assert np.array_equal(varying.point_weights(points), np.ones(len(points)))
    custom = np.arange(1.0, len(points) + 1)
    evaluation = point_evaluation(forecast, fixed_targets(points), weights=custom)
    assert np.array_equal(evaluation.weights, custom) and evaluation.plan()["weights"] == "array"


def test_file_weights_and_regions_on_a_fixed_set(tmp_path):
    grid = lam_grid()
    forecast = forecast_of(truth(grid))
    points = random_points(np.random.default_rng(7), 10)
    np.save(tmp_path / "w.npy", np.linspace(1, 2, len(points)))
    np.save(tmp_path / "r.npy", np.arange(len(points)) % 2 == 0)
    evaluation = point_evaluation(
        forecast,
        fixed_targets(points),
        weights={"file": str(tmp_path / "w.npy")},
        regions={"even": {"file": str(tmp_path / "r.npy")}},
    )
    assert np.array_equal(evaluation.weights, np.linspace(1, 2, len(points)))
    assert evaluation.regions["even"].tolist() == (np.arange(len(points)) % 2 == 0).tolist()
    evaluation.run()
    np.save(tmp_path / "short.npy", np.ones(3))
    with pytest.raises(ValueError, match="expected"):
        point_evaluation(forecast, fixed_targets(points), weights={"file": str(tmp_path / "short.npy")}).weights


REFUSALS = {
    "no operator": "need an observation_operator",
    "identity": "need an observation_operator",
    "grid targets": "targets are on the grid",
    "grid derived": "take no observation",
    "grid output": "must map the forecast",
    "points input": "takes points frames",
    "bracketing": "time stencil",
    "anomaly": "anomaly metrics",
    "area weights": "per model node",
    "voronoi": "per model node",
    "sub-grid region": "sub-grid regions",
    "file weights": "needs a fixed point",
    "file region": "needs a fixed",
    "array weights": "needs a fixed point set",
    "array region": "needs a fixed point set",
    "weights shape": "expected",
    "region shape": "expected",
    "no constants": "with a provider that serves them",
    "no elevation": "elevation",
    "stray constants": "constants are",
    "bound": "bound to a model before",
    "needless constants": "constants are only read by an operator step",
    "unknown variable": "cannot deliver",
}


def refusal(label, tmp_path):
    """Forecast, targets and keyword arguments of the refusal `label`."""
    grid = lam_grid()
    gridded = truth(grid)
    points = random_points(np.random.default_rng(8), 10)
    fixed = fixed_targets(points)
    fields = {date: values for date, (_, values) in fixed.data.items()}
    no_elevation = ArrayPointTargets.fixed(PointSet(points.latitudes, points.longitudes), VARIABLES, fields)
    orography = ArrayConstants({"orography": (np.zeros(grid.n), "m")})
    np.save(tmp_path / "w.npy", np.ones(10))
    bound = Nearest(max_distance_km=MAX_KM)
    Chain([bound]).bind(ModelSide("grid", VARIABLES, grid=grid))
    lapse = Chain([Nearest(max_distance_km=MAX_KM), LapseRate()])
    nearest = Nearest(max_distance_km=MAX_KM)
    w = str(tmp_path / "w.npy")
    cases = {
        "no operator": (varying_targets(), dict(observation_operator=None)),
        "identity": (varying_targets(), dict(observation_operator=Identity())),
        "grid targets": (gridded, dict(observation_operator=nearest)),
        "grid derived": (gridded, dict(observation_operator=GridWindSpeed())),
        "grid output": (varying_targets(), dict(observation_operator=GridWindSpeed())),
        "points input": (varying_targets(), dict(observation_operator=WindSpeed())),
        "bracketing": (varying_targets(), dict(observation_operator=Chain([nearest, Bracketing()]))),
        "anomaly": (varying_targets(), dict(metrics=["acc"])),
        "area weights": (varying_targets(), dict(weights={"graph_attribute": "area_weight"})),
        "voronoi": (varying_targets(), dict(weights={"spherical_voronoi": {}})),
        "sub-grid region": (varying_targets(), dict(regions={"g": {"grid": 0}})),
        "file weights": (varying_targets(), dict(weights={"file": w})),
        "file region": (varying_targets(), dict(regions={"r": {"file": w}})),
        "array weights": (varying_targets(), dict(weights=np.ones(10))),
        "array region": (varying_targets(), dict(regions={"r": np.ones(10, bool)})),
        "weights shape": (fixed, dict(weights=np.ones(11))),
        "region shape": (fixed, dict(regions={"r": np.ones(9, bool)})),
        "no constants": (fixed, dict(observation_operator=lapse)),
        "no elevation": (no_elevation, dict(observation_operator=lapse, constants=orography)),
        "stray constants": (gridded, dict(observation_operator=None, constants=orography)),
        "bound": (fixed, dict(observation_operator=bound)),
        "needless constants": (fixed, dict(observation_operator=nearest, constants=orography)),
        "unknown variable": (fixed, dict(variables=["ws"])),
    }
    assert set(cases) == set(REFUSALS)
    return forecast_of(gridded), *cases[label]


@pytest.mark.parametrize("label", list(REFUSALS))
def test_construction_refusals(label, tmp_path):
    forecast, targets, kwargs = refusal(label, tmp_path)
    kwargs = {"observation_operator": Nearest(max_distance_km=MAX_KM), "bins": "none", **kwargs}
    metrics = kwargs.pop("metrics", METRICS)
    with pytest.raises(ValueError, match=REFUSALS[label]):
        Evaluation(forecast, targets, INITS, LEAD, metrics, **kwargs)


def test_constant_errors_surface_before_the_model_runs():
    grid = lam_grid()
    forecast = forecast_of(truth(grid), NoModel)
    operator = Chain([Nearest(max_distance_km=MAX_KM), LapseRate()])
    for constants, match in (
        (ArrayConstants({"orography": (np.zeros(grid.n), "m**2 s**-2")}), "is in"),
        (ArrayConstants({"orography": (np.zeros(grid.n - 1), "m")}), "shape"),
    ):
        with pytest.raises((ValueError, KeyError), match=match):
            point_evaluation(forecast, varying_targets(), operator, constants=constants).plan()
        with pytest.raises((ValueError, KeyError), match=match):  # NoModel would raise AssertionError
            point_evaluation(forecast, varying_targets(), operator, constants=constants).run()
    with pytest.raises(ValueError, match="provider serves \\['height'\\]"):  # at construction
        point_evaluation(
            forecast, varying_targets(), operator, constants=ArrayConstants({"height": (np.zeros(grid.n), "m")})
        )


def test_derived_variable_selected_by_name():
    grid = lam_grid()
    forecast = forecast_of(truth(grid, ["2t", "10u", "10v"]))
    targets = varying_targets(["ws", "2t"])
    evaluation = point_evaluation(forecast, targets, Chain([Nearest(max_distance_km=MAX_KM), WindSpeed()]))
    assert evaluation.variables == ["2t", "ws"] and evaluation.forecast_variables == ["2t", "10u", "10v"]
    frame, _ = next(evaluation.pairs([INITS[0]]))
    points = targets.data[frame.valid_time][0]
    nodes, distance = brute_nearest(points, grid)
    gridded = next(iter(forecast.frames(INITS[0], LEAD, ["2t", "10u", "10v"], "cpu"))).data[:, :, nodes].double()
    inside = distance <= MAX_KM
    speed = torch.sqrt(gridded[:, 1] ** 2 + gridded[:, 2] ** 2).float()
    assert frame.variables == ["2t", "ws"]
    assert torch.equal(frame.data[:, 1, inside], speed[:, inside])
    assert torch.equal(frame.data[:, 0, inside], gridded[:, 0, inside].float())
    # a grid -> grid step before the spatial step is driven too: the same speeds, derived on the grid
    on_grid = point_evaluation(forecast, targets, Chain([GridWindSpeed(), Nearest(max_distance_km=MAX_KM)]))
    first, _ = next(on_grid.pairs([INITS[0]]))
    assert on_grid.forecast_variables == ["2t", "10u", "10v"] and first.variables == frame.variables
    torch.testing.assert_close(first.data, frame.data, rtol=0, atol=0, equal_nan=True)
    # a lapse-rate chain scoring only 2d does not ask the forecast for 2t
    lapse = Chain([Nearest(max_distance_km=MAX_KM), LapseRate()])
    constants = ArrayConstants({"orography": (np.zeros(grid.n), "m")})
    only_2d = point_evaluation(
        forecast_of(truth(grid)), varying_targets(), lapse, variables=["2d"], constants=constants
    )
    assert only_2d.forecast_variables == ["2d"]
    only_2d.run()


def test_one_operator_instance_serves_two_evaluations():
    grid = lam_grid()
    gridded = truth(grid)
    targets = varying_targets()
    operator = Chain([Nearest(max_distance_km=MAX_KM), LapseRate()])
    constants = ArrayConstants({"orography": (np.zeros(grid.n), "m")})
    model = point_evaluation(forecast_of(gridded), targets, operator, constants=constants)
    persistence = point_evaluation(PersistenceForecastSource(gridded, "6h", 3), targets, operator, constants=constants)
    model_state, persistence_state = model.run(), persistence.run()
    assert model.observation_operator is not persistence.observation_operator
    assert model.observation_operator.steps[0] is not operator.steps[0] and not operator.bound
    assert torch.equal(model_state.weights, persistence_state.weights)
    assert not torch.equal(model_state.sums["error"], persistence_state.sums["error"])
    assert model.run().sums["error"].equal(model_state.sums["error"])  # a second run reuses the bound copy


def test_sharded_point_run_merges_to_the_unsharded_one(tmp_path):
    grid = lam_grid()
    forecast = forecast_of(truth(grid))
    evaluation = point_evaluation(forecast, varying_targets(), bins="season")
    full = evaluation.run()
    paths = []
    for i in range(2):
        state = evaluation.run(evaluation.shard(i, 2))
        state.attrs["shard"] = f"{i}/2"
        paths.append(tmp_path / f"shard{i}.nc")
        write(state.to_xarray(evaluation.metrics), paths[-1])
    merged = merge(paths)
    assert torch.equal(merged.weights, full.weights) and torch.equal(merged.n_init, full.n_init)
    assert torch.equal(merged.out_of_domain, full.out_of_domain)
    for name, total in full.sums.items():
        torch.testing.assert_close(merged.sums[name], total, rtol=1e-12, atol=1e-12)
    for key in ("observation_operator", "point_targets"):
        assert merged.attrs[key] == full.attrs[key]
    assert load_state(paths[0]).out_of_domain is not None

    def shards(first, second):
        return [first.run(first.shard(0, 2)), second.run(second.shard(1, 2))]

    other = point_evaluation(forecast, varying_targets(), Nearest(max_distance_km=2 * MAX_KM), bins="season")
    with pytest.raises(ValueError, match="observation_operator"):
        merge(shards(evaluation, other))
    renamed = point_evaluation(forecast, varying_targets(cls=OtherSource), bins="season")
    with pytest.raises(ValueError, match="point_targets"):
        merge(shards(evaluation, renamed))
    lapse = [
        point_evaluation(
            forecast,
            varying_targets(),
            Chain([Nearest(max_distance_km=MAX_KM), LapseRate()]),
            bins="season",
            constants=ArrayConstants({"orography": (np.full(grid.n, height), "m")}),
        )
        for height in (0.0, 1000.0)
    ]
    merge(shards(lapse[0], lapse[0]))
    with pytest.raises(ValueError, match="observation_operator"):
        merge(shards(*lapse))
    scaled = [
        point_evaluation(
            forecast, varying_targets(), Chain([Nearest(max_distance_km=MAX_KM), Scaled(f)]), bins="season"
        )
        for f in (1.0, 2.0, 2.0)
    ]
    merged = merge(shards(scaled[1], scaled[2]))
    parameters = json.loads(merged.attrs["observation_operator"])["steps"][1]["parameters"]
    assert parameters == {"factor": 2.0, "size": None}  # the values before the bind
    with pytest.raises(ValueError, match="observation_operator"):
        merge(shards(scaled[0], scaled[1]))
    numpy = [
        point_evaluation(
            forecast, varying_targets(), Chain([Nearest(max_distance_km=MAX_KM), Scaled(np.float32(f))]), bins="season"
        )
        for f in (1, 2)
    ]
    with pytest.raises(ValueError, match="observation_operator"):
        merge(shards(*numpy))  # numpy scalars are parameters too
    gridded = Evaluation(forecast, truth(grid), INITS, LEAD, METRICS, bins="season", weights={"uniform": {}})
    with pytest.raises(ValueError, match="observation_operator"):
        merge(shards(evaluation, gridded))
    with pytest.raises(ValueError, match="point targets"):
        shards(evaluation, gridded)[0].merge(gridded.run(gridded.shard(1, 2)))


def test_identity_json_is_exact_and_strict():
    assert identity_json({"b": 1, "a": 0.1}) == '{"a": 0.1, "b": 1}'
    assert identity_json(np.float32(0.1)) == repr(float(np.float32(0.1))) != identity_json(0.1)
    assert identity_json([np.int64(3), np.bool_(True), torch.tensor(2.5), np.array(1.5), (1, "a")]) == (
        '[3, true, 2.5, 1.5, [1, "a"]]'
    )
    assert identity_json([math.nan, math.inf, -math.inf]) == "[NaN, Infinity, -Infinity]"
    small = json.loads(identity_json(np.arange(IDENTITY_ARRAY_VALUES, dtype=np.int32).reshape(8, -1)))
    assert small == {"array": "int32", "shape": [8, 8], "values": np.arange(64).reshape(8, 8).tolist()}
    large = json.loads(identity_json(np.arange(IDENTITY_ARRAY_VALUES + 1.0)))
    assert set(large) == {"array", "shape", "sha256"} and large["shape"] == [65] and large["array"] == "float64"
    assert identity_json(np.arange(66.0).reshape(6, 11)) != identity_json(np.arange(66.0).reshape(11, 6))
    assert identity_json(np.arange(66.0, dtype=">f8")) == identity_json(np.arange(66.0))  # byte order aside
    assert identity_json(torch.arange(66.0)) == identity_json(np.arange(66.0, dtype=np.float32))
    when = datetime.datetime(2024, 1, 1, 6, 30, 0, 5)
    tagged = {
        "path": Path("/data/obs.zarr"),
        "datetime": when,
        "date": when.date(),
        "timedelta": datetime.timedelta(hours=6, microseconds=1),
        "datetime64": np.datetime64("2024-01-01T06", "h"),
        "timedelta64": np.timedelta64(600, "s"),
        "nat": np.timedelta64("NaT", "s"),
        "steps": np.timedelta64(3, "10m"),
    }
    assert json.loads(identity_json(tagged)) == {
        "path": {"__path__": "/data/obs.zarr"},
        "datetime": {"__datetime__": "2024-01-01T06:30:00.000005"},
        "date": {"__date__": "2024-01-01"},
        "timedelta": {"__timedelta__": [21_600_000_001_000, "ns"]},
        "datetime64": {"__datetime64__": ["2024-01-01T06", "h"]},
        "timedelta64": {"__timedelta64__": [600, "s"]},
        "nat": {"__timedelta64__": ["NaT", "s"]},
        "steps": {"__timedelta64__": [3, "10m"]},
    }
    assert identity_json(tagged) == identity_json(dict(reversed(tagged.items())))  # stable
    assert identity_json(Path("/data/obs.zarr")) != identity_json("/data/obs.zarr")  # not the plain string
    assert identity_json(when) != identity_json(when.isoformat())
    assert identity_json(np.datetime64("2024-01-01", "D")) != identity_json(np.datetime64("2024-01-01T00", "h"))
    # pandas keeps its nanoseconds; a pandas duration in whole microseconds equals the datetime.timedelta
    assert identity_json(pd.Timedelta(1999, "ns")) == '{"__timedelta__": [1999, "ns"]}'
    assert identity_json(pd.Timedelta(1000, "ns")) == identity_json(datetime.timedelta(microseconds=1))
    assert identity_json(pd.Timedelta(-1, "ns")) == '{"__timedelta__": [-1, "ns"]}'
    stamp = pd.Timestamp("2024-01-01T00:00:00.000000001")
    assert identity_json(stamp) == '{"__datetime__": "2024-01-01T00:00:00.000000001"}'
    assert identity_json(stamp) != identity_json(stamp.floor("us")) == identity_json(datetime.datetime(2024, 1, 1))
    nested = {"z": [np.ones(3), {"y": np.float32(0.5)}], "a": torch.ones(70), "n": None, "t": tagged}
    assert identity_json(json.loads(identity_json(nested))) == identity_json(nested)  # what merge compares
    for value, match in (
        (
            {850: 1.0},
            r"^the merge identity: the key 850 \(int\) is not a str, and a merge identity takes str keys only$",
        ),
        ({"a": [{1: 2}]}, r"^the merge identity\.a\[0\]: the key 1 "),
        ({"files": {"x"}}, r"^the merge identity\.files: set cannot be held exactly by a merge identity"),
        (datetime.time(6), ": time cannot be held"),
        (object(), ": object cannot be held"),
        (np.array(["a", None], dtype=object), ": a object array cannot be held"),
        (np.ones(2, dtype=np.complex128), ": a complex128 array"),
        (np.array(["2024-01-01"], dtype="datetime64[s]"), r": a datetime64\[s\] array"),
        (np.ma.masked_array(np.zeros(100), mask=np.arange(100) % 2 == 0), ": a masked array cannot be held"),
        (np.ma.masked_array(np.zeros(4), mask=[True, False, False, False]), ": a masked array cannot be held"),
        (torch.ones(2, dtype=torch.bfloat16), ": a torch.bfloat16 tensor"),
    ):
        with pytest.raises(TypeError, match=match):
            identity_json(value)


def test_merge_identities_are_exact_canonical_and_checked_early(tmp_path):
    grid = lam_grid()
    forecast = forecast_of(truth(grid))

    def evaluation(table, targets=None, **kwargs):
        operator = Chain([Nearest(max_distance_km=MAX_KM), Tabled(table)])
        return point_evaluation(forecast, targets or varying_targets(), operator, bins="season", **kwargs)

    def shards(first, second):
        return [first.run(first.shard(0, 2)), second.run(second.shard(1, 2))]

    for table in (np.array([0.1, 0.2, 0.3]), np.linspace(0.0, 1.0, 5000)):  # values, then a sha256
        other = table.copy()
        other[len(other) // 2] += 1e-10
        merged = merge(shards(evaluation(table), evaluation(table.copy())))
        recorded = json.loads(merged.attrs["observation_operator"])["steps"][1]["parameters"]["table"]
        assert ("values" in recorded) == (table.size <= IDENTITY_ARRAY_VALUES) and recorded["shape"] == [table.size]
        with pytest.raises(ValueError, match="observation_operator"):
            merge(shards(evaluation(table), evaluation(other)))
    tensor = evaluation(torch.tensor([1.0, 2.0]))
    merge(shards(tensor, evaluation(np.array([1.0, 2.0], dtype=np.float32))))  # the same values
    with pytest.raises(ValueError, match="observation_operator"):
        merge(shards(tensor, evaluation(torch.tensor([1.0, 2.5]))))
    ordered = evaluation({"b": 1.0, "a": [1, 2]}), evaluation({"a": [1, 2], "b": 1.0})
    states = shards(*ordered)
    assert states[0].attrs["observation_operator"] == states[1].attrs["observation_operator"]
    merge(states)

    def calibrated(offsets):
        targets = varying_targets(cls=Calibrated)
        targets.offsets = offsets
        return evaluation(1.0, targets)

    merged = merge(shards(calibrated(np.array([0.0, 0.5])), calibrated(np.array([0.0, 0.5]))))
    assert json.loads(merged.attrs["point_targets"])["offsets"]["values"] == [0.0, 0.5]
    with pytest.raises(ValueError, match="point_targets"):
        merge(shards(calibrated(np.array([0.0, 0.5])), calibrated(np.array([0.0, 0.5 + 1e-10]))))
    # a source opened with a path and a date, recorded as a dataset source records its open arguments, binds
    opened = {"args": [Path("/data/obs.zarr")], "kwargs": {"start": T0}}
    merged = merge(shards(calibrated(opened), calibrated(dict(opened))))
    assert json.loads(merged.attrs["point_targets"])["offsets"] == {
        "args": [{"__path__": "/data/obs.zarr"}],
        "kwargs": {"start": {"__datetime__": "2024-01-01T00:00:00"}},
    }
    with pytest.raises(ValueError, match="point_targets"):
        merge(shards(calibrated(opened), calibrated({**opened, "args": ["/data/obs.zarr"]})))

    # shards whose JSON differs in key order and whitespace only merge: merge compares the canonical form
    def shuffled(value):
        if isinstance(value, dict):
            return {key: shuffled(value[key]) for key in reversed(list(value))}
        return [shuffled(item) for item in value] if isinstance(value, list) else value

    run = point_evaluation(
        forecast,
        varying_targets(),
        Chain([Nearest(max_distance_km=MAX_KM), LapseRate(), Tabled({"b": 1.0, "a": np.ones(3)})]),
        bins="season",
        constants=ArrayConstants({"orography": (np.zeros(grid.n), "m")}),
    )
    paths = []
    for i, state in enumerate(shards(run, run)):
        state.attrs["shard"] = f"{i}/2"
        if i:
            for key in ("observation_operator", "point_targets"):
                written = json.dumps(shuffled(json.loads(state.attrs[key])), indent=2)
                assert written != state.attrs[key] and json.loads(written) == json.loads(state.attrs[key])
                state.attrs[key] = written
        paths.append(tmp_path / f"shard{i}.nc")
        write(state.to_xarray(run.metrics), paths[-1])
    merged = merge(paths)
    assert merged.attrs["observation_operator"] == load_state(paths[0]).attrs["observation_operator"]

    # the identities written are those encoded at bind: a source changed during the run neither fails late nor
    # changes what is recorded
    targets = varying_targets(cls=Calibrated)
    targets.offsets = np.array([0.0, 0.5])
    changing = evaluation(1.0, targets)
    before = changing.attrs()
    targets.offsets = object()
    assert changing.run().attrs["point_targets"] == before["point_targets"]

    # a value the identity cannot hold fails at bind and in the dry run, before any frame or observation is read,
    # and leaves the operator unbound, so a second run fails at the same point
    silent = forecast_of(truth(grid), NoModel)
    orography = {"orography": (np.zeros(grid.n), "m")}
    for operator, constants, offsets, match in (
        (Tabled({850: 1.0}), None, None, r"merge_identity\(\)\.steps\[1\]\.parameters\.table: the key 850"),
        (Tabled(object()), None, None, r"merge_identity\(\)\.steps\[1\]\.parameters\.table: object cannot"),
        (LapseRate(), Located(orography), None, r"merge_identity\(\)\.constants\.files: set cannot"),
        (Tabled(1.0), None, 1j, r"Calibrated\.merge_identity\(\)\.offsets: complex cannot"),
    ):
        targets = varying_targets(cls=Calibrated)
        targets.offsets = offsets
        chain = Chain([Nearest(max_distance_km=MAX_KM), operator])
        bad = point_evaluation(silent, targets, chain, constants=constants)
        for call in (bad.plan, bad.run, bad.run, bad.attrs):
            with pytest.raises(TypeError, match=match):
                call()
            assert not bad.observation_operator.bound
        assert targets.reads == 0

    # a multi-dataset run without a shared rollout checks every dataset before any of them runs
    first, second = varying_targets(), varying_targets(cls=Calibrated)
    second.offsets = 1j
    multi = MultiEvaluation(
        {
            "a": point_evaluation(silent, first),
            "b": point_evaluation(silent, second, Chain([Nearest(max_distance_km=MAX_KM), Tabled(1.0)])),
        }
    )
    for call in (multi.run, lambda: next(multi.pairs())):
        with pytest.raises(TypeError, match=r"Calibrated\.merge_identity\(\)\.offsets: complex cannot"):
            call()
    assert first.reads == second.reads == 0


def test_points_at_grid_nodes_match_the_gridded_run():
    rng = np.random.default_rng(10)
    grid = Grid(rng.uniform(-90, 90, 300), rng.uniform(0, 360, 300))
    gridded = truth(grid, gaps=True)
    forecast = forecast_of(gridded)
    order = rng.permutation(grid.n)
    points = PointSet(grid.latitudes[order], grid.longitudes[order])
    targets = ArrayPointTargets.fixed(points, VARIABLES, {d: f[:, order] for d, f in gridded.fields.items()})
    regions = {"global": "all", "box": {"bbox": {"north": 60, "west": -30, "south": 0, "east": 40}}}
    common = dict(bins="season", regions=regions, weights={"uniform": {}})
    on_grid = Evaluation(forecast, gridded, INITS, LEAD, METRICS, **common).run()
    evaluation = Evaluation(
        forecast, targets, INITS, LEAD, METRICS, observation_operator=Nearest(max_distance_km=1.0), **common
    )
    on_points = evaluation.run()
    diagnostics = json.loads(on_points.attrs["observation_diagnostics"])
    assert diagnostics["nearest_node_max_km"] < 1e-6 and not on_points.out_of_domain.any()
    assert torch.equal(on_points.weights, on_grid.weights) and torch.equal(on_points.n_init, on_grid.n_init)
    for name, total in on_grid.sums.items():
        torch.testing.assert_close(on_points.sums[name], total, rtol=1e-13, atol=1e-13)
    shards = [evaluation.run(evaluation.shard(i, 2)) for i in range(2)]
    assert merge(shards).attrs["observation_diagnostics"] == on_points.attrs["observation_diagnostics"]
    shards[1].attrs["observation_diagnostics"] = json.dumps({"nearest_node_max_km": 1e-7})  # informational only
    assert merge(shards).attrs["observation_diagnostics"] == on_points.attrs["observation_diagnostics"]


def gridded_reference(forecast, targets, inits, lead):
    """Summed statistics and weights per (lead, variable) over every node with uniform weights, in numpy float64."""
    leads = forecast.lead_times(lead)
    shape = (len(leads), len(targets.variables))
    sums, total = {name: np.zeros(shape) for name in STATISTICS}, np.zeros(shape)
    for init in inits:
        for frame in forecast.frames(init, lead, list(targets.variables), "cpu"):
            data, truth_field = frame.data.numpy(), targets.fields[frame.valid_time]
            error = data.astype(np.float64).mean(0) - truth_field.astype(np.float64)
            valid = np.isfinite(data).all(0) & np.isfinite(truth_field)
            k = leads.index(frame.lead_time)
            total[k] += valid.sum(1)
            for name, function in STATISTICS.items():
                sums[name][k] += np.where(valid, function(error), 0.0).sum(1)
    return sums, total


@pytest.mark.parametrize("operator", [None, Identity(), Chain([Identity()])], ids=["none", "identity", "chain"])
def test_identity_operator_is_the_gridded_path(fake_sources, operator):
    forecast, targets = fake_sources()
    evaluation = Evaluation(forecast, targets, INITS, LEAD, METRICS, bins="none", observation_operator=operator)
    state = evaluation.run()
    sums, total = gridded_reference(forecast, targets, INITS, LEAD)
    assert torch.equal(state.weights[:, 0, :, 0], torch.from_numpy(total))
    for name in STATISTICS:
        np.testing.assert_allclose(state.sums[name][:, 0, :, 0].numpy(), sums[name], rtol=1e-12, atol=1e-12)
    assert state.out_of_domain is None and evaluation.observation_operator is None
    dataset = state.to_xarray(evaluation.metrics)
    assert not {"out_of_domain", "state_out_of_domain"} & set(dataset.data_vars)
    assert not {"observation_operator", "point_targets", "observation_diagnostics"} & set(dataset.attrs)


def test_dry_run_on_points():
    grid = lam_grid()
    forecast = forecast_of(truth(grid), NoModel)
    points = random_points(np.random.default_rng(11), 10)
    operator = Chain([Nearest(max_distance_km=MAX_KM), LapseRate()])
    constants = ArrayConstants({"orography": (np.zeros(grid.n), "m")})
    evaluation = point_evaluation(
        forecast, fixed_targets(points), operator, regions={"a": "all", "b": "all"}, constants=constants
    )
    plan = evaluation.plan()
    assert plan["points"]["fixed"] and plan["points"]["max_points"] == len(points)
    _, distance = brute_nearest(points, grid)
    inside = distance <= MAX_KM
    assert plan["points"]["out_of_domain"] == np.count_nonzero(~inside) == 2
    assert plan["points"]["diagnostics"]["nearest_node_max_km"] == pytest.approx(distance[inside].max(), rel=1e-9)
    assert plan["bytes"]["aggregator_weights"] == len(points) * 2 * 8
    assert plan["bytes"]["frame"] == 3 * 2 * grid.n * 4
    assert plan["observation_operator"]["steps"][0]["operator"] == "Nearest"
    assert plan["observation_operator"]["constants"]["sha256"]
    plan = point_evaluation(forecast, varying_targets()).plan()
    assert not plan["points"]["fixed"] and plan["points"]["max_points"] is None and "note" in plan["points"]
    assert "aggregator_weights" not in plan["bytes"] and plan["frames"]["with_targets"] == 6
    assert "diagnostics" not in plan["points"]
    with pytest.raises(ValueError, match="cannot be written to a config"):
        point_evaluation(forecast, varying_targets()).to_config()
    with pytest.raises(ValueError, match="ArrayPointTargets cannot be written to a config"):
        varying_targets().to_config()


def test_multi_dataset_rollout_with_a_point_dataset():
    grid = lam_grid()
    variables = ["2t", "10u", "10v"]
    gridded = truth(grid, variables)
    sources = {"grid": forecast_of(gridded), "obs": forecast_of(gridded)}
    targets = varying_targets(["ws", "2t"])
    operator = Chain([Nearest(max_distance_km=MAX_KM), WindSpeed()])

    def evaluations():
        return {
            "grid": Evaluation(sources["grid"], gridded, INITS, LEAD, METRICS, include_lead_zero=True),
            "obs": point_evaluation(sources["obs"], targets, operator, include_lead_zero=True, bins="season"),
        }

    single = {name: evaluation.run() for name, evaluation in evaluations().items()}
    multi = MultiEvaluation(evaluations(), MultiFake(sources))
    assert multi.evaluations["obs"].forecast_variables == variables != multi.variables["obs"]
    states = multi.run()
    for name, state in states.items():
        assert torch.equal(state.weights, single[name].weights)
        assert all(torch.equal(state.sums[key], single[name].sums[key]) for key in state.sums)
    assert (
        torch.equal(states["obs"].out_of_domain, single["obs"].out_of_domain) and states["grid"].out_of_domain is None
    )
    assert states["obs"].n_init[0].sum() == 3  # the lead-0 frames of the three init times, mapped too
    triples = list(multi.pairs([INITS[0]]))
    shapes = {name: tuple(target.shape) for name, frame, target in triples}
    assert shapes["grid"] == (1, 3, grid.n) and shapes["obs"][:2] == (1, 2)
    assert [frame.lead_time for name, frame, _ in triples if name == "obs"][0] == datetime.timedelta(0)


def test_explicit_bind_and_a_dry_run_on_a_throwaway_copy():
    grid = lam_grid()
    points = random_points(np.random.default_rng(11), 10)
    operator = Chain([Nearest(max_distance_km=MAX_KM), LapseRate()])
    constants = ArrayConstants({"orography": (np.zeros(grid.n), "m")})
    # a dry run for a CUDA run on a node without a GPU: the copy it describes is bound on the CPU and thrown away
    forecast = forecast_of(truth(grid), NoModel)
    dry = point_evaluation(forecast, fixed_targets(points), operator, constants=constants, device="cuda")
    plan = dry.plan()
    assert plan["device"] == "cuda" and plan["points"]["out_of_domain"] == 2
    assert plan["observation_operator"]["constants"]["sha256"]
    assert not dry.observation_operator.bound and not dry._points._cache and dry._points.diagnostics is None

    evaluation = point_evaluation(forecast_of(truth(grid)), fixed_targets(points), operator, constants=constants)
    assert not evaluation.observation_operator.bound
    before = evaluation.attrs()  # binds, so the operator's attr names its constants already
    assert evaluation.observation_operator.bound and json.loads(before["observation_operator"])["constants"]
    state = evaluation.run()
    assert state.attrs["observation_operator"] == before["observation_operator"]
    assert "observation_diagnostics" not in before and "observation_diagnostics" in state.attrs
    assert evaluation.plan()["points"]["out_of_domain"] == 2  # after the run too: a fresh copy, not the bound one
    fresh = point_evaluation(forecast_of(truth(grid)), varying_targets(), operator, constants=constants)
    next(fresh.pairs([INITS[0]]))
    assert fresh.observation_operator.bound


def test_a_reused_key_must_come_with_the_same_points():
    grid = lam_grid()
    forecast = forecast_of(truth(grid))
    points = random_points(np.random.default_rng(15), 12, key="window")
    values = np.random.default_rng(16).normal(280, 5, (2, len(points))).astype(np.float32)
    elevation = points.attribute("elevation")
    for change in (
        dict(attributes={"elevation": elevation + 100.0}),
        dict(attributes={"elevation": elevation, "kind": np.zeros(len(points))}),
    ):
        changed = PointSet(points.latitudes, points.longitudes, **{"attributes": {"elevation": elevation}, **change})
        changed = PointSet(changed.latitudes, changed.longitudes, changed.timedeltas, changed.attributes, key="window")
        targets = ArrayPointTargets(VARIABLES, {DATES[1]: (points, values), DATES[2]: (changed, values)}, ())
        with pytest.raises(ValueError, match="differ in their points"):
            point_evaluation(forecast, targets).run()
    same = PointSet(points.latitudes, points.longitudes, attributes={"elevation": elevation.copy()}, key="window")
    targets = ArrayPointTargets(VARIABLES, {DATES[1]: (points, values), DATES[2]: (same, values)}, ())
    point_evaluation(forecast, targets).run()
    zero = PointSet(points.latitudes, points.longitudes, np.zeros(len(points)), {"elevation": elevation}, key="window")
    targets = ArrayPointTargets(VARIABLES, {DATES[1]: (points, values), DATES[2]: (zero, values)}, ())
    point_evaluation(forecast, targets).run()  # zero timedeltas are the absent ones
    # the driver refuses non-zero timedeltas before comparing, so the timedeltas comparison is checked directly
    moved = PointSet(points.latitudes, points.longitudes, np.full(len(points), 600.0), {"elevation": elevation})
    assert same_points(points, zero) and not same_points(points, moved)


def test_equal_point_sets_with_nan_and_nat_attributes_are_the_same_points():
    grid = lam_grid()
    forecast = forecast_of(truth(grid))
    points = random_points(np.random.default_rng(17), 6)
    elevation = points.attribute("elevation").copy()
    elevation[0] = np.nan
    seen = np.array(["2020-01-01T00:00"] * len(points), dtype="datetime64[s]")
    seen[1] = np.datetime64("NaT")
    values = np.random.default_rng(18).normal(280, 5, (2, len(points))).astype(np.float32)

    def rebuilt(key):  # a new object per read, as a tabular reader builds it
        return PointSet(
            points.latitudes,
            points.longitudes,
            attributes={"elevation": elevation.copy(), "seen": seen.copy()},
            key=key,
        )

    for key in (None, "window"):
        targets = ArrayPointTargets(VARIABLES, {DATES[1]: (rebuilt(key), values), DATES[2]: (rebuilt(key), values)}, ())
        point_evaluation(forecast, targets).run()


def test_a_skipped_step_needs_no_constants_or_attributes():
    grid = lam_grid()
    variables = ["2t", "10u"]
    forecast = forecast_of(truth(grid, variables))  # serves no constants: a needed orography would be refused
    targets = varying_targets(variables)
    bare = ArrayPointTargets(
        variables, {d: (PointSet(p.latitudes, p.longitudes), v) for d, (p, v) in targets.data.items()}
    )
    lapse = Chain([Nearest(max_distance_km=MAX_KM), LapseRate("2t")])
    evaluation = point_evaluation(forecast, bare, lapse, variables=["10u"])
    assert evaluation.constants is None and evaluation.forecast_variables == ["10u"]
    plan = evaluation.plan()
    assert "constants" not in plan["observation_operator"]
    state = evaluation.run()
    assert "constants" not in json.loads(state.attrs["observation_operator"])
    nearest = point_evaluation(forecast, bare, variables=["10u"]).run()
    assert torch.equal(state.sums["error"], nearest.sums["error"]) and torch.equal(state.weights, nearest.weights)
    with pytest.raises(ValueError, match="elevation"):
        point_evaluation(forecast, bare, lapse, variables=["2t"])


def test_point_time_offsets_are_refused():
    grid = lam_grid()
    forecast = forecast_of(truth(grid))
    targets = varying_targets()
    shifted = ArrayPointTargets(
        VARIABLES,
        {
            date: (PointSet(p.latitudes, p.longitudes, np.full(len(p), -3 * 3600.0), p.attributes), values)
            for date, (p, values) in targets.data.items()
        },
    )
    with pytest.raises(ValueError, match="time offsets .* not supported yet"):
        point_evaluation(forecast, shifted).run()
    points = random_points(np.random.default_rng(19), 6)
    for offset in (600.0, 1e-3):  # any non-zero offset, float noise included
        moved = PointSet(points.latitudes, points.longitudes, np.full(len(points), offset), points.attributes)
        with pytest.raises(ValueError, match="fixed point set .* time offsets .* leave timedeltas out, or zero"):
            point_evaluation(forecast, fixed_targets(moved))


def test_aggregator_for_points():
    aggregator = Aggregator.for_points(2, {}, None)
    with pytest.raises(ValueError, match="needs the W"):
        aggregator.add(None, None, None)
    gridded = Aggregator(np.ones(3), np.ones((3, 1), bool), {}, None)
    with pytest.raises(ValueError, match="has its own W"):
        gridded.add(None, None, None, W=torch.ones(3, 1))
    with pytest.raises(ValueError, match="at least one region"):
        Aggregator.for_points(0, {}, None)
    assert aggregator.W is None and aggregator.n_regions == 2 and aggregator.statistics == {}
    for args, kwargs in (((np.ones(3), None), {}), ((np.ones(3), np.ones((3, 1), bool)), {"n_regions": 1})):
        with pytest.raises(ValueError, match="weights and masks, or n_regions"):
            Aggregator(*args, {}, None, **kwargs)


def test_constants_digest_identifies_the_values():
    a = ArrayConstants({"orography": (np.zeros(4), "m")}).describe()
    b = ArrayConstants({"orography": (np.ones(4), "m")}).describe()
    c = ArrayConstants({"orography": (np.zeros(4), "km")}).describe()
    assert len({a["sha256"], b["sha256"], c["sha256"]}) == 3
    assert ArrayConstants({"orography": (np.zeros(4), "m")}).describe() == a


def test_bad_observation_values_are_refused():
    grid = lam_grid()
    forecast = forecast_of(truth(grid))
    targets = varying_targets()
    targets.observations = lambda valid_time, variables, device: Observations(
        targets.data[DATES[1]][0],
        torch.zeros(1, 2, 3, dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="expected float32"):
        point_evaluation(forecast, targets).run()
    targets.observations = lambda valid_time, variables, device: targets.data[DATES[1]]
    with pytest.raises(TypeError, match="expected Observations"):
        point_evaluation(forecast, targets).run()
