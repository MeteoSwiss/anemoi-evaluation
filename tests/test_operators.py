"""The observation operator contract, proven with throwaway operators that live only here."""

import copy
import datetime
import math

import numpy as np
import pytest
import torch
from anemoi.graphs import EARTH_RADIUS

from anemoi.evaluation import Frame
from anemoi.evaluation import Grid
from anemoi.evaluation import PointSet
from anemoi.evaluation.operators import ArrayConstants
from anemoi.evaluation.operators import Chain
from anemoi.evaluation.operators import ConstantNeed
from anemoi.evaluation.operators import Identity
from anemoi.evaluation.operators import LapseRate
from anemoi.evaluation.operators import ModelSide
from anemoi.evaluation.operators import Nearest
from anemoi.evaluation.operators import NearestNodes
from anemoi.evaluation.operators import ObservationOperatorBase

HOUR = datetime.timedelta(hours=1)
T0 = datetime.datetime(2024, 1, 1)
VARIABLES = ["2t", "10u", "10v"]


class WindSpeed(ObservationOperatorBase):
    """Toy derived variable: two reads, one product; acts on point values."""

    reads = ("10u", "10v")
    produces = ("ws",)

    def apply(self, frames, plan, valid_time):
        frame = frames[0]
        u, v = (frame.data[:, frame.variables.index(name)].double() for name in self.reads)
        data = torch.cat([frame.data, torch.sqrt(u * u + v * v).float()[:, None]], dim=1)
        return self.output_frame(frames, valid_time, self.output_variables(frame.variables), data)


class GridWindSpeed(WindSpeed):
    """The same derivation on the model grid, before the spatial step."""

    input_space = output_space = "grid"


class TimeInterpolation(ObservationOperatorBase):
    """Toy per-point linear interpolation in time between the frames bracketing `valid_time + timedelta`."""

    stencil = "bracketing"

    def prepare(self, points, constants):
        return torch.from_numpy(points.timedeltas.astype(np.float64))

    def apply(self, frames, plan, valid_time):
        offsets = torch.tensor([(f.valid_time - valid_time).total_seconds() for f in frames], dtype=torch.float64)
        if not (offsets[0] <= plan.min() and plan.max() <= offsets[-1]):
            raise ValueError("the frames do not cover the points")
        stacked = torch.stack([f.data.double() for f in frames])  # (T, M, V, P)
        hi = torch.searchsorted(offsets, plan).clamp(1, len(frames) - 1)
        lo = hi - 1
        w = (plan - offsets[lo]) / (offsets[hi] - offsets[lo])
        points = torch.arange(plan.shape[0])
        data = (1 - w) * stacked[lo, :, :, points].permute(1, 2, 0) + w * stacked[hi, :, :, points].permute(1, 2, 0)
        return self.output_frame(frames, valid_time, list(frames[0].variables), data.float())


class PointNativeForecast:
    """Fake model decoding onto the points it is handed: an analytic field of coordinates and time."""

    variables = VARIABLES
    members = 2

    def frame(self, points: PointSet, valid_time: datetime.datetime) -> Frame:
        lat, lon = torch.tensor(points.latitudes), torch.tensor(points.longitudes)
        fields = torch.stack([280 + 0.1 * lat, torch.cos(torch.deg2rad(lon)), torch.sin(torch.deg2rad(lat))])
        data = torch.stack([fields + m for m in range(self.members)]).float()
        return Frame(T0, valid_time - T0, valid_time, list(self.variables), data)


def grid_frames(grid: Grid, valid_times, members=3, seed=0, variables=VARIABLES):
    """Random frames (M, V, N) with NaN gaps at nodes 0-2, one per valid time."""
    rng = np.random.default_rng(seed)
    frames = []
    for valid_time in valid_times:
        data = rng.normal(280, 5, (members, len(variables), grid.n)).astype(np.float32)
        data[:, :, :3] = np.nan
        frames.append(Frame(T0, valid_time - T0, valid_time, list(variables), torch.from_numpy(data)))
    return frames


def lam_grid(n=400, seed=0):
    """A limited-area grid over the Alps."""
    rng = np.random.default_rng(seed)
    return Grid(rng.uniform(44, 49, n), rng.uniform(5, 12, n))


def model_side(grid, orography=None, units="m", variables=VARIABLES):
    constants = None if orography is None else ArrayConstants({"orography": (orography, units)})
    return ModelSide("grid", variables, grid=grid, constants=constants)


def bound(steps, side):
    chain = Chain(steps)
    chain.bind(side)
    return chain


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS * np.arcsin(np.sqrt(a))


def test_point_set_validation_and_identity():
    points = PointSet([46.0, 47.5], [7.0, 8.5], attributes={"elevation": [500.0, 1200.0]})
    assert points.latitudes.dtype == np.float64 and points.timedeltas.dtype == np.float32 and len(points) == 2
    assert not points.timedeltas.any() and not points.latitudes.flags.writeable
    same = PointSet(np.array([46.0, 47.5], np.float32), [7.0, 8.5], attributes={"elevation": [500.0, 1200.0]})
    assert same.identity == points.identity
    moved = PointSet([46.0, 47.5], [7.0, 8.5], timedeltas=[0, -600], attributes={"elevation": [500.0, 1200.0]})
    higher = PointSet([46.0, 47.5], [7.0, 8.5], attributes={"elevation": [500.0, 1300.0]})
    assert len({points.identity, moved.identity, higher.identity}) == 3
    assert PointSet([0.0], [0.0], key="swissmetnet").identity == "swissmetnet"
    assert len(PointSet([], [])) == 0
    for kwargs, error, match in (
        ({"latitudes": [0.0, 1.0], "longitudes": [0.0]}, ValueError, "same length"),
        ({"latitudes": [91.0], "longitudes": [0.0]}, ValueError, r"\[-90, 90\]"),
        ({"latitudes": [np.nan], "longitudes": [0.0]}, ValueError, "finite"),
        ({"latitudes": ["a"], "longitudes": [0.0]}, ValueError, "real numbers"),
        ({"latitudes": [0.0], "longitudes": [0.0], "timedeltas": [0.0, 1.0]}, ValueError, "timedeltas"),
        ({"latitudes": [0.0], "longitudes": [0.0], "attributes": {"elevation": [1.0, 2.0]}}, ValueError, "elevation"),
        ({"latitudes": [0.0], "longitudes": [0.0], "key": ["unhashable"]}, TypeError, "unhashable"),
    ):
        with pytest.raises(error, match=match):
            PointSet(**kwargs)
    with pytest.raises(KeyError, match="elevation"):
        PointSet([0.0], [0.0]).attribute("elevation")


def test_identity_is_a_no_op():
    grid = lam_grid()
    chain = bound([Identity()], model_side(grid))
    (frame,) = grid_frames(grid, [T0 + 6 * HOUR])
    plan = chain.prepare(PointSet(grid.latitudes, grid.longitudes))
    assert chain.apply([frame], plan, frame.valid_time) is frame
    values = np.arange(grid.n, dtype=np.float64)
    assert chain.map_constant(values, plan) is values
    assert chain.to_config() == [{"identity": {}}] and chain.output_space == "grid"


def test_nearest_is_a_great_circle_argmin_and_a_pure_gather():
    rng = np.random.default_rng(5)
    n = 3000
    grid = Grid(np.degrees(np.arcsin(rng.uniform(-1, 1, n))), rng.uniform(-180, 180, n))
    lats = np.concatenate([[90.0, -90.0, 0.5, -0.5, 60.0], rng.uniform(-90, 90, 500)])
    lons = np.concatenate([[0.0, 123.0, 179.99, -179.99, 359.9], rng.uniform(0, 360, 500)])  # [0, 360) on [-180, 180)
    points = PointSet(lats, lons)
    expected = np.array(
        [np.argmin(haversine_km(la, lo, grid.latitudes, grid.longitudes)) for la, lo in zip(lats, lons)]
    )
    nodes, distance_km, in_domain = NearestNodes(grid).query(points, None)
    np.testing.assert_array_equal(nodes, expected)
    reference = haversine_km(lats, lons, grid.latitudes[expected], grid.longitudes[expected])
    np.testing.assert_allclose(distance_km, reference, rtol=0, atol=1e-8)
    assert in_domain.all()

    grid = lam_grid()
    nodes = np.array([5, 0, 399, 5, 123])  # duplicates, NaN gap nodes, any order
    chain = bound([Nearest(max_distance_km=50)], model_side(grid))
    (frame,) = grid_frames(grid, [T0 + 6 * HOUR])
    plan = chain.prepare(PointSet(grid.latitudes[nodes], grid.longitudes[nodes]))
    out = chain.apply([frame], plan, frame.valid_time)
    assert torch.equal(out.data.isnan(), frame.data[:, :, nodes].isnan())
    assert torch.equal(torch.nan_to_num(out.data), torch.nan_to_num(frame.data[:, :, nodes]))
    assert plan.plans[0].distance_km.max() < 1e-6 and not chain.out_of_domain(plan).any()
    assert out.variables == VARIABLES and out.data.isnan().any()


def test_nearest_out_of_domain_and_max_distance():
    grid = lam_grid()
    chain = bound([Nearest(max_distance_km=50)], model_side(grid))
    (frame,) = grid_frames(grid, [T0 + 6 * HOUR])
    far = PointSet([46.5, 60.0, -10.0], [8.0, 8.0, 100.0])  # one inside, two far outside the domain
    plan = chain.prepare(far)
    out = chain.apply([frame], plan, frame.valid_time)
    assert chain.out_of_domain(plan).tolist() == [False, True, True]
    assert out.data[:, :, 1:].isnan().all() and not out.data[:, :, 0].isnan().any() and out.data.shape == (3, 3, 3)
    mapped = chain.map_constant(np.ones(grid.n), plan)
    assert mapped[0] == 1.0 and np.isnan(mapped[1:]).all()

    # a great-circle distance: a node at (0, 0), a point 30 degrees away, where the chord is 1.1 % shorter
    sphere = Grid(np.array([0.0, 0.0]), np.array([0.0, 180.0]))
    arc = EARTH_RADIUS * math.radians(30.0)
    point = PointSet([30.0], [0.0])
    for distance, outside in ((arc * 1.001, False), (arc * 0.999, True)):
        nearest = bound([Nearest(max_distance_km=distance)], model_side(sphere, variables=["2t"]))
        assert nearest.out_of_domain(nearest.prepare(point)).tolist() == [outside]
    assert Nearest(max_distance_km=None).to_config() == {"nearest": {"max_distance_km": None}}
    for value, error in (("5", ValueError), (-1.0, ValueError), (math.inf, ValueError), (True, ValueError)):
        with pytest.raises(error, match="max_distance_km"):
            Nearest(max_distance_km=value)
    with pytest.raises(TypeError):
        Nearest()  # required: None has to be written
    with pytest.raises(RuntimeError, match="bind"):
        Chain([Nearest(max_distance_km=None)]).prepare(point)


def test_out_of_domain_counts_the_spatial_step_only():
    class Above(ObservationOperatorBase):
        """A points step that cannot compute north of 47 N: NaN there, reported, and not counted by the chain."""

        touches = ("2t",)

        def prepare(self, points, constants):
            return points.latitudes > 47.0

        def apply(self, frames, plan, valid_time):
            frame = frames[0]
            data = frame.data.clone()
            data[:, frame.variables.index("2t"), torch.from_numpy(plan)] = float("nan")
            return self.output_frame(frames, valid_time, list(frame.variables), data)

        def out_of_domain(self, plan):
            return plan

    class Masked(Nearest):
        """A spatial step reporting a given mask."""

        def __init__(self, mask):
            super().__init__(max_distance_km=50)
            self._mask = mask

        def out_of_domain(self, plan):
            return self._mask

    grid = lam_grid()
    points = PointSet([46.5, 47.5, 60.0, 48.0], [8.0, 8.0, 8.0, 8.0])  # the third is far outside the domain
    chain = bound([Nearest(max_distance_km=50), Above()], model_side(grid))
    plan = chain.prepare(points)
    assert chain.out_of_domain(plan).tolist() == [False, False, True, False]
    (frame,) = grid_frames(grid, [T0 + 6 * HOUR])
    out = chain.apply([frame], plan, frame.valid_time)
    assert out.data[:, 0, 1:].isnan().all() and not out.data[:, 0, 0].isnan().any()
    assert out.data[:, 1, 2].isnan().all() and not out.data[:, 1, [0, 1, 3]].isnan().any()
    assert bound([Above()], ModelSide("points", VARIABLES)).prepare(points).out_of_domain is None  # no spatial step
    accepted = bound([Masked(torch.tensor([False, True, False, False]))], model_side(grid))
    assert accepted.out_of_domain(accepted.prepare(points)).tolist() == [False, True, False, False]
    for mask in (np.array([True]), np.zeros(4, dtype=np.int64), np.zeros((4, 1), dtype=bool)):
        chain = bound([Masked(mask)], model_side(grid))
        with pytest.raises(ValueError, match=r"Masked\.out_of_domain returned \w+ \(.*\), expected bool \(4,\)"):
            chain.prepare(points)


def test_nearest_and_lapse_rate_against_a_float64_reference():
    grid = lam_grid()
    rng = np.random.default_rng(1)
    orography = rng.uniform(200, 3000, grid.n)
    p = 40
    lats, lons = rng.uniform(44, 49, p), rng.uniform(5, 12, p)
    lats[-1], lons[-1] = 30.0, 8.0  # out of the domain
    elevation = rng.uniform(200, 3500, p)
    elevation[0] = np.nan  # unknown: a NaN forecast there, not an error
    points = PointSet(lats, lons, attributes={"elevation": elevation, "station_type": np.array(["a"] * p)})
    chain = Chain([Nearest(max_distance_km=100), LapseRate()])
    assert chain.touches == ("2t",) and chain.point_attributes == ("elevation",)
    assert [need.name for need in chain.constants] == ["orography"]
    chain.bind(model_side(grid, orography))
    (frame,) = grid_frames(grid, [T0 + 6 * HOUR])
    plan = chain.prepare(points)
    out = chain.apply([frame], plan, frame.valid_time).data.numpy()

    nodes = plan.plans[0].nodes
    data = frame.data.numpy()
    expected = np.empty((3, 3, p), np.float32)
    for m in range(3):
        for i in range(3):
            for k in range(p):
                value = np.float64(data[m, i, nodes[k]])
                if i == 0:
                    value = value - 0.0065 * (elevation[k] - orography[nodes[k]])
                expected[m, i, k] = np.float32(value)
    expected[:, :, -1] = np.nan
    np.testing.assert_array_equal(out, expected)  # bit for bit, NaN gaps and out-of-domain points included
    assert np.isnan(out[:, 0, 0]).all() and not np.isnan(out[:, 1:, 0]).any()
    assert chain.out_of_domain(plan).sum() == 1
    assert chain.to_config() == [
        {"nearest": {"max_distance_km": 100}},
        {"lapse_rate": {"variable": "2t", "rate": 0.0065, "elevation": "elevation", "orography": "orography"}},
    ]
    assert chain.merge_identity()["constants"]["constants"] == {"orography": "m"}
    # the parameters are the public scalar attributes, never the tree or the device built at bind
    parameters = {"variable": "2t", "rate": 0.0065, "elevation": "elevation", "orography": "orography"}
    assert chain.steps[0].parameters() == {"max_distance_km": 100.0}
    assert chain.merge_identity()["steps"][1]["parameters"] == chain.describe()["steps"][1]["parameters"] == parameters
    with pytest.raises(TypeError):
        LapseRate("2t", 0.0065)  # keyword-only after the variable

    # the elevation enters the point set's identity: a new elevation is a new plan, not the old height differences
    moved = PointSet(lats, lons, attributes={"elevation": elevation + 100, "station_type": np.array(["a"] * p)})
    corrected = chain.apply([frame], chain.prepare(moved), frame.valid_time).data.numpy()
    np.testing.assert_allclose(corrected[:, 0, 1:-1], out[:, 0, 1:-1] - np.float32(0.65), atol=1e-4)

    # scoring only another variable does not need the touched one from the model: the step is passed through
    assert chain.input_variables(["2d"]) == ["2d"] and chain.input_variables(["2t", "2d"]) == ["2t", "2d"]
    only = bound([Nearest(max_distance_km=100), LapseRate()], model_side(grid, orography, variables=["2d"]))
    (frame,) = grid_frames(grid, [T0 + 6 * HOUR], variables=["2d"])
    out = only.apply([frame], only.prepare(points), frame.valid_time)
    assert out.variables == ["2d"]
    np.testing.assert_array_equal(out.data[:, :, :-1].numpy(), frame.data[:, :, nodes[:-1]].numpy())
    # a step passed through imposes nothing: no orography, no elevation, no prepare
    assert chain.needs(["2d"]) == ((), ()) and chain.needs(["2t"])[0] == ("elevation",)
    bare = bound([Nearest(max_distance_km=100), LapseRate()], model_side(grid, variables=["2d"]))
    assert bare.running() == [True, False] and bare.describe().get("constants") is None
    plan = bare.prepare(PointSet(lats, lons))
    assert plan.plans[1] is None and plan.constants == {}
    np.testing.assert_array_equal(bare.apply([frame], plan, frame.valid_time).data.numpy(), out.data.numpy())
    (full,) = grid_frames(grid, [T0 + 6 * HOUR])
    with pytest.raises(ValueError, match="LapseRate runs on frames holding"):
        bare.apply([full], plan, full.valid_time)


def test_derived_variables_and_time_stencil():
    grid = lam_grid()
    valid_time = T0 + 12 * HOUR
    frames = grid_frames(grid, [T0, T0 + 6 * HOUR, valid_time])
    rng = np.random.default_rng(2)
    p = 30
    timedeltas = rng.uniform(-12, 0, p).astype(np.float32) * 3600  # a (-12h, 0] window over a 6-hourly model
    points = PointSet(rng.uniform(45, 48, p), rng.uniform(6, 11, p), timedeltas=timedeltas)
    chain = Chain([Nearest(max_distance_km=None), WindSpeed(), TimeInterpolation()])
    assert chain.reads == ("10u", "10v") and chain.produces == ("ws",) and chain.stencil == "bracketing"
    assert chain.input_variables(["ws", "2t"]) == ["2t", "10u", "10v"] and chain.input_variables(["2t"]) == ["2t"]
    assert chain.output_variables(["2t", "10u", "10v"]) == ["2t", "10u", "10v", "ws"]
    assert chain.output_variables(["2t", "10u"]) == ["2t", "10u"]  # nothing asks for ws, so 10v is not needed
    with pytest.raises(ValueError, match="already hold"):
        chain.output_variables(["ws", "10u", "10v"])
    assert chain.time_window(points) == (float(timedeltas.min()), float(timedeltas.max()))
    chain.bind(model_side(grid))
    plan = chain.prepare(points)
    out = chain.apply(frames, plan, valid_time)
    assert out.variables == ["2t", "10u", "10v", "ws"] and out.valid_time == valid_time

    nodes = plan.plans[0].nodes
    stack = np.stack([f.data.numpy().astype(np.float64) for f in frames])[:, :, :, nodes]  # (T, M, V, P)
    ws = np.sqrt(stack[:, :, 1] ** 2 + stack[:, :, 2] ** 2).astype(np.float32).astype(np.float64)
    stack = np.concatenate([stack, ws[:, :, None]], axis=2)
    times = np.array([-12.0, -6.0, 0.0]) * 3600
    expected = np.empty((3, 4, p))
    for k in range(p):
        for m in range(3):
            for v in range(4):
                expected[m, v, k] = np.interp(np.float64(timedeltas[k]), times, stack[:, m, v, k])
    np.testing.assert_allclose(out.data.numpy(), expected.astype(np.float32), rtol=1e-6)
    with pytest.raises(ValueError, match="do not cover"):
        chain.apply(frames[1:], plan, valid_time)
    single = bound([Nearest(max_distance_km=None)], model_side(grid))
    with pytest.raises(ValueError, match="exactly the frame"):
        single.apply(frames, single.prepare(points), valid_time)

    # a grid -> grid step before the spatial step
    gridded = bound([GridWindSpeed(), Nearest(max_distance_km=None)], model_side(grid))
    out = gridded.apply(frames[-1:], gridded.prepare(points), valid_time)
    u, v = frames[-1].data[:, 1].double(), frames[-1].data[:, 2].double()
    assert out.variables == ["2t", "10u", "10v", "ws"]
    np.testing.assert_array_equal(out.data[:, 3].numpy(), torch.sqrt(u * u + v * v).float()[:, nodes].numpy())
    values = np.arange(grid.n, dtype=np.float64)
    np.testing.assert_array_equal(gridded.map_constant(values, gridded.prepare(points)), values[nodes])


def test_points_to_points_on_a_point_native_forecast():
    forecast = PointNativeForecast()
    chain = bound([WindSpeed()], ModelSide("points", forecast.variables))
    points = PointSet([46.0, 47.0, 45.5], [7.0, 9.0, 6.0], timedeltas=[0, -3600, -1800])
    frame = forecast.frame(points, T0 + 6 * HOUR)
    out = chain.apply([frame], chain.prepare(points), frame.valid_time)
    u, v = frame.data[:, 1].double(), frame.data[:, 2].double()
    assert torch.equal(out.data[:, :3], frame.data)
    assert torch.equal(out.data[:, 3], torch.sqrt(u * u + v * v).float())
    # members are never mixed: permuting the members of the input permutes the output
    swapped = Frame(frame.init_time, frame.lead_time, frame.valid_time, frame.variables, frame.data.flip(0))
    assert torch.equal(chain.apply([swapped], chain.prepare(points), frame.valid_time).data, out.data.flip(0))

    for steps, match in (
        ([Nearest(max_distance_km=None)], "takes grid frames"),
        ([LapseRate()], "only available on points once a grid -> points step"),
    ):
        with pytest.raises(ValueError, match=match):
            Chain(steps).bind(ModelSide("points", forecast.variables))

    class WrongSize(ObservationOperatorBase):
        def apply(self, frames, plan, valid_time):
            return self.output_frame(frames, valid_time, frames[0].variables, frames[0].data[:, :, :2])

    wrong = bound([WrongSize()], ModelSide("points", forecast.variables))
    with pytest.raises(ValueError, match="on 2 locations"):
        wrong.apply([frame], wrong.prepare(points), frame.valid_time)


def test_varying_point_sets_through_one_bound_chain():
    grid = lam_grid()
    chain = bound([Nearest(max_distance_km=None)], model_side(grid))
    rng = np.random.default_rng(3)
    frames = grid_frames(grid, [T0 + k * HOUR for k in range(1, 6)])
    sizes = []
    for k, frame in enumerate(frames):
        points = PointSet(rng.uniform(45, 48, 5 + k), rng.uniform(6, 11, 5 + k))
        sizes.append(chain.apply([frame], chain.prepare(points), frame.valid_time).data.shape[-1])
    assert sizes == [5, 6, 7, 8, 9]


def test_bound_operators_are_not_shared():
    grid = lam_grid()
    points = PointSet([46.0, 47.0], [7.0, 8.0], attributes={"elevation": [1000.0, 1000.0]})
    (frame,) = grid_frames(grid, [T0 + 6 * HOUR])
    chain = Chain([Nearest(max_distance_km=100), LapseRate()])
    twin = copy.deepcopy(chain)
    chain.bind(model_side(grid, np.zeros(grid.n)))
    twin.bind(model_side(grid, np.full(grid.n, 1000.0)))
    low = chain.apply([frame], chain.prepare(points), frame.valid_time).data
    high = twin.apply([frame], twin.prepare(points), frame.valid_time).data
    np.testing.assert_allclose((high - low)[:, 0].numpy(), 6.5, atol=1e-4)
    assert torch.equal(high[:, 1:], low[:, 1:])
    for again in (chain, Chain([chain.steps[0]]), copy.deepcopy(chain)):
        with pytest.raises(RuntimeError, match="already bound"):
            again.bind(model_side(grid, np.zeros(grid.n)))


def test_default_parameters_take_numpy_scalars_and_refuse_other_values():
    class Scaled(ObservationOperatorBase):
        def __init__(self, factor):
            self.factor = factor

    class Tabled(Scaled):
        def parameters(self):
            return {"factor": self.factor.tolist()}

    grid = lam_grid()
    for factor, recorded in (
        (np.float32(2), 2.0),
        (np.int64(2), 2),
        (np.bool_(True), True),
        (torch.tensor(2.0), 2.0),
        ((np.float32(1), 2), [1.0, 2]),
    ):
        chain = bound([Nearest(max_distance_km=None), Scaled(factor)], model_side(grid))
        assert chain.merge_identity()["steps"][1]["parameters"] == {"factor": recorded}
        assert type(chain.steps[1].parameters()["factor"]) is type(recorded)
    for factor in (np.ones(3), torch.ones(3), [np.ones(3)], {"a": 1}):
        chain = Chain([Nearest(max_distance_km=None), Scaled(factor)])
        with pytest.raises(TypeError, match=r"Scaled\.factor \(\w+\) is not a parameter value.*override parameters"):
            chain.bind(model_side(grid))
        assert not chain.bound and not hasattr(chain.steps[0], "model")  # refused before any step is bound
    chain = bound([Nearest(max_distance_km=None), Tabled(np.ones(3))], model_side(grid))
    assert chain.merge_identity()["steps"][1]["parameters"] == {"factor": [1.0, 1.0, 1.0]}


def test_chain_refusals():
    class Moves(ObservationOperatorBase):
        input_space, output_space = "points", "grid"

    class TwoStencils(TimeInterpolation):
        pass

    for steps, match in (
        ([], "at least one step"),
        ([LapseRate(), Nearest(max_distance_km=None)], "takes grid"),
        ([Nearest(max_distance_km=None), Nearest(max_distance_km=None)], "takes grid"),
        ([Moves()], "not an operator space"),
        ([Nearest(max_distance_km=None), WindSpeed(), WindSpeed()], "already produced"),
        ([Nearest(max_distance_km=None), TimeInterpolation(), TwoStencils()], "at most one step"),
    ):
        with pytest.raises(ValueError, match=match):
            Chain(steps)
    grid = lam_grid()
    with pytest.raises(ValueError, match="takes points frames"):
        Chain([WindSpeed()]).bind(model_side(grid))
    with pytest.raises(ValueError, match="already hold"):
        Chain([Nearest(max_distance_km=None), WindSpeed()]).bind(model_side(grid, variables=[*VARIABLES, "ws"]))
    nested = Chain([Chain([Nearest(max_distance_km=None)]), Chain([WindSpeed()])])
    assert [type(step) for step in nested.steps] == [Nearest, WindSpeed]
    with pytest.raises(ValueError, match="cannot be written"):
        nested.to_config()


def test_declared_needs_are_checked():
    grid = lam_grid()
    orography = np.full(grid.n, 500.0)
    chain = Chain([Nearest(max_distance_km=None), LapseRate()])
    with pytest.raises(ValueError, match="elevation"):
        chain.check_attributes(["stalt"])
    chain.check_attributes(["elevation", "stalt"])
    for side, error, match in (
        (model_side(grid), ValueError, "no provider"),
        (model_side(grid, orography, units="m2 s-2"), ValueError, "in 'm2 s-2'"),
        (model_side(grid, orography[:-1]), ValueError, "shape"),
        (ModelSide("grid", VARIABLES, grid=grid, constants=ArrayConstants({})), KeyError, "no model constant"),
    ):
        with pytest.raises(error, match=match):
            chain.bind(side)
    chain.bind(model_side(grid, orography))
    with pytest.raises(ValueError, match="point attributes"):
        chain.prepare(PointSet([46.0], [7.0]))
    with pytest.raises(ValueError, match="numeric"):
        chain.prepare(PointSet([46.0], [7.0], attributes={"elevation": np.array(["high"])}))

    class Wrong(ObservationOperatorBase):
        constants = (ConstantNeed("orography", units="km"),)

    with pytest.raises(ValueError, match="in units 'm' and 'km'"):
        Chain([Nearest(max_distance_km=None), LapseRate(), Wrong()])
    with pytest.raises(ValueError, match="produced and read"):
        Chain([type("Both", (ObservationOperatorBase,), {"reads": ("a",), "produces": ("a",)})()])


def test_constants_grid_check_runs_at_bind_before_any_constant():
    class Checked(ArrayConstants):
        def __init__(self, constants, on_grid):
            super().__init__(constants)
            self.on_grid, self.calls = on_grid, []

        def check_grid(self, grid):
            self.calls.append(("check_grid", grid))
            if not self.on_grid:
                raise ValueError("constants on another grid")

        def constant(self, name):
            self.calls.append(("constant", name))
            return super().constant(name)

    grid = lam_grid()
    orography = {"orography": (np.zeros(grid.n), "m")}
    elsewhere = Checked(orography, on_grid=False)
    with pytest.raises(ValueError, match="another grid"):
        Chain([Nearest(max_distance_km=None), LapseRate()]).bind(
            ModelSide("grid", VARIABLES, grid=grid, constants=elsewhere)
        )
    assert elsewhere.calls == [("check_grid", grid)]
    Chain([Nearest(max_distance_km=None)]).bind(
        ModelSide("grid", VARIABLES, grid=grid, constants=elsewhere)
    )  # needs none
    assert elsewhere.calls == [("check_grid", grid)]
    here = Checked(orography, on_grid=True)
    Chain([Nearest(max_distance_km=None), LapseRate()]).bind(ModelSide("grid", VARIABLES, grid=grid, constants=here))
    assert here.calls == [("check_grid", grid), ("constant", "orography")]
