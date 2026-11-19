"""Gridded station datasets as point targets (`StationDatasetTargets`), the constants providers that read their fields
(`ForecastConstants`, `DatasetConstants`) and the inference source's constant forcings; on fake anemoi datasets read
through `DatasetTargets`, against numpy float64 references."""

import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_point_targets import DATES
from test_point_targets import INITS
from test_point_targets import MAX_KM
from test_point_targets import STATISTICS
from test_point_targets import VARIABLES
from test_point_targets import forecast_of
from test_point_targets import lam_grid
from test_point_targets import point_evaluation
from test_point_targets import reference
from test_point_targets import truth

from anemoi.evaluation import PointSet
from anemoi.evaluation import StationDatasetTargets
from anemoi.evaluation.frame import Grid
from anemoi.evaluation.operators import DEFAULT_CONSTANTS
from anemoi.evaluation.operators import ArrayConstants
from anemoi.evaluation.operators import Chain
from anemoi.evaluation.operators import DatasetConstants
from anemoi.evaluation.operators import ForecastConstants
from anemoi.evaluation.operators import LapseRate
from anemoi.evaluation.operators import ModelSide
from anemoi.evaluation.operators import Nearest
from anemoi.evaluation.output import merge
from anemoi.evaluation.sources.anemoi_dataset import DatasetTargets
from anemoi.evaluation.sources.anemoi_inference import DatasetForecast
from anemoi.evaluation.sources.anemoi_inference import InferenceForecastSource
from anemoi.evaluation.sources.base import MissingTargetError
from anemoi.evaluation.sources.fake import ArrayPointTargets
from anemoi.evaluation.sources.fake import FakeForecastSource
from anemoi.evaluation.sources.persistence import PersistenceForecastSource

HOUR = datetime.timedelta(hours=1)
T0 = DATES[0]
CPU = torch.device("cpu")
G = 9.80665
RAW = ["tre200s0", "tde200s0", "prestas0"]
MAPPING = {"tre200s0": {"name": "2t", "offset": 273.15}, "tde200s0": {"name": "2d", "offset": 273.15}}
ALL_NAN, MISSING = 2, 3  # dataset date indices: every station NaN at T0 + 12 h, T0 + 18 h a missing date
LAPSE = Chain([Nearest(max_distance_km=MAX_KM), LapseRate("2t", rate=0.0065)])


class FakeDataset:
    """What `DatasetTargets` reads from an anemoi dataset: `fields[i]` (V, N) at date `i`, counting reads."""

    def __init__(self, variables, latitudes, longitudes, fields, missing=()):
        self.variables = list(variables)
        self.latitudes, self.longitudes = np.asarray(latitudes), np.asarray(longitudes)
        self.dates = np.array(DATES, dtype="datetime64[s]")
        self.missing = set(missing)
        self.fields = fields
        self.shape = (len(DATES), len(self.variables), 1, len(self.latitudes))
        self.calls = 0

    def __getitem__(self, i):
        self.calls += 1
        return self.fields[i][:, None, :]


def stations(seed=20, p=12):
    """`p` station coordinates over the Alps with 6 decimals, one far outside the model domain, and elevations."""
    rng = np.random.default_rng(seed)
    lats = np.round(np.append(rng.uniform(45, 48, p - 1), 30.0), 6)
    lons = np.round(np.append(rng.uniform(6, 11, p - 1), 345.0), 6)
    return lats, lons, np.round(rng.uniform(200, 3500, p), 1)


def station_dataset(seed=21):
    """Raw station values in degC (and hPa) with NaN gaps, every value NaN at `ALL_NAN`, `MISSING` missing."""
    lats, lons, _ = stations()
    rng = np.random.default_rng(seed)
    fields = {}
    for i in range(len(DATES)):
        field = np.stack(
            [rng.normal(7, 5, len(lats)), rng.normal(2, 5, len(lats)), rng.normal(900, 50, len(lats))]
        ).astype(np.float32)
        field[0, rng.integers(0, len(lats), 2)] = np.nan
        field[1, 3] = np.nan
        if i == ALL_NAN:
            field[:] = np.nan
        fields[i] = field
    return FakeDataset(RAW, lats, lons, fields, missing=(MISSING,))


def station_file(path, lats=None, lons=None, elevation=None, rows=None):
    """A station attribute file as `build_station_files.py` writes them: one row per station in dataset order."""
    default = stations()
    lats, lons = default[0] if lats is None else lats, default[1] if lons is None else lons
    elevation = default[2] if elevation is None else elevation
    lines = ["station,latitude,longitude,elevation,distance_km"]
    lines += [f"S{i:02d},{_cell(lats[i])},{_cell(lons[i])},{elevation[i]},0.0" for i in range(len(lats))]
    path.write_text("\n".join((rows or lines)) + "\n")
    return path


def _cell(value):
    return repr(float(value)) if not isinstance(value, str) else value


def station_targets(tmp_path, dataset=None, variables=MAPPING, name="stations.csv", **kwargs):
    targets = DatasetTargets.from_dataset(dataset or station_dataset(), kwargs={"dataset": "smn"})
    file = kwargs.pop("stations", None) or station_file(tmp_path / name)
    return StationDatasetTargets(
        targets, variables, file, kwargs.pop("attributes", {"elevation": "elevation"}), **kwargs
    )


def converted(dataset, mapping=MAPPING):
    """The reference observations: raw * scale + offset in float64, then float32, every non-missing, not-all-NaN
    date, as `ArrayPointTargets` on the same point set."""
    return {
        i: np.stack(
            [
                (
                    dataset.fields[i][RAW.index(raw)].astype(np.float64) * spec.get("scale", 1.0)
                    + spec.get("offset", 0.0)
                )
                for raw, spec in mapping.items()
            ]
        ).astype(np.float32)
        for i in range(len(DATES))
        if i not in (ALL_NAN, MISSING)
    }


class ConstantForecast(FakeForecastSource):
    """A fake forecast that serves a geopotential `z` as an inference checkpoint serves its constant forcings, with
    typed variables."""

    constant_variables = ["z"]

    def __init__(self, *args, z, **kwargs):
        super().__init__(*args, **kwargs)
        self.z, self.reads = z, []

    def constant_field(self, name, date):
        self.reads.append((name, date))
        return self.z.astype(np.float64)

    def variable_info(self, name):
        return {"param": f"forecast-{name}", "level": 0}


def model(grid, seed=9):
    z = (np.random.default_rng(seed).uniform(0, 2500, grid.n) * G).astype(np.float32)
    return forecast_of(truth(grid), ConstantForecast, z=z), z


def analysis_dataset(grid, z, seed=1):
    """A gridded analysis on `grid` with `2t`, `2d` and the geopotential `z`, the fields of `truth(grid, seed)`."""
    fields = truth(grid, seed=seed).fields
    rows = {i: np.concatenate([fields[date], z[None]]).astype(np.float32) for i, date in enumerate(DATES)}
    return FakeDataset(["2t", "2d", "z"], grid.latitudes, grid.longitudes, rows)


def test_station_targets_convert_and_hand_out_the_station_set(tmp_path):
    dataset = station_dataset()
    targets = station_targets(tmp_path, dataset, {**MAPPING, "prestas0": {"name": "sp", "scale": 100}})
    lats, lons, elevation = stations()
    points = targets.fixed_points
    assert targets.variables == ["2t", "2d", "sp"] and targets.point_attributes == ("elevation",)
    assert targets.max_points == len(lats) and points.key[0] == "StationDatasetTargets"
    np.testing.assert_array_equal(points.latitudes, lats)
    np.testing.assert_array_equal(points.timedeltas, 0)
    np.testing.assert_array_equal(points.attribute("elevation"), elevation)
    mapping = {**MAPPING, "prestas0": {"scale": 100}}
    expected = converted(dataset, mapping)
    observed = targets.observations(DATES[1], ["sp", "2t"], CPU)
    values = observed.values
    assert observed.points is points and values.dtype == torch.float32 and tuple(values.shape) == (1, 2, len(lats))
    np.testing.assert_array_equal(values[0].numpy(), expected[1][[2, 0]])  # exact, NaN gaps kept
    assert torch.isnan(values[0, 1]).sum() == np.isnan(dataset.fields[1][0]).sum() > 0
    with pytest.raises(MissingTargetError, match="every station value"):
        targets.observations(DATES[ALL_NAN], ["2t"], CPU)
    with pytest.raises(MissingTargetError, match="missing"):
        targets.observations(DATES[MISSING], ["2t"], CPU)
    with pytest.raises(ValueError, match="cannot deliver"):
        targets.observations(DATES[1], ["tre200s0"], CPU)  # the dataset names are not offered
    assert targets.available(DATES[ALL_NAN]) and not targets.available(DATES[MISSING])  # optimistic
    assert targets.variable_info("2t") == {"param": None, "level": None}
    described = targets.describe()
    assert described["variables"]["2t"] == {"variable": "tre200s0", "scale": 1.0, "offset": 273.15}
    assert described["variables"]["sp"] == {"variable": "prestas0", "scale": 100.0, "offset": 0.0}
    assert described["stations"]["sha256"] == hashlib.sha256((tmp_path / "stations.csv").read_bytes()).hexdigest()
    assert described["stations"]["max_coordinate_offset_deg"] < 1e-12 and described["points"] == len(lats)
    assert described["row_bytes"] == 3 * len(lats) * 4 and described["dataset"]["kwargs"] == {"dataset": "smn"}
    json.dumps(described)
    with pytest.raises(ValueError, match="cannot be written to a config"):
        targets.to_config()
    # without a station file: the dataset's coordinates, no attributes
    bare = StationDatasetTargets(DatasetTargets.from_dataset(dataset), {"tre200s0": "2t"})
    assert bare.point_attributes == () and bare.describe()["stations"] is None and bare.variables == ["2t"]
    np.testing.assert_array_equal(bare.observations(DATES[0], ["2t"], CPU).values[0, 0], dataset.fields[0][0])


@pytest.mark.parametrize(
    "case, match",
    [
        ("latitude", "differ from the dataset's coordinates"),
        ("rows", "has 11 stations, the dataset 12"),
        ("text", "line 4: column 'elevation' must be a number, got 'high'"),
        (
            "empty latitude",
            "differ from the dataset's coordinates by more than 1e-06 degrees or have none, e.g. line 3",
        ),
        ("semicolons", "one column: not ','-separated\\? pass delimiter="),
        ("column", "no columns \\['height'\\]"),
        ("no file", "need a station file"),
        ("same name", "given to 'tre200s0' and 'tde200s0'"),
        ("unknown", "cannot deliver"),
        ("scale", "scale must be a finite number"),
        ("key", "unknown keys \\['units'\\]"),
    ],
)
def test_station_refusals(tmp_path, case, match):
    lats, lons, elevation = stations()
    path = tmp_path / "bad.csv"
    kwargs = {}
    variables = MAPPING
    if case == "latitude":
        station_file(path, lats=lats + np.where(np.arange(len(lats)) == 5, 2e-6, 0.0))
    elif case == "rows":
        station_file(path, lats[:-1], lons[:-1], elevation[:-1])
    elif case == "text":
        station_file(path, elevation=[*elevation[:2], "high", *elevation[3:]])
    elif case == "empty latitude":
        station_file(path, lats=[lats[0], "", *lats[2:]])
    elif case == "semicolons":
        path.write_text(station_file(path).read_text().replace(",", ";"))
    elif case == "column":
        station_file(path)
        kwargs["attributes"] = {"elevation": "height"}
    elif case == "no file":
        with pytest.raises(ValueError, match=match):
            StationDatasetTargets(DatasetTargets.from_dataset(station_dataset()), MAPPING, None, {"elevation": "x"})
        return
    else:
        station_file(path)
        variables = {
            "same name": {"tre200s0": "2t", "tde200s0": "2t"},
            "unknown": {"tre200s1": "2t"},
            "scale": {"tre200s0": {"name": "2t", "scale": float("nan")}},
            "key": {"tre200s0": {"name": "2t", "units": "K"}},
        }[case]
    with pytest.raises(ValueError, match=match):
        station_targets(tmp_path, variables=variables, stations=path, **kwargs)


def test_station_file_tolerance_longitudes_and_unknown_values(tmp_path):
    lats, lons, elevation = stations()
    shifted = station_file(tmp_path / "shifted.csv", lats=lats + 2e-6)
    assert station_targets(tmp_path, stations=shifted, tolerance=1e-5).describe()["stations"][
        "max_coordinate_offset_deg"
    ] == pytest.approx(2e-6, rel=1e-3)
    # the file in [-180, 180), the dataset in [0, 360): the same stations; an empty elevation is unknown
    wrapped = station_file(tmp_path / "wrapped.csv", lons=np.where(lons > 180, lons - 360, lons))
    targets = station_targets(tmp_path, stations=wrapped)
    np.testing.assert_array_equal(targets.fixed_points.longitudes, lons)
    lines = (tmp_path / "wrapped.csv").read_text().splitlines()
    lines[3] = ",".join(cell if i != 3 else "" for i, cell in enumerate(lines[3].split(",")))
    blank = station_file(tmp_path / "blank.csv", rows=lines)
    height = station_targets(tmp_path, stations=blank).fixed_points.attribute("elevation")
    assert np.isnan(height[2]) and np.isfinite(np.delete(height, 2)).all()
    # another delimiter, quoted fields holding it and blank lines; errors name the line of the file
    lines = station_file(tmp_path / "plain.csv").read_text().splitlines()
    semicolons = [line.replace(",", ";") for line in lines]
    semicolons[1] = '"Arosa; GR"' + semicolons[1][3:]
    semicolons.insert(2, "")
    path = station_file(tmp_path / "semicolons.csv", rows=semicolons)
    targets = station_targets(tmp_path, stations=path, delimiter=";")
    np.testing.assert_array_equal(targets.fixed_points.attribute("elevation"), elevation)
    semicolons[5] = semicolons[5].replace(f";{elevation[3]};", ";high;")
    with pytest.raises(ValueError, match="line 6: column 'elevation'"):
        station_targets(tmp_path, stations=station_file(path, rows=semicolons), delimiter=";")


@pytest.mark.parametrize("provider", ["forecast", "dataset"])
def test_stations_through_nearest_and_lapse_rate_match_numpy(tmp_path, provider):
    grid = lam_grid()
    forecast, z = model(grid)
    dataset = station_dataset()
    targets = station_targets(tmp_path, dataset)
    if provider == "forecast":
        constants = None  # the default: ForecastConstants, the orography from z at the first init time
    else:
        constants = DatasetConstants(DatasetTargets.from_dataset(analysis_dataset(grid, z)))
    evaluation = point_evaluation(forecast, targets, LAPSE, constants=constants)
    assert isinstance(evaluation.constants, ForecastConstants if provider == "forecast" else DatasetConstants)
    plan = evaluation.plan()
    assert plan["points"]["out_of_domain"] == 1 and plan["variables"]["param"] == ["forecast-2t", "forecast-2d"]
    state = evaluation.run()
    points = targets.fixed_points
    array = ArrayPointTargets(VARIABLES, {DATES[i]: (points, v) for i, v in converted(dataset).items()})
    sums, weights, outside = reference(forecast, array, z.astype(np.float64) / G)
    assert torch.equal(state.weights[:, 0], torch.from_numpy(weights))
    for name in STATISTICS:
        np.testing.assert_allclose(state.sums[name][:, 0].numpy(), sums[name], rtol=1e-12, atol=1e-12)
    # 12 h is all NaN and 18 h missing: they count as missing, not as init times with zero weight
    assert state.n_init[:, 0].tolist() == [1, 1] and state.out_of_domain[:, 0].tolist() == outside.tolist() == [1, 1]
    # one dataset read per frame whose date the dataset has (the all-NaN 12 h twice, the missing 18 h never)
    assert dataset.calls == targets.stats["reads"] == 4
    assert forecast.reads == ([("z", INITS[0])] if provider == "forecast" else [])
    attrs = state.attrs
    operator = json.loads(attrs["observation_operator"])
    assert operator["constants"]["constants"] == {
        "orography": {"variable": "z", "scale": 1 / G, "offset": 0.0, "units": "m"}
    }
    assert operator["constants"]["sha256"] and "date" not in operator["constants"]
    assert json.loads(attrs["point_targets"])["stations"] == {
        "sha256": targets.stations_sha256,
        "attributes": {"elevation": "elevation"},
    }
    assert json.loads(attrs["point_targets_description"]) == json.loads(json.dumps(targets.describe()))
    assert json.loads(attrs["observation_diagnostics"])["nearest_node_max_km"] <= MAX_KM
    with pytest.raises(MissingTargetError, match="every station value"):
        point_evaluation(forecast, station_targets(tmp_path), LAPSE, on_missing_target="raise").run()


def test_merge_identity_covers_what_changes_the_numbers_only(tmp_path):
    grid = lam_grid()
    forecast, z = model(grid)

    def evaluation(variables=MAPPING, name="stations.csv", elevation=None, constants=None, **options):
        (tmp_path / name).parent.mkdir(exist_ok=True)
        station_file(tmp_path / name, elevation=elevation)
        dataset = DatasetTargets.from_dataset(station_dataset(), kwargs={"dataset": "smn"}, **options)
        targets = StationDatasetTargets(dataset, variables, tmp_path / name, {"elevation": "elevation"})
        return point_evaluation(forecast, targets, LAPSE, bins="season", constants=constants)

    def shards(first, second):
        return [first.run(first.shard(0, 2)), second.run(second.shard(1, 2))]

    base = evaluation()
    full = base.run()
    # execution settings, the station file's directory and where the same constants come from do not count
    same = DatasetConstants(DatasetTargets.from_dataset(analysis_dataset(grid, z), cache_bytes=10**6))
    for other in (
        evaluation(cache_bytes=10**6),
        evaluation(prefetch=4),
        evaluation(name="elsewhere/stations.csv"),
        evaluation(constants=same),
    ):
        merged = merge(shards(base, other))
        assert torch.equal(merged.weights, full.weights)
        for key, total in full.sums.items():
            torch.testing.assert_close(merged.sums[key], total, rtol=1e-12, atol=1e-12)
    # a dataset opened with a path, as from Python, binds and is recorded as a tagged path, not as the string
    pathed = evaluation(args=(Path("/data/smn.zarr"),))
    assert json.loads(pathed.attrs()["point_targets"])["dataset"]["args"] == [{"__path__": "/data/smn.zarr"}]
    merge(shards(pathed, evaluation(args=(Path("/data/smn.zarr"),))))
    with pytest.raises(ValueError, match="point_targets"):
        merge(shards(pathed, evaluation(args=("/data/smn.zarr",))))
    # the conversions, the file's content and the orography do
    offset = {**MAPPING, "tre200s0": {"name": "2t", "offset": 273.0}}
    with pytest.raises(ValueError, match="point_targets"):
        merge(shards(base, evaluation(variables=offset)))
    higher = stations()[2] + np.eye(len(stations()[2]))[0]
    with pytest.raises(ValueError, match="point_targets"):
        merge(shards(base, evaluation(name="other.csv", elevation=higher)))
    lower = DatasetConstants(DatasetTargets.from_dataset(analysis_dataset(grid, z - np.float32(G))))
    with pytest.raises(ValueError, match="observation_operator"):
        merge(shards(base, evaluation(constants=lower)))


def test_forecast_constants():
    grid = lam_grid()
    forecast, z = model(grid)
    constants = ForecastConstants(forecast, date=T0)
    assert constants.specs["orography"].variable == "z" and DEFAULT_CONSTANTS["orography"]["units"] == "m"
    orography = constants.constant("orography")
    np.testing.assert_array_equal(orography.values, z.astype(np.float64) * (1 / G))
    assert orography.units == "m" and orography.values.dtype == np.float64
    assert constants.constant("orography") is orography and forecast.reads == [("z", T0)]  # read once
    with pytest.raises(KeyError, match="no model constant 'lsm'"):
        constants.constant("lsm")
    digest = constants.describe()
    assert digest["date"] == T0.isoformat() and digest["served"] == ["orography"]
    for other in (
        {"orography": {"variable": "z", "scale": 1 / 9.81, "units": "m"}},
        {"orography": {"variable": "z", "scale": 1 / G, "units": "m", "offset": 1.0}},
        {"orography": {"variable": "z", "scale": 1 / G, "units": "km"}},
    ):
        provider = ForecastConstants(forecast, other, date=T0)
        provider.constant("orography")
        assert provider.describe()["sha256"] != digest["sha256"]
    same = ForecastConstants(forecast, date=T0)
    same.constant("orography")
    assert same.describe() == digest
    with pytest.raises(ValueError, match="FakeForecastSource cannot serve the model constants.*serves none.*Dataset"):
        ForecastConstants(forecast_of(truth(grid)), date=T0)
    with pytest.raises(ValueError, match=r"'lsm'.*it serves \['z'\]"):
        ForecastConstants(forecast, {"land": {"variable": "lsm"}}, date=T0)
    with pytest.raises(ValueError, match="unknown keys"):
        ForecastConstants(forecast, {"orography": {"variable": "z", "unit": "m"}}, date=T0)
    # without a date, the evaluation it is given to sets its first init time
    undated = ForecastConstants(forecast)
    with pytest.raises(ValueError, match="no date"):
        undated.constant("orography")
    points = PointSet(grid.latitudes[:5], grid.longitudes[:5], attributes={"elevation": np.zeros(5)})
    fixed = ArrayPointTargets.fixed(points, VARIABLES, {date: np.zeros((2, 5), np.float32) for date in DATES})
    later = [INITS[2], INITS[1]]
    dated = point_evaluation(forecast, fixed, LAPSE, inits=later, constants=undated)
    assert dated.constants.date == INITS[1] and dated.constants is not undated
    dated.run()
    assert undated.date is None and not undated._served  # the provider given is untouched
    assert "date" not in undated.merge_identity()
    # a constant the provider does not serve is refused at construction
    height = Chain([Nearest(max_distance_km=MAX_KM), LapseRate(orography="height")])
    for provider in (None, ArrayConstants({"orography": (z / G, "m")})):
        with pytest.raises(ValueError, match=r"needs the model constants \['height'\].*serves \['orography'\]"):
            point_evaluation(forecast, fixed, height, constants=provider)
    # through the evaluation: the default provider is built only when the operator needs constants
    nodes = PointSet(grid.latitudes[:5], grid.longitudes[:5])
    targets = ArrayPointTargets.fixed(nodes, VARIABLES, {date: np.zeros((2, 5), np.float32) for date in DATES})
    assert point_evaluation(forecast, targets).constants is None


def test_dataset_constants(tmp_path):
    grid = lam_grid()
    forecast, z = model(grid)
    dataset = analysis_dataset(grid, z)
    spec = {
        "orography": {"variable": "z", "scale": 1 / G, "units": "m"},
        "temperature": {"variable": "2t", "units": "K"},
    }
    constants = DatasetConstants(DatasetTargets.from_dataset(dataset, kwargs={"dataset": "analysis"}), spec)
    assert constants.date == T0  # the first available date
    np.testing.assert_array_equal(constants.constant("orography").values, z.astype(np.float64) * (1 / G))
    constants.constant("temperature")
    assert dataset.calls == 1  # one read for every constant
    described = constants.describe()
    assert described["dataset"] == {"source": "anemoi_dataset", "args": [], "kwargs": {"dataset": "analysis"}}
    with pytest.raises(ValueError, match="has no date 2023-01-01"):
        DatasetConstants(DatasetTargets.from_dataset(dataset), date=datetime.datetime(2023, 1, 1))
    with pytest.raises(ValueError, match="no variables \\['orog'\\]"):
        DatasetConstants(DatasetTargets.from_dataset(dataset), {"orography": {"variable": "orog"}})
    # another grid is refused when the operator is bound, before the model runs
    other = lam_grid(seed=5)
    elsewhere = DatasetConstants(DatasetTargets.from_dataset(analysis_dataset(other, z)))
    evaluation = point_evaluation(forecast, station_targets(tmp_path), LAPSE, constants=elsewhere)
    with pytest.raises(ValueError, match="not on the forecast grid"):
        evaluation.plan()
    with pytest.raises(ValueError, match="not on the forecast grid"):
        Chain([Nearest(max_distance_km=MAX_KM), LapseRate()]).bind(
            ModelSide("grid", VARIABLES, grid=grid, constants=elsewhere)
        )
    assert dataset.calls == 1


def test_persistence_of_the_analysis_through_the_operator(tmp_path):
    grid = lam_grid()
    _, z = model(grid)
    analysis = analysis_dataset(grid, z, seed=7)
    persistence = PersistenceForecastSource(DatasetTargets.from_dataset(analysis), "6h")
    dataset = station_dataset()
    targets = station_targets(tmp_path, dataset)
    with pytest.raises(ValueError, match="PersistenceForecastSource cannot serve.*DatasetConstants"):
        point_evaluation(persistence, targets, LAPSE)
    constants = DatasetConstants(DatasetTargets.from_dataset(analysis))
    evaluation = point_evaluation(persistence, targets, LAPSE, constants=constants)
    assert evaluation.variables == VARIABLES and evaluation.forecast_variables == VARIABLES
    state = evaluation.run()
    array = ArrayPointTargets(VARIABLES, {DATES[i]: (targets.fixed_points, v) for i, v in converted(dataset).items()})
    sums, weights, _ = reference(persistence, array, z.astype(np.float64) / G)
    assert torch.equal(state.weights[:, 0], torch.from_numpy(weights))
    for name in STATISTICS:
        np.testing.assert_allclose(state.sums[name][:, 0].numpy(), sums[name], rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="cannot serve model constants such as 'z'"):
        persistence.constant_field("z", T0)
    # the model's own constants for its persistence baseline: the same orography, the same numbers
    forecast, _ = model(grid)
    borrowed = point_evaluation(persistence, targets, LAPSE, constants=ForecastConstants(forecast))
    other = borrowed.run()
    assert forecast.reads == [("z", INITS[0])]
    assert torch.equal(other.weights, state.weights)
    assert all(torch.equal(other.sums[name], state.sums[name]) for name in state.sums)


class FakeConstantInput:
    """A constant-forcings input of anemoi-inference: one input state of every retrieved constant forcing per call."""

    def __init__(self, n):
        self.variables = ["z", "lsm"]
        self.n, self.calls = n, []

    def create_input_state(self, *, dates, **kwargs):
        self.calls.append(list(dates))
        z = np.arange(self.n, dtype=np.float32) * G
        return {"fields": {"z": np.stack([z] * len(dates)), "lsm": np.ones((len(dates), self.n), np.float32)}}


def inference_source(n=6):
    source = InferenceForecastSource.__new__(InferenceForecastSource)
    source.dataset_names, source.quiet, source._constant_fields = ["ds"], False, {}
    source.runner = SimpleNamespace(constant_forcings_inputs={"ds": FakeConstantInput(n)})
    source.datasets = {"ds": SimpleNamespace(grid=Grid(np.zeros(n), np.arange(n, dtype=np.float64)))}
    return source


def test_inference_constant_forcings():
    source = inference_source()
    inputs = source.runner.constant_forcings_inputs["ds"]
    assert source.constant_variables == ["z", "lsm"]
    z = source.constant_field("z", T0)
    assert z.dtype == np.float64 and z.shape == (6,)
    np.testing.assert_array_equal(z, (np.arange(6, dtype=np.float32) * G).astype(np.float32).astype(np.float64))
    np.testing.assert_array_equal(source.constant_field("lsm", T0), 1.0)
    assert inputs.calls == [[T0]]  # one input state per date
    source.constant_field("z", T0 + 6 * HOUR)
    assert inputs.calls == [[T0], [T0 + 6 * HOUR]]
    with pytest.raises(ValueError, match="'cos_latitude' is not a constant forcing.*DatasetConstants"):
        source.constant_field("cos_latitude", T0)
    provider = ForecastConstants(source, date=T0)
    np.testing.assert_allclose(provider.constant("orography").values, np.arange(6), rtol=1e-6)
    # a dataset view serves its own dataset's
    view = DatasetForecast.__new__(DatasetForecast)
    view.source, view.name = source, "ds"
    assert view.constant_variables == ["z", "lsm"]
    np.testing.assert_array_equal(view.constant_field("z", T0), z)
    # a grid the input state does not match
    source.datasets["ds"] = SimpleNamespace(grid=Grid(np.zeros(5), np.zeros(5)))
    source._constant_fields.clear()
    with pytest.raises(ValueError, match="constant forcing 'z' has shape"):
        source.constant_field("z", T0)
    source.runner.constant_forcings_inputs["ds"].variables = []
    with pytest.raises(ValueError, match="cannot serve the model constants.*serves none"):
        ForecastConstants(source, date=T0)
