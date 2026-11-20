"""Variable metadata: the targets' against the checkpoint's at construction (units, time processing, accumulation
period, type of level, as anemoi-transform compares them), and the `period` of every variable in the result file."""

import logging

import _multi_dataset as fixture
import pytest
import xarray as xr
from anemoi.transform.variables import Variable

from anemoi.evaluation.evaluate import Evaluation
from anemoi.evaluation.output import load_state
from anemoi.evaluation.output import merge
from anemoi.evaluation.output import write
from anemoi.evaluation.sources.anemoi_dataset import DatasetTargets
from anemoi.evaluation.sources.base import compare_variables
from anemoi.evaluation.sources.base import name_list
from anemoi.evaluation.sources.persistence import PersistenceForecastSource

ONE_HOUR = {"period": ["0h", "1h"]}
WARNING = "the targets describe scored variables differently than the checkpoint (checkpoint vs targets): "
MISMATCH = WARNING + "tp: Periods are not compatible: 6:00:00 vs 1:00:00;"
UNCHECKED = "variable metadata recorded on one side only, not checked: "


def targets(name="era5", changes=None, drop=()):
    """The fixture's targets, with the `variables_metadata` entries of `changes` updated and of `drop` removed."""
    dataset = fixture.FixtureDataset(name)
    for variable, change in (changes or {}).items():
        entry = {**dataset.typed_variables[variable].data, **change}
        dataset.typed_variables[variable] = Variable.from_dict(
            variable, {k: v for k, v in entry.items() if v is not None}
        )
    for variable in drop:
        del dataset.typed_variables[variable]
    return DatasetTargets.from_dataset(dataset, kwargs={"dataset": f"fixture-{name}"})


def ours(caplog) -> str:
    """What anemoi-evaluation logged, without the runner's warnings."""
    return "\n".join(r.getMessage() for r in caplog.records if r.name.startswith("anemoi.evaluation"))


def evaluation(forecast, targets, variables=("2t", "tp"), **options):
    init_times = fixture.init_times("era5")
    return Evaluation(
        forecast, targets, init_times, "6h", ["rmse"], list(variables), weights={"uniform": {}}, **options
    )


def no_variables_metadata(metadata):
    """A `save_checkpoint` patch: a checkpoint whose training dataset recorded no `variables_metadata`."""
    metadata["dataset"]["era5"]["variables_metadata"] = {}


def kelvin(metadata):
    """A `save_checkpoint` patch: a checkpoint that records the units of `2t`."""
    metadata["dataset"]["era5"]["variables_metadata"]["2t"]["units"] = "K"


def test_periods_are_checked_against_the_checkpoint(monkeypatch, tmp_path, caplog):
    forecast = fixture.forecast_source(monkeypatch, tmp_path, ("era5",))
    assert forecast.recorded_variables()["tp"].period == 6 * fixture.HOUR
    hourly = targets(changes={"tp": ONE_HOUR})
    with caplog.at_level(logging.INFO):
        caplog.clear()
        evaluation(forecast, targets())
        evaluation(forecast, hourly, ["2t"])  # tp is not scored
        evaluation(PersistenceForecastSource(hourly, "6h"), hourly)  # a baseline records no metadata
        assert ours(caplog) == ""
        evaluation(forecast, targets(changes={"tp": {"period": None}}))
        assert UNCHECKED + "processing period of tp" in caplog.text and WARNING not in caplog.text
        caplog.clear()
        evaluation(forecast, hourly)
        assert MISMATCH in caplog.text and "dataset" not in caplog.text


def test_units_and_levels_are_checked_and_request_keys_are_not(monkeypatch, tmp_path, caplog):
    """Units and the type of level are compared like the period; the MARS request keys (class, stream, the GRIB
    param name) legitimately differ between sources of truth and are not."""
    forecast = fixture.forecast_source(monkeypatch, tmp_path, ("era5",), patch=kelvin)
    request = {"mars": {**fixture.FixtureDataset("era5").typed_variables["tp"].data["mars"], "class": "od"}}
    with caplog.at_level(logging.INFO):
        caplog.clear()
        evaluation(forecast, targets(changes={"2t": {"units": "K"}, "tp": request}))
        assert ours(caplog) == ""
        evaluation(forecast, targets())
        assert UNCHECKED + "units of 2t" in caplog.text and WARNING not in caplog.text
        caplog.clear()
        surface = fixture.FixtureDataset("era5").typed_variables["2t"].data["mars"]
        evaluation(forecast, targets(changes={"2t": {"units": "degC", "mars": {**surface, "levtype": "pl"}}}))
        assert WARNING + "2t: Units are not compatible: K vs degC; " in caplog.text
        caplog.clear()
        evaluation(forecast, targets(changes={"2t": {"units": "K", "mars": {**surface, "levtype": "pl"}}}))
        assert WARNING + "2t: Pressure level status is not compatible: False vs True;" in caplog.text
    assert not [record for record in caplog.records if record.name.startswith("anemoi.transform")]  # skipped instead


def test_what_transform_reads_as_no_metadata_is_not_compared(monkeypatch, tmp_path, caplog):
    """A `levtype` of `unknown` (some datasets write it for pressure levels) and a malformed `period` are not
    recorded, so they are skipped instead of warned about or raised on; another time processing is a difference."""
    forecast = fixture.forecast_source(monkeypatch, tmp_path, ("era5",))
    mars = fixture.FixtureDataset("era5").typed_variables["2t"].data["mars"]
    with caplog.at_level(logging.INFO):
        caplog.clear()
        run = evaluation(
            forecast, targets(changes={"2t": {"mars": {**mars, "levtype": "unknown"}}, "tp": {"period": [None, "6h"]}})
        )
        assert UNCHECKED + "type of level of 2t; processing period of tp" in ours(caplog) and WARNING not in caplog.text
        assert run.variable_coords()["period"] == ["", "6h"]  # the checkpoint's
        evaluation(forecast, targets(changes={"tp": {"process": "average"}}))
        assert WARNING + "tp: Time processinging types are not compatible: accumulation vs average;" in caplog.text


def test_differences_and_unknowns_are_grouped():
    def variable(name, **entry):
        return Variable.from_dict(name, {"mars": {"levtype": "sfc"}, **entry})

    names = [f"v{i}" for i in range(7)]
    checkpoint = {name: variable(name, units="K") for name in names}
    targets = {name: variable(name, units="degC" if name < "v2" else None) for name in names}
    differ, unknown = compare_variables(checkpoint, {**targets, "w": variable("w")}, [*names, "w"])
    assert differ == {"Units are not compatible: K vs degC": ["v0", "v1"]}
    assert unknown == {"units": names[2:], "any metadata": ["w"]}
    assert name_list(names) == "v0, v1, v2, v3, v4 and 2 more" and name_list(names[:5]) == "v0, v1, v2, v3, v4"


def test_metadata_the_checkpoint_does_not_record_is_unknown(monkeypatch, tmp_path, caplog):
    """anemoi-inference fills an absent `variables_metadata` with assumed entries, a 6h tp among them: not recorded."""
    forecast = fixture.forecast_source(monkeypatch, tmp_path, ("era5",), patch=no_variables_metadata)
    assert forecast.typed_variables["tp"].period == 6 * fixture.HOUR  # the legacy fill
    assert forecast.recorded_variables() == {}
    with caplog.at_level(logging.INFO):
        run = evaluation(forecast, targets(changes={"tp": ONE_HOUR}, drop=["2t"]))
        assert UNCHECKED + "any metadata of 2t, tp" in caplog.text and WARNING not in caplog.text
    assert run.variable_coords()["period"] == ["", "1h"]  # the targets' period


def test_a_multi_dataset_warning_names_the_dataset(monkeypatch, tmp_path, caplog):
    forecast = fixture.forecast_source(monkeypatch, tmp_path)
    assert {name: view.recorded_variables()["tp"].period for name, view in forecast.datasets.items()} == {
        "era5": 6 * fixture.HOUR,
        "realch1": 6 * fixture.HOUR,
    }
    with caplog.at_level(logging.WARNING):
        evaluation(forecast.datasets["era5"], targets(changes={"tp": ONE_HOUR}), dataset="era5")
        assert f"dataset 'era5': {MISMATCH}" in caplog.text


def test_the_period_is_written_and_merged(monkeypatch, tmp_path):
    forecast = fixture.forecast_source(monkeypatch, tmp_path, ("era5",))
    run = evaluation(PersistenceForecastSource(targets(), "6h"), targets())
    assert run.variable_coords()["period"] == ["", "6h"]  # the targets' period when the forecast reports none
    assert evaluation(forecast, targets(changes={"tp": ONE_HOUR})).variable_coords()["period"] == [
        "",
        "6h",
    ]  # the model's
    paths = []
    for i in range(2):
        paths.append(tmp_path / f"shard-{i}.nc")
        write(run.run(run.shard(i, 2)).to_xarray(run.metrics), paths[-1])
    with xr.open_dataset(paths[0]) as written:
        assert written["period"].dims == ("variable",) and written["period"].values.tolist() == ["", "6h"]
    assert load_state(paths[0]).variable_coords["period"] == ["", "6h"]
    assert merge(paths).variable_coords["period"] == ["", "6h"]
    older = load_state(paths[1])
    del older.variable_coords["period"]  # a file written before the period was recorded
    merged = merge([paths[0], older])
    assert "period" not in merged.variable_coords and merged.variable_coords["param"] == ["2t", "tp"]
    other = load_state(paths[1])
    other.variable_coords["period"] = ["", "1h"]
    with pytest.raises(ValueError, match="different variable coordinate 'period'"):
        merge([paths[0], other])
