"""Accumulation periods: the targets' against the checkpoint's at construction, and the `period` of every variable
in the result file."""

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
from anemoi.evaluation.sources.persistence import PersistenceForecastSource

ONE_HOUR_TP = {"process": "accumulation", "period": ["0h", "1h"]}
MISMATCH = "the targets accumulate over another period than the forecast: tp 1h against 6h in the model"


def targets(name="era5", tp=None):
    """The fixture's targets, with `tp` described by the metadata `tp` when given."""
    dataset = fixture.FixtureDataset(name)
    if tp is not None:
        dataset.typed_variables["tp"] = Variable.from_dict("tp", tp)
    return DatasetTargets.from_dataset(dataset, kwargs={"dataset": f"fixture-{name}"})


def evaluation(forecast, targets, variables=("2t", "tp"), **options):
    init_times = fixture.init_times("era5")
    return Evaluation(
        forecast, targets, init_times, "6h", ["rmse"], list(variables), weights={"uniform": {}}, **options
    )


def no_variables_metadata(metadata):
    """A `save_checkpoint` patch: a checkpoint whose training dataset recorded no `variables_metadata`."""
    metadata["dataset"]["era5"]["variables_metadata"] = {}


def test_periods_are_checked_against_the_checkpoint(monkeypatch, tmp_path, caplog):
    forecast = fixture.forecast_source(monkeypatch, tmp_path, ("era5",))
    assert forecast.accumulation_periods() == {"tp": 6 * fixture.HOUR}
    hourly = targets(tp=ONE_HOUR_TP)
    with caplog.at_level(logging.INFO):
        evaluation(forecast, targets())
        evaluation(forecast, hourly, ["2t"])  # tp is not scored
        evaluation(PersistenceForecastSource(hourly, "6h"), hourly)  # a baseline reports no accumulations
        assert "accumulat" not in caplog.text
        evaluation(forecast, targets(tp={"process": "accumulation"}))
        assert "the accumulation period of ['tp'] is unknown in the targets or the forecast" in caplog.text
        caplog.clear()
        evaluation(forecast, hourly)
        assert MISMATCH in caplog.text and "dataset" not in caplog.text


def test_a_period_the_checkpoint_does_not_record_is_unknown(monkeypatch, tmp_path, caplog):
    """anemoi-inference fills an absent `variables_metadata` with an assumed 6h tp: not a recorded period."""
    forecast = fixture.forecast_source(monkeypatch, tmp_path, ("era5",), patch=no_variables_metadata)
    assert forecast.typed_variables["tp"].period == 6 * fixture.HOUR  # the legacy fill
    assert forecast.accumulation_periods() == {"tp": None}
    with caplog.at_level(logging.INFO):
        evaluation(forecast, targets(tp=ONE_HOUR_TP))
        assert "the accumulation period of ['tp'] is unknown" in caplog.text and MISMATCH not in caplog.text


def test_a_multi_dataset_warning_names_the_dataset(monkeypatch, tmp_path, caplog):
    forecast = fixture.forecast_source(monkeypatch, tmp_path)
    assert {name: view.accumulation_periods() for name, view in forecast.datasets.items()} == {
        "era5": {"tp": 6 * fixture.HOUR},
        "realch1": {"tp": 6 * fixture.HOUR},
    }
    with caplog.at_level(logging.WARNING):
        evaluation(forecast.datasets["era5"], targets(tp=ONE_HOUR_TP), dataset="era5")
        assert f"dataset 'era5': {MISMATCH}" in caplog.text


def test_the_period_is_written_and_merged(monkeypatch, tmp_path):
    forecast = fixture.forecast_source(monkeypatch, tmp_path, ("era5",))
    run = evaluation(PersistenceForecastSource(targets(), "6h"), targets())
    assert run.variable_coords()["period"] == ["", "6h"]  # the targets' period when the forecast reports none
    assert evaluation(forecast, targets(tp=ONE_HOUR_TP)).variable_coords()["period"] == ["", "6h"]  # the model's
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
