"""The checks on the targets run at construction: accumulation periods against the checkpoint's, and target dates
anemoi-datasets made up. The targets are small real zarrs, so the wrappers are anemoi-datasets' own."""

import logging

import _multi_dataset as fixture
import pytest

from anemoi.evaluation.evaluate import Evaluation
from anemoi.evaluation.evaluate import build
from anemoi.evaluation.sources.anemoi_dataset import DatasetTargets
from anemoi.evaluation.sources.persistence import PersistenceForecastSource

ONE_HOUR_TP = {"tp": {"process": "accumulation", "period": ["0h", "1h"]}}


def evaluation(forecast, targets, variables=("2t", "tp"), **options):
    return Evaluation(forecast, targets, fixture.init_times("era5"), "6h", ["rmse"], list(variables), **options)


def no_variables_metadata(metadata):
    """A `save_checkpoint` patch: a checkpoint whose training dataset recorded no `variables_metadata`."""
    metadata["dataset"]["era5"]["variables_metadata"] = {}


@pytest.mark.parametrize(
    "options, found",
    [
        ({"interpolate_frequency": "1h"}, "InterpolateFrequency"),
        ({"fill_missing_dates": "closest"}, "MissingDatesClosest"),
        ({"fill_missing_dates": "interpolate"}, "MissingDatesInterpolate"),
        ({"fill_missing_dates": "mean"}, "MissingDatesFillStatistics"),
        ({"rolling_average": [-1, 0, "freq"]}, "RollingAverage"),
        ({"fill_missing_dates": "nan"}, None),
        ({"set_missing_dates": ["2020-01-02T00:00:00"]}, None),
        ({}, None),
    ],
)
def test_made_up_dates_are_read_off_the_opened_dataset(tmp_path, options, found):
    path = fixture.write_zarr(tmp_path / "era5.zarr", "era5", missing=(3,))
    targets = DatasetTargets(dataset=path, **options)
    made_up = targets.made_up_dates()
    assert len(made_up) == (1 if found else 0)
    assert all(text.startswith(found) and text.endswith(" over era5") for text in made_up)
    assert targets.interpolate_frequency == (["1h"] if "interpolate_frequency" in options else [])


def test_made_up_dates_name_the_element(tmp_path):
    path = fixture.write_zarr(tmp_path / "era5.zarr", "era5", missing=(3,))
    targets = DatasetTargets(
        join=[{"dataset": path, "select": ["2t"], "fill_missing_dates": "closest"}, {"dataset": path, "select": ["tp"]}]
    )
    [made_up] = targets.made_up_dates()
    assert made_up.startswith("MissingDatesClosest") and made_up.endswith(" in Join[0] over era5")
    assert targets.interpolate_frequency == []


def test_made_up_dates_warn_raise_or_are_ignored(monkeypatch, tmp_path, caplog):
    forecast = fixture.forecast_source(monkeypatch, tmp_path, ("era5",))
    path = fixture.write_zarr(tmp_path / "era5.zarr", "era5", missing=(3,))
    plain, filled = DatasetTargets(dataset=path), DatasetTargets(dataset=path, fill_missing_dates="closest")
    with caplog.at_level(logging.WARNING):
        evaluation(forecast, plain)
        assert "make up values" not in caplog.text
        evaluation(forecast, filled)
        assert "the targets make up values at some dates: MissingDatesClosest" in caplog.text
        assert "set on_made_up_dates: ignore to silence" in caplog.text
        caplog.clear()
        evaluation(forecast, filled, dataset="era5")
        assert "dataset 'era5': the targets make up values" in caplog.text
        caplog.clear()
        evaluation(forecast, filled, on_made_up_dates="ignore")
        assert caplog.text == ""
    with pytest.raises(ValueError, match="make up values at some dates: MissingDatesClosest"):
        evaluation(forecast, filled, on_made_up_dates="raise")
    with pytest.raises(ValueError, match="on_made_up_dates must be 'warn', 'raise' or 'ignore'"):
        evaluation(forecast, plain, on_made_up_dates="loud")


def test_accumulation_periods_are_checked_against_the_checkpoint(monkeypatch, tmp_path, caplog):
    forecast = fixture.forecast_source(monkeypatch, tmp_path, ("era5",))
    assert forecast.accumulation_periods() == {"tp": 6 * fixture.HOUR}
    same = DatasetTargets(dataset=fixture.write_zarr(tmp_path / "same.zarr", "era5"))
    hourly = DatasetTargets(dataset=fixture.write_zarr(tmp_path / "hourly.zarr", "era5", ONE_HOUR_TP))
    unknown = fixture.write_zarr(tmp_path / "unknown.zarr", "era5", {"tp": {"process": "accumulation"}})
    with caplog.at_level(logging.INFO):
        evaluation(forecast, same)
        assert "accumulat" not in caplog.text
        evaluation(forecast, DatasetTargets(dataset=unknown))
        assert "the accumulation period of ['tp'] is unknown in the targets or the forecast" in caplog.text
        caplog.clear()
        evaluation(forecast, hourly)
        assert "accumulate over another period than the forecast: tp 1h against 6h in the model" in caplog.text
        caplog.clear()
        evaluation(forecast, hourly, on_period_mismatch="ignore")
        evaluation(forecast, hourly, ["2t"], on_period_mismatch="raise")  # tp is not scored
        evaluation(PersistenceForecastSource(hourly, "6h"), hourly)  # a baseline knows no accumulations
        assert "accumulat" not in caplog.text
    with pytest.raises(ValueError, match="tp 1h against 6h in the model.*set on_period_mismatch: warn or ignore"):
        evaluation(forecast, hourly, on_period_mismatch="raise")


def test_a_period_the_checkpoint_does_not_record_is_unknown(monkeypatch, tmp_path, caplog):
    """anemoi-inference fills an absent `variables_metadata` with an assumed 6h tp: not a recorded period."""
    forecast = fixture.forecast_source(monkeypatch, tmp_path, ("era5",), patch=no_variables_metadata)
    assert forecast.typed_variables["tp"].period == 6 * fixture.HOUR  # the legacy fill
    assert forecast.accumulation_periods() == {"tp": None}
    hourly = DatasetTargets(dataset=fixture.write_zarr(tmp_path / "hourly.zarr", "era5", ONE_HOUR_TP))
    with caplog.at_level(logging.INFO):
        evaluation(forecast, hourly, on_period_mismatch="raise")
        assert "the accumulation period of ['tp'] is unknown" in caplog.text


def test_a_multi_dataset_view_reports_its_own_periods(monkeypatch, tmp_path):
    forecast = fixture.forecast_source(monkeypatch, tmp_path)
    assert {name: view.accumulation_periods() for name, view in forecast.datasets.items()} == {
        "era5": {"tp": 6 * fixture.HOUR},
        "realch1": {"tp": 6 * fixture.HOUR},
    }


def test_the_settings_reach_the_checks_config_and_plan(monkeypatch, tmp_path):
    forecast = fixture.forecast_source(monkeypatch, tmp_path, ("era5",))
    path = fixture.write_zarr(tmp_path / "hourly.zarr", "era5", ONE_HOUR_TP, missing=(3,))
    hourly, filled = DatasetTargets(dataset=path), DatasetTargets(dataset=path, fill_missing_dates="closest")
    config = {
        "forecast": {"anemoi_inference": {"checkpoint": "unused"}},
        "lead_time": "6h",
        "init_times": {"dates": [date.isoformat() for date in fixture.init_times("era5")]},
        "variables": ["2t", "tp"],
        "weights": {"uniform": {}},
    }
    with pytest.raises(ValueError, match="tp 1h against 6h"):
        build({**config, "on_period_mismatch": "raise"}, forecast=forecast, targets=hourly)
    with pytest.raises(ValueError, match="make up values at some dates"):
        build(
            {**config, "on_period_mismatch": "ignore", "on_made_up_dates": "raise"}, forecast=forecast, targets=filled
        )
    with pytest.raises(ValueError, match="on_made_up_dates"):
        build({**config, "on_made_up_dates": "loud"}, forecast=forecast, targets=hourly)
    settings = {"on_period_mismatch": "raise", "on_made_up_dates": "ignore"}
    by_hand = evaluation(PersistenceForecastSource(filled, "6h"), filled, weights={"uniform": {}}, **settings)
    assert {key: by_hand.to_config()[key] for key in settings} == settings
    assert {key: by_hand.plan()[key] for key in settings} == settings
