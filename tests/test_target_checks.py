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


def evaluation(forecast, targets, **options):
    return Evaluation(forecast, targets, fixture.init_times("era5"), "6h", ["rmse"], ["2t", "tp"], **options)


@pytest.mark.parametrize(
    "options, found",
    [
        ({"interpolate_frequency": "1h"}, "InterpolateFrequency(frequency=1h) over era5"),
        ({"fill_missing_dates": "closest"}, "MissingDatesClosest(closest=up) over era5"),
        ({"fill_missing_dates": "interpolate"}, "MissingDatesInterpolate over era5"),
        ({"fill_missing_dates": "mean"}, "MissingDatesFillStatistics(method=mean) over era5"),
        ({"rolling_average": [-1, 0, "freq"]}, "RollingAverage(window=-1-to-1) over era5"),
        ({"fill_missing_dates": "nan"}, None),
        ({"set_missing_dates": ["2020-01-02T00:00:00"]}, None),
        ({}, None),
    ],
)
def test_made_up_dates_are_read_off_the_opened_dataset(tmp_path, options, found):
    path = fixture.write_zarr(tmp_path / "era5.zarr", "era5", missing=(3,))
    targets = DatasetTargets(dataset=path, **options)
    assert targets.made_up_dates() == ([found] if found else [])
    assert targets.interpolate_frequency == (["1h"] if "interpolate_frequency" in options else [])


def test_made_up_dates_name_the_element(tmp_path):
    path = fixture.write_zarr(tmp_path / "era5.zarr", "era5", missing=(3,))
    targets = DatasetTargets(
        join=[{"dataset": path, "select": ["2t"], "fill_missing_dates": "closest"}, {"dataset": path, "select": ["tp"]}]
    )
    assert targets.made_up_dates() == ["MissingDatesClosest(closest=up) in Join[0] over era5"]
    assert targets.interpolate_frequency == []


def test_made_up_dates_warn_raise_or_are_ignored(monkeypatch, tmp_path, caplog):
    forecast = fixture.forecast_source(monkeypatch, tmp_path, ("era5",))
    path = fixture.write_zarr(tmp_path / "era5.zarr", "era5", missing=(3,))
    plain, filled = DatasetTargets(dataset=path), DatasetTargets(dataset=path, fill_missing_dates="closest")
    with caplog.at_level(logging.WARNING):
        evaluation(forecast, plain)
        assert "make up values" not in caplog.text
        evaluation(forecast, filled)
        assert "the targets make up values at some dates: MissingDatesClosest(closest=up) over era5" in caplog.text
        assert "set on_made_up_dates: ignore to silence" in caplog.text
        caplog.clear()
        evaluation(forecast, filled, on_made_up_dates="ignore")
        assert caplog.text == ""
    with pytest.raises(ValueError, match="make up values at some dates: MissingDatesClosest"):
        evaluation(forecast, filled, on_made_up_dates="raise")
    with pytest.raises(ValueError, match="on_made_up_dates must be one of warn, raise, ignore"):
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
        evaluation(PersistenceForecastSource(hourly, "6h"), hourly)  # a baseline knows no accumulations
        assert "accumulat" not in caplog.text
    with pytest.raises(ValueError, match="tp 1h against 6h in the model.*set on_period_mismatch: warn or ignore"):
        evaluation(forecast, hourly, on_period_mismatch="raise")


def test_the_config_keys_reach_the_checks(monkeypatch, tmp_path):
    forecast = fixture.forecast_source(monkeypatch, tmp_path, ("era5",))
    hourly = DatasetTargets(dataset=fixture.write_zarr(tmp_path / "hourly.zarr", "era5", ONE_HOUR_TP))
    config = {
        "forecast": {"anemoi_inference": {"checkpoint": "unused"}},
        "lead_time": "6h",
        "init_times": {"dates": [date.isoformat() for date in fixture.init_times("era5")]},
        "variables": ["2t", "tp"],
        "weights": {"uniform": {}},
    }
    built = build({**config, "on_period_mismatch": "ignore"}, forecast=forecast, targets=hourly)
    assert built.plan()["on_period_mismatch"] == "ignore" and built.to_config()["on_made_up_dates"] == "warn"
    with pytest.raises(ValueError, match="tp 1h against 6h"):
        build({**config, "on_period_mismatch": "raise"}, forecast=forecast, targets=hourly)
    with pytest.raises(ValueError, match="on_made_up_dates"):
        build({**config, "on_made_up_dates": "loud"}, forecast=forecast, targets=hourly)
