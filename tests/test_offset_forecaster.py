"""Offset forecasters, through the fixture runner (`tests/_multi_dataset.py`) with the timing of the ckpt-gen dummy:
inputs `[-12h, 0h]`, outputs `[6h, 12h, 1d]`, rollout shift `1d`, on 6 h data."""

import datetime
import logging

import _multi_dataset as fixture
import numpy as np
import pytest
import torch
import xarray as xr
import yaml

from anemoi.evaluation.__main__ import main
from anemoi.evaluation.evaluate import Evaluation
from anemoi.evaluation.sources.anemoi_inference import member_seed

HOUR = fixture.HOUR
LEADS_48H = [h * HOUR for h in (6, 12, 24, 30, 36, 48)]
INITS = [datetime.datetime(2020, 1, 1, 12), datetime.datetime(2020, 1, 2)]


def offset_source(monkeypatch, tmp_path, **options):
    return fixture.forecast_source(monkeypatch, tmp_path, ("era5",), patch=fixture.offset_forecaster, **options)


def test_lead_times_and_model_calls_come_from_the_offsets(tmp_path, monkeypatch, caplog):
    forecast = offset_source(monkeypatch, tmp_path)
    assert forecast.input_offsets == [-12 * HOUR, 0 * HOUR] and forecast.output_offsets == LEADS_48H[:3]
    assert forecast.output_horizon == 24 * HOUR and forecast.frames_per_pass == 3
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert forecast.lead_times(48 * HOUR) == LEADS_48H
    assert not caplog.records
    assert forecast.model_calls(48 * HOUR) == (2, 48 * HOUR)
    # not on the grid: computed up to the next call's end, the outputs beyond it discarded
    evaluation = Evaluation(forecast, fixture.make_targets("era5"), INITS, "36h", ["rmse"], weights={"uniform": {}})
    assert evaluation.lead_times == LEADS_48H[:5] and evaluation.model_calls_per_init() == (2, 48 * HOUR)
    assert "the model computes 2d (2 calls of 1d)" in caplog.text
    plan = evaluation.plan()
    assert plan["computed_lead_time"] == "2d" and "computed and discarded" in plan["note"]
    assert plan["forecast"]["input_offsets"] == ["-12h", "0h"] and plan["forecast"]["output_offsets"] == [
        "6h",
        "12h",
        "1d",
    ]
    assert forecast.lead_times(9 * HOUR) == [6 * HOUR] and forecast.model_calls(9 * HOUR) == (1, 24 * HOUR)


def test_the_init_time_check_reads_the_input_offsets(tmp_path, monkeypatch):
    forecast = offset_source(monkeypatch, tmp_path)
    dataset = forecast.runner.prognostics_inputs["era5"].ds
    # 2020-01-01T06 needs 2019-12-31T18, before the fixture starts
    with pytest.raises(ValueError, match=r"dates \[datetime.datetime\(2019, 12, 31, 18, 0\)\]"):
        forecast.check_init_times([datetime.datetime(2020, 1, 1, 6)], 48 * HOUR)
    forecast.check_init_times(INITS, 48 * HOUR)
    # init - 6h is not an input of this model, init - 12h is
    dataset.missing = {dataset.index(INITS[1] - 6 * HOUR)}
    forecast.check_init_times(INITS[1:], 48 * HOUR)
    dataset.missing = {dataset.index(INITS[1] - 12 * HOUR)}
    with pytest.raises(ValueError, match="1 missing"):
        forecast.check_init_times(INITS[1:], 48 * HOUR)


@pytest.mark.parametrize("lead_time", [48 * HOUR, 36 * HOUR], ids=["on-the-grid", "between-calls"])
def test_the_rollout_reseeds_each_member_before_its_model_call(tmp_path, monkeypatch, lead_time):
    forecast = offset_source(monkeypatch, tmp_path, members=2, seed=7)
    events = []
    manual_seed, predict_step = torch.manual_seed, forecast.runner.predict_step
    monkeypatch.setattr(torch, "manual_seed", lambda seed: events.append(("seed", seed)) or manual_seed(seed))
    forecast.runner.predict_step = lambda *args, **kwargs: events.append(("call",)) or predict_step(*args, **kwargs)
    for frame in forecast.frames(INITS[0], lead_time, ["2t"], torch.device("cpu")):
        assert frame.valid_time == INITS[0] + frame.lead_time
        events.append(("frame", frame.lead_time))

    def call(k):
        return [event for member in range(2) for event in (("seed", member_seed(7, INITS[0], member, k)), ("call",))]

    frames = [("frame", lead) for lead in forecast.lead_times(lead_time)]
    assert events == [*call(0), *frames[:3], *call(1), *frames[3:]]


def test_an_offset_forecaster_runs_shards_and_merges_on_its_lead_times(tmp_path, monkeypatch, capsys):
    """The dry run, the sharded run and `merge` on the irregular lead-time axis; lead 0 from the `0h` input."""
    checkpoint = fixture.save_checkpoint(tmp_path / "offset.ckpt", ("era5",), patch=fixture.offset_forecaster)
    fixture.patch_create_runner(monkeypatch, checkpoint, ("era5",))
    fixture.patch_targets_from_forecast(monkeypatch)
    config = {
        "forecast": {"anemoi_inference": {"checkpoint": str(checkpoint), "device": "cpu"}},
        "lead_time": "48h",
        "init_times": {"dates": [date.isoformat() for date in INITS]},
        "metrics": ["rmse", "bias"],
        "weights": {"uniform": {}},
        "bins": {"time": "none"},
        "include_lead_zero": True,
        "output": {"path": str(tmp_path / "run-{shard}-of-{shards}.nc")},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    main(["run", str(path), "--dry-run"])
    plan = yaml.safe_load(capsys.readouterr().out)
    assert plan["lead_times"] == {"count": 7, "first": "0h", "last": "2d"}
    assert plan["model_calls"] == 4 and plan["computed_lead_time"] == "2d" and "note" not in plan
    main(["run", str(path), "--shard", "0/1"])
    for shard in range(2):
        main(["run", str(path), "--shard", f"{shard}/2"])
    parts = [str(tmp_path / f"run-{shard}-of-2.nc") for shard in range(2)]
    for part in parts:  # one init time per shard, every lead time on each
        with xr.open_dataset(part, decode_timedelta=True) as one:
            assert one.sizes["lead_time"] == 7
    main(["merge", *parts, "-o", str(tmp_path / "merged.nc")])
    with xr.open_dataset(tmp_path / "run-0-of-1.nc", decode_timedelta=True) as whole:
        with xr.open_dataset(tmp_path / "merged.nc", decode_timedelta=True) as merged:
            whole, merged = whole.load(), merged.load()
    expected = np.array([datetime.timedelta(0), *LEADS_48H], dtype="timedelta64[ns]")
    np.testing.assert_array_equal(whole["lead_time"].values, expected)
    xr.testing.assert_equal(merged, whole)
    assert int(whole.attrs["model_calls"]) == 4
    rmse = whole["rmse"].sel(bin="all", region="global")
    assert (rmse.isel(lead_time=0) == 0).sum() > 0 and np.isfinite(rmse.isel(lead_time=slice(1, None))).all()
