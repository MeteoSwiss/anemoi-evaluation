import datetime

import pytest
import yaml
from pydantic import ValidationError

from anemoi.evaluation import config as config_module
from anemoi.evaluation.config import load_config
from anemoi.evaluation.evaluate import Evaluation
from anemoi.evaluation.evaluate import init_times

CONFIG = {
    "forecast": {
        "anemoi_inference": {
            "checkpoint": "/x/inference-last.ckpt",
            "device": "cuda",
            "members": 2,
            "seed": 3,
            "quiet": False,
        }
    },
    "lead_time": "24h",
    "init_times": {"start": "2024-01-01", "end": "2024-01-01T12", "frequency": "12h"},
    "metrics": [
        "rmse",
        {"crps": {"alpha": 0.5}},
        {"csi": {"label": "heavy", "thresholds": {"t": 0.5}}},
        {"reliability": {"label": "heavy", "thresholds": {"t": 0.5}}},
        "rank_histogram",
    ],
    "weights": {"uniform": {}},
    "regions": {"global": "all", "box": {"bbox": {"north": 60, "west": -10, "south": 30, "east": 40}}},
    "bins": {"time": "none"},
    "output": {"path": "results/out.nc"},
}


def test_init_times_forms_and_inference_block():
    config = load_config(CONFIG)
    dates = [datetime.datetime(2024, 1, 1), datetime.datetime(2024, 1, 1, 12)]
    assert config.init_times.resolve() == dates == init_times("2024-01-01", "2024-01-01T12", "12h")
    as_list = load_config({**CONFIG, "init_times": {"dates": ["2024-01-01", "2024-01-01T12:00"]}})
    assert as_list.init_times.resolve() == dates
    assert config.lead_time == datetime.timedelta(hours=24)
    block = config.forecast.anemoi_inference
    assert (block.members, block.seed, block.quiet) == (2, 3, False)
    assert block.model_dump() == {**CONFIG["forecast"]["anemoi_inference"], "forcings_cache_bytes": 2**30}


def test_validation_errors(fake_sources):
    invalid = [
        {**CONFIG, "device": "cuda"},
        {**CONFIG, "quiet_inference": True},
        {**CONFIG, "bins": {"time": "none", "month": True}},
        {**CONFIG, "bins": {"season": "init_time"}},
        {**CONFIG, "metrics": ["nope"]},
        {**CONFIG, "weights": {"uniform": {}, "file": "w.npy"}},
        {**CONFIG, "lead_time": "-6h"},
        {**CONFIG, "forecast": {}},
        {**CONFIG, "metrics": [{"crps": {"alpah": 0.5}}]},
        {**CONFIG, "metrics": [{"csi": {"thresholds": {"t": 0.5}}}]},
        {**CONFIG, "metrics": [{"csi": {"label": "x"}}]},
        {**CONFIG, "metrics": [{"csi": {"label": "x", "thresholds": {"t": 0.5}, "labl": "y"}}]},
        {**CONFIG, "metrics": [{"csi": {"label": "a b", "thresholds": {"t": 0.5}}}]},
        {**CONFIG, "metrics": [{"csi": {"label": "x", "thresholds": {}}}]},
        {**CONFIG, "metrics": [{"csi": {"label": "x", "thresholds": [1, 2]}}]},
        {**CONFIG, "metrics": [{"csi": {"label": "x", "thresholds": {"t": "0.5"}}}]},
        {**CONFIG, "metrics": [{"csi": {"label": "x", "thresholds": {"t": True}}}]},
        {**CONFIG, "metrics": [{"csi": {"label": "x", "thresholds": {"t": float("nan")}}}]},
        {**CONFIG, "metrics": [{"rank_histogram": {"members": 8}}]},  # M comes from the run, never from the spec
        {**CONFIG, "metrics": [{"reliability": {"thresholds": {"t": 0.5}}}]},
        {**CONFIG, "metrics": [{"reliability": {"label": "x"}}]},
        {**CONFIG, "metrics": [{"reliability": {"label": "x", "thresholds": {"t": 0.5}, "members": 8}}]},
        {
            **CONFIG,
            "metrics": [  # one label, one threshold map, across the metric families too
                {"csi": {"label": "x", "thresholds": {"t": 0.5}}},
                {"reliability": {"label": "x", "thresholds": {"t": 1.5}}},
            ],
        },
        {
            **CONFIG,
            "metrics": [
                {"csi": {"label": "x", "thresholds": {"t": 0.5}}},
                {"pod": {"label": "x", "thresholds": {"t": 1.5}}},
            ],
        },
    ]
    for config in invalid:
        with pytest.raises(ValidationError):
            load_config(config)
    with pytest.raises(ValidationError, match="rank_histogram"):
        load_config({**CONFIG, "metrics": [{"rank_histogram": {"members": 8}}]})
    with pytest.raises(ValidationError, match="one label, one threshold map"):
        load_config(
            {
                **CONFIG,
                "metrics": [
                    {"csi": {"label": "x", "thresholds": {"t": 0.5}}},
                    {"pod": {"label": "x", "thresholds": {"t": 1.5}}},
                ],
            }
        )
    # the config layer knows no member count, so a metric that needs one validates unbound
    assert load_config({**CONFIG, "metrics": ["rank_histogram", "outlier_fraction"]}).metrics == [
        "rank_histogram",
        "outlier_fraction",
    ]
    forecast, targets = fake_sources()
    with pytest.raises(ValueError):
        Evaluation.from_config({**CONFIG, "lead_time": "9h"}, forecast=forecast, targets=targets)


def test_config_round_trip(fake_sources):
    forecast, targets = fake_sources()
    evaluation = Evaluation.from_config(CONFIG, forecast=forecast, targets=targets)
    config = evaluation.to_config()
    assert Evaluation.from_config(config, forecast=forecast, targets=targets).to_config() == config
    assert config["lead_time"] == "1d" and config["metrics"] == CONFIG["metrics"]
    assert config["include_lead_zero"] is False and evaluation.lead_times[0] == datetime.timedelta(hours=6)
    with_zero = Evaluation.from_config({**CONFIG, "include_lead_zero": True}, forecast=forecast, targets=targets)
    assert with_zero.to_config()["include_lead_zero"] is True and with_zero.lead_times[0] == datetime.timedelta(0)
    assert list(evaluation.regions) == ["global", "box"] and evaluation.binning.kind == "none"
    assert config["bins"] == {"time": "none", "by": "init_time"}


def write(path, config):
    """Write a YAML config and return its path."""
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def test_base_merges_and_chains(tmp_path):
    write(
        tmp_path / "common.yaml",
        {
            "lead_time": "24h",
            "metrics": ["rmse", "mae"],
            "regions": {"global": "all", "box": CONFIG["regions"]["box"]},
            "forecast": {"anemoi_inference": {"checkpoint": "/x/inference-last.ckpt", "device": "cuda", "members": 2}},
        },
    )
    write(tmp_path / "middle.yaml", {"base": "common.yaml", "weights": {"uniform": {}}, "bins": {"time": "none"}})
    leaf = write(
        tmp_path / "leaf.yaml",
        {
            "base": "middle.yaml",
            "forecast": {"anemoi_inference": {"members": 4}},
            "metrics": ["bias"],
            "regions": {"lam": {"graph_attribute": "cutout_mask"}},
            "init_times": {"dates": ["2024-01-01"]},
            "output": {"path": "out.nc"},
        },
    )
    config = load_config(leaf)
    block = config.forecast.anemoi_inference
    assert (block.members, block.model_extra["checkpoint"], block.model_extra["device"]) == (
        4,
        "/x/inference-last.ckpt",
        "cuda",
    )
    assert config.metrics == ["bias"]
    assert list(config.regions) == ["global", "box", "lam"]
    assert config.lead_time == datetime.timedelta(hours=24)
    assert config.bins.time == "none" and config.weights.spec() == {"uniform": {}}


def test_base_errors(tmp_path):
    write(tmp_path / "missing-base.yaml", {"base": "nowhere.yaml", **CONFIG})
    with pytest.raises(ValueError, match="not found"):
        load_config(tmp_path / "missing-base.yaml")
    write(tmp_path / "a.yaml", {"base": "b.yaml", **CONFIG})
    write(tmp_path / "b.yaml", {"base": "a.yaml"})
    with pytest.raises(ValueError, match="cycle"):
        load_config(tmp_path / "a.yaml")
    write(tmp_path / "list.yaml", ["not", "a", "mapping"])
    write(tmp_path / "on-list.yaml", {"base": "list.yaml", **CONFIG})
    with pytest.raises(ValueError, match="must be a mapping"):
        load_config(tmp_path / "on-list.yaml")


RECORDED = {"dataset": {"cutout": ["lam", "global"]}, "frequency": "6h", "select": ["2t", "10u"]}


def test_from_checkpoint(monkeypatch):
    read = []

    def arguments(checkpoint):
        read.append(checkpoint)
        return (RECORDED,), {"start": None, "end": 2023}

    monkeypatch.setattr(config_module, "checkpoint_dataset_arguments", arguments)
    inference = CONFIG["forecast"]["anemoi_inference"]
    config = load_config(
        {
            **CONFIG,
            "forecast": {
                "anemoi_inference": {
                    **inference,
                    "input": {"dataset": {"from_checkpoint": True, "start": 2024, "end": 2024}},
                }
            },
            "targets": {"anemoi_dataset": {"from_checkpoint": True, "end": 2025, "cache_bytes": "1GiB"}},
        }
    )
    assert config.forecast.anemoi_inference.model_extra["input"]["dataset"] == {**RECORDED, "start": 2024, "end": 2024}
    targets = config.targets.anemoi_dataset
    assert targets.open_dataset_kwargs() == {**RECORDED, "start": None, "end": 2025}
    assert targets.cache_bytes == 2**30
    assert read == [inference["checkpoint"], inference["checkpoint"]]

    persistence = {"persistence": {"timestep": "6h"}}
    with_path = load_config(
        {**CONFIG, "forecast": persistence, "targets": {"anemoi_dataset": {"from_checkpoint": "/other.ckpt"}}}
    )
    assert with_path.targets.anemoi_dataset.open_dataset_kwargs() == {**RECORDED, "start": None, "end": 2023}
    assert read[-1] == "/other.ckpt"
    disabled = load_config({**CONFIG, "targets": {"anemoi_dataset": {"from_checkpoint": False, "dataset": "d"}}})
    assert disabled.targets.anemoi_dataset.open_dataset_kwargs() == {"dataset": "d"}
    with pytest.raises(ValueError, match="needs forecast.anemoi_inference.checkpoint"):
        load_config({**CONFIG, "forecast": persistence, "targets": {"anemoi_dataset": {"from_checkpoint": True}}})
    with pytest.raises(ValueError, match="only resolved in"):
        load_config({**CONFIG, "targets": {"anemoi_dataset": {"dataset": {"from_checkpoint": True}}}})
