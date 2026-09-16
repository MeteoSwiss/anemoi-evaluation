import datetime
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from anemoi.evaluation.config import TargetsConfig
from anemoi.evaluation.sources.anemoi_dataset import DatasetTargets
from anemoi.evaluation.sources.anemoi_inference import member_seed
from anemoi.evaluation.sources.anemoi_inference import missing_dates
from anemoi.evaluation.sources.anemoi_inference import required_dates
from anemoi.evaluation.sources.base import MissingTargetError

HOUR = datetime.timedelta(hours=1)
T0 = datetime.datetime(2024, 1, 1)
CPU = torch.device("cpu")


class FakeDataset:
    """What DatasetTargets reads from an anemoi dataset; values encode (date, variable, node)."""

    def __init__(self, n_dates: int = 4, missing: tuple = (), grids: tuple = (4, 6), fail: int | None = None) -> None:
        self.calls, self.fail = 0, fail
        self.dates = np.array([T0 + 6 * k * HOUR for k in range(n_dates)], dtype="datetime64[s]")
        self.missing = set(missing)
        self.variables = ["a", "b", "c"]
        self.name_to_index = {name: i for i, name in enumerate(self.variables)}
        self.typed_variables = {"a": SimpleNamespace(param="t", level=850)}
        self.grids = grids
        n = sum(grids)
        self.latitudes, self.longitudes = np.linspace(-80, 80, n), np.linspace(0, 350, n)
        self.shape = (n_dates, 3, 1, n)

    def __getitem__(self, i: int) -> np.ndarray:
        self.calls += 1
        if i == self.fail:
            raise OSError(f"cannot read row {i}")
        values = 100 * i + 10 * np.arange(3)[:, None] + np.arange(self.shape[3])[None, :]
        return values.astype(np.float32)[:, None, :]


def test_member_seed_and_required_dates():
    assert member_seed(0, T0, 1, 2) == member_seed(0, T0, 1, 2)
    seeds = {
        member_seed(0, T0),
        member_seed(0, T0 + 12 * HOUR),
        member_seed(1, T0),
        member_seed(0, T0, 1),
        member_seed(0, T0, 0, 1),
    }
    assert len(seeds) == 5 and all(0 <= seed < 2**63 for seed in seeds)
    assert required_dates(T0, 6 * HOUR, 2) == [T0 - 6 * HOUR, T0]
    assert required_dates(T0, 6 * HOUR, 1, 6 * HOUR) == [T0]
    required = required_dates(T0, 6 * HOUR, 2, 18 * HOUR)
    assert required == [T0 - 6 * HOUR, T0, T0 + 6 * HOUR, T0 + 12 * HOUR]
    assert required_dates(T0, 6 * HOUR, 2, 18 * HOUR, 6 * HOUR) == required
    for lead in (9 * HOUR, 12 * HOUR):  # a 6-output hourly model makes two calls and loads forcings for the first only
        assert required_dates(T0, HOUR, 7, lead, 6 * HOUR) == [T0 + k * HOUR for k in range(-6, 7)]
    assert required_dates(T0, HOUR, 7, 6 * HOUR, 6 * HOUR) == [T0 + k * HOUR for k in range(-6, 1)]
    available = {T0, T0 + 6 * HOUR, T0 + 12 * HOUR}
    assert missing_dates(required, available, {T0 + 6 * HOUR}) == [T0 - 6 * HOUR, T0 + 6 * HOUR]


def test_dataset_targets():
    dataset = FakeDataset(missing=(2,))
    targets = DatasetTargets.from_dataset(dataset, kwargs={"dataset": "fake", "start": 2024})
    assert targets.grid.n == 10 and targets.variables == ["a", "b", "c"]
    frame = targets.frame(T0 + 6 * HOUR, ["c", "a"], CPU)
    assert frame.shape == (1, 2, 10) and frame.dtype == torch.float32
    np.testing.assert_array_equal(frame[0].numpy(), np.stack([120 + np.arange(10), 100 + np.arange(10)]))
    with pytest.raises(MissingTargetError):
        targets.frame(T0 + 3 * HOUR, ["a"], CPU)
    with pytest.raises(MissingTargetError):
        targets.frame(T0 + 12 * HOUR, ["a"], CPU)
    with pytest.raises(ValueError):
        targets.frame(T0, ["nope"], CPU)
    assert targets.grid_mask(1).tolist() == [False] * 4 + [True] * 6
    with pytest.raises(ValueError):
        targets.grid_mask(2)
    block = TargetsConfig.model_validate(targets.to_config()).anemoi_dataset
    assert block.open_dataset_kwargs() == {"dataset": "fake", "start": 2024} and (
        block.prefetch,
        block.cache_bytes,
    ) == (None, 0)
    targets.prefetch_hint(6)
    assert targets.lookahead == 6 and "prefetch" not in targets.to_config()["anemoi_dataset"]
    assert targets.variable_info("a") == {"param": "t", "level": 850}
    assert targets.variable_info("b") == {"param": None, "level": None}
    reduced = DatasetTargets.from_dataset(dataset, grid_indices=np.array([0, 5, 9]))
    assert reduced.grid.n == 3 and reduced.grid_mask(0).tolist() == [True, False, False]
    assert reduced.frame(T0, ["b"], CPU)[0, 0].tolist() == [10.0, 15.0, 19.0]

    dataset = FakeDataset(missing=(2,), fail=3)
    ahead = DatasetTargets.from_dataset(dataset, prefetch=2, cache_bytes=100)
    ahead.prefetch([T0 + 6 * k * HOUR for k in range(4)], ["c", "a"])
    first = ahead.frame(T0, ["c", "a"], CPU)
    assert torch.equal(first, targets.frame(T0, ["c", "a"], CPU))
    assert torch.equal(ahead.frame(T0, ["c", "a"], CPU), first) and ahead.stats["cache_hits"] == 1
    ahead.frame(T0 + 6 * HOUR, ["c", "a"], CPU)
    with pytest.raises(MissingTargetError):
        ahead.frame(T0 + 12 * HOUR, ["c", "a"], CPU)
    with pytest.raises(OSError):
        ahead.frame(T0 + 18 * HOUR, ["c", "a"], CPU)
    assert dataset.calls == 3 and ahead.stats["reads"] == 2 and ahead.stats["read_seconds"] > 0
    assert torch.equal(ahead.frame(T0, ["c", "a"], CPU), first) and dataset.calls == 4  # evicted by the 100-byte budget
    ahead.prefetch_hint(6)
    assert ahead.lookahead == 2 and ahead.to_config()["anemoi_dataset"] == {"prefetch": 2, "cache_bytes": 100}
    assert TargetsConfig.model_validate({"anemoi_dataset": {"cache_bytes": "2GiB"}}).anemoi_dataset.cache_bytes == 2**31
    ahead.close()
    ahead.close()
