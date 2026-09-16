import datetime

import numpy as np
import pytest

from anemoi.evaluation.frame import Grid
from anemoi.evaluation.sources.fake import ArrayTargets
from anemoi.evaluation.sources.fake import FakeForecastSource

HOUR = datetime.timedelta(hours=1)
T0 = datetime.datetime(2024, 1, 1)


def make_sources(
    n: int = 30,
    members: int = 3,
    drift: float = 0.5,
    spread: float = 0.2,
    frames_per_pass: int = 1,
    missing: tuple[datetime.datetime, ...] = (),
    seed: int = 0,
) -> tuple[FakeForecastSource, ArrayTargets]:
    """Constant-in-time targets on a random grid, so the forecast error at step k is exactly k * drift."""
    rng = np.random.default_rng(seed)
    grid = Grid(rng.uniform(-90, 90, n), rng.uniform(0, 360, n))
    field = (rng.integers(-8, 9, (2, n)) / 4).astype(np.float32)
    dates = [T0 + k * 6 * HOUR for k in range(7)]
    targets = ArrayTargets(grid, ["t", "q"], {d: field for d in dates if d not in missing})
    forecast = FakeForecastSource(targets, 6 * HOUR, members, drift, spread, frames_per_pass, device="cpu")
    return forecast, targets


@pytest.fixture
def fake_sources():
    """Factory of (FakeForecastSource, ArrayTargets) pairs; see `make_sources`."""
    return make_sources
