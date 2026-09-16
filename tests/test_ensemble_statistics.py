import numpy as np
import pytest
import torch

from anemoi.evaluation.metrics import CRPS
from anemoi.evaluation.statistics import EnsembleVariance
from anemoi.evaluation.statistics import Error
from anemoi.evaluation.statistics import MemberSquaredError
from anemoi.evaluation.statistics import Pairs
from anemoi.evaluation.statistics import Skill
from anemoi.evaluation.statistics import SquaredError


def naive_crps(pred: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    """Pairwise loop of anemoi-training's kcrps.py naive backend, members on axis 0."""
    members = pred.shape[0]
    mae = np.abs(target[None] - pred).mean(0)
    coefficient = -(alpha / (members * (members - 1)) + (1.0 - alpha) / members**2)
    ens_var = np.zeros_like(mae)
    for i in range(members - 1):
        ens_var += np.abs(pred[i][None] - pred[i + 1 :]).sum(0)
    return mae + coefficient * ens_var


def crps_from_primitives(pred: torch.Tensor, target: torch.Tensor, alpha: float) -> np.ndarray:
    """Per-node CRPS the way the framework derives it: `from_means` on the per-node skill and pairs."""
    means = {"skill": Skill().compute(pred, target), "pairs": Pairs().compute(pred, target)}
    return CRPS(alpha).from_means(means, pred.shape[0]).numpy()


@pytest.mark.parametrize("members", [2, 3, 5])
@pytest.mark.parametrize("alpha", [0.0, 0.95, 1.0])
def test_crps_matches_naive_pairwise(members, alpha):
    rng = np.random.default_rng(members)
    pred, target = rng.standard_normal((members, 2, 7)), rng.standard_normal((1, 2, 7))
    expected = naive_crps(pred, target[0], alpha)
    got = crps_from_primitives(torch.from_numpy(pred), torch.from_numpy(target), alpha)
    np.testing.assert_allclose(got, expected, rtol=1e-12)
    got32 = crps_from_primitives(
        torch.from_numpy(pred.astype(np.float32)), torch.from_numpy(target.astype(np.float32)), alpha
    )
    np.testing.assert_allclose(got32, expected, rtol=1e-5, atol=1e-6)


def test_ensemble_variance_aux_and_member_requirement():
    rng = np.random.default_rng(1)
    pred, target = rng.standard_normal((4, 2, 7)), rng.standard_normal((1, 2, 7))
    pred_t, target_t = torch.from_numpy(pred), torch.from_numpy(target)
    got = EnsembleVariance().compute(pred_t, target_t).numpy()
    np.testing.assert_allclose(got, pred.var(0, ddof=1), rtol=1e-12)
    mean = torch.full((2, 7), 3.0, dtype=torch.float64)
    assert torch.equal(Error().compute(pred_t, target_t, {"ensemble_mean": mean}), mean - target_t[0])
    np.testing.assert_allclose(Skill().compute(pred_t[:1], target_t).numpy(), np.abs(pred[0] - target[0]))
    for statistic in (Pairs(), EnsembleVariance()):
        with pytest.raises(ValueError):
            statistic.compute(pred_t[:1], target_t)
    with pytest.raises(ValueError):
        CRPS(alpha=1.5)


def test_member_squared_error_identity():
    rng = np.random.default_rng(4)
    for members in (2, 5):
        pred, target = rng.standard_normal((members, 2, 7)), rng.standard_normal((1, 2, 7))
        pred_t, target_t = torch.from_numpy(pred), torch.from_numpy(target)
        got = MemberSquaredError().compute(pred_t, target_t).numpy()
        np.testing.assert_allclose(got, ((pred - target) ** 2).mean(0), rtol=1e-12)
        identity = (pred.mean(0) - target[0]) ** 2 + (members - 1) / members * pred.var(0, ddof=1)
        np.testing.assert_allclose(got, identity, rtol=1e-12)
    single = MemberSquaredError().compute(pred_t[:1], target_t)
    np.testing.assert_allclose(single.numpy(), SquaredError().compute(pred_t[:1], target_t).numpy(), rtol=1e-15)


def test_large_offset_small_spread_in_float32():
    rng = np.random.default_rng(2)
    pred = (1e5 + 0.2 * rng.standard_normal((4, 2, 7))).astype(np.float32)
    target = (1e5 + 0.3 * rng.standard_normal((1, 2, 7))).astype(np.float32)
    pred64, target64 = pred.astype(np.float64), target.astype(np.float64)
    got = EnsembleVariance().compute(torch.from_numpy(pred), torch.from_numpy(target)).numpy()
    np.testing.assert_allclose(got, pred64.var(0, ddof=1), rtol=1e-6)
    for alpha in (0.0, 1.0):
        got = crps_from_primitives(torch.from_numpy(pred), torch.from_numpy(target), alpha)
        np.testing.assert_allclose(got, naive_crps(pred64, target64[0], alpha), rtol=1e-6)
