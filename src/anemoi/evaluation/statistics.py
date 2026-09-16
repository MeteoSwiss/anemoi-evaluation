"""Per-element statistics: (M, V, N) prediction x (1, V, N) target -> (V, N) float64.

Members are reduced in float64 without an (M, V, N) float64 copy: the mean accumulates in float64 and
the ensemble statistics work one variable at a time, so large offsets (pressure in Pa, temperature in K)
with a small spread do not cancel in float32. The ensemble statistics are the WeatherBench-X primitives
(`skill`, `pairs`, `ensemble_variance`); CRPS for any alpha and the spread are derived from their means. The anomaly
statistics need a climatology in `aux` (declared by `Statistic.aux`) and zero their own contribution where it is
non-finite, so those nodes leave the anomaly correlation only.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def ensemble_mean(pred: torch.Tensor, aux: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
    """Member mean in float64, taken from `aux` when the aggregator already computed it for the frame."""
    if aux and "ensemble_mean" in aux:
        return aux["ensemble_mean"]
    return pred.mean(0, dtype=torch.float64)


def per_variable(pred: torch.Tensor, target: torch.Tensor, function: Callable) -> torch.Tensor:
    """Apply `function((M, N) float64 members, (N,) float64 target)` to each variable and stack the results."""
    return torch.stack([function(pred[:, j].double(), target[j].double()) for j in range(pred.shape[1])])


def anomalies(
    mean: torch.Tensor, target: torch.Tensor, climatology: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ensemble-mean and target anomalies from the (V, N) climatology in float64, one variable at a time, and the mask
    of finite climatology nodes."""
    forecast, truth = torch.empty_like(mean), torch.empty_like(mean)
    for j in range(mean.shape[0]):
        reference = climatology[j].double()
        forecast[j] = mean[j] - reference
        truth[j] = target[j].double() - reference
    return forecast, truth, torch.isfinite(climatology)


def climatology_anomalies(
    pred: torch.Tensor, target: torch.Tensor, aux: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The anomalies the aggregator put in `aux`, or computed here from `aux["climatology"]` when called standalone."""
    if "forecast_anomaly" in aux:
        return aux["forecast_anomaly"], aux["target_anomaly"], aux["climatology_finite"]
    if "climatology" not in aux:
        raise ValueError("the anomaly statistics need a climatology in aux")
    return anomalies(ensemble_mean(pred, aux), target, aux["climatology"])


class Statistic:
    """Base class; `compute` receives the raw tensors, the aggregator masks non-finite elements afterwards."""

    name: str
    min_members: int = 1
    aux: tuple[str, ...] = ()

    def compute(
        self, pred: torch.Tensor, target: torch.Tensor, aux: dict[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        """Statistic per variable and node; `aux` carries per-frame fields such as the ensemble mean."""
        if pred.shape[0] < self.min_members:
            raise ValueError(f"{self.name} needs at least {self.min_members} members, got {pred.shape[0]}")
        return self._compute(pred, target[0], aux or {})

    def _compute(self, pred: torch.Tensor, target: torch.Tensor, aux: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class Error(Statistic):
    """Ensemble mean minus target."""

    name = "error"

    def _compute(self, pred: torch.Tensor, target: torch.Tensor, aux: dict[str, torch.Tensor]) -> torch.Tensor:
        return ensemble_mean(pred, aux) - target


class SquaredError(Statistic):
    """Squared error of the ensemble mean."""

    name = "squared_error"

    def _compute(self, pred: torch.Tensor, target: torch.Tensor, aux: dict[str, torch.Tensor]) -> torch.Tensor:
        return (ensemble_mean(pred, aux) - target) ** 2


class AbsoluteError(Statistic):
    """Absolute error of the ensemble mean."""

    name = "absolute_error"

    def _compute(self, pred: torch.Tensor, target: torch.Tensor, aux: dict[str, torch.Tensor]) -> torch.Tensor:
        return (ensemble_mean(pred, aux) - target).abs()


class EnsembleVariance(Statistic):
    """Unbiased variance over members."""

    name = "ensemble_variance"
    min_members = 2

    def _compute(self, pred: torch.Tensor, target: torch.Tensor, aux: dict[str, torch.Tensor]) -> torch.Tensor:
        return per_variable(pred, target, lambda members, _: members.var(0, unbiased=True))


class Skill(Statistic):
    """Mean absolute member error, the first CRPS term."""

    name = "skill"

    def _compute(self, pred: torch.Tensor, target: torch.Tensor, aux: dict[str, torch.Tensor]) -> torch.Tensor:
        return per_variable(pred, target, lambda members, truth: (members - truth).abs().mean(0))


class Pairs(Statistic):
    """Sum over member pairs of |m_i - m_j|, the second CRPS term, from the sorted centred members."""

    name = "pairs"
    min_members = 2

    def _compute(self, pred: torch.Tensor, target: torch.Tensor, aux: dict[str, torch.Tensor]) -> torch.Tensor:
        members = pred.shape[0]
        k = torch.arange(1, members + 1, dtype=torch.float64, device=pred.device)[:, None]
        weights = 2 * k - members - 1
        return per_variable(pred, target, lambda values, truth: (weights * (values - truth).sort(dim=0).values).sum(0))


class MemberSquaredError(Statistic):
    """Squared error averaged over members, mean_m (m - y)^2: the squared error of the ensemble mean plus (M - 1) / M
    times the unbiased ensemble variance; equals `squared_error` for one member."""

    name = "member_squared_error"

    def _compute(self, pred: torch.Tensor, target: torch.Tensor, aux: dict[str, torch.Tensor]) -> torch.Tensor:
        return per_variable(pred, target, lambda members, truth: ((members - truth) ** 2).mean(0))


class AnomalyProduct(Statistic):
    """Product of the ensemble-mean and target anomalies from the climatology; zero where the climatology is not finite."""

    name = "anomaly_product"
    aux = ("climatology",)

    def _compute(self, pred: torch.Tensor, target: torch.Tensor, aux: dict[str, torch.Tensor]) -> torch.Tensor:
        forecast, truth, finite = climatology_anomalies(pred, target, aux)
        return torch.where(finite, forecast * truth, 0.0)


class ForecastAnomalySquared(Statistic):
    """Squared anomaly of the ensemble mean from the climatology; zero where the climatology is not finite."""

    name = "forecast_anomaly_squared"
    aux = ("climatology",)

    def _compute(self, pred: torch.Tensor, target: torch.Tensor, aux: dict[str, torch.Tensor]) -> torch.Tensor:
        forecast, _, finite = climatology_anomalies(pred, target, aux)
        return torch.where(finite, forecast * forecast, 0.0)


class TargetAnomalySquared(Statistic):
    """Squared anomaly of the target from the climatology; zero where the climatology is not finite."""

    name = "target_anomaly_squared"
    aux = ("climatology",)

    def _compute(self, pred: torch.Tensor, target: torch.Tensor, aux: dict[str, torch.Tensor]) -> torch.Tensor:
        _, truth, finite = climatology_anomalies(pred, target, aux)
        return torch.where(finite, truth * truth, 0.0)


def crps_coefficient(alpha: float, members: int) -> float:
    """Weight of the `pairs` term in CRPS_alpha = skill - coefficient * pairs; alpha 0 standard, 1 fair."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return alpha / (members * (members - 1)) + (1.0 - alpha) / members**2
