"""Metrics are functions of mean statistics; names map to YAML specs."""

from __future__ import annotations

import torch

from anemoi.evaluation import statistics as st


class Metric:
    """Base class: `statistics` it needs by name, `from_means` maps their means to the metric."""

    name: str
    statistics: dict[str, st.Statistic]

    @property
    def spec(self) -> str | dict:
        """YAML form: the name, or `{name: kwargs}` for parametrised metrics."""
        return self.name

    @property
    def min_members(self) -> int:
        """Smallest ensemble size the statistics accept."""
        return max(s.min_members for s in self.statistics.values())

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        """Metric from the weighted means of its statistics; `members` is the ensemble size of the run."""
        raise NotImplementedError


class Bias(Metric):
    """Mean error of the ensemble mean."""

    name = "bias"

    def __init__(self) -> None:
        self.statistics = {"error": st.Error()}

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        return means["error"]


class MAE(Metric):
    """Mean absolute error of the ensemble mean."""

    name = "mae"

    def __init__(self) -> None:
        self.statistics = {"absolute_error": st.AbsoluteError()}

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        return means["absolute_error"]


class RMSE(Metric):
    """Root mean squared error of the ensemble mean."""

    name = "rmse"

    def __init__(self) -> None:
        self.statistics = {"squared_error": st.SquaredError()}

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        return means["squared_error"].sqrt()


class CRPS(Metric):
    """Mean kernel CRPS from the `skill` and `pairs` means; `crps` for alpha 0, `fair_crps` for alpha 1."""

    def __init__(self, alpha: float = 0.0) -> None:
        st.crps_coefficient(alpha, 2)
        self.alpha = alpha
        self.statistics = {"skill": st.Skill(), "pairs": st.Pairs()}
        self.name = {0.0: "crps", 1.0: "fair_crps"}.get(alpha, f"crps_{alpha:g}".replace(".", "p"))

    @property
    def spec(self) -> str | dict:
        return self.name if self.alpha in (0.0, 1.0) else {"crps": {"alpha": self.alpha}}

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        return means["skill"] - st.crps_coefficient(self.alpha, members) * means["pairs"]


class FairCRPS(CRPS):
    """Mean fair CRPS (alpha 1)."""

    def __init__(self) -> None:
        super().__init__(alpha=1.0)


class Spread(Metric):
    """Root mean ensemble variance."""

    name = "spread"

    def __init__(self) -> None:
        self.statistics = {"ensemble_variance": st.EnsembleVariance()}

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        return means["ensemble_variance"].sqrt()


class SpreadSkill(Metric):
    """Spread divided by RMSE."""

    name = "spread_skill"

    def __init__(self) -> None:
        self.statistics = {"ensemble_variance": st.EnsembleVariance(), "squared_error": st.SquaredError()}

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        return (means["ensemble_variance"] / means["squared_error"]).sqrt()


class MemberRMSE(Metric):
    """Root mean member squared error: in expectation the RMSE of a randomly drawn member, never below `rmse`."""

    name = "member_rmse"

    def __init__(self) -> None:
        self.statistics = {"member_squared_error": st.MemberSquaredError()}

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        return means["member_squared_error"].sqrt()


class MemberMAE(Metric):
    """Mean absolute member error (the CRPS `skill` term): in expectation the MAE of a randomly drawn member."""

    name = "member_mae"

    def __init__(self) -> None:
        self.statistics = {"skill": st.Skill()}

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        return means["skill"]


class ACC(Metric):
    """Anomaly correlation of the ensemble mean against a climatology, pooled over nodes and init times within a bin:
    mean(fa * ta) / sqrt(mean(fa^2) * mean(ta^2)).

    Per-init or per-season values come from the `init_time` / `season` binning (each bin pooled), not from averaging
    per-init correlations. Nodes where the climatology is non-finite contribute zero to the three anomaly sums while the
    shared weights still count them, so only this ratio is meaningful, not the raw means of the anomaly statistics.
    """

    name = "acc"

    def __init__(self) -> None:
        self.statistics = {
            "anomaly_product": st.AnomalyProduct(),
            "forecast_anomaly_squared": st.ForecastAnomalySquared(),
            "target_anomaly_squared": st.TargetAnomalySquared(),
        }

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        return means["anomaly_product"] / (means["forecast_anomaly_squared"] * means["target_anomaly_squared"]).sqrt()


REGISTRY: dict[str, type[Metric]] = {
    "bias": Bias,
    "mae": MAE,
    "rmse": RMSE,
    "crps": CRPS,
    "fair_crps": FairCRPS,
    "spread": Spread,
    "spread_skill": SpreadSkill,
    "member_rmse": MemberRMSE,
    "member_mae": MemberMAE,
    "acc": ACC,
}


def from_spec(spec: str | dict | Metric) -> Metric:
    """Metric from a name, a `{name: kwargs}` dict, or an instance."""
    if isinstance(spec, Metric):
        return spec
    name, kwargs = spec, {}
    if isinstance(spec, dict):
        if len(spec) != 1:
            raise ValueError(f"metric spec must have exactly one key, got {spec}")
        ((name, kwargs),) = spec.items()
    if name not in REGISTRY:
        raise ValueError(f"unknown metric {name!r}, expected one of {sorted(REGISTRY)}")
    return REGISTRY[name](**(kwargs or {}))


def unique_statistics(metrics: list[Metric]) -> dict[str, st.Statistic]:
    """Statistics needed by `metrics`, deduplicated by name."""
    result: dict[str, st.Statistic] = {}
    for metric in metrics:
        result.update(metric.statistics)
    return result


def required_aux(metrics: list[Metric]) -> set[str]:
    """Names of the per-frame `aux` fields the statistics of `metrics` need (e.g. `climatology`)."""
    return {name for statistic in unique_statistics(metrics).values() for name in statistic.aux}


def min_members(metrics: list[Metric]) -> int:
    """Smallest ensemble size that all `metrics` accept."""
    return max((metric.min_members for metric in metrics), default=1)
