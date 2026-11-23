"""Metrics are functions of mean statistics; names map to YAML specs."""

from __future__ import annotations

import inspect

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

    @property
    def output_dim(self) -> tuple[str, list] | None:
        """An extra output dimension this metric's values carry, as (name, coordinate values), or None.

        A metric declaring one returns from `from_means` an array with that axis last, and `to_xarray` writes the
        dimension and its coordinate; the stored statistics keep the four axes of the state.
        """
        return None

    def bind_members(self, members: int) -> None:
        """Tell the metric the ensemble size of the run, so it can create the statistics that depend on it; the
        metrics whose statistics are fixed at construction do not care."""

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
    """Mean absolute error of the ensemble median."""

    name = "mae"

    def __init__(self) -> None:
        self.statistics = {"absolute_median_error": st.AbsoluteMedianError()}

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        return means["absolute_median_error"]


class RMSE(Metric):
    """Root mean squared error of the ensemble mean."""

    name = "rmse"

    def __init__(self) -> None:
        self.statistics = {"squared_error": st.SquaredError()}

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        return means["squared_error"].sqrt()


class ErrorStd(Metric):
    """Standard deviation of the ensemble-mean error over the cell, `sqrt(mse - bias^2)`: the RMSE with the bias
    removed, so `rmse^2 = bias^2 + error_std^2`.

    The difference is clamped at zero before the root: where the error is constant over the cell the two means agree
    to rounding and could otherwise give a negative variance, hence NaN. A NaN mean stays NaN. The one-pass form
    loses about `(bias / error_std)^2 * eps` of relative accuracy, so it reads 0 where the bias dwarfs the spread by
    eight orders of magnitude; real forecast errors are far from that.
    """

    name = "error_std"

    def __init__(self) -> None:
        self.statistics = {"error": st.Error(), "squared_error": st.SquaredError()}

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        return (means["squared_error"] - means["error"] ** 2).clamp(min=0.0).sqrt()


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
    """Spread divided by the unbiased RMSE of the ensemble mean, `sqrt(var / (mse - var / M))`.

    For `M` members drawn from the predictive distribution the squared error of the ensemble mean exceeds the
    predictive variance by a factor `1 + 1 / M`; subtracting `ensemble_variance / M` (linear, so it applies to the
    means) removes that bias, and a calibrated ensemble gives 1 for every `M` (WeatherBench-X
    `UnbiasedSpreadSkillRatio`). Where the pooled `mse - var / M` is not positive, which happens with few samples or a
    strongly over-dispersive ensemble, the value is NaN, or inf for exactly zero.
    """

    name = "spread_skill"

    def __init__(self) -> None:
        self.statistics = {"ensemble_variance": st.EnsembleVariance(), "squared_error": st.SquaredError()}

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        unbiased_mse = means["squared_error"] - means["ensemble_variance"] / members
        return (means["ensemble_variance"] / unbiased_mse).sqrt()


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


class ActivityRatio(Metric):
    """Activity of the ensemble-mean anomalies against the observed ones, sqrt(mean(fa^2) / mean(ta^2)), pooled like
    `acc` and needing the same climatology. Below 1 the forecast is smoother than the truth.

    As for `acc`, the nodes where the climatology is non-finite contribute zero to both sums while the shared weights
    still count them, which the ratio cancels; the raw means of the two statistics are not meaningful on their own.
    """

    name = "activity_ratio"

    def __init__(self) -> None:
        self.statistics = {
            "forecast_anomaly_squared": st.ForecastAnomalySquared(),
            "target_anomaly_squared": st.TargetAnomalySquared(),
        }

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        return (means["forecast_anomaly_squared"] / means["target_anomaly_squared"]).sqrt()


class LabelledMetric:
    """Mixin for a metric named `<kind>_<label>` by a user label and a per-variable threshold map.

    It carries nothing but the label, the name and the spec, so the families that build their statistics at
    construction and the families that build them once the ensemble size is known cannot drift apart on the one thing
    the result file records. It inherits from nothing, so a metric mixing it in front of `Metric` or of one of its
    subclasses linearises to a chain ending in `Metric`. The label and the map exist from construction, before any
    binding, which is what lets `build` compare them across families.
    """

    kind: str
    label: str
    thresholds: dict[str, float]

    def _set_thresholds(self, label: str | None, thresholds: dict[str, float] | None) -> None:
        """Validate the label and the map and take the metric's name from them."""
        self.label, self.thresholds = st.validate_thresholds(label, thresholds)
        self.name = f"{self.kind}_{self.label}"

    @property
    def spec(self) -> str | dict:
        return {self.kind: {"label": self.label, "thresholds": dict(self.thresholds)}}


class ThresholdMetric(LabelledMetric, Metric):
    """Base class of the scores of one binarised event: a label and a per-variable threshold map, named `<kind>_<label>`.

    The arguments are validated by `statistics.validate_thresholds`, so a missing or malformed one is a `ValueError`
    and reaches the config layer as a validation error. Metrics sharing a kind of statistic and a label share the
    stored sums, and every metric carrying a label must give it the same map (`build`).
    """

    needs: tuple[type[st.ThresholdStatistic], ...]

    def __init__(self, label: str | None = None, thresholds: dict[str, float] | None = None) -> None:
        self._set_thresholds(label, thresholds)
        built = [statistic(label=self.label, thresholds=self.thresholds) for statistic in self.needs]
        self.statistics = {statistic.name: statistic for statistic in built}

    def _mean(self, means: dict[str, torch.Tensor], kind: str) -> torch.Tensor:
        """The mean of one of this label's statistics."""
        return means[f"{kind}_{self.label}"]

    def _cells(self, means: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The four contingency cells `h, m, fa, cn`; the fourth from `h + m + fa + cn == 1`."""
        hit, miss = self._mean(means, "hit"), self._mean(means, "miss")
        false_alarm = self._mean(means, "false_alarm")
        return hit, miss, false_alarm, 1.0 - hit - miss - false_alarm

    def _stored_cells(
        self, means: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The four contingency cells, each from its own stored sum, so an empty cell is an exact zero.

        The remainder of `_cells` is a rounding residue of order 1e-16 where the true `cn` is zero (a forecast that
        says yes everywhere in the cell), which a score dividing by `m fa` or taking `log(1 - F)` turns into an
        infinity of either sign or a spurious 1, and a sharded run into a different value than the unsharded one.
        The scores that existed before `correct_negative` keep `_cells`, so their values are unchanged.
        """
        hit, miss = self._mean(means, "hit"), self._mean(means, "miss")
        return hit, miss, self._mean(means, "false_alarm"), self._mean(means, "correct_negative")


class POD(ThresholdMetric):
    """Probability of detection, the fraction of observed events that were forecast."""

    kind = "pod"
    needs = (st.Hit, st.Miss)

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        hit, miss = self._mean(means, "hit"), self._mean(means, "miss")
        return hit / (hit + miss)


class FAR(ThresholdMetric):
    """False alarm ratio, the fraction of forecast events that did not happen."""

    kind = "far"
    needs = (st.Hit, st.FalseAlarm)

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        hit, false_alarm = self._mean(means, "hit"), self._mean(means, "false_alarm")
        return false_alarm / (hit + false_alarm)


class CSI(ThresholdMetric):
    """Critical success index, hits over everything forecast or observed."""

    kind = "csi"
    needs = (st.Hit, st.Miss, st.FalseAlarm)

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        hit, miss, false_alarm, _ = self._cells(means)
        return hit / (hit + miss + false_alarm)


class ETS(ThresholdMetric):
    """Equitable threat score, the critical success index with the hits of a random forecast removed."""

    kind = "ets"
    needs = (st.Hit, st.Miss, st.FalseAlarm)

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        hit, miss, false_alarm, _ = self._cells(means)
        random_hits = (hit + false_alarm) * (hit + miss)
        return (hit - random_hits) / (hit + miss + false_alarm - random_hits)


class FrequencyBias(ThresholdMetric):
    """Forecast events over observed events: above 1 the model over-forecasts the event."""

    kind = "frequency_bias"
    needs = (st.Hit, st.Miss, st.FalseAlarm)

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        hit, miss, false_alarm, _ = self._cells(means)
        return (hit + false_alarm) / (hit + miss)


class HSS(ThresholdMetric):
    """Heidke skill score, the accuracy against the accuracy of a random forecast with the same margins."""

    kind = "hss"
    needs = (st.Hit, st.Miss, st.FalseAlarm)

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        hit, miss, false_alarm, negative = self._cells(means)
        numerator = 2.0 * (hit * negative - miss * false_alarm)
        return numerator / ((hit + miss) * (miss + negative) + (hit + false_alarm) * (false_alarm + negative))


class PSS(ThresholdMetric):
    """Peirce skill score, the hit rate minus the false alarm rate; close to `pod` for rare events."""

    kind = "pss"
    needs = (st.Hit, st.Miss, st.FalseAlarm)

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        hit, miss, false_alarm, negative = self._cells(means)
        return hit / (hit + miss) - false_alarm / (false_alarm + negative)


class POFD(ThresholdMetric):
    """Probability of false detection (the false alarm rate), the fraction of observed non-events that were forecast;
    not to be confused with `far`, the false alarm ratio."""

    kind = "pofd"
    needs = (st.FalseAlarm, st.CorrectNegative)

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        false_alarm, negative = self._mean(means, "false_alarm"), self._mean(means, "correct_negative")
        return false_alarm / (false_alarm + negative)


class SEDI(ThresholdMetric):
    """Symmetric extremal dependence index of Ferro and Stephenson (2011), from the hit rate `H` and the false alarm
    rate `F`: `(ln F - ln H - ln(1 - F) + ln(1 - H)) / (ln F + ln H + ln(1 - F) + ln(1 - H))`.

    It does not degenerate to 0 or 1 for rare events, which is what it is for. It is undefined where `H` or `F` is 0
    or 1, a perfect forecast included, and the logarithms then give NaN.
    """

    kind = "sedi"
    needs = (st.Hit, st.Miss, st.FalseAlarm, st.CorrectNegative)

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        hit, miss, false_alarm, negative = self._stored_cells(means)
        hit_rate, false_alarm_rate = hit / (hit + miss), false_alarm / (false_alarm + negative)
        log_h, log_f = hit_rate.log(), false_alarm_rate.log()
        log_not_h, log_not_f = (1.0 - hit_rate).log(), (1.0 - false_alarm_rate).log()
        return (log_f - log_h - log_not_f + log_not_h) / (log_f + log_h + log_not_f + log_not_h)


class OddsRatio(ThresholdMetric):
    """Odds ratio `(h cn) / (m fa)`, the odds of a hit over the odds of a false alarm; infinite where there is no miss
    or no false alarm but a hit and a correct negative, NaN where both products are zero."""

    kind = "odds_ratio"
    needs = (st.Hit, st.Miss, st.FalseAlarm, st.CorrectNegative)

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        hit, miss, false_alarm, negative = self._stored_cells(means)
        return (hit * negative) / (miss * false_alarm)


class ORSS(ThresholdMetric):
    """Odds ratio skill score (Yule's Q), `(h cn - m fa) / (h cn + m fa)`, the odds ratio mapped to [-1, 1]."""

    kind = "orss"
    needs = (st.Hit, st.Miss, st.FalseAlarm, st.CorrectNegative)

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        hit, miss, false_alarm, negative = self._stored_cells(means)
        return (hit * negative - miss * false_alarm) / (hit * negative + miss * false_alarm)


class F1(ThresholdMetric):
    """F1 score `2 h / (2 h + m + fa)`, the harmonic mean of `pod` and the success ratio `1 - far`."""

    kind = "f1"
    needs = (st.Hit, st.Miss, st.FalseAlarm)

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        hit, miss, false_alarm = (self._mean(means, kind) for kind in ("hit", "miss", "false_alarm"))
        return 2.0 * hit / (2.0 * hit + miss + false_alarm)


class BrierScore(ThresholdMetric):
    """Brier score of the ensemble exceedance fraction; at one member it is `miss + false_alarm`."""

    kind = "brier"
    needs = (st.Brier,)

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        return self._mean(means, "brier")


class BSS(ThresholdMetric):
    """Brier skill score against the base rate of the same cell, so it does not compare across cells."""

    kind = "bss"
    needs = (st.Brier, st.EventFrequency)

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        base_rate = self._mean(means, "event_frequency")
        return 1.0 - self._mean(means, "brier") / (base_rate * (1.0 - base_rate))


class EventFrequency(ThresholdMetric):
    """Base rate: how often the event was observed, the number that says whether the threshold is scorable."""

    kind = "event_frequency"
    needs = (st.EventFrequency,)

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        return self._mean(means, "event_frequency")


class EnsembleSizeMetric(Metric):
    """Base class of the metrics whose statistics depend on the run's ensemble size (the rank bins, the
    reliability levels).

    The metric is a valid object before it learns that size, so the config layer can validate a spec naming it; it
    gains its statistics in `bind_members`, called by `build` from the evaluation and from the merge CLI. Rebinding to
    a different size would change the statistics, the coordinate and every stored sum under an evaluation still
    holding the instance, and raises.
    """

    def __init__(self) -> None:
        self.members: int | None = None
        self.statistics: dict[str, st.Statistic] = {}

    needs_members: int = 2

    @property
    def min_members(self) -> int:
        """The ensemble size this family needs, whether or not the metric is bound; the base class would look at an
        empty statistics dict."""
        return self.needs_members

    def bind_members(self, members: int) -> None:
        if self.members is not None:
            if self.members != members:
                raise ValueError(f"{self.name} is already bound to {self.members} members, cannot rebind to {members}")
            return
        if members < self.min_members:
            raise ValueError(f"{self.name} needs at least {self.min_members} members, the run has {members}")
        self.members = members
        self.statistics = self._build_statistics(members)

    def _build_statistics(self, members: int) -> dict[str, st.Statistic]:
        raise NotImplementedError

    def _bound(self, members: int) -> int:
        """The ensemble size the metric was bound to, checked against the one the state carries."""
        if self.members is None:
            raise ValueError(f"{self.name} was not bound to the run's ensemble size")
        if members != self.members:
            raise ValueError(f"{self.name} is bound to {self.members} members but the state has {members}")
        return self.members


class RankHistogram(EnsembleSizeMetric):
    """Rank histogram: the `M + 1` mean rank bins, stacked on a `rank` axis.

    Flat is calibrated, a U is under-dispersed, a slope is a bias. At lead 0 every member equals every other one, so
    the target is either tied with all of them (the mass spread evenly, a flat histogram, the case when the targets
    come from the model's input dataset) or outside all of them (the whole mass in bin 0 or bin M); either way it says
    nothing about calibration.
    """

    name = "rank_histogram"

    @property
    def output_dim(self) -> tuple[str, list] | None:
        """The `rank` axis, `0..M`, once the ensemble size is known."""
        return None if self.members is None else ("rank", list(range(self.members + 1)))

    def _build_statistics(self, members: int) -> dict[str, st.Statistic]:
        return {f"rank_bin_{k}": st.RankBin(k, members) for k in range(members + 1)}

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        bound = self._bound(members)
        return torch.stack([means[f"rank_bin_{k}"] for k in range(bound + 1)], dim=-1)


class OutlierFraction(EnsembleSizeMetric):
    """Weighted fraction of targets that fell outside the ensemble range, the two end bins of the rank histogram.

    A calibrated `M`-member ensemble gives `2 / (M + 1)`, an under-dispersed one more, an over-dispersed one less.
    """

    name = "outlier_fraction"

    def _build_statistics(self, members: int) -> dict[str, st.Statistic]:
        return {name: st.RankBin(k, members) for k, name in ((0, "rank_bin_0"), (members, f"rank_bin_{members}"))}

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        bound = self._bound(members)
        return means["rank_bin_0"] + means[f"rank_bin_{bound}"]


class ReliabilityMetric(LabelledMetric, EnsembleSizeMetric):
    """Base of the level-resolved scores of one binarised event: a label, a threshold map and the run's ensemble size.

    An `M`-member exceedance fraction takes only the `M + 1` values `k / M`, so those levels are the bins of the
    reliability diagram and nothing is configured. Each level stores how much weight it carried and how much of it was
    observed, which is enough for the diagram, for the forecast-probability distribution under it and for the
    decomposition `BS = REL - RES + UNC`; a coarser diagram is a sum of levels and needs no further statistics. The
    family is legal at one member, where the two levels are the contingency table of a deterministic forecast.
    """

    needs_members = 1
    needs_events: bool = True

    def __init__(self, label: str | None = None, thresholds: dict[str, float] | None = None) -> None:
        EnsembleSizeMetric.__init__(self)
        self._set_thresholds(label, thresholds)

    @property
    def probability_axis(self) -> tuple[str, list] | None:
        """The `probability` axis, `k / M` for `k` in `0..M`, once the ensemble size is known."""
        if self.members is None:
            return None
        return ("probability", [k / self.members for k in range(self.members + 1)])

    def _build_statistics(self, members: int) -> dict[str, st.Statistic]:
        """The `2 (M + 1)` level statistics, or only the counts for a metric that needs no observed events."""
        built: list[st.Statistic] = [
            st.ReliabilityCount(self.label, self.thresholds, k=k, members=members) for k in range(members + 1)
        ]
        if self.needs_events:
            built += [
                st.ReliabilityEvent(self.label, self.thresholds, k=k, members=members) for k in range(members + 1)
            ]
        return {statistic.name: statistic for statistic in built}

    def _counts(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        """This label's level weights, stacked on a trailing level axis."""
        return torch.stack([means[f"reliability_count_{self.label}_{k}"] for k in range(members + 1)], dim=-1)

    def _events(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        """This label's observed weight within each level, stacked on a trailing level axis."""
        return torch.stack([means[f"reliability_event_{self.label}_{k}"] for k in range(members + 1)], dim=-1)

    def _levels(
        self, means: dict[str, torch.Tensor], members: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The level weights, the observed frequency within each level, the total weight and the base rate, the last
        two from the level sums themselves so that each metric of the family is self-contained."""
        counts, events = self._counts(means, members), self._events(means, members)
        total = counts.sum(-1)
        return counts, events / counts, total, events.sum(-1) / total

    def _probabilities(self, counts: torch.Tensor, members: int) -> torch.Tensor:
        """The forecast probability of each level, `k / M`, shaped to broadcast against the level axis."""
        return torch.arange(members + 1, dtype=counts.dtype, device=counts.device) / members

    def _rates(self, means: dict[str, torch.Tensor], members: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The hit rate and the false alarm rate of the decision "forecast the event when `p >= k / M`", one per level
        `k` on a trailing axis, from the level sums cumulated from the top.

        Level 0 is the decision that always forecasts the event, `(1, 1)` exactly since its cumulated sum is the
        denominator itself. Both rates are 0/0 NaN where the cell observed no event, or no non-event.

        The non-events of a level are the difference `n_k - h_k`, unlike the contingency remainder this is safe: a
        level whose nodes were all observed has identical count and event vectors, so the two sums are bit-identical
        and the difference is an exact zero (a merge adds identical shard sums); otherwise the event values never
        exceed the counts elementwise, the rounded sums keep that order, and the true difference is at least one
        node's weight, far above the rounding.
        """
        counts, events = self._counts(means, members), self._events(means, members)
        hits = events.flip(-1).cumsum(-1).flip(-1)
        alarms = (counts - events).flip(-1).cumsum(-1).flip(-1)
        return hits / hits[..., :1], alarms / alarms[..., :1]

    def _component(self, counts: torch.Tensor, difference: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
        """A weighted mean of squared differences over the levels, skipping the empty ones.

        An empty level has an undefined observed frequency, and `0 * NaN` is NaN, so its term is replaced by zero. The
        test is on the weight, which is NaN rather than zero for a variable the threshold map does not name, so that
        metric stays NaN as every threshold score does.
        """
        term = torch.where(counts == 0.0, torch.zeros_like(counts), counts * difference**2)
        return term.sum(-1) / total


class ReliabilityDiagram(ReliabilityMetric):
    """Reliability diagram: the observed event frequency at each forecast probability `k / M`, stacked on a
    `probability` axis.

    On the diagonal is calibrated, flatter than it is over-confident, steeper is under-confident. A level no forecast
    ever fell into is NaN, and `forecast_frequency` says which points carry enough sample to read.
    """

    kind = "reliability"

    @property
    def output_dim(self) -> tuple[str, list] | None:
        """The `probability` axis."""
        return self.probability_axis

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        bound = self._bound(members)
        return self._levels(means, bound)[1]


class ForecastFrequency(ReliabilityMetric):
    """How often each forecast probability `k / M` was issued, the histogram under the reliability diagram; it sums to
    1 over the `probability` axis and needs no observations."""

    kind = "forecast_frequency"
    needs_events = False

    @property
    def output_dim(self) -> tuple[str, list] | None:
        """The `probability` axis."""
        return self.probability_axis

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        counts = self._counts(means, self._bound(members))
        return counts / counts.sum(-1, keepdim=True)


class BrierReliability(ReliabilityMetric):
    """The reliability component `REL` of the Brier score: how far the observed frequencies sit from the forecast
    probabilities, weighted by the sample of each level. Small is good, zero is a perfectly calibrated forecast."""

    kind = "brier_reliability"

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        bound = self._bound(members)
        counts, observed, total, _ = self._levels(means, bound)
        return self._component(counts, self._probabilities(counts, bound) - observed, total)


class BrierResolution(ReliabilityMetric):
    """The resolution component `RES` of the Brier score: how far the observed frequencies of the levels sit from the
    base rate, weighted by the sample of each level. Large is good, zero is a forecast that never separates events."""

    kind = "brier_resolution"

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        bound = self._bound(members)
        counts, observed, total, base_rate = self._levels(means, bound)
        return self._component(counts, observed - base_rate.unsqueeze(-1), total)


class ROCPOD(ReliabilityMetric):
    """Probability of detection (hit rate) of the ROC curve, the fraction of observed events forecast by "the event
    when `p >= k / M`", stacked on the `probability` axis; level 0 is 1, the decision that always forecasts the event.
    The decision "never", `(0, 0)`, is not stored: a plotted curve has to append it."""

    kind = "roc_pod"

    @property
    def output_dim(self) -> tuple[str, list] | None:
        """The `probability` axis."""
        return self.probability_axis

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        return self._rates(means, self._bound(members))[0]


class ROCPOFD(ReliabilityMetric):
    """Probability of false detection (false alarm rate) of the ROC curve, the fraction of observed non-events
    forecast by "the event when `p >= k / M`", stacked on the `probability` axis; level 0 is 1 and `(0, 0)` is not
    stored."""

    kind = "roc_pofd"

    @property
    def output_dim(self) -> tuple[str, list] | None:
        """The `probability` axis."""
        return self.probability_axis

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        return self._rates(means, self._bound(members))[1]


class ROCArea(ReliabilityMetric):
    """Area under the ROC curve by the trapezoid rule through `(0, 0)`, the `M` levels `k >= 1` and `(1, 1)`.

    0.5 is no discrimination, 1 a perfect one. At one member the curve has a single interior point and the area is
    `(1 + pod - pofd) / 2`. The trapezoids join the points by straight lines, so the area of a small ensemble is
    below that of the smooth curve it samples.
    """

    kind = "roc_area"

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        hit_rate, false_alarm_rate = self._rates(means, self._bound(members))
        zero = torch.zeros_like(hit_rate[..., :1])
        hit_rate, false_alarm_rate = torch.cat([hit_rate, zero], -1), torch.cat([false_alarm_rate, zero], -1)
        widths = false_alarm_rate[..., :-1] - false_alarm_rate[..., 1:]
        return (widths * (hit_rate[..., :-1] + hit_rate[..., 1:]) / 2.0).sum(-1)


def validate_cost_loss(cost_loss: list[float] | None) -> list[float]:
    """The cost/loss ratios of a relative economic value, or a ValueError naming the fault: a non-empty, strictly
    increasing list of numbers strictly between 0 and 1."""
    example = "{rev: {label: heavy, thresholds: {tp: 0.005}, cost_loss: [0.05, 0.1, 0.2, 0.5]}}"
    if cost_loss is None:
        raise ValueError(f"rev needs cost_loss ratios, e.g. {example}")
    if not isinstance(cost_loss, (list, tuple)) or not cost_loss:
        raise ValueError(f"cost_loss must be a non-empty list of ratios, e.g. {example}, got {cost_loss!r}")
    values = []
    for value in cost_loss:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 < value < 1.0:
            raise ValueError(f"a cost/loss ratio must be a number strictly between 0 and 1, got {value!r}")
        values.append(float(value))
    if any(later <= earlier for earlier, later in zip(values, values[1:])):
        raise ValueError(f"cost_loss must be strictly increasing, got {values}")
    return values


class RelativeEconomicValue(ReliabilityMetric):
    """Relative economic value (Richardson 2000) of the decision "protect when `p >= k / M`" for a user with
    cost/loss ratio `a`, the best over the `M` levels `k >= 1`, on a `cost_loss` axis.

    With base rate `s`, hit rate `H` and false alarm rate `F`, the value of one level is
    `(min(a, s) - F a (1 - s) + H s (1 - a) - s) / (min(a, s) - s a)`: 1 is a perfect forecast, 0 climatology (the
    better of always and never protecting), and it is negative where the level does worse than that. Taking the best
    level is the envelope of the ensemble's value curves; at one member there is one level, the deterministic
    forecast. NaN where the cell observed no event or no non-event.
    """

    kind = "rev"

    def __init__(
        self,
        label: str | None = None,
        thresholds: dict[str, float] | None = None,
        cost_loss: list[float] | None = None,
    ) -> None:
        super().__init__(label, thresholds)
        self.cost_loss = validate_cost_loss(cost_loss)

    @property
    def spec(self) -> str | dict:
        return {
            self.kind: {"label": self.label, "thresholds": dict(self.thresholds), "cost_loss": list(self.cost_loss)}
        }

    @property
    def output_dim(self) -> tuple[str, list] | None:
        """The `cost_loss` axis, known from the spec alone."""
        return ("cost_loss", list(self.cost_loss))

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        bound = self._bound(members)
        hit_rate, false_alarm_rate = self._rates(means, bound)
        base_rate = self._events(means, bound).sum(-1) / self._counts(means, bound).sum(-1)
        ratio = torch.tensor(self.cost_loss, dtype=hit_rate.dtype, device=hit_rate.device).unsqueeze(-1)  # (C, 1)
        hit_rate, false_alarm_rate = hit_rate[..., None, 1:], false_alarm_rate[..., None, 1:]  # (..., 1, M)
        s = base_rate[..., None, None]
        climate = torch.minimum(ratio, s)
        value = (climate - false_alarm_rate * ratio * (1.0 - s) + hit_rate * s * (1.0 - ratio) - s) / (
            climate - s * ratio
        )
        return value.amax(-1)


class BrierUncertainty(ThresholdMetric):
    """The uncertainty component `UNC` of the Brier score, `obar (1 - obar)`: the difficulty of the threshold in this
    cell, a property of the observations alone. It needs neither levels nor an ensemble size, so it costs the one sum
    `bss` and `event_frequency` already store."""

    kind = "brier_uncertainty"
    needs = (st.EventFrequency,)

    def from_means(self, means: dict[str, torch.Tensor], members: int) -> torch.Tensor:
        base_rate = self._mean(means, "event_frequency")
        return base_rate * (1.0 - base_rate)


REGISTRY: dict[str, type[Metric]] = {
    "bias": Bias,
    "mae": MAE,
    "rmse": RMSE,
    "error_std": ErrorStd,
    "crps": CRPS,
    "fair_crps": FairCRPS,
    "spread": Spread,
    "spread_skill": SpreadSkill,
    "member_rmse": MemberRMSE,
    "member_mae": MemberMAE,
    "acc": ACC,
    "activity_ratio": ActivityRatio,
    "pod": POD,
    "far": FAR,
    "csi": CSI,
    "ets": ETS,
    "frequency_bias": FrequencyBias,
    "hss": HSS,
    "pss": PSS,
    "pofd": POFD,
    "sedi": SEDI,
    "odds_ratio": OddsRatio,
    "orss": ORSS,
    "f1": F1,
    "brier": BrierScore,
    "bss": BSS,
    "event_frequency": EventFrequency,
    "reliability": ReliabilityDiagram,
    "forecast_frequency": ForecastFrequency,
    "brier_reliability": BrierReliability,
    "brier_resolution": BrierResolution,
    "brier_uncertainty": BrierUncertainty,
    "roc_pod": ROCPOD,
    "roc_pofd": ROCPOFD,
    "roc_area": ROCArea,
    "rev": RelativeEconomicValue,
    "rank_histogram": RankHistogram,
    "outlier_fraction": OutlierFraction,
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
    try:
        return REGISTRY[name](**(kwargs or {}))
    except TypeError as error:
        accepted = sorted(inspect.signature(REGISTRY[name]).parameters)
        raise ValueError(f"metric {name!r} does not accept {kwargs}: {error}; accepted keywords {accepted}") from None


def unique_statistics(metrics: list[Metric]) -> dict[str, st.Statistic]:
    """Statistics needed by `metrics`, deduplicated by name; a name meaning two different statistics is an error."""
    result: dict[str, st.Statistic] = {}
    owner: dict[str, str] = {}
    for metric in metrics:
        for name, statistic in metric.statistics.items():
            if name in result and result[name] != statistic:
                raise ValueError(
                    f"{metric.name} and {owner[name]} both need a statistic named {name!r} "
                    f"with different parameters, {statistic.parameters} and {result[name].parameters}; "
                    "give them different labels"
                )
            result.setdefault(name, statistic)
            owner.setdefault(name, metric.name)
    return result


def build(
    specs: list[str | dict | Metric], variables: list[str] | None = None, members: int | None = None
) -> list[Metric]:
    """Metrics from their specs: duplicate names, a label used with two threshold maps, two metrics giving one extra
    axis different coordinates (two `rev` with different `cost_loss` lists) and conflicting statistics are refused,
    and the metrics and their statistics are bound to `members` and to `variables` when they are known.

    The ensemble size is bound first, because a metric whose statistics depend on it has none until then and the
    statistic checks below would see nothing to check. The label check needs no binding, so it catches a label meaning
    two events at config-validation time, where the statistic identity guard can only catch it once the statistics
    exist; the guard stays as the second line of defence, and the only one for two statistics of one name arriving
    from something other than a labelled metric."""
    built = [from_spec(spec) for spec in specs]
    if members is not None:
        for metric in built:
            metric.bind_members(members)
    counts: dict[str, int] = {}
    for metric in built:
        counts[metric.name] = counts.get(metric.name, 0) + 1
    repeated = [f"{name!r} appears {count} times" for name, count in counts.items() if count > 1]
    if repeated:
        raise ValueError(f"duplicate metric names: {'; '.join(repeated)}; give the metrics different labels")
    by_label: dict[str, tuple[str, dict[str, float]]] = {}
    for metric in built:
        if not isinstance(metric, LabelledMetric):
            continue
        owner, thresholds = by_label.setdefault(metric.label, (metric.name, metric.thresholds))
        if thresholds != metric.thresholds:
            raise ValueError(
                f"label {metric.label!r} is used with different thresholds by {owner} ({thresholds}) "
                f"and {metric.name} ({metric.thresholds}); one label, one threshold map"
            )
    axes: dict[str, tuple[str, list]] = {}
    for metric in built:
        extra = metric.output_dim
        if extra is None:
            continue
        owner, values = axes.setdefault(extra[0], (metric.name, list(extra[1])))
        if values != list(extra[1]):
            raise ValueError(
                f"{owner} and {metric.name} disagree on the {extra[0]!r} axis, {values} and {list(extra[1])}; "
                f"the metrics of a run share one {extra[0]!r} coordinate"
            )
    unique_statistics(built)
    if variables is not None:
        for metric in built:
            for statistic in metric.statistics.values():
                statistic.bind_variables(variables)
    return built


def required_aux(metrics: list[Metric]) -> set[str]:
    """Names of the per-frame `aux` fields the statistics of `metrics` need (e.g. `climatology`)."""
    return {name for statistic in unique_statistics(metrics).values() for name in statistic.aux}


def min_members(metrics: list[Metric]) -> int:
    """Smallest ensemble size that all `metrics` accept."""
    return max((metric.min_members for metric in metrics), default=1)
