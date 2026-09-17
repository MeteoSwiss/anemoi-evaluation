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

    def _levels(
        self, means: dict[str, torch.Tensor], members: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The level weights, the observed frequency within each level, the total weight and the base rate, the last
        two from the level sums themselves so that each metric of the family is self-contained."""
        counts = self._counts(means, members)
        events = torch.stack([means[f"reliability_event_{self.label}_{k}"] for k in range(members + 1)], dim=-1)
        total = counts.sum(-1)
        return counts, events / counts, total, events.sum(-1) / total

    def _probabilities(self, counts: torch.Tensor, members: int) -> torch.Tensor:
        """The forecast probability of each level, `k / M`, shaped to broadcast against the level axis."""
        return torch.arange(members + 1, dtype=counts.dtype, device=counts.device) / members

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
    "crps": CRPS,
    "fair_crps": FairCRPS,
    "spread": Spread,
    "spread_skill": SpreadSkill,
    "member_rmse": MemberRMSE,
    "member_mae": MemberMAE,
    "acc": ACC,
    "pod": POD,
    "far": FAR,
    "csi": CSI,
    "ets": ETS,
    "frequency_bias": FrequencyBias,
    "hss": HSS,
    "pss": PSS,
    "brier": BrierScore,
    "bss": BSS,
    "event_frequency": EventFrequency,
    "reliability": ReliabilityDiagram,
    "forecast_frequency": ForecastFrequency,
    "brier_reliability": BrierReliability,
    "brier_resolution": BrierResolution,
    "brier_uncertainty": BrierUncertainty,
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
    """Metrics from their specs: duplicate names, a label used with two threshold maps and conflicting statistics are
    refused, and the metrics and their statistics are bound to `members` and to `variables` when they are known.

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
