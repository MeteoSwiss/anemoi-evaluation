"""Per-element statistics: (M, V, N) prediction x (1, V, N) target -> (V, N) float64.

Members are reduced in float64 without an (M, V, N) float64 copy: the mean accumulates in float64 and
the ensemble statistics work one variable at a time, so large offsets (pressure in Pa, temperature in K)
with a small spread do not cancel in float32. The ensemble statistics are the WeatherBench-X primitives
(`skill`, `pairs`, `ensemble_variance`); CRPS for any alpha and the spread are derived from their means. The anomaly
statistics need a climatology in `aux` (declared by `Statistic.aux`) and zero their own contribution where it is
non-finite, so those nodes leave the anomaly correlation only.

The threshold statistics binarise the target, and the ensemble mean or the members, with a strict `>` against a
per-variable threshold map carrying a user-given label, so their names are `<kind>_<label>`. They learn the variable order of the run from
`bind_variables`, and a variable the map does not name is NaN in every value, hence NaN in every sum derived from it.

The rank bins hold the position of the target among the members, one statistic per bin of `0..M`, with tied members
spread deterministically over the bins the target could occupy, so no seed is involved. Their two counts are the same
for every bin of a frame, so the first bin computes them and leaves them in `aux` for the others: `aux` is rebuilt per
frame and shared by reference, so a statistic may stash an intermediate of its own in it as long as it reads it back
through a helper that recomputes it when it is absent, as `rank_counts` does.

The reliability levels split one binarised event by the integer number of members above the threshold, one pair of
statistics per level of `0..M`: how much weight the level carries and how much of it was observed. The test is an
equality on the count, which is exact, rather than on the exceedance fraction. All `2 (M + 1)` statistics of a label
need the same count and the same observed event, so they share them through `exceedance_counts`, which caches them in
`aux` per frame under the label.
"""

from __future__ import annotations

import math
import re
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


def exceedance(field: torch.Tensor, thresholds: list[float | None]) -> torch.Tensor:
    """(V, N) float64 indicator 1{field > t} per variable, NaN where the variable has no threshold."""
    out = torch.empty(field.shape, dtype=torch.float64, device=field.device)
    for j, threshold in enumerate(thresholds):
        out[j] = float("nan") if threshold is None else (field[j].double() > threshold).double()
    return out


def exceedance_count(pred: torch.Tensor, thresholds: list[float | None]) -> torch.Tensor:
    """(V, N) float64 number of members above the threshold, one variable at a time, NaN where none.

    The count is a whole number well below 2^53, so an equality against another whole number is exact, which is what
    the reliability levels compare against.
    """
    out = torch.empty(pred.shape[1:], dtype=torch.float64, device=pred.device)
    for j, threshold in enumerate(thresholds):
        if threshold is None:
            out[j] = float("nan")
        else:
            out[j] = (pred[:, j].double() > threshold).double().sum(0)
    return out


def exceedance_fraction(pred: torch.Tensor, thresholds: list[float | None]) -> torch.Tensor:
    """(V, N) float64 fraction of members above the threshold, NaN where the variable has no threshold."""
    return exceedance_count(pred, thresholds) / pred.shape[0]


def rank_counts(
    pred: torch.Tensor, target: torch.Tensor, aux: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """(V, N) float64 counts of the members below and tied with the target, computed once per frame and kept in `aux`.

    The comparisons are exact in the frame's own dtype, so nothing is widened to float64 before them; only the counts
    are accumulated in float64, which is exact well beyond any ensemble size. `aux` is per frame: the counts are
    computed when they are absent and read back when they are there, so a mapping reused for a second frame would
    return the first one's counts.
    """
    if "rank_below" not in aux or "rank_ties" not in aux:
        below = torch.empty(target.shape, dtype=torch.float64, device=pred.device)
        ties = torch.empty(target.shape, dtype=torch.float64, device=pred.device)
        for j in range(target.shape[0]):
            members, truth = pred[:, j], target[j]
            below[j] = (members < truth).sum(0, dtype=torch.float64)
            ties[j] = (members == truth).sum(0, dtype=torch.float64)
        aux["rank_below"], aux["rank_ties"] = below, ties
    return aux["rank_below"], aux["rank_ties"]


def exceedance_counts(
    pred: torch.Tensor,
    target: torch.Tensor,
    thresholds: list[float | None],
    label: str,
    aux: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """(V, N) float64 member exceedance count and observed event of one label, computed once per frame and kept in
    `aux` under `exceedance_count_<label>` and `exceedance_event_<label>`.

    `aux` is per frame: the two fields are computed when they are absent and read back when they are there, so a
    mapping reused for a second frame would return the first one's values. The label alone is the key because a label
    carries one threshold map across the whole config (`metrics.build` refuses anything else) and because every
    statistic reading it carries the label in its own name; a future family that reads it without that property would
    have to put the thresholds in the key too.
    """
    count_key, event_key = f"exceedance_count_{label}", f"exceedance_event_{label}"
    if count_key not in aux or event_key not in aux:
        aux[count_key] = exceedance_count(pred, thresholds)
        aux[event_key] = exceedance(target, thresholds)
    return aux[count_key], aux[event_key]


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


def validate_thresholds(label: str | None, thresholds: dict[str, float] | None) -> tuple[str, dict[str, float]]:
    """The label and the finite per-variable thresholds of a threshold score, or a ValueError naming the fault.

    It lives here rather than in `ThresholdStatistic.__init__` because a metric that cannot build its statistics until
    it knows the ensemble size still has to validate its label and its map at construction, when the config layer
    builds it unbound.
    """
    example = "{csi: {label: heavy, thresholds: {tp: 0.005}}}"
    if label is None:
        raise ValueError(f"a threshold metric needs a label, e.g. {example}")
    if thresholds is None:
        raise ValueError(f"a threshold metric needs thresholds, e.g. {example}")
    if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9_]+", label):
        raise ValueError(f"label {label!r} must be letters, digits and underscores")
    if not isinstance(thresholds, dict):
        raise ValueError(f"thresholds must map variables to numbers, e.g. {example}, got {thresholds!r}")
    if not thresholds:
        raise ValueError("thresholds must name at least one variable")
    values: dict[str, float] = {}
    for name, value in thresholds.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"threshold for {name!r} must be a finite number, got {value!r}")
        values[name] = float(value)
    return label, values


class Statistic:
    """Base class; `compute` receives the raw tensors, the aggregator masks non-finite elements afterwards."""

    name: str
    min_members: int = 1
    aux: tuple[str, ...] = ()

    @property
    def parameters(self) -> tuple:
        """The values that make two statistics of the same class the same statistic."""
        return ()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Statistic):
            return NotImplemented
        return type(self) is type(other) and self.parameters == other.parameters

    def __hash__(self) -> int:
        return hash((type(self), self.parameters))

    def bind_variables(self, variables: list[str]) -> None:
        """Tell the statistic the variable order of the run; the default statistics do not care."""

    def compute(
        self, pred: torch.Tensor, target: torch.Tensor, aux: dict[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        """Statistic per variable and node; `aux` carries per-frame fields such as the ensemble mean.

        `aux` belongs to **one frame**: it is shared with the other statistics of that frame, and a statistic may add
        a per-frame intermediate of its own to it (the rank bins add `rank_below` and `rank_ties`), so the mapping a
        caller passes may gain keys and must not be reused for another frame, which would silently return the
        previous frame's values. The aggregator copies it per frame; a caller passing nothing gets an empty mapping.
        """
        if pred.shape[0] < self.min_members:
            raise ValueError(f"{self.name} needs at least {self.min_members} members, got {pred.shape[0]}")
        return self._compute(pred, target[0], {} if aux is None else aux)

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


class ThresholdStatistic(Statistic):
    """Base class of the statistics of one binarised event: a per-variable threshold map and the label naming it.

    The event is `1{value > threshold}`, strict and in float64 on both sides. `bind_variables` turns the map into a
    list aligned with the variable axis of the run, so `_compute` only ever sees `float | None` per column.
    """

    kind: str

    def __init__(self, label: str | None = None, thresholds: dict[str, float] | None = None) -> None:
        self.label, self.thresholds = validate_thresholds(label, thresholds)
        self.name = f"{self.kind}_{self.label}"
        self._variables: list[str] | None = None
        self._per_variable: list[float | None] | None = None

    @property
    def parameters(self) -> tuple:
        """The label and the thresholds; the bound variables are run state, not identity."""
        return (self.label, tuple(sorted(self.thresholds.items())))

    def bind_variables(self, variables: list[str]) -> None:
        """Align the threshold map with the variable order of the run; an unknown variable is an error.

        Binding the same variables again is a no-op, so one run may be repeated; binding a *different* order would
        silently move the thresholds to other columns of an evaluation still holding this instance, and raises.
        """
        variables = list(variables)
        if self._variables is not None and self._variables != variables:
            raise ValueError(f"{self.name} is already bound to {self._variables}, cannot rebind to {variables}")
        unknown = [name for name in self.thresholds if name not in variables]
        if unknown:
            raise ValueError(f"the thresholds of {self.name} name variables the run does not have: {unknown}")
        self._variables = variables
        self._per_variable = [self.thresholds.get(name) for name in variables]

    def _bound(self, n_variables: int) -> list[float | None]:
        if self._per_variable is None:
            raise ValueError(f"{self.name} was not bound to the run's variables")
        if len(self._per_variable) != n_variables:
            raise ValueError(
                f"{self.name} is bound to {len(self._per_variable)} variables but the frame has {n_variables}"
            )
        return self._per_variable

    def _events(
        self, pred: torch.Tensor, target: torch.Tensor, aux: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The binarised ensemble mean and the binarised target, (V, N) float64."""
        thresholds = self._bound(target.shape[0])
        return exceedance(ensemble_mean(pred, aux), thresholds), exceedance(target, thresholds)


class Hit(ThresholdStatistic):
    """Event forecast and observed, `f * o`."""

    kind = "hit"

    def _compute(self, pred: torch.Tensor, target: torch.Tensor, aux: dict[str, torch.Tensor]) -> torch.Tensor:
        forecast, observed = self._events(pred, target, aux)
        return forecast * observed


class Miss(ThresholdStatistic):
    """Event observed but not forecast, `(1 - f) * o`."""

    kind = "miss"

    def _compute(self, pred: torch.Tensor, target: torch.Tensor, aux: dict[str, torch.Tensor]) -> torch.Tensor:
        forecast, observed = self._events(pred, target, aux)
        return (1.0 - forecast) * observed


class FalseAlarm(ThresholdStatistic):
    """Event forecast but not observed, `f * (1 - o)`."""

    kind = "false_alarm"

    def _compute(self, pred: torch.Tensor, target: torch.Tensor, aux: dict[str, torch.Tensor]) -> torch.Tensor:
        forecast, observed = self._events(pred, target, aux)
        return forecast * (1.0 - observed)


class Brier(ThresholdStatistic):
    """Squared error of the ensemble exceedance fraction, `(p - o)^2`; at one member `miss + false_alarm`."""

    kind = "brier"

    def _compute(self, pred: torch.Tensor, target: torch.Tensor, aux: dict[str, torch.Tensor]) -> torch.Tensor:
        thresholds = self._bound(target.shape[0])
        return (exceedance_fraction(pred, thresholds) - exceedance(target, thresholds)) ** 2


class EventFrequency(ThresholdStatistic):
    """The observed event itself, `o`: its mean is the base rate of the threshold."""

    kind = "event_frequency"

    def _compute(self, pred: torch.Tensor, target: torch.Tensor, aux: dict[str, torch.Tensor]) -> torch.Tensor:
        return exceedance(target, self._bound(target.shape[0]))


class ReliabilityLevel(ThresholdStatistic):
    """Base of the two statistics of one probability level of the reliability diagram, named `<kind>_<label>_<k>`.

    The level is the integer number of members above the threshold, `c`, and the levels `0..M` partition the nodes of a
    variable the map names. The test is `c == k` on the count itself, which is exact for any ensemble size, rather than
    `p == k / M` on the exceedance fraction. A variable the map does not name has a NaN count, so both statistics are
    NaN for it, as for every other threshold statistic.
    """

    def __init__(
        self,
        label: str | None = None,
        thresholds: dict[str, float] | None = None,
        *,
        k: int,
        members: int,
    ) -> None:
        super().__init__(label=label, thresholds=thresholds)
        if members < 1:
            raise ValueError(f"a reliability level needs at least 1 member, got {members}")
        if not 0 <= k <= members:
            raise ValueError(f"reliability level {k} is outside 0..{members}")
        self.k = k
        self.members = members
        self.name = f"{self.kind}_{self.label}_{k}"

    @property
    def parameters(self) -> tuple:
        """The label, the thresholds, the level and the ensemble size it was built for; all four change the meaning."""
        return (self.label, tuple(sorted(self.thresholds.items())), self.k, self.members)

    def _level(
        self, pred: torch.Tensor, target: torch.Tensor, aux: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The `1{c == k}` indicator and the observed event, (V, N) float64 and NaN where the variable has no
        threshold."""
        if pred.shape[0] != self.members:
            raise ValueError(f"{self.name} is built for {self.members} members but the frame has {pred.shape[0]}")
        count, observed = exceedance_counts(pred, target, self._bound(target.shape[0]), self.label, aux)
        return torch.where(count.isnan(), count, (count == float(self.k)).double()), observed


class ReliabilityCount(ReliabilityLevel):
    """The weight the forecast probability `k / M` carries, `1{c == k}`."""

    kind = "reliability_count"

    def _compute(self, pred: torch.Tensor, target: torch.Tensor, aux: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._level(pred, target, aux)[0]


class ReliabilityEvent(ReliabilityLevel):
    """The observed events within the level, `1{c == k} * o`; never above `reliability_count`."""

    kind = "reliability_event"

    def _compute(self, pred: torch.Tensor, target: torch.Tensor, aux: dict[str, torch.Tensor]) -> torch.Tensor:
        level, observed = self._level(pred, target, aux)
        return level * observed


class RankBin(Statistic):
    """One bin of the rank histogram: the mass the target's rank puts in bin `k` of `0..M`, named `rank_bin_<k>`.

    With `below` members under the target and `ties` equal to it, the rank under uniform random tie-breaking is uniform
    over `below .. below + ties`, and the value stored is that expectation, `1{below <= k <= below + ties}
    / (ties + 1)`. The `M + 1` bins of a frame therefore sum to 1 per element, exactly when `ties + 1` is a power of two
    and to a few ulp otherwise. A node with a non-finite member or target lands in some finite bin (a NaN member in
    bin 0, a `-inf` member counted as below), which the aggregator's shared validity mask replaces with 0 before the
    weighted sum, so no NaN enters a rank sum.
    """

    min_members = 2

    def __init__(self, k: int, members: int) -> None:
        if members < self.min_members:
            raise ValueError(f"a rank bin needs at least {self.min_members} members, got {members}")
        if not 0 <= k <= members:
            raise ValueError(f"rank bin {k} is outside 0..{members}")
        self.k = k
        self.members = members
        self.name = f"rank_bin_{k}"

    @property
    def parameters(self) -> tuple:
        """The bin and the ensemble size it was built for; both change what the statistic means."""
        return (self.k, self.members)

    def _compute(self, pred: torch.Tensor, target: torch.Tensor, aux: dict[str, torch.Tensor]) -> torch.Tensor:
        if pred.shape[0] != self.members:
            raise ValueError(f"{self.name} is built for {self.members} members but the frame has {pred.shape[0]}")
        below, ties = rank_counts(pred, target, aux)
        return ((below <= self.k) & (self.k <= below + ties)).double() / (ties + 1.0)


def crps_coefficient(alpha: float, members: int) -> float:
    """Weight of the `pairs` term in CRPS_alpha = skill - coefficient * pairs; alpha 0 standard, 1 fair."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return alpha / (members * (members - 1)) + (1.0 - alpha) / members**2
