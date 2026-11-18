"""Checks on the targets, run when an evaluation is built: metadata only, so a dry run shows them."""

from __future__ import annotations

import logging

from anemoi.utils.dates import frequency_to_string

LOG = logging.getLogger(__name__)
ACTIONS = ("warn", "raise", "ignore")


def check_action(key: str, value: str) -> str:
    """`value` if it is one of `ACTIONS`; raises naming `key` otherwise."""
    if value not in ACTIONS:
        raise ValueError(f"{key} must be one of {', '.join(ACTIONS)}, got {value!r}")
    return value


def check_targets(
    forecast: object,
    targets: object,
    variables: list[str],
    on_period_mismatch: str = "warn",
    on_made_up_dates: str = "warn",
) -> None:
    """Report scored accumulations whose period differs between the targets and the forecast, and target dates
    anemoi-datasets made up; each problem is warned about, raised as a ValueError or ignored as configured."""
    check_action("on_period_mismatch", on_period_mismatch)
    check_action("on_made_up_dates", on_made_up_dates)
    _report(on_period_mismatch, "on_period_mismatch", _period_mismatch(forecast, targets, variables))
    _report(on_made_up_dates, "on_made_up_dates", _made_up_dates(forecast, targets))


def _report(action: str, key: str, message: str | None) -> None:
    if message is None or action == "ignore":
        return
    if action == "raise":
        raise ValueError(f"{message} (set {key}: warn or ignore to score anyway)")
    LOG.warning("%s (set %s: ignore to silence)", message, key)


def _period_mismatch(forecast: object, targets: object, variables: list[str]) -> str | None:
    """The accumulations whose periods differ; the ones with a period unknown on either side are logged."""
    model = getattr(forecast, "accumulation_periods", dict)()
    truth = getattr(targets, "accumulation_periods", dict)()
    unknown, differ = [], []
    for name in variables:
        if name not in model:
            continue  # not an accumulation the forecast knows of: baselines and own sources are never checked
        if model[name] is None or truth.get(name) is None:
            unknown.append(name)
        elif model[name] != truth[name]:
            differ.append(f"{name} {frequency_to_string(truth[name])} against {frequency_to_string(model[name])}")
    if unknown:
        LOG.info("the accumulation period of %s is unknown in the targets or the forecast, not checked", unknown)
    if not differ:
        return None
    return (
        f"the targets accumulate over another period than the forecast: {', '.join(differ)} in the model; "
        "the scores compare totals over different periods"
    )


def _made_up_dates(forecast: object, targets: object) -> str | None:
    """The wrappers that make target values up at some dates, in the downscaler wording when one interpolates."""
    made_up = getattr(targets, "made_up_dates", list)()
    if not made_up:
        return None
    where = "; ".join(made_up)
    interpolated = getattr(targets, "interpolate_frequency", None)
    if getattr(forecast, "temporal_downscaling", False) and interpolated:
        return (
            f"the targets are opened with interpolate_frequency {interpolated}: between the dates of the dataset it "
            "interpolates, the truth is itself a linear interpolation, not an analysis; score a region whose data "
            f"is real at every lead time ({where})"
        )
    return (
        f"the targets make up values at some dates: {where}; there the truth is computed by anemoi-datasets, "
        "not read from the zarr"
    )
