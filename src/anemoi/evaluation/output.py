"""xarray output, netcdf round trip of the raw state, and merging of states."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from anemoi.utils.humanize import compress_dates

from anemoi.evaluation.aggregation import AggregationState
from anemoi.evaluation.metrics import Metric

LOG = logging.getLogger(__name__)
BIN_ALL = "all"
TIMING_ATTRS = ("time_model_s", "time_target_s", "time_target_read_s", "time_statistics_s", "time_total_s")
COUNT_ATTRS = ("target_reads", "target_cache_hits", "model_calls")
MAX_ATTRS = ("peak_gpu_memory_bytes",)
_DIMS = ("lead_time", "bin", "variable", "region")
_STATE_DIMS = ("lead_time", "state_bin", "variable", "region")
_STATE_SUM = "state_sum_"


def to_xarray(state: AggregationState, metrics: list[Metric], attrs: dict | None = None) -> xr.Dataset:
    """Metrics on (lead_time, bin, variable, region), `bin` being the stored bins plus a derived `all`, the raw
    state on `state_bin` (stored bins only) and the contributed init times as `init_time`; attrs extend `state.attrs`."""
    state = state.cpu()
    sums = {name: total.numpy() for name, total in state.sums.items()}
    weights, n_init, bins = state.weights.numpy(), state.n_init.numpy(), list(state.bins)
    if bins != [BIN_ALL]:
        sums = {name: _with_total(total) for name, total in sums.items()}
        weights, n_init, bins = _with_total(weights), _with_total(n_init), bins + [BIN_ALL]
    means = {name: torch.from_numpy(total) / torch.from_numpy(weights) for name, total in sums.items()}
    data_vars = {}
    for metric in metrics:
        missing = set(metric.statistics) - set(means)
        if missing:
            raise ValueError(f"state has no sums for statistics {sorted(missing)} needed by {metric.name}")
        data_vars[metric.name] = (_DIMS, metric.from_means(means, state.members).numpy())
    data_vars["n_init"] = (_DIMS[:2], n_init)
    data_vars["weight_sum"] = (_DIMS, weights)
    for name, total in state.sums.items():
        data_vars[_STATE_SUM + name] = (_STATE_DIMS, total.numpy())
    data_vars["state_weights"] = (_STATE_DIMS, state.weights.numpy())
    data_vars["state_n_init"] = (_STATE_DIMS[:2], state.n_init.numpy())
    coords = {
        "lead_time": np.array(state.lead_times, dtype="timedelta64[ns]"),
        "bin": bins,
        "state_bin": list(state.bins),
        "variable": list(state.variables),
        "region": list(state.regions),
        "init_time": np.array(state.init_times, dtype="datetime64[ns]"),
        **{name: ("variable", list(values)) for name, values in state.variable_coords.items()},
    }
    attrs = {**state.attrs, **(attrs or {}), "members": state.members, "metrics": json.dumps([m.spec for m in metrics])}
    return xr.Dataset(data_vars, coords, attrs)


def write(dataset: xr.Dataset, path: str | Path) -> None:
    """Write a results Dataset to netcdf, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(path, engine="netcdf4")


def load_state(path: str | Path) -> AggregationState:
    """Read the raw aggregation state back from a results file; attrs are kept on the state."""
    with xr.open_dataset(path, engine="netcdf4", decode_timedelta=True) as dataset:
        dataset = dataset.load()
    lead_times = [value.astype("timedelta64[us]").item() for value in dataset["lead_time"].values]
    sums = {
        name[len(_STATE_SUM) :]: torch.from_numpy(np.asarray(dataset[name].values, dtype=np.float64))
        for name in dataset.data_vars
        if name.startswith(_STATE_SUM)
    }
    variable_coords = {
        name: coord.values.tolist()
        for name, coord in dataset.coords.items()
        if coord.dims == ("variable",) and name != "variable"
    }
    return AggregationState(
        lead_times,
        dataset["state_bin"].values.tolist(),
        dataset["variable"].values.tolist(),
        dataset["region"].values.tolist(),
        sums,
        torch.from_numpy(np.asarray(dataset["state_weights"].values, dtype=np.float64)),
        torch.from_numpy(np.asarray(dataset["state_n_init"].values, dtype=np.int64)),
        int(dataset.attrs["members"]),
        variable_coords,
        dataset["init_time"].values.astype("datetime64[us]").tolist(),
        {key: value for key, value in dataset.attrs.items() if key != "members"},
    )


def merge(items: Iterable[AggregationState | str | Path], *, partial: bool = False) -> AggregationState:
    """Sum states or result files with disjoint init times, whose `shard` tags must be the complete `0..n-1` of one
    run unless `partial`; attrs come from the first except the timing totals and the counters (summed),
    `peak_gpu_memory_bytes` (the maximum), `init_times` (the union), `config` (a JSON `{"merge": [...]}` of every
    input's config) and `shard` (dropped, or the shards merged when `partial` allowed an incomplete set)."""
    items = list(items)
    states = [load_state(item) if isinstance(item, (str, Path)) else item for item in items]
    if not states:
        raise ValueError("nothing to merge")
    labels = [str(item) if isinstance(item, (str, Path)) else f"input {i}" for i, item in enumerate(items)]
    shard = _merged_shard(states, labels, partial)
    result = states[0]
    for state in states[1:]:
        result = result.merge(state)
    attrs = dict(states[0].attrs)
    attrs.pop("shard", None)
    for keys, cast, combine in ((TIMING_ATTRS, float, sum), (COUNT_ATTRS, int, sum), (MAX_ATTRS, int, max)):
        for key in keys:
            values = [state.attrs[key] for state in states if key in state.attrs]
            if values:
                attrs[key] = cast(combine(values))
    attrs["init_times"] = ", ".join(compress_dates(result.init_times)) if result.init_times else ""
    attrs["config"] = json.dumps({"merge": [state.attrs.get("config") for state in states]})
    paths = [str(item) for item in items if isinstance(item, (str, Path))]
    if paths:
        attrs["merged_from"] = json.dumps(paths)
    if shard is not None:
        attrs["shard"] = shard
    result.attrs = attrs
    return result


def _shard_tag(value: str, label: str) -> tuple[frozenset[int], int]:
    """Parse a `shard` attr: the `i/n` of a run, or the `i,j,k/n` of a partial merge."""
    try:
        indices, total = value.split("/")
        tag, count = frozenset(int(part) for part in indices.split(",")), int(total)
    except ValueError:
        raise ValueError(f"{label} has shard attr {value!r}, expected 'i/n'") from None
    if not tag or any(not 0 <= index < count for index in tag):
        raise ValueError(f"{label} has shard attr {value!r}, expected 0 <= i < n") from None
    return tag, count


def _merged_shard(states: list[AggregationState], labels: list[str], partial: bool) -> str | None:
    """The `shard` attr of a merge of these inputs, refusing anything but the complete `0..n-1` shards of one run:
    None when they are complete or carry no tag at all, `i,j,k/n` when `partial` allowed an incomplete set."""
    tags = [
        _shard_tag(state.attrs["shard"], label) if "shard" in state.attrs else None
        for state, label in zip(states, labels)
    ]
    if all(tag is None for tag in tags):
        return None
    untagged = [label for label, tag in zip(labels, tags) if tag is None]
    if untagged:
        raise ValueError(
            f"{len(untagged)} of {len(tags)} inputs have no shard attr ({', '.join(untagged)}): "
            "merge either all the shards of one run or only untagged results"
        )
    counts: dict[int, str] = {}
    for (_, count), label in zip(tags, labels):
        counts.setdefault(count, label)
    if len(counts) > 1:
        found = ", ".join(f"{count} ({label})" for count, label in sorted(counts.items()))
        raise ValueError(f"inputs disagree on the number of shards: {found}")
    count = next(iter(counts))
    seen: dict[int, list[str]] = {}
    for (indices, _), label in zip(tags, labels):
        for index in sorted(indices):
            seen.setdefault(index, []).append(label)
    duplicates = {index: names for index, names in sorted(seen.items()) if len(names) > 1}
    if duplicates:
        found = ", ".join(f"{index} ({', '.join(names)})" for index, names in duplicates.items())
        raise ValueError(f"duplicate shards of {count}: {found}")
    configs = {state.attrs["config"] for state in states if "config" in state.attrs}
    if len(configs) > 1:
        LOG.warning("the shards carry %d different configs, are they shards of the same run?", len(configs))
    missing = sorted(set(range(count)) - set(seen))
    if not missing:
        return None
    if not partial:
        listed = ", ".join(str(index) for index in missing[:8])
        listed += f" and {len(missing) - 8} more" if len(missing) > 8 else ""
        raise ValueError(
            f"{len(seen)} of the {count} shards of the run were given, missing {listed}; "
            "pass --partial (partial=True) to merge an incomplete set"
        )
    return ",".join(str(index) for index in sorted(seen)) + f"/{count}"


def _with_total(array: np.ndarray) -> np.ndarray:
    return np.concatenate([array, array.sum(axis=1, keepdims=True)], axis=1)
