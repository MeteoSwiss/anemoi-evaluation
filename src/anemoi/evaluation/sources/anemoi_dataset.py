"""Targets read from an anemoi-datasets zarr at valid time, read ahead on a worker thread."""

from __future__ import annotations

import datetime
import time
from collections import OrderedDict
from collections import deque
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch

from anemoi.evaluation.frame import Grid
from anemoi.evaluation.sources.base import MissingTargetError
from anemoi.evaluation.sources.base import TargetSourceBase
from anemoi.evaluation.sources.base import select_variables
from anemoi.evaluation.sources.base import variable_info

DEFAULT_PREFETCH = 2


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


class DatasetTargets(TargetSourceBase):
    """Targets from `anemoi.datasets.open_dataset(*args, **kwargs)`, one zarr read per valid time.

    Valid times announced with `prefetch()` are read `prefetch` rows ahead on a single worker thread, which is then the
    only thread touching the dataset; decoded, variable-selected rows are kept in a host LRU cache of `cache_bytes`.
    With `prefetch=None` the depth is 2, raised to the forecast source's frames per model call by `prefetch_hint`.
    """

    def __init__(self, *args: Any, prefetch: int | None = None, cache_bytes: int = 0, **kwargs: Any) -> None:
        from anemoi.datasets import open_dataset

        self._setup(open_dataset(*args, **kwargs), args, kwargs, prefetch=prefetch, cache_bytes=cache_bytes)

    @classmethod
    def from_dataset(
        cls,
        dataset: Any,
        args: tuple = (),
        kwargs: dict | None = None,
        grid_indices: Any = None,
        prefetch: int | None = None,
        cache_bytes: int = 0,
    ) -> DatasetTargets:
        """Targets from an already open dataset; `grid_indices` reduces the grid as the checkpoint does."""
        targets = cls.__new__(cls)
        targets._setup(dataset, args, kwargs or {}, grid_indices, prefetch, cache_bytes)
        return targets

    @classmethod
    def from_forecast(cls, forecast: Any, prefetch: int | None = None, cache_bytes: int = 0) -> DatasetTargets:
        """Targets from the dataset the forecast source's runner reads, so both use the same zarr."""
        from anemoi.datasets import open_dataset

        args, kwargs = forecast.dataset_args_kwargs()
        dataset = open_dataset(*args, **kwargs)
        targets = cls.from_dataset(dataset, args, kwargs, forecast.grid_indices, prefetch, cache_bytes)
        targets.prefetch_hint(getattr(forecast, "frames_per_pass", 1))
        return targets

    def _setup(
        self,
        dataset: Any,
        args: tuple,
        kwargs: dict,
        grid_indices: Any = None,
        prefetch: int | None = None,
        cache_bytes: int = 0,
    ) -> None:
        if dataset.shape[2] != 1:
            raise ValueError(f"target datasets must have one member, got {dataset.shape[2]}")
        if (prefetch is not None and prefetch < 0) or cache_bytes < 0:
            raise ValueError(f"prefetch and cache_bytes must not be negative, got {prefetch} and {cache_bytes}")
        self.ds = dataset
        self.args, self.kwargs = tuple(_plain(list(args))), _plain(dict(kwargs))
        self.grid_indices = slice(None) if grid_indices is None else grid_indices
        self.grid = Grid(dataset.latitudes[self.grid_indices], dataset.longitudes[self.grid_indices])
        self.variables = list(dataset.variables)
        try:
            self.typed_variables = dict(dataset.typed_variables)
        except (AttributeError, KeyError, TypeError):
            self.typed_variables = {}
        dates = dataset.dates.astype("datetime64[us]").tolist()
        self._dates = dates
        self._index = {date: i for i, date in enumerate(dates)}
        self._missing = set(dataset.missing)
        self.prefetch_given = prefetch
        self.lookahead = DEFAULT_PREFETCH if prefetch is None else prefetch
        self.cache_bytes = cache_bytes
        self._executor: ThreadPoolExecutor | None = None
        self._queue: deque = deque()
        self._pending: dict[tuple, Future] = {}
        self._cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._cache_size = 0
        self._stats = {"reads": 0, "read_seconds": 0.0, "cache_hits": 0}

    @property
    def stats(self) -> dict:
        """Rows read (and the seconds spent reading them, on the worker when there is one) and cache hits."""
        return dict(self._stats)

    def available(self, valid_time: datetime.datetime) -> bool:
        """Whether the dataset has `valid_time` and it is not a missing date."""
        index = self._index.get(valid_time)
        return index is not None and index not in self._missing

    def prefetch_hint(self, frames_per_pass: int) -> None:
        """Without an explicit `prefetch`, read at least the frames of one model call ahead (they land together)."""
        if self.prefetch_given is None:
            self.lookahead = max(DEFAULT_PREFETCH, int(frames_per_pass))

    def prefetch(self, valid_times: list[datetime.datetime], variables: list[str]) -> None:
        """Announce the valid times to read next, in order; reads run ahead on the worker thread."""
        if self.lookahead == 0:
            return
        if self._executor is None:
            self._executor = ThreadPoolExecutor(1, thread_name_prefix="anemoi-evaluation-targets")
        key_variables = tuple(variables)
        self._queue = deque((valid_time, key_variables) for valid_time in valid_times)
        self._pump()

    def _pump(self) -> None:
        while self._queue and len(self._pending) < self.lookahead:
            key = self._queue.popleft()
            if key in self._pending or key in self._cache or not self.available(key[0]):
                continue
            self._pending[key] = self._executor.submit(self._read, key)

    def frame(self, valid_time: datetime.datetime, variables: list[str], device: torch.device) -> torch.Tensor:
        """Target for `valid_time` as (1, V, N); MissingTargetError when the date is absent or missing."""
        index = self._index.get(valid_time)
        if index is None:
            raise MissingTargetError(f"no target date {valid_time} in the dataset")
        if index in self._missing:
            raise MissingTargetError(f"target date {valid_time} is missing from the dataset")
        select_variables(list(variables), self.variables)
        key = (valid_time, tuple(variables))
        if key in self._cache:
            self._cache.move_to_end(key)
            data = self._cache[key]
            self._stats["cache_hits"] += 1
        else:
            future = self._pending.pop(key, None)
            if future is None and self._executor is not None:
                future = self._executor.submit(self._read, key)
            data, seconds = future.result() if future is not None else self._read(key)
            self._stats["reads"] += 1
            self._stats["read_seconds"] += seconds
            self._store(key, data)
        self._pump()
        return torch.from_numpy(data).to(device)[None]

    def _read(self, key: tuple) -> tuple[np.ndarray, float]:
        """Decoded, variable-selected row and its read time; the only method that touches the dataset after setup."""
        valid_time, variables = key
        columns = select_variables(list(variables), self.variables)
        start = time.perf_counter()
        data = self.ds[self._index[valid_time]][columns, 0][:, self.grid_indices]
        return np.ascontiguousarray(data, dtype=np.float32), time.perf_counter() - start

    def _store(self, key: tuple, data: np.ndarray) -> None:
        if self.cache_bytes == 0 or data.nbytes > self.cache_bytes:
            return
        while self._cache and self._cache_size + data.nbytes > self.cache_bytes:
            _, evicted = self._cache.popitem(last=False)
            self._cache_size -= evicted.nbytes
        self._cache[key] = data
        self._cache_size += data.nbytes

    def close(self) -> None:
        """Stop the worker thread and drop pending reads and the cache."""
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        self._queue.clear()
        self._pending.clear()
        self._cache.clear()
        self._cache_size = 0

    def grid_mask(self, i: int) -> np.ndarray:
        """Nodes of source `i` of a cutout or join dataset, from the per-source counts in `ds.grids`."""
        grids = [int(count) for count in self.ds.grids]
        if not 0 <= i < len(grids):
            raise ValueError(f"grid index {i} out of range for {len(grids)} grids")
        mask = np.zeros(sum(grids), dtype=bool)
        start = sum(grids[:i])
        mask[start : start + grids[i]] = True
        return mask[self.grid_indices]

    def variable_info(self, name: str) -> dict:
        """`param` and `level` of a variable from the dataset's typed variables; None when unknown."""
        return variable_info(self.typed_variables.get(name))

    def describe(self) -> dict:
        """Dataset arguments, date range, row size and the prefetch and cache settings."""
        itemsize = np.dtype(getattr(self.ds, "dtype", np.float32)).itemsize
        return {
            "source": "anemoi_dataset",
            "args": list(self.args),
            "kwargs": dict(self.kwargs),
            "dates": {
                "first": self._dates[0].isoformat(),
                "last": self._dates[-1].isoformat(),
                "count": len(self._dates),
                "missing": len(self._missing),
            },
            "nodes": self.grid.n,
            "variables": len(self.variables),
            "row_bytes": int(np.prod(self.ds.shape[1:])) * itemsize,
            "prefetch": self.lookahead,
            "cache_bytes": self.cache_bytes,
        }

    def to_config(self) -> dict:
        """The `targets:` block that rebuilds this source; raises for positional arguments other than one mapping."""
        if len(self.args) > 1 or (self.args and not isinstance(self.args[0], dict)):
            raise ValueError("targets opened with positional arguments cannot be serialised, use keyword arguments")
        options = {}
        if self.prefetch_given is not None:
            options["prefetch"] = self.prefetch_given
        if self.cache_bytes:
            options["cache_bytes"] = self.cache_bytes
        return {"anemoi_dataset": {**(self.args[0] if self.args else {}), **self.kwargs, **options}}
