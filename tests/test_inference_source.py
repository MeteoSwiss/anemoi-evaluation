"""The inference source's pieces that need no checkpoint: the chunk defaults, the forcings cache and the chunk report."""

import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from anemoi.evaluation.sources.anemoi_inference import CHUNKS_ENV
from anemoi.evaluation.sources.anemoi_inference import PROCESSOR_CHUNKS_ENV
from anemoi.evaluation.sources.anemoi_inference import ForcingsCache
from anemoi.evaluation.sources.anemoi_inference import SharedForcings
from anemoi.evaluation.sources.anemoi_inference import inference_chunks
from anemoi.evaluation.sources.anemoi_inference import inference_env

HOUR = datetime.timedelta(hours=1)
T0 = datetime.datetime(2024, 1, 1)


def test_inference_env_adds_the_processor_default_only_when_chunking():
    assert inference_env({}, environ={}) == {}
    assert inference_env({"env": {"FOO": "bar"}}, environ={}) == {"FOO": "bar"}
    # chunking asked for in the block: the processor count defaults to 1
    assert inference_env({"env": {CHUNKS_ENV: 8}}, environ={}) == {CHUNKS_ENV: 8, PROCESSOR_CHUNKS_ENV: 1}
    # chunking asked for in the process environment: the same
    assert inference_env({}, environ={CHUNKS_ENV: "8"}) == {PROCESSOR_CHUNKS_ENV: 1}
    # a processor count given anywhere is left alone
    assert inference_env({"env": {CHUNKS_ENV: 8, PROCESSOR_CHUNKS_ENV: 4}}, environ={}) == {
        CHUNKS_ENV: 8,
        PROCESSOR_CHUNKS_ENV: 4,
    }
    assert inference_env({"env": {CHUNKS_ENV: 8}}, environ={PROCESSOR_CHUNKS_ENV: "2"}) == {CHUNKS_ENV: 8}
    # the caller's block is not mutated
    block = {"env": {CHUNKS_ENV: 8}}
    inference_env(block, environ={})
    assert block == {"env": {CHUNKS_ENV: 8}}


class Provider:
    """A forcings provider as anemoi-inference's tensor handler sees it: values encode (variable, date, node)."""

    def __init__(self, n: int = 5) -> None:
        self.variables = ["a", "b"]
        self.mask = np.array([3, 4])
        self.kinds = {"computed": True}
        self.n = n
        self.calls = []

    def load_forcings_array(self, dates, state):
        self.calls.append(list(dates))
        hours = np.array([(d - T0) / HOUR for d in dates], dtype=np.float32)
        return (
            np.arange(2, dtype=np.float32)[:, None, None] * 100
            + hours[None, :, None]
            + np.arange(self.n)[None, None, :]
        ).astype(np.float32)


def state(n: int = 5, shift: float = 0.0) -> dict:
    return {"latitudes": np.arange(n, dtype=np.float64) + shift, "longitudes": np.arange(n, dtype=np.float64) * 2}


def test_shared_forcings_serves_one_array_per_dates_on_one_grid():
    provider = Provider()
    cache = ForcingsCache(10 * 2 * 3 * 5 * 4)  # room for ten (2, 3, 5) float32 arrays
    shared = SharedForcings(provider, cache)
    assert shared.variables == ["a", "b"] and shared.kinds == {"computed": True}
    np.testing.assert_array_equal(shared.mask, [3, 4])
    s1 = state()
    dates = [T0, T0 + 6 * HOUR]
    first = shared.load_forcings_array(dates, s1)
    again = shared.load_forcings_array(list(dates), s1)  # another member, same dates, same state objects
    assert again is first and provider.calls == [dates]
    # a later init time on the same grid, new numpy objects with equal values: still a hit
    later = shared.load_forcings_array(dates, state())
    assert later is first and cache.stats == {"computed": 1, "hits": 2, "bypassed": 0}
    # other dates are computed, and a single date is accepted as the runner passes it
    other = shared.load_forcings_array([T0 + 12 * HOUR], s1)
    assert other.shape == (2, 1, 5) and other[0, 0, 0] == 12.0
    np.testing.assert_array_equal(shared.load_forcings_array(T0 + 12 * HOUR, s1), other)
    assert cache.stats == {"computed": 2, "hits": 3, "bypassed": 0}
    # a different grid bypasses the cache and is neither served nor stored
    off = shared.load_forcings_array(dates, state(shift=0.5))
    np.testing.assert_array_equal(off, first)  # the fake provider ignores the grid; a real one would not
    assert off is not first and cache.stats["bypassed"] == 1 and len(provider.calls) == 3
    assert shared.load_forcings_array(dates, s1) is first
    assert repr(shared).startswith("Shared(")


def test_forcings_cache_budget_and_clear():
    provider = Provider(n=4)
    one = 2 * 1 * 4 * 4  # bytes of one (2, 1, 4) float32 array
    cache = ForcingsCache(2 * one)
    shared = SharedForcings(provider, cache)
    s = state(4)
    arrays = [shared.load_forcings_array([T0 + k * HOUR], s) for k in range(3)]
    assert cache.stats["computed"] == 3 and cache._size == 2 * one
    assert shared.load_forcings_array([T0 + 2 * HOUR], s) is arrays[2]  # newest kept
    assert shared.load_forcings_array([T0], s) is not arrays[0]  # oldest evicted and recomputed
    big = ForcingsCache(one // 2)
    SharedForcings(provider, big).load_forcings_array([T0], s)
    assert big._size == 0 and big.stats["computed"] == 1  # too large to store, still served
    cache.clear()
    assert cache._size == 0 and shared.load_forcings_array([T0 + 2 * HOUR], s) is not arrays[2]
    with pytest.raises(ValueError, match="negative"):
        ForcingsCache(-1)


def test_inference_chunks_reports_the_larger_of_env_and_checkpoint_mapper_counts(monkeypatch):
    import anemoi.evaluation.sources.anemoi_inference as module

    monkeypatch.setattr(module, "_chunk_constants", lambda: {"processor": 1, "mapper": 8})
    inner = SimpleNamespace(
        encoder={"ds": SimpleNamespace(num_chunks=4)}, decoder={"ds": SimpleNamespace(num_chunks=4)}
    )
    assert inference_chunks(SimpleNamespace(model=inner)) == {"processor": 1, "mapper": 8}
    monkeypatch.setattr(module, "_chunk_constants", lambda: {"processor": 1, "mapper": 2})
    assert inference_chunks(SimpleNamespace(model=inner)) == {"processor": 1, "mapper": 4}
    assert inference_chunks(SimpleNamespace()) == {"processor": 1, "mapper": 2}
    monkeypatch.setattr(module, "_chunk_constants", lambda: None)  # anemoi-models not installed
    assert inference_chunks(SimpleNamespace(model=inner)) == {}


def test_graph_node_attribute_reads_both_graph_data_forms():
    """anemoi-models exposes `graph_data` either as the HeteroData or as a dict of them keyed by
    dataset name; indexing a HeteroData with an attribute name silently yields an empty store."""
    import torch
    from torch_geometric.data import HeteroData

    from anemoi.evaluation.sources.anemoi_inference import InferenceForecastSource

    graph = HeteroData()
    graph["data"].area_weight = torch.tensor([[0.25], [0.75]])
    source = InferenceForecastSource.__new__(InferenceForecastSource)
    source.dataset_name = "ds"

    source.runner = SimpleNamespace(model=SimpleNamespace(graph_data=graph))
    np.testing.assert_array_equal(source.graph_node_attribute("area_weight"), [0.25, 0.75])
    source.runner = SimpleNamespace(model=SimpleNamespace(graph_data={"ds": graph}))
    np.testing.assert_array_equal(source.graph_node_attribute("area_weight"), [0.25, 0.75])
    source.runner = SimpleNamespace(model=SimpleNamespace(graph_data={"other": graph}))
    np.testing.assert_array_equal(source.graph_node_attribute("area_weight"), [0.25, 0.75])
    with pytest.raises(ValueError, match="no attribute 'cutout_mask'"):
        source.graph_node_attribute("cutout_mask")
