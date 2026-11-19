"""Member-parallel building blocks on gloo, in spawned CPU processes with a file store and a 30 s group timeout."""

import datetime
import json
import os
import time

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from anemoi.evaluation import metrics
from anemoi.evaluation.aggregation import AggregationState
from anemoi.evaluation.aggregation import Aggregator
from anemoi.evaluation.binning import SeasonBinning
from anemoi.evaluation.distributed import abort_on_error
from anemoi.evaluation.distributed import blocks
from anemoi.evaluation.distributed import concat_states
from anemoi.evaluation.distributed import gather_state
from anemoi.evaluation.distributed import merge_attrs
from anemoi.evaluation.distributed import swap
from anemoi.evaluation.frame import Frame
from anemoi.evaluation.output import merge
from anemoi.evaluation.output import to_xarray
from anemoi.evaluation.output import write

HOUR = datetime.timedelta(hours=1)
INITS = [datetime.datetime(2024, 1, 1), datetime.datetime(2024, 1, 2)]
LEADS = [6 * HOUR, 12 * HOUR]
REGIONS = ["all", "half", "third"]
WET = {"label": "wet", "thresholds": {"tp": 0.5}}
SPECS = ["rmse", "fair_crps", "spread", "member_rmse", "acc", "rank_histogram"]
SPECS += [{"csi": WET}, {"brier": WET}, {"reliability": WET}]


def _spawn(worker, nprocs: int, *args, timeout: float = 60.0) -> None:
    """Run `worker(rank, nprocs, *args)` in `nprocs` spawned processes, re-raising a worker's error; fail on a hang."""
    context = mp.start_processes(worker, (nprocs, *args), nprocs=nprocs, join=False, start_method="spawn")
    deadline = time.monotonic() + timeout
    while not context.join(timeout=1):
        if time.monotonic() > deadline:
            for process in context.processes:
                process.kill()
            pytest.fail(f"the workers did not finish within {timeout} s")


def _init(rank: int, size: int, store: str) -> None:
    torch.set_num_threads(2)
    dist.init_process_group(
        "gloo", init_method=f"file://{store}", rank=rank, world_size=size, timeout=datetime.timedelta(seconds=30)
    )


def _case(members: int, n_variables: int) -> dict:
    """Frames with spread, NaNs in the members, the target and the climatology, and a `tp` with a threshold."""
    rng = np.random.default_rng(members * 10 + n_variables)
    n = 1001
    variables = ["t", "tp", "u", "v", "z"][:n_variables]
    frames = []
    for init in INITS:
        for lead in LEADS:
            pred = rng.normal(280, 2, (members, n_variables, n)).astype(np.float32)
            target = rng.normal(280, 2, (1, n_variables, n)).astype(np.float32)
            pred[:, 1] = rng.gamma(0.5, 1.0, (members, n))
            target[:, 1] = rng.gamma(0.5, 1.0, (1, n))
            pred[1, 0, :4], target[0, 1, 10:13] = np.nan, np.nan
            frames.append((init, lead, torch.from_numpy(pred), torch.from_numpy(target)))
    climatology = rng.normal(280, 1, (n_variables, n)).astype(np.float32)
    climatology[-1, 20:25] = np.nan
    return {
        "members": members,
        "variables": variables,
        "weights": rng.random(n) + 0.1,
        "masks": np.stack([np.ones(n, dtype=bool), rng.random(n) < 0.5, rng.random(n) < 0.3], axis=1),
        "frames": frames,
        "climatology": torch.from_numpy(climatology),
    }


def _aggregate(case: dict, statistics: dict, block: slice, frames: list) -> AggregationState:
    """The state of the variables `block` over `frames`, whose members already hold only that block."""
    aggregator = Aggregator(case["weights"], case["masks"], statistics, SeasonBinning("init_time"))
    variables = case["variables"][block]
    coords = {"param": variables, "level": [float(len(name)) for name in variables]}
    state = aggregator.new_state(LEADS, variables, REGIONS, case["members"], coords)
    for init, lead, pred, target in frames:
        frame = Frame(init, lead, init + lead, variables, pred)
        aggregator.add(state, frame, target[:, block].contiguous(), {"climatology": case["climatology"][block]})
    return state


def _identical(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Same dtype, shape and NaN positions, and the same bits everywhere else (signed zeros included)."""
    if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a.isnan(), b.isnan()):
        return False
    if not a.is_floating_point():
        return torch.equal(a, b)
    bits = torch.int64 if a.dtype == torch.float64 else torch.int32
    return torch.equal(torch.where(a.isnan(), 0.0, a).view(bits), torch.where(b.isnan(), 0.0, b).view(bits))


def _split_worker(rank: int, size: int, store: str, out: str) -> None:
    _init(rank, size, store)
    pair = dist.new_group([1, 2])
    for group, members, n_variables in ((None, 7, 5), (pair, 7, 5), (None, 5, 2)):
        if dist.get_rank(group) < 0:
            continue
        group_rank, group_size = dist.get_rank(group), dist.get_world_size(group)
        label = f"M={members} V={n_variables} R={group_size}"
        case = _case(members, n_variables)
        built = metrics.build(SPECS, case["variables"], members)
        statistics = metrics.unique_statistics(built)
        block = blocks(n_variables, group_size)[group_rank]
        mine = blocks(members, group_size)[group_rank]
        restricted = {name: statistic.restrict(block) for name, statistic in statistics.items()}
        gathered = []
        for init in INITS:
            frames = []
            for frame_init, lead, pred, target in case["frames"]:
                if frame_init == init:
                    swapped = swap(pred[mine].contiguous(), members, group)
                    assert _identical(swapped, pred[:, block].contiguous()), f"{label} rank {group_rank}"
                    frames.append((frame_init, lead, swapped, target))
            gathered.append(gather_state(_aggregate(case, restricted, block, frames), group))
        if group_rank != 0:
            assert gathered == [None, None]
            continue
        full = _aggregate(case, statistics, slice(None), case["frames"])
        path = f"{out}/{label.replace(' ', '_')}.nc"
        write(to_xarray(gathered[1], built), path)
        for state in (gathered[0].merge(gathered[1]), merge([gathered[0], path])):
            assert state.coords == full.coords and state.members == full.members, label
            assert state.variable_coords == full.variable_coords and state.init_times == full.init_times, label
            assert state.sums.keys() == full.sums.keys(), label
            assert _identical(state.weights, full.weights) and _identical(state.n_init, full.n_init), label
            for name, total in state.sums.items():
                assert _identical(total, full.sums[name]), f"{label} {name}"
        wet = full.sums["brier_wet"][full.n_init > 0]
        assert wet[:, 0].isnan().all() and wet[:, 1].isfinite().all(), label
    dist.destroy_process_group()


def test_member_split_scores_like_the_unsplit_run(tmp_path):
    """Members split over 3 ranks and over a 2-rank subgroup, swapped to variable blocks, scored with restricted
    statistics and gathered give the unsplit state bit for bit, also after a netcdf round trip and `merge`; with
    `V < R` one rank scores no variable, and every rank but one lacks the threshold variable in some case."""
    _spawn(_split_worker, 3, str(tmp_path / "store"), str(tmp_path))


def _abort_worker(rank: int, size: int, store: str, done: str) -> None:
    _init(rank, size, store)
    if rank == 1:
        try:
            with abort_on_error():
                raise RuntimeError("boom")
        except RuntimeError as error:
            assert str(error) == "boom"
        else:
            raise AssertionError("abort_on_error swallowed the error")
        deadline = time.monotonic() + 30
        while not os.path.exists(done) and time.monotonic() < deadline:
            time.sleep(0.05)
        return
    start = time.perf_counter()
    try:
        swap(torch.zeros(1, 2, 5), 2)
    except RuntimeError:
        elapsed = time.perf_counter() - start
    else:
        raise AssertionError("the swap succeeded without the failed rank")
    finally:
        open(done, "w").close()
    assert elapsed < 10, f"the swap took {elapsed:.1f} s to fail"


def test_abort_on_error_unblocks_the_other_ranks(tmp_path):
    """A rank that raises aborts the group and re-raises, and the rank waiting for it in a swap fails promptly while
    the failed process is still alive, instead of waiting for the group timeout."""
    _spawn(_abort_worker, 2, str(tmp_path / "store"), str(tmp_path / "done"))


def test_blocks_concat_and_attrs():
    assert [(b.start, b.stop) for b in blocks(7, 3)] == [(0, 3), (3, 5), (5, 7)]
    assert [b.stop - b.start for b in blocks(2, 3)] == [1, 1, 0]
    for count in range(12):
        for parts in range(1, 6):
            sizes = [b.stop - b.start for b in blocks(count, parts)]
            assert sum(sizes) == count and max(sizes) - min(sizes) <= 1
    with pytest.raises(ValueError):
        blocks(3, 0)
    with pytest.raises(ValueError, match="bound"):
        metrics.build([{"brier": WET}])[0].statistics["brier_wet"].restrict(slice(0, 1))

    def state(variables, n_init=1, inits=INITS[:1], attrs=None):
        result = AggregationState.zeros(LEADS, ["all"], variables, REGIONS, ["error"], 3, "cpu", {"param": variables})
        result.n_init += n_init
        result.init_times, result.attrs = list(inits), dict(attrs or {})
        return result

    joined = concat_states([state(["a", "b"]), state([]), state(["c"])])
    assert joined.variables == ["a", "b", "c"] and joined.variable_coords == {"param": ["a", "b", "c"]}
    assert joined.sums["error"].shape == (2, 1, 3, 3) and joined.n_init.tolist() == [[1], [1]]
    with pytest.raises(ValueError, match="other frames"):
        concat_states([state(["a"]), state(["b"], n_init=2)])
    with pytest.raises(ValueError, match="init_times"):
        concat_states([state(["a"]), state(["b"], inits=INITS[1:])])

    ranks = [
        {"time_model_s": 1.5, "target_reads": 3, "forcings_hits": 2, "peak_gpu_memory_bytes": 10, "model_calls": 4},
        {"time_model_s": 2.0, "target_reads": 1, "forcings_hits": 1, "peak_gpu_memory_bytes": 30, "model_calls": 4},
    ]
    for item, nonfinite in zip(ranks, ({"t": 0, "tp": 2}, {"tp": 2, "u": 1})):
        item.update(gpu="A100", checkpoint="x.ckpt", climatology_nonfinite_nodes=json.dumps(nonfinite))
    merged = merge_attrs(ranks)
    assert merged["time_model_s"] == 3.5 and merged["target_reads"] == 4 and merged["forcings_hits"] == 3
    assert merged["peak_gpu_memory_bytes"] == 30 and merged["model_calls"] == 4 and merged["checkpoint"] == "x.ckpt"
    assert json.loads(merged["climatology_nonfinite_nodes"]) == {"t": 0, "tp": 2, "u": 1}
    with pytest.raises(ValueError, match="gpu"):
        merge_attrs([ranks[0], {**ranks[1], "gpu": "PG506-232"}])
    with pytest.raises(ValueError, match="inference_chunks_processor"):
        merge_attrs([{**ranks[0], "inference_chunks_processor": 2}, ranks[1]])
