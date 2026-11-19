"""Member-parallel building blocks on gloo, in spawned CPU processes with a file store and a 30 s group timeout."""

import datetime
import json
import logging
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
from anemoi.evaluation.distributed import TIMING_ATTRS
from anemoi.evaluation.distributed import abort
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


def _rank_attrs(rank: int, case: dict, block: slice) -> dict:
    """Attrs as `Evaluation.attrs` writes them, with numbers that differ per rank and the non-finite climatology nodes
    of the rank's own variables only, so the merge rules can be told apart."""
    nonfinite = (~case["climatology"][block].isfinite()).sum(-1).tolist()
    return {
        "checkpoint": "x.ckpt",
        "checkpoint_uuid": "1234",
        "gpu": "A100",
        "config": "forecast: {}",
        "package_versions": json.dumps({"torch": "2"}),
        "init_times": "2024-01-01",
        **{key: 1.5 * (rank + 1) for key in TIMING_ATTRS},
        "model_calls": 2,
        "target_reads": 4 + rank,
        "target_cache_hits": rank,
        "forcings_computed": 3,
        "forcings_hits": 10 * rank,
        "peak_gpu_memory_bytes": 100 * (3 - rank),
        "climatology_nonfinite_nodes": json.dumps(dict(zip(case["variables"][block], nonfinite))),
    }


def _check_merged_attrs(merged: dict, case: dict, size: int, label: str) -> None:
    ranks = [_rank_attrs(rank, case, part) for rank, part in enumerate(blocks(len(case["variables"]), size))]
    for key in TIMING_ATTRS:
        assert merged[key] == sum(item[key] for item in ranks), f"{label} {key}"
    assert merged["target_reads"] == sum(4 + rank for rank in range(size)), label
    assert merged["forcings_hits"] == sum(10 * rank for rank in range(size)) and merged["forcings_computed"] == 3 * size
    assert merged["peak_gpu_memory_bytes"] == 300 and merged["model_calls"] == 2, label
    whole = dict(zip(case["variables"], (~case["climatology"].isfinite()).sum(-1).tolist()))
    assert json.loads(merged["climatology_nonfinite_nodes"]) == whole and whole[case["variables"][-1]] == 5, label
    assert {key: merged[key] for key in ("checkpoint", "gpu", "config", "init_times")} == {
        key: ranks[0][key] for key in ("checkpoint", "gpu", "config", "init_times")
    }, label


def _split_worker(rank: int, size: int, store: str, out: str) -> None:
    _init(rank, size, store)
    pair = dist.new_group([1, 2])
    # the last case leaves rank 2 with neither members nor variables
    for group, members, n_variables in ((None, 7, 5), (pair, 7, 5), (None, 5, 2), (None, 2, 2)):
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
            state = _aggregate(case, restricted, block, frames)
            state.attrs = _rank_attrs(group_rank, case, block)
            gathered.append(gather_state(state, group))
        if group_rank != 0:
            assert gathered == [None, None]
            continue
        for state in gathered:
            _check_merged_attrs(state.attrs, case, group_size, label)
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
    statistics and gathered give the unsplit state bit for bit, also after a netcdf round trip and `merge`, and the
    attrs merged by the rules of `merge_attrs`; with `V < R` one rank scores no variable, with `M, V < R` one rank has
    neither members nor variables, and every rank but one lacks the threshold variable in some case."""
    _spawn(_split_worker, 3, str(tmp_path / "store"), str(tmp_path))


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _abort_worker(rank: int, size: int, store: str, done: str) -> None:
    _init(rank, size, store)
    dist.barrier()
    if rank == 1:
        time.sleep(0.3)  # rank 0 is inside the swap by now
        try:
            with abort_on_error():
                raise RuntimeError("boom")
        except RuntimeError as error:
            assert str(error) == "boom"
        else:
            raise AssertionError("abort_on_error swallowed the error")
        records = _Records()
        logging.getLogger("anemoi.evaluation.distributed").addHandler(records)
        abort()  # the group is gone: one line, no traceback, no error
        assert [(r.levelno, r.exc_info) for r in records.records] == [(logging.WARNING, None)], records.records
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

    def state(variables, n_init=1, inits=INITS[:1], attrs=None, members=3):
        result = AggregationState.zeros(
            LEADS, ["all"], variables, REGIONS, ["error"], members, "cpu", {"param": variables}
        )
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
    with pytest.raises(ValueError, match="members"):
        concat_states([state(["a"]), state(["b"], members=4)])
    with pytest.raises(ValueError, match="overlap"):
        concat_states([state(["a", "b"]), state(["b"])])

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
    for key in ("checkpoint_uuid", "config", "package_versions"):
        with pytest.raises(ValueError, match=key):
            merge_attrs([{**ranks[0], key: "a"}, {**ranks[1], key: "b"}])


# The attrs a member group's file may differ in from the unsplit run's: the timings and the read and forcings
# counters, summed over the ranks, and the group size.
GROUP_ATTRS = (*TIMING_ATTRS, "target_reads", "target_cache_hits", "member_ranks")
FORCINGS = ("forcings_computed", "forcings_hits", "forcings_bypassed")
NOISY = ["rmse", "fair_crps", "spread", "rank_histogram", {"brier": {"label": "wet", "thresholds": {"tp": 0.0}}}]


def _noisy_runners(monkeypatch, checkpoint, names) -> None:
    """The fixture runner, its model adding a standard normal draw of the global RNG to every output, so the members
    differ and depend on their seeds; the targets come from the fixture too."""
    import _multi_dataset as fixture
    import anemoi.inference.runners as runners

    def create_runner(config):
        runner = fixture.make_runner(checkpoint, names)
        predict = runner.predict_step

        def noisy(*args, **kwargs):
            outputs = predict(*args, **kwargs)
            if isinstance(outputs, dict):
                return {name: value + torch.randn_like(value) for name, value in outputs.items()}
            return outputs + torch.randn_like(outputs)

        runner.predict_step = noisy
        return runner

    monkeypatch.setattr(runners, "create_runner", create_runner)
    fixture.patch_targets_from_forecast(monkeypatch)


def _patch(monkeypatch, case: dict) -> None:
    if case["kind"] == "downscaler":
        from test_temporal_downscaler import patch_downscaler_runner

        patch_downscaler_runner(monkeypatch, case["checkpoint"], noise=1.0)
        import _multi_dataset as fixture

        fixture.patch_targets_from_forecast(monkeypatch)
    else:
        _noisy_runners(monkeypatch, case["checkpoint"], case["names"])


def _cli_worker(rank: int, size: int, cases: list[dict]) -> None:
    from anemoi.evaluation.__main__ import main

    torch.set_num_threads(1)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    with pytest.MonkeyPatch.context() as monkeypatch:
        for case in cases:
            if rank >= case["ranks"]:
                continue
            _patch(monkeypatch, case)
            os.environ.update(RANK=str(rank), WORLD_SIZE=str(case["ranks"]), MASTER_PORT=str(case["port"]))
            main(["run", case["config"], "--member-group", str(case["ranks"]), *case["args"]])


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _same_file(split: str, unsplit: str, ranks: int) -> None:
    """The netcdf files hold the same values, bit for bit, and the same attrs but `GROUP_ATTRS` and the forcings
    counters, which add up as the docs say."""
    import xarray as xr

    with xr.open_dataset(split, decode_timedelta=True) as a, xr.open_dataset(unsplit, decode_timedelta=True) as b:
        a, b = a.load(), b.load()
    for name in a.variables:
        x, y = a[name].values, b[name].values
        if x.dtype.kind == "f":
            assert _identical(torch.from_numpy(x), torch.from_numpy(y)), f"{split} {name}"
        else:
            assert np.array_equal(x, y), f"{split} {name}"
    xr.testing.assert_identical(a.drop_attrs(), b.drop_attrs())
    import yaml

    def kept(attrs: dict) -> dict:
        config = {key: value for key, value in yaml.safe_load(attrs["config"]).items() if key != "output"}
        return {**{k: v for k, v in attrs.items() if k not in GROUP_ATTRS + FORCINGS}, "config": config}

    assert kept(a.attrs) == kept(b.attrs), split  # the two configs differ in output.path only
    assert a.attrs["member_ranks"] == ranks and "member_ranks" not in b.attrs
    for key in ("target_reads", "target_cache_hits"):
        assert a.attrs[key] == ranks * b.attrs[key], f"{split} {key}"
    if "forcings_computed" in b.attrs:
        assert a.attrs["forcings_computed"] == ranks * b.attrs["forcings_computed"], split
        assert sum(a.attrs[key] for key in FORCINGS) == sum(b.attrs[key] for key in FORCINGS), split


def _config(tmp_path, label: str, checkpoint, members: int, **overrides) -> dict:
    import _multi_dataset as fixture

    return {
        "forecast": {"anemoi_inference": {"checkpoint": str(checkpoint), "device": "cpu", "members": members}},
        "lead_time": "12h",
        "init_times": {"dates": [date.isoformat() for date in fixture.init_times()]},
        "metrics": NOISY,
        "weights": {"uniform": {}},
        "bins": {"time": "none"},
        "output": {"path": str(tmp_path / f"{label}-{{shard}}-{{shards}}.nc")},
        **overrides,
    }


def test_a_member_group_writes_the_file_of_the_unsplit_run(tmp_path, monkeypatch):
    """`run --member-group R` on R spawned gloo ranks writes the file of the unsplit run of the same config, bit
    for bit apart from the summed counters: a forecaster at R = 3 with the lead-0 frame, a climatology and a
    threshold, with `V < R` (a rank scores no variable), over two datasets (`V < R` in one), a temporal downscaler at
    R = 2, and a shard of a member group (R = 2) merges with the other shard of an unsplit run."""
    import _multi_dataset as fixture
    import xarray as xr
    import yaml
    from test_temporal_downscaler import T0
    from test_temporal_downscaler import downscaler_patch

    from anemoi.evaluation.__main__ import main
    from anemoi.evaluation.output import merge
    from anemoi.evaluation.sources.climatology import ArrayClimatology

    era5 = fixture.save_checkpoint(tmp_path / "era5.ckpt", ("era5",))
    both = fixture.save_checkpoint(tmp_path / "both.ckpt")
    downscaler = fixture.save_checkpoint(tmp_path / "td.ckpt", ("era5",), patch=downscaler_patch())
    targets = fixture.make_targets("era5")
    rng = np.random.default_rng(0)
    fields = {hour: rng.normal(0, 1, (7, targets.grid.n)).astype(np.float32) for hour in (0, 6, 12, 18)}
    fields[6][3, :4] = np.nan
    climatology = tmp_path / "climatology.nc"
    ArrayClimatology(targets.grid, targets.variables[:7], fields).to_netcdf(climatology)
    thorough = {
        "metrics": [*NOISY, "acc"],
        "climatology": {"file": str(climatology)},
        "include_lead_zero": True,
    }
    configs = {
        "era5": (_config(tmp_path, "era5", era5, 5, **thorough), "forecaster", era5, ("era5",)),
        "narrow": (_config(tmp_path, "narrow", era5, 4, variables=["tp", "2t"]), "forecaster", era5, ("era5",)),
        "both": (
            _config(
                tmp_path,
                "both",
                both,
                4,
                variables={"datasets": {"era5": targets.variables[:7], "realch1": ["tp", "msl"]}},
                output={"path": str(tmp_path / "both-{dataset}-{shard}-{shards}.nc")},
            ),
            "forecaster",
            both,
            fixture.DATASET_NAMES,
        ),
        "td": (
            {
                **_config(tmp_path, "td", downscaler, 3),
                "lead_time": "18h",
                "init_times": {"dates": [T0.isoformat(), (T0 + 6 * HOUR).isoformat()]},
            },
            "downscaler",
            downscaler,
            ("era5",),
        ),
    }
    cases, runs = [], []
    for label, (config, kind, checkpoint, names) in configs.items():
        unsplit = {**config, "output": {"path": config["output"]["path"].replace(label, f"{label}-unsplit")}}
        for name, content in ((label, config), (f"{label}-unsplit", unsplit)):
            (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(content))
        case = {"kind": kind, "checkpoint": str(checkpoint), "names": names}
        _patch(monkeypatch, case)
        for args in (["--shard", "0/1"], ["--shard", "1/2"]) if label == "era5" else [[]]:
            main(["run", str(tmp_path / f"{label}-unsplit.yaml"), *args])
        for ranks in (3,) if kind == "forecaster" else (2,):
            config_path = tmp_path / f"{label}-{ranks}.yaml"
            output = config["output"]["path"].replace(label, f"{label}-{ranks}")
            config_path.write_text(yaml.safe_dump({**config, "output": {"path": output}}))
            cases.append({**case, "config": str(config_path), "ranks": ranks, "port": _free_port(), "args": []})
            runs.append((label, ranks, names))
    shard = tmp_path / "era5-shard.yaml"
    shard.write_text(
        yaml.safe_dump({**configs["era5"][0], "output": {"path": str(tmp_path / "era5-g-{shard}-{shards}.nc")}})
    )
    cases.append({**cases[0], "config": str(shard), "ranks": 2, "port": _free_port(), "args": ["--shard", "0/2"]})
    _spawn(_cli_worker, 3, cases, timeout=180.0)

    for label, ranks, names in runs:
        datasets = names if len(names) > 1 else [None]
        for dataset in datasets:
            middle = "" if dataset is None else f"{dataset}-"
            _same_file(
                str(tmp_path / f"{label}-{ranks}-{middle}0-1.nc"),
                str(tmp_path / f"{label}-unsplit-{middle}0-1.nc"),
                ranks,
            )
    with xr.open_dataset(tmp_path / "era5-unsplit-0-1.nc", decode_timedelta=True) as reference:
        assert (reference["spread"].isel(lead_time=slice(1, None)).values > 0.5).all()  # the members differ
        assert reference["acc"].notnull().any() and reference.sizes["variable"] == 7
    merged = merge([tmp_path / "era5-g-0-2.nc", tmp_path / "era5-unsplit-1-2.nc"])
    whole = merge([tmp_path / "era5-unsplit-0-1.nc"])
    assert merged.coords == whole.coords and merged.init_times == whole.init_times
    for key, total in whole.sums.items():
        assert _identical(merged.sums[key], total), key


def test_member_group_plan_shards_and_refusals(tmp_path, monkeypatch, capsys):
    """The plan of rank 0 of a member group, the shards of a member-group step, and what a member group refuses."""
    import _multi_dataset as fixture
    import yaml

    from anemoi.evaluation.__main__ import main
    from anemoi.evaluation.__main__ import resolve_shard
    from anemoi.evaluation.distributed import MemberGroup
    from anemoi.evaluation.evaluate import Evaluation
    from anemoi.evaluation.sources.persistence import PersistenceForecastSource

    checkpoint = fixture.save_checkpoint(tmp_path / "era5.ckpt", ("era5",))
    _noisy_runners(monkeypatch, checkpoint, ("era5",))
    for name in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("SLURM_STEP_NUM_TASKS", "1")

    def path(members: int) -> str:
        config = tmp_path / f"m{members}.yaml"
        metrics = NOISY if members > 1 else ["rmse"]
        config.write_text(yaml.safe_dump(_config(tmp_path, f"m{members}", checkpoint, members, metrics=metrics)))
        return str(config)

    evaluation = Evaluation.from_config(path(5))
    evaluation.split_members(MemberGroup(0, 2))
    plan, calls, n = evaluation.plan(step_time=2.0), evaluation.model_calls_per_init()[0], 288
    assert plan["member_group"] == {"ranks": 2, "rank": 0, "members": 3, "variables": ["2t", "10u", "10v", "msl"]}
    assert plan["time"]["model_per_init_s"] == calls * 3 * 2.0
    assert plan["time"]["run"]["gpu_hours"] == round(2 * calls * 3 * 2.0 * 2 / 3600, 2)
    sizes = plan["bytes"]
    assert sizes["frame"] == 5 * 4 * n * 4 and sizes["ensemble_temporaries"] == 4 * 5 * n * 8
    assert sizes["swap_peak"] == 3 * 7 * n * 4 + 3 * 3 * n * 4 + 5 * 4 * n * 4
    assert sizes["member_states"] == 3 * plan["forecast"]["per_member_state_bytes"]
    main(["run", path(5), "--member-group", "2", "--dry-run", "--step-time", "1"])
    assert "member_group:" in capsys.readouterr().out

    env = {"SLURM_STEP_NUM_TASKS": "4", "SLURM_PROCID": "2", "SLURM_ARRAY_TASK_ID": "3", "SLURM_ARRAY_TASK_MIN": "2"}
    env["SLURM_ARRAY_TASK_COUNT"] = "3"
    assert resolve_shard(None, env) == (6, 12) and resolve_shard(None, env, member_group=True) == (1, 3)

    with pytest.raises(SystemExit, match="one member"):
        main(["run", path(1), "--member-group", "2", "--dry-run"])
    with pytest.raises(SystemExit, match="every rank needs a member"):
        main(["run", path(2), "--member-group", "3", "--dry-run"])
    with pytest.raises(SystemExit, match="as many tasks"):
        main(["run", path(5), "--member-group", "2"])
    with pytest.raises(SystemExit, match="MASTER_ADDR and MASTER_PORT"):
        main(["run", path(5), "--member-group", "1"])
    targets = fixture.make_targets("era5")
    persistence = Evaluation(PersistenceForecastSource(targets, "6h", 2), targets, fixture.init_times(), "6h", ["rmse"])
    with pytest.raises(ValueError, match="PersistenceForecastSource cannot split its members"):
        persistence.split_members(MemberGroup(0, 2))
