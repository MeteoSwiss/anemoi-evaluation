"""Member-parallel evaluation: the members of one init time spread over the ranks of a process group.

Rank `r` of `R` runs the member block `blocks(M, R)[r]` and scores the variable block `blocks(V, R)[r]` of every
member, after `swap` has exchanged the frames. Each rank aggregates its variables with statistics bound to the full
variable list and `restrict`-ed to its block, and `gather_state` concatenates the states on rank 0 into the state an
unsplit run would have produced. Nothing here is imported by a run that does not split its members.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import json
import logging
import os
from collections.abc import Iterator
from collections.abc import Mapping

import torch
import torch.distributed as dist

from anemoi.evaluation.aggregation import AggregationState
from anemoi.evaluation.output import MAX_ATTRS
from anemoi.evaluation.output import TIMING_ATTRS

LOG = logging.getLogger(__name__)
SUMMED_ATTRS = ("target_reads", "target_cache_hits")
SUMMED_PREFIXES = ("forcings_",)
IDENTITY_ATTRS = ("gpu", "checkpoint_uuid", "config", "package_versions")
IDENTITY_PREFIXES = ("inference_chunks_",)
GROUP_TIMEOUT = datetime.timedelta(minutes=30)


def blocks(count: int, parts: int) -> list[slice]:
    """`count` items in `parts` contiguous blocks whose sizes differ by at most one, the larger ones first; a block is
    empty when `count < parts`."""
    if count < 0 or parts < 1:
        raise ValueError(f"cannot split {count} items into {parts} blocks")
    size, extra = divmod(count, parts)
    bounds = [part * size + min(part, extra) for part in range(parts + 1)]
    return [slice(start, stop) for start, stop in zip(bounds[:-1], bounds[1:])]


@dataclasses.dataclass(frozen=True)
class MemberGroup:
    """Rank `rank` of a member group of `size` ranks, the world group of the run."""

    rank: int
    size: int

    def __post_init__(self) -> None:
        if not 0 <= self.rank < self.size:
            raise ValueError(f"rank {self.rank} is out of range for a member group of {self.size}")

    def block(self, count: int) -> slice:
        """This rank's block of `count` members or variables."""
        return blocks(count, self.size)[self.rank]


def task_env(env: Mapping[str, str] | None = None) -> tuple[int, int, int]:
    """Rank, task count and node-local rank of this process: torchrun's `RANK`, `WORLD_SIZE` and `LOCAL_RANK`, else
    srun's `SLURM_PROCID`, `SLURM_STEP_NUM_TASKS` and `SLURM_LOCALID`, else one task."""
    env = os.environ if env is None else env
    for names in (("RANK", "WORLD_SIZE", "LOCAL_RANK"), ("SLURM_PROCID", "SLURM_STEP_NUM_TASKS", "SLURM_LOCALID")):
        if names[0] in env and names[1] in env:
            return int(env[names[0]]), int(env[names[1]]), int(env.get(names[2], 0))
    return 0, 1, 0


def init_group(group: MemberGroup, device: torch.device, timeout: datetime.timedelta = GROUP_TIMEOUT) -> None:
    """Join the world group of `group`: NCCL on a CUDA device, gloo otherwise, rendezvous at `MASTER_ADDR` and
    `MASTER_PORT`, then a barrier, so that every rank is up before the model loads and every later batch of sends and
    receives may leave ranks out (NCCL needs all of them in the first one). With NCCL the CUDA device must be set
    first."""
    missing = [name for name in ("MASTER_ADDR", "MASTER_PORT") if not os.environ.get(name)]
    if missing:
        raise ValueError(
            f"a member group needs {' and '.join(missing)} in the environment, the address and a free port of rank 0, "
            "for example MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) MASTER_PORT=29500"
        )
    nccl = device.type == "cuda"
    dist.init_process_group(
        "nccl" if nccl else "gloo",
        init_method="env://",
        rank=group.rank,
        world_size=group.size,
        timeout=timeout,
        **({"device_id": torch.device("cuda", torch.cuda.current_device())} if nccl else {}),
    )
    dist.barrier()


def _rank_and_size(group: dist.ProcessGroup | None) -> tuple[int, int]:
    return dist.get_rank(group), dist.get_world_size(group)


def _global(group: dist.ProcessGroup | None, rank: int) -> int:
    return rank if group is None else dist.get_global_rank(group, rank)


def swap(data: torch.Tensor, members: int, group: dist.ProcessGroup | None = None) -> torch.Tensor:
    """This rank's (M_r, V, N) member block -> the (M, V_r, N) variable block of every member, in global member order.

    The members of rank `r` are `blocks(members, R)[r]` and its variables `blocks(V, R)[r]`. Every rank of `group` must
    call it with the same `members`, `V` and `N`, including the ranks whose member or variable block is empty; the
    blocks are derived from them, so ranks that disagree exchange messages of the wrong size. It is one batch of
    point-to-point sends and receives, which gloo and NCCL both run (gloo has no list all-to-all and NCCL serialises
    unbatched ones); with NCCL the current CUDA device must be set, and the first operation on the group must involve
    all its ranks, which a rank with neither members nor variables would skip.

    The peak memory is the input, the contiguous copies sent to the other ranks and the output, about three times the
    (M_r, V, N) frame when the blocks are even.
    """
    rank, size = _rank_and_size(group)
    member_blocks, variable_blocks = blocks(members, size), blocks(data.shape[1], size)
    mine = member_blocks[rank]
    if data.shape[0] != mine.stop - mine.start:
        raise ValueError(f"rank {rank} holds {data.shape[0]} members, its block of {members} is {mine}")
    out = torch.empty(
        (members, variable_blocks[rank].stop - variable_blocks[rank].start, data.shape[2]),
        dtype=data.dtype,
        device=data.device,
    )
    out[mine] = data[:, variable_blocks[rank]]
    ops = []
    for peer in range(size):
        if peer == rank:
            continue
        outgoing = data[:, variable_blocks[peer]].contiguous()
        if outgoing.numel():
            ops.append(dist.P2POp(dist.isend, outgoing, _global(group, peer), group))
        incoming = out[member_blocks[peer]]
        if incoming.numel():
            ops.append(dist.P2POp(dist.irecv, incoming, _global(group, peer), group))
    if ops:
        for work in dist.batch_isend_irecv(ops):
            work.wait()
    return out


def merge_attrs(attrs: list[dict]) -> dict:
    """The attrs of a member group's result from its ranks', in rank order.

    Timings and the target-read and forcings counters are summed, `peak_gpu_memory_bytes` is the maximum, the
    `climatology_nonfinite_nodes` maps are unioned, and everything else, `model_calls` included (every rank makes the
    same calls, each for its own members), is rank 0's. Ranks on different GPU models or inference chunk counts are refused,
    since the bit-exact members depend on both, and so are ranks that disagree on the checkpoint uuid, the config or
    the package versions.
    """
    if not attrs:
        raise ValueError("no attrs to merge")
    keys = {key for item in attrs for key in item}
    for key in sorted(keys):
        if key in IDENTITY_ATTRS or key.startswith(IDENTITY_PREFIXES):
            values = [item.get(key) for item in attrs]
            if len({str(value) for value in values}) > 1:
                raise ValueError(
                    f"the ranks disagree on {key!r}: {values}; a member group runs one setup on one GPU model"
                )
    result = dict(attrs[0])
    for key in keys:
        values = [item[key] for item in attrs if key in item]
        if key in TIMING_ATTRS:
            result[key] = float(sum(values))
        elif key in SUMMED_ATTRS or key.startswith(SUMMED_PREFIXES):
            result[key] = int(sum(values))
        elif key in MAX_ATTRS:
            result[key] = int(max(values))
    if "climatology_nonfinite_nodes" in keys:
        union: dict[str, int] = {}
        for item in attrs:
            for name, count in json.loads(item.get("climatology_nonfinite_nodes", "{}")).items():
                if union.setdefault(name, count) != count:
                    raise ValueError(f"the ranks disagree on the non-finite climatology nodes of {name!r}")
        result["climatology_nonfinite_nodes"] = json.dumps(union)
    return result


def concat_states(states: list[AggregationState]) -> AggregationState:
    """The states of consecutive variable blocks, in rank order, as one state over all the variables; the ranks must
    agree on everything but the variables."""
    if not states:
        raise ValueError("no states to concatenate")
    first = states[0]
    names = [name for state in states for name in state.variables]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"the variable blocks overlap: {duplicates} are in more than one")
    for rank, state in enumerate(states[1:], start=1):
        for name in ("lead_times", "bins", "regions", "members", "init_times"):
            if getattr(state, name) != getattr(first, name):
                raise ValueError(f"rank {rank} has {name} {getattr(state, name)}, rank 0 {getattr(first, name)}")
        if not torch.equal(state.n_init.cpu(), first.n_init.cpu()):
            raise ValueError(f"rank {rank} added other frames than rank 0, n_init {state.n_init} and {first.n_init}")
        if state.sums.keys() != first.sums.keys() or state.variable_coords.keys() != first.variable_coords.keys():
            raise ValueError(f"rank {rank} has other statistics or variable coordinates than rank 0")
    states = [state.cpu() for state in states]
    return AggregationState(
        list(first.lead_times),
        list(first.bins),
        names,
        list(first.regions),
        {name: torch.cat([state.sums[name] for state in states], dim=2) for name in first.sums},
        torch.cat([state.weights for state in states], dim=2),
        states[0].n_init.clone(),
        first.members,
        {name: [value for state in states for value in state.variable_coords[name]] for name in first.variable_coords},
        list(first.init_times),
        merge_attrs([state.attrs for state in states]),
    )


def gather_state(state: AggregationState, group: dist.ProcessGroup | None = None) -> AggregationState | None:
    """Rank 0 of `group` gets the states of all its ranks concatenated by `concat_states`, on the CPU; the others get
    None. The states travel pickled (`gather_object`), so with NCCL the current CUDA device must be set."""
    rank, size = _rank_and_size(group)
    gathered = [None] * size if rank == 0 else None
    dist.gather_object(state.cpu(), gathered, dst=_global(group, 0), group=group)
    return concat_states(gathered) if rank == 0 else None


def abort() -> None:
    """Tear down the world group, and every group made from it, without waiting for the other ranks: NCCL
    communicators are aborted, any other backend is destroyed; a failure to do so (a group already gone) is logged in
    one line, not raised."""
    try:
        if "nccl" in str(dist.get_backend()) and hasattr(dist.distributed_c10d, "_abort_process_group"):
            dist.distributed_c10d._abort_process_group()
        else:
            dist.destroy_process_group()
    except Exception as error:
        LOG.warning("could not abort the process group: %s", error)


@contextlib.contextmanager
def abort_on_error() -> Iterator[None]:
    """Abort the world group and re-raise when the block raises, so the other ranks fail instead of waiting for this
    one.

    Only the world group: destroying a gloo subgroup does not unblock its peers, they wait until this process exits.
    With gloo the other ranks see the connections close and raise in their next or current operation. With NCCL the
    abort frees only this rank's communicators: a rank already inside a collective waits for NCCL's timeout, and
    torch's abort is meant to run with `TORCH_NCCL_ASYNC_ERROR_HANDLING=0`, since otherwise NCCL's watchdog may abort
    the group by itself first. Run a member group under `srun --kill-on-bad-exit=1`, so that Slurm kills the other
    tasks of the step as soon as one exits with an error.
    """
    try:
        yield
    except BaseException:
        LOG.error("this rank of the member group failed, aborting the group")
        abort()
        raise
