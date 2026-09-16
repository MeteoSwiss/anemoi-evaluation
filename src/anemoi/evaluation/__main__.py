"""Command line interface: ``anemoi-evaluation run <config.yaml> [--shard i/n] [--dry-run [--step-time s]]``
and ``anemoi-evaluation merge <nc>... -o <nc> [--partial]``."""

import argparse
import json
import logging
import os
from collections.abc import Mapping

LOG = logging.getLogger(__name__)


def resolve_shard(shard: str | None, env: Mapping[str, str] | None = None) -> tuple[int, int]:
    """Shard index and count: ``--shard i/n``, else the Slurm job-array task times the job-step task, else 0 of 1.

    The task count comes from ``SLURM_STEP_NUM_TASKS`` (set by ``srun`` for every task of a step) rather than
    ``SLURM_NTASKS``, which a batch script also sees with ``SLURM_PROCID`` 0 and would silently evaluate one shard.
    """
    if shard is not None:
        try:
            index, count = (int(part) for part in shard.split("/"))
        except ValueError:
            raise ValueError(f"--shard must be of the form i/n, got {shard!r}") from None
    else:
        env = os.environ if env is None else env

        def value(name: str, default: int) -> int:
            return int(env.get(name, default))

        tasks = value("SLURM_STEP_NUM_TASKS", 1)
        array = value("SLURM_ARRAY_TASK_ID", 0) - value("SLURM_ARRAY_TASK_MIN", 0)
        count = value("SLURM_ARRAY_TASK_COUNT", 1) * tasks
        index = array * tasks + value("SLURM_PROCID", 0)
    if not 0 <= index < count:
        raise ValueError(f"shard index {index} is out of range for {count} shards")
    return index, count


def format_output_path(path: str, index: int, count: int) -> str:
    """`path` with ``{shard}`` and ``{shards}`` filled in; the ``{shard}`` placeholder is required for several shards."""
    if count > 1 and "{shard}" not in path:
        raise ValueError(f"output.path needs a {{shard}} placeholder to write {count} shards, got {path!r}")
    return path.format(shard=index, shards=count)


def _humanise(value: object, key: str = "", in_bytes: bool = False) -> object:
    if isinstance(value, dict):
        return {k: _humanise(v, str(k), key == "bytes") for k, v in value.items()}
    if isinstance(value, int) and not isinstance(value, bool) and (in_bytes or key.endswith("bytes")):
        from anemoi.utils.humanize import bytes_to_human

        return f"{bytes_to_human(value)} ({value})"
    if isinstance(value, (int, float)) and not isinstance(value, bool) and key.endswith("_s"):
        from anemoi.utils.humanize import seconds_to_human

        return f"{seconds_to_human(value)} ({value:.1f})"
    return value


def _headline(plan: dict, count: int) -> str:
    """One line of the dry run's time estimate, or how to ask for one."""
    estimate = plan.get("time")
    if estimate is None:
        return "no time estimate (pass --step-time <seconds per model call per member>, docs/benchmarks.md section 3)"
    from anemoi.utils.humanize import seconds_to_human

    over = f" per shard over {count} shards" if count > 1 else ""
    return (
        f"about {estimate['run']['gpu_hours']:.2f} GPU-hours of model time, about "
        f"{seconds_to_human(estimate['run']['wall_per_shard_s'])}{over}"
    )


def _run(args: argparse.Namespace) -> None:
    if args.step_time is not None and not args.dry_run:
        raise SystemExit("--step-time is only used by --dry-run")
    import torch
    import yaml
    from anemoi.utils.humanize import compress_dates

    from anemoi.evaluation.config import load_config
    from anemoi.evaluation.evaluate import Evaluation
    from anemoi.evaluation.output import write

    config = load_config(args.config)
    index, count = resolve_shard(args.shard)
    if config.output is None and not args.dry_run:
        raise SystemExit("the config needs output.path to run from the command line")
    path = format_output_path(config.output.path, index, count) if config.output is not None else None
    logging.basicConfig(level=config.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    evaluation = Evaluation.from_config(config)
    init_times = evaluation.shard(index, count)
    LOG.info("shard %d/%d: %d init times %s", index, count, len(init_times), ", ".join(compress_dates(init_times)))
    if not init_times:
        LOG.warning("shard %d/%d has no init times, an empty state will be written", index, count)
    if args.dry_run:
        plan = evaluation.plan(init_times, step_time=args.step_time, shards=count)
        plan["shard"] = {"index": index, "count": count, "output": path}
        print(yaml.safe_dump(_humanise(plan), sort_keys=False, width=120))
        LOG.info("dry run: %s; model loaded: %s", _headline(plan, count), plan["forecast"].get("model_loaded", "n/a"))
        evaluation.close()
        return
    device = evaluation.device
    if device.type == "cuda":
        LOG.info("device %s: %s, %d visible", device, torch.cuda.get_device_name(device), torch.cuda.device_count())
    else:
        LOG.info("device %s", device)
    try:
        state = evaluation.run(init_times)
    finally:
        evaluation.close()
    state.attrs["shard"] = f"{index}/{count}"
    write(state.to_xarray(evaluation.metrics), path)
    LOG.info("wrote %s", path)


def _merge(args: argparse.Namespace) -> None:
    from anemoi.evaluation import metrics
    from anemoi.evaluation.output import merge
    from anemoi.evaluation.output import to_xarray
    from anemoi.evaluation.output import write

    state = merge(args.inputs, partial=args.partial)
    write(to_xarray(state, [metrics.from_spec(spec) for spec in json.loads(state.attrs["metrics"])]), args.output)


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and dispatch to ``run`` or ``merge``."""
    parser = argparse.ArgumentParser(prog="anemoi-evaluation", description="Evaluate anemoi forecasts in memory.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run an evaluation from a YAML config and write the results netcdf.")
    run.add_argument("config", help="Path to the YAML config.")
    run.add_argument(
        "--shard",
        metavar="i/n",
        help="Evaluate every n-th init time starting at i and write output.path with {shard} filled in; "
        "without it, Slurm job-array and job-step variables shard automatically (0/1 disables that).",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the config and sources, check the dates and print the plan without running the model.",
    )
    run.add_argument(
        "--step-time",
        metavar="seconds",
        type=float,
        help="Seconds of model time per model call per member, to add a time and GPU-hour estimate to the plan; "
        "docs/benchmarks.md section 3 reports it per checkpoint, and one call of a multi-step-output model "
        "produces several frames.",
    )
    merge = commands.add_parser("merge", help="Sum the aggregation states of several result files.")
    merge.add_argument("inputs", nargs="+", help="Result netcdf files to merge.")
    merge.add_argument("-o", "--output", required=True, help="Merged result netcdf.")
    merge.add_argument(
        "--partial",
        action="store_true",
        help="Merge an incomplete set of shards, recording the shards merged in the output's shard attr; "
        "without it the inputs must be the complete 0..n-1 shards of one run.",
    )
    args = parser.parse_args(argv)
    {"run": _run, "merge": _merge}[args.command](args)


if __name__ == "__main__":
    main()
