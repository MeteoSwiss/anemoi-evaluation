"""Compare target prefetch depths and cache sizes on one config, in one process.

    python tools/ab_prefetch.py <config.yaml> --variants p0 p2 p2c2 p0

Needs a GPU: it loads the checkpoint of the config's `forecast.anemoi_inference` block and runs the whole config once
per variant. Repeat a variant (`p0 ... p0`) to bracket the measurement against drift. Single-dataset checkpoints only:
it reads a plain `regions` block.

A variant is `p<prefetch>[c<GiB>][t]`, e.g. `p6c2` = read six rows ahead with a 2 GiB decoded-row cache, `t` =
`numcodecs.blosc.use_threads = True` for that variant only. One `InferenceForecastSource` serves every variant; the
warm-up reads every valid time once (page cache) and runs one model step (triton and cuBLAS first-call overheads); each
variant then gets a fresh `DatasetTargets` and `Evaluation`.

Prints `AB:` lines: the warm-up cost, then per variant the timing totals (`model`, `target`, `statistics`, `total`), the
target source's counters (rows read, worker seconds, cache hits) and the peak GPU memory, and finally whether every
variant's aggregation state is bit-identical to the first one's.

`ANEMOI_INFERENCE_NUM_CHUNKS` is set to 8 unless the environment already sets it (needed at 1 km resolution to avoid
running out of GPU memory in the mapper).
"""

import argparse
import logging
import os
import re
import time

os.environ.setdefault("ANEMOI_INFERENCE_NUM_CHUNKS", "8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

import torch  # noqa: E402

import anemoi.evaluation as ae  # noqa: E402
from anemoi.evaluation.config import load_config  # noqa: E402
from anemoi.evaluation.sources.base import MissingTargetError  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("--variants", nargs="+", default=["p0", "p2", "p0"])
args = parser.parse_args()


def log(*items):
    print("AB:", *items, flush=True)


config = load_config(args.config)
forecast = ae.InferenceForecastSource(**config.forecast.anemoi_inference.model_dump())
regions = {key: (r if isinstance(r, str) else r.spec()) for key, r in config.regions.items()}
weights = config.weights.spec() if config.weights is not None else None


def build(prefetch, cache_bytes, init_times=None, lead_time=None):
    targets = ae.DatasetTargets.from_forecast(forecast, prefetch=prefetch, cache_bytes=cache_bytes)
    return ae.Evaluation(
        forecast,
        targets,
        init_times or config.init_times.resolve(),
        lead_time or config.lead_time,
        config.metrics,
        config.variables,
        weights,
        regions,
        config.bins.model_dump(),
        on_missing_target=config.on_missing_target,
    )


# warm-up: every target row once (page cache), one model step (triton/cuBLAS first-call overheads)
ev = build(0, 0)
t0, n = time.perf_counter(), 0
for valid in sorted({init + lead for init in ev.init_times for lead in ev.lead_times}):
    try:
        ev.targets.frame(valid, ev.variables, ev.device)
        n += 1
    except MissingTargetError:
        pass
log(f"warm-up: {n} target rows read in {time.perf_counter() - t0:.1f}s")
build(0, 0, ev.init_times[:1], forecast.timestep).run()
log("warm-up: one model step done")

states = []
for label in args.variants:
    match = re.fullmatch(r"p(\d+)(?:c(\d+))?(t)?", label)
    prefetch, cache, threads = int(match[1]), int(match[2] or 0) * 2**30, bool(match[3])
    import numcodecs.blosc

    previous = numcodecs.blosc.use_threads
    numcodecs.blosc.use_threads = True if threads else previous
    ev = build(prefetch, cache)
    state = ev.run()
    ev.targets.close()
    numcodecs.blosc.use_threads = previous
    timing = {k: round(v, 2) for k, v in ev.timing.items()}
    log(
        f"{label}: prefetch {prefetch} cache {cache / 2**30:.0f} GiB blosc threads {threads}: timing {timing}, targets {ev.targets.stats}, "
        f"peak {ev.peak_memory_bytes / 2**30:.2f} GiB"
    )
    states.append((label, state))

ref_label, ref = states[0]
for label, state in states[1:]:
    exact = all(torch.equal(ref.sums[name], state.sums[name]) for name in ref.sums)
    rel = max(float(((ref.sums[k] - state.sums[k]).abs() / ref.sums[k].abs().clamp_min(1e-30)).max()) for k in ref.sums)
    log(f"state {label} vs {ref_label}: {'bit-exact' if exact else f'max relative difference {rel:.3e}'}")
forecast.close()
