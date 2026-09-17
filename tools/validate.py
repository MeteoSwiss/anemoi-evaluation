"""Check the framework's metrics against a direct `runner.run()` loop scored in numpy float64.

    python tools/validate.py <config.yaml> --init 2024-01-02T00 [--steps 2 | --lead 9h]
                             [--members 1 4] [--lead-zero] [--rtol 1e-5] [-o out.nc]

Needs a GPU: it loads the checkpoint of the config's `forecast.anemoi_inference` block and runs it.

One process: the framework `Evaluation` (config members) and the direct loop share the loaded model and the seeding
(`member_seed(seed, init, member, call)` before every member's model call), so the member draws are identical. For every
--members value the direct loop times every `next()` (model calls flagged: the first `next()` of each block of
`frames_per_pass` frames) and records its peak GPU memory; the numpy statistics (rmse, mae, bias, crps, fair_crps, spread,
spread_skill, member_rmse, member_mae, acc when the config has a climatology, and the threshold metrics, the rank
histogram and the reliability levels the config asks for) are compared for the config's member count; a metric carrying an extra axis, such as
`rank_histogram`, is compared element by element and reported by its worst element; a threshold metric must be NaN for
a variable its map omits, and a framework NaN against a finite reference, or the reverse, is a mismatch rather than a
skipped comparison.
With --lead-zero the framework's lead-0 rmse/mae/bias must be exactly 0.0 for the variables present in the input
state and NaN for the others.

Prints `VALIDATE:` lines: the checkpoint facts, the framework run's timing, model calls and peak memory, the direct
loop's per-`next()` times and peak memory per member count, every mismatch above --rtol, the maximum relative difference
per metric, and a final PASS or FAIL (also the exit status, 0 or 1). The framework run is written to --output
(default: `validate-<config stem>.nc` next to the config's `output.path`).

`ANEMOI_INFERENCE_NUM_CHUNKS` is set to 8 unless the environment already sets it (needed at 1 km resolution to avoid
running out of GPU memory in the mapper).
"""

import argparse
import itertools
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("ANEMOI_INFERENCE_NUM_CHUNKS", "8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from anemoi.datasets import open_dataset  # noqa: E402
from anemoi.utils.dates import as_datetime  # noqa: E402
from anemoi.utils.dates import frequency_to_timedelta  # noqa: E402

import anemoi.evaluation as ae  # noqa: E402
from anemoi.evaluation.config import load_config  # noqa: E402
from anemoi.evaluation.output import write  # noqa: E402
from anemoi.evaluation.sources.anemoi_inference import member_seed  # noqa: E402
from anemoi.evaluation.sources.anemoi_inference import quiet_inference  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("--init", required=True)
parser.add_argument("--steps", type=int, default=None)
parser.add_argument("--lead", default=None)
parser.add_argument("--members", type=int, nargs="+", default=None)
parser.add_argument("--rtol", type=float, default=1e-5)
parser.add_argument("--lead-zero", action="store_true")
parser.add_argument(
    "-o", "--output", default=None, help="Framework results file (default: beside the config's output.path)."
)
args = parser.parse_args()


def log(*items):
    print("VALIDATE:", *items, flush=True)


config = load_config(args.config)
block = config.forecast.anemoi_inference.model_dump()
init = as_datetime(args.init)
forecast = ae.InferenceForecastSource(**block)
targets = ae.DatasetTargets.from_forecast(forecast)
lead = frequency_to_timedelta(args.lead) if args.lead else (args.steps or 2) * forecast.timestep
members_list = args.members or [forecast.members]
name = forecast.dataset_name
runner = forecast.runner
device = forecast.device
mso = forecast.frames_per_pass
log(
    f"checkpoint {forecast.checkpoint}: timestep {forecast.timestep}, multi_step_input {forecast.multi_step_input}, "
    f"multi_step_output {forecast.multi_step_output}, lead {lead}, targets prefetch {targets.lookahead}"
)

climatology = None
if getattr(config, "climatology", None) is not None:
    from anemoi.evaluation.sources.climatology import ArrayClimatology

    climatology = ArrayClimatology.from_netcdf(config.climatology.file)
    log(f"climatology {config.climatology.file}: {climatology.describe()}")

# 1. framework
regions = {key: (r if isinstance(r, str) else r.spec()) for key, r in config.regions.items()}
extra = {"include_lead_zero": args.lead_zero}
if climatology is not None:
    extra["climatology"] = climatology
evaluation = ae.Evaluation(
    forecast,
    targets,
    [init],
    lead,
    config.metrics,
    config.variables,
    config.weights.spec() if config.weights is not None else None,
    regions,
    config.bins.model_dump(),
    on_missing_target=config.on_missing_target,
    **extra,
)
variables = evaluation.variables
assert "weights" not in evaluation.__dict__, "weights resolved at construction"
fields0 = {}
if args.lead_zero:  # the init-time slice of the input state; the framework reuses the memoised state
    t0 = time.perf_counter()
    state0 = forecast._initial_state(init)["fields"]
    fields0 = {v: np.asarray(state0[v][-1], dtype=np.float64) for v in variables if v in state0}
    log(
        f"initial state built in {time.perf_counter() - t0:.1f}s: {len(state0)} input fields, "
        f"{len(fields0)} of {len(variables)} evaluated variables present (NaN at lead 0 for the others)"
    )
state = evaluation.run()
log(
    f"contributed init times {state.init_times}; members {state.members}; bins {state.bins}; lead times {[str(t) for t in state.lead_times]}"
)
log("attrs", {k: v for k, v in state.attrs.items() if k != "config"})
framework = state.to_xarray(evaluation.metrics)
out = Path(args.output) if args.output else Path(config.output.path).parent / f"validate-{Path(args.config).stem}.nc"
write(framework, out)
log(
    f"framework run written to {out}; timing {evaluation.timing}; model calls {evaluation.model_calls}; "
    f"peak {evaluation.peak_memory_bytes / 2**30:.2f} GiB"
)

# 2. direct loop, per member count
dataset = open_dataset(*forecast.dataset_args_kwargs()[0], **forecast.dataset_args_kwargs()[1])
dates = dataset.dates.astype("datetime64[us]").tolist()
date_index = {date: i for i, date in enumerate(dates)}
columns = [dataset.name_to_index[v] for v in variables]
gi = slice(None) if forecast.grid_indices is None else forecast.grid_indices
weights = evaluation.weights
masks = evaluation.regions
fields_by_members = {}
for members in members_list:
    prognostic = runner.prognostics_inputs[name].create_input_state(date=init)
    constant = runner.constant_forcings_inputs[name].create_input_state(date=init)
    dynamic = runner.dynamic_forcings_inputs[name].create_input_state(date=init)
    initial = runner._combine_states(prognostic, constant, dynamic)
    runner.reference_date = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    generators = [runner.run(input_states={name: initial}, lead_time=lead, return_numpy=True) for _ in range(members)]
    steps, next_times = [], []
    t0 = time.perf_counter()
    try:
        with quiet_inference(), torch.inference_mode():
            for step_index in itertools.count():
                call, first = divmod(step_index, mso)
                outputs = []
                for member, generator in enumerate(generators):
                    if first == 0:
                        torch.manual_seed(member_seed(forecast.seed, init, member, call))
                    outputs.append(next(generator, None))
                if any(o is None for o in outputs):
                    assert all(o is None for o in outputs), "members out of step"
                    break
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                next_times.append((step_index, t1 - t0, first == 0))
                step, date = outputs[0][name]["step"], outputs[0][name]["date"]
                arrays = np.stack(
                    [
                        np.stack([np.asarray(o[name]["fields"][v], dtype=np.float64)[gi] for v in variables])
                        for o in outputs
                    ]
                )
                steps.append((step, date, arrays))
                t0 = time.perf_counter()
    finally:
        for generator in generators:
            generator.close()
    assert not torch.is_inference_mode_enabled()
    peak = torch.cuda.max_memory_allocated() / 2**30
    calls = [f"{s:.2f}" for _, s, is_call in next_times if is_call]
    others = [s for _, s, is_call in next_times if not is_call]
    log(
        f"M={members}: {len(steps)} states, model-call next() times {calls} s (first includes input tensor preparation), "
        f"other next() max {max(others) * 1e3 if others else 0:.1f} ms, peak GPU {peak:.2f} GiB"
    )
    got_steps = [s for s, _, _ in steps]
    expected_steps = evaluation.forecast.lead_times(lead)
    assert got_steps == expected_steps, (got_steps, expected_steps)
    fields_by_members[members] = steps

# 3. numpy float64 statistics for the framework's member count
steps = fields_by_members[forecast.members]
M = forecast.members
worst = {}
nan_expected = []  # (metric, lead, variable, region) a threshold metric must be NaN for: its map omits the variable
nan_mismatches = 0  # one side NaN and the other a number, which a relative difference cannot express
shape_mismatches = 0  # the framework and the reference disagree on the shape of a metric, which cannot be compared
for step, date, arrays in steps:
    y = np.asarray(dataset[date_index[date]][columns, 0], dtype=np.float64)[:, gi]
    c = None
    if climatology is not None:
        c = np.asarray(climatology.frame(date, variables, torch.device("cpu")).numpy(), dtype=np.float64)
    mean = arrays.mean(0)
    finite = np.isfinite(arrays).all(0) & np.isfinite(y)
    for region, mask in masks.items():
        for j, variable in enumerate(variables):
            ok = finite[j] & mask
            w = weights[ok]
            e = mean[j][ok] - y[j][ok]
            ref = {
                "rmse": np.sqrt((w * e**2).sum() / w.sum()),
                "mae": (w * np.abs(e)).sum() / w.sum(),
                "bias": (w * e).sum() / w.sum(),
            }
            m = arrays[:, j][:, ok]
            skill = np.abs(m - y[j][ok]).mean(0)
            ref["member_mae"] = (w * skill).sum() / w.sum()
            ref["member_rmse"] = np.sqrt((w * ((m - y[j][ok]) ** 2).mean(0)).sum() / w.sum())
            if M > 1:
                pairs = sum(np.abs(m[a] - m[b]) for a, b in itertools.combinations(range(M), 2))
                for alpha, key in ((0.0, "crps"), (1.0, "fair_crps")):
                    crps = skill - (alpha / (M * (M - 1)) + (1 - alpha) / M**2) * pairs
                    ref[key] = (w * crps).sum() / w.sum()
                var = m.var(0, ddof=1)
                ref["spread"] = np.sqrt((w * var).sum() / w.sum())
                ref["spread_skill"] = ref["spread"] / ref["rmse"]
            if c is not None:
                okc = ok & np.isfinite(c[j])
                wc = weights[okc]
                fa, ta = mean[j][okc] - c[j][okc], y[j][okc] - c[j][okc]
                ref["acc"] = (wc * fa * ta).sum() / np.sqrt((wc * fa * fa).sum() * (wc * ta * ta).sum())
            for metric in evaluation.metrics:
                thresholds = getattr(metric, "thresholds", None)
                if thresholds is None:
                    continue
                if variable not in thresholds:
                    nan_expected.append((metric.name, step, variable, region))
                    continue
                threshold = thresholds[variable]
                o = (y[j][ok] > threshold).astype(np.float64)
                binary = (mean[j][ok] > threshold).astype(np.float64)
                p = (m > threshold).astype(np.float64).mean(0)
                hits = (w * binary * o).sum() / w.sum()
                misses = (w * (1 - binary) * o).sum() / w.sum()
                alarms = (w * binary * (1 - o)).sum() / w.sum()
                negatives = (w * (1 - binary) * (1 - o)).sum() / w.sum()
                brier = (w * (p - o) ** 2).sum() / w.sum()
                obar = (w * o).sum() / w.sum()
                random_hits = (hits + alarms) * (hits + misses)
                with np.errstate(divide="ignore", invalid="ignore"):  # a degenerate cell is legitimately 0/0
                    formulas = {
                        "pod": hits / (hits + misses),
                        "far": alarms / (hits + alarms),
                        "csi": hits / (hits + misses + alarms),
                        "ets": (hits - random_hits) / (hits + misses + alarms - random_hits),
                        "frequency_bias": (hits + alarms) / (hits + misses),
                        "hss": 2
                        * (hits * negatives - misses * alarms)
                        / ((hits + misses) * (misses + negatives) + (hits + alarms) * (alarms + negatives)),
                        "pss": hits / (hits + misses) - alarms / (alarms + negatives),
                        "brier": brier,
                        "bss": 1 - brier / (obar * (1 - obar)),
                        "event_frequency": obar,
                        "brier_uncertainty": obar * (1.0 - obar),
                    }
                if metric.kind in formulas:  # the level-resolved metrics are handled below, from their own counts
                    ref[metric.name] = formulas[metric.kind]
            rank_metrics = [
                metric
                for metric in evaluation.metrics
                if isinstance(metric, (ae.metrics.RankHistogram, ae.metrics.OutlierFraction))
            ]
            if rank_metrics:  # the two counts are the same for every bin and for both metrics
                below = (m < y[j][ok]).sum(0)
                ties = (m == y[j][ok]).sum(0)
                bins = np.stack([((below <= k) & (k <= below + ties)) / (ties + 1.0) for k in range(M + 1)])
                histogram = (w * bins).sum(1) / w.sum()
                for metric in rank_metrics:
                    is_histogram = isinstance(metric, ae.metrics.RankHistogram)
                    ref[metric.name] = histogram if is_histogram else float(histogram[0] + histogram[-1])
            level_metrics = [
                metric
                for metric in evaluation.metrics
                if isinstance(metric, ae.metrics.ReliabilityMetric) and variable in metric.thresholds
            ]
            for label in sorted({metric.label for metric in level_metrics}):  # the counts serve every level metric
                threshold = next(metric.thresholds[variable] for metric in level_metrics if metric.label == label)
                count = (m > threshold).sum(0)  # (n_valid,) integer member exceedance count
                observed = (y[j][ok] > threshold).astype(np.float64)
                n = np.array([(w * (count == k)).sum() for k in range(M + 1)])
                h = np.array([(w * (count == k) * observed).sum() for k in range(M + 1)])
                total, p_k = n.sum(), np.arange(M + 1) / M  # named _k so as not to shadow the per-node p and o above
                o_k = np.divide(h, n, out=np.full_like(h, np.nan), where=n > 0)
                obar_k = h.sum() / total
                levels = {
                    "reliability": o_k,
                    "forecast_frequency": n / total,
                    "brier_reliability": np.nansum(np.where(n > 0, n * (p_k - o_k) ** 2, 0.0)) / total,
                    "brier_resolution": np.nansum(np.where(n > 0, n * (o_k - obar_k) ** 2, 0.0)) / total,
                }
                for metric in level_metrics:
                    if metric.label == label:
                        ref[metric.name] = levels[metric.kind]
            for key, value in ref.items():
                if key not in framework:
                    continue
                # A metric with an extra axis (rank_histogram) leaves an array here, a scalar one a 0-d array; the
                # comparison is elementwise either way and reports the worst element.
                got = np.asarray(
                    framework[key].sel(lead_time=np.timedelta64(step), bin="all", variable=variable, region=region),
                    dtype=np.float64,
                )
                value = np.asarray(value, dtype=np.float64)
                if got.shape != value.shape:
                    log(f"MISMATCH {key} lead={step} {variable} {region}: shapes {got.shape} and {value.shape}")
                    shape_mismatches += 1
                    continue
                pattern = np.isnan(got) != np.isnan(value)  # one NaN and one number is a mismatch, not rel = NaN
                if pattern.any():
                    log(f"MISMATCH {key} lead={step} {variable} {region}: framework {got!r} numpy {value!r} (NaN)")
                    nan_mismatches += 1
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    elementwise = np.abs(got - value) / np.maximum(np.abs(value), 1e-30)
                rel = float(np.max(np.where(np.isnan(got) | (got == value), 0.0, elementwise), initial=0.0))
                worst[key] = max(worst.get(key, 0.0), rel)
                if rel > args.rtol:
                    log(
                        f"MISMATCH {key} lead={step} {variable} {region}: framework {got!r} numpy {value!r} rel {rel:.2e}"
                    )
for key, rel in worst.items():
    log(f"max relative difference {key}: {rel:.2e}")
ok = all(rel <= args.rtol for rel in worst.values()) and nan_mismatches == 0 and shape_mismatches == 0
if nan_mismatches:
    log(f"{nan_mismatches} comparisons where exactly one of the framework and numpy was NaN")
if shape_mismatches:
    log(f"{shape_mismatches} comparisons where the framework and numpy disagreed on the shape")
missing_nan = 0
for key, step, variable, region in nan_expected:
    if key not in framework:
        continue
    got = np.asarray(
        framework[key].sel(lead_time=np.timedelta64(step), bin="all", variable=variable, region=region),
        dtype=np.float64,
    )
    if not np.isnan(got).all():
        log(f"MISMATCH {key} lead={step} {variable} {region}: framework {got!r}, expected NaN (no threshold)")
        missing_nan += 1
if nan_expected:
    log(f"threshold metrics without a threshold for the variable: {len(nan_expected)} checked, {missing_nan} not NaN")
ok &= missing_nan == 0

# 4. lead 0: exactly zero for the variables in the input state, NaN for the others
if args.lead_zero:
    zero = framework.sel(lead_time=np.timedelta64(0, "ns"), bin="all")
    lead0_ok = True
    for j, variable in enumerate(variables):
        for key in ("rmse", "mae", "bias"):
            if key not in framework:
                continue
            values = zero[key].sel(variable=variable).values
            if variable in fields0:
                good = bool(np.all(values == 0.0))
            else:
                good = bool(np.all(np.isnan(values)))
            if not good:
                log(f"LEAD0 MISMATCH {key} {variable}: {values} (expected {'0.0' if variable in fields0 else 'NaN'})")
            lead0_ok &= good
    excluded = 1 - zero["weight_sum"] / zero["weight_sum"].max("variable")
    log(
        f"lead 0: {sum(v in fields0 for v in variables)} variables exactly zero, "
        f"{[v for v in variables if v not in fields0]} NaN; excluded weight fraction per variable "
        f"{dict(zip(variables, np.round(excluded.sel(region=list(masks)[0]).values, 3)))}; {'OK' if lead0_ok else 'FAIL'}"
    )
    ok &= lead0_ok
log("PASS" if ok else "FAIL", f"(rtol {args.rtol})")
evaluation.close()
sys.exit(0 if ok else 1)
