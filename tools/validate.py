"""Check the framework's metrics against a direct `runner.run()` loop scored in numpy float64.

    python tools/validate.py <config.yaml> --init 2024-01-02T00 [--steps 2 | --lead 9h]
                             [--members 1 4] [--lead-zero] [--rtol 1e-5] [-o out.nc]
    python tools/validate.py <config.yaml> --init 2024-03-01T00,2024-08-15T12 --lead 48h --stations <zarr>
                             --obs-variable tre200s0=2t:offset=273.15 [--station-file stations.csv] [--lapse-rate]

Needs a GPU: it loads the checkpoint of the config's `forecast.anemoi_inference` block and runs it.
Single-dataset checkpoints only: it reads `forecast.dataset_name` and a plain `regions` block.
The lead defaults to the config's `lead_time`; --steps (model timesteps) or --lead override it.

A temporal downscaler is scored over the one window starting at --init (the lead is its window, --steps is ignored
and --lead-zero refused), and its direct loop is anemoi-inference's own `execute()`, one call per member with the
member's seed before it and the written states captured in memory, so the window input state the framework builds
itself is checked against the one the runner builds.

One process: the framework `Evaluation` (config members) and the direct loop share the loaded model and the seeding
(`member_seed(seed, init, member, call)` before every member's model call), so the member draws are identical. For every
--members value the direct loop times every `next()` (model calls flagged: the first `next()` of each model call, from
the source's `rollout_schedule`), checks that the runner's steps are the source's `lead_times` (`k * rollout shift +
offset` for an offset forecaster) and records its peak GPU memory; the numpy statistics (rmse, mae, bias, crps, fair_crps, spread,
spread_skill, member_rmse, member_mae, error_std, acc and activity_ratio when the config has a climatology, and the
threshold metrics, the rank histogram and the reliability-level metrics, ROC and rev included, the config asks for)
are compared for the config's member count; a metric carrying an extra axis, such as `rank_histogram`, is compared
element by element and reported by its worst element; a threshold metric must be NaN for a variable its map omits, and
a framework NaN against a finite reference, or the reverse, is a mismatch rather than a skipped comparison.
With --lead-zero the framework's lead-0 rmse/mae/bias must be exactly 0.0 for the variables present in the input
state and NaN for the others.

Prints `VALIDATE:` lines: the checkpoint facts, the framework run's timing, model calls and peak memory, the direct
loop's per-`next()` times and peak memory per member count, every mismatch above --rtol, the maximum relative difference
per metric, and a final PASS or FAIL (also the exit status, 0 or 1). The framework run is written to --output
(default: `validate-<config stem>.nc` next to the config's `output.path`).

Station mode (`--stations <zarr>`, with `--obs-variable RAW=NAME[:offset=X][:scale=Y]` repeated, optionally
`--station-file <csv>` with an elevation column, `--lapse-rate [K/m]` and `--max-distance-km`): point targets are
Python-only in the package, so the config gives only the checkpoint (its `forecast` block, whose input dataset must
cover the init times) and the lead; `--init` takes comma-separated init times. The model is scored against the
gridded station dataset through `Nearest` (+ `LapseRate` on 2t), in one rollout per init time whose frames feed two
evaluations (uniform weights, and seeded non-dyadic weights in [0.5, 1.5) per station, so the summation order shows)
and a numpy reference with its own KD-tree, its own `z / g` read and its own observation read. PASS or FAIL: the
chain's mapped frames equal the numpy ones bit for bit; the sums equal `math.fsum` references within --rtol (default
1e-13 in this mode; the signed error sum relative to the sum of its absolute terms), the uniform weight sums exactly
(they are the finite-pair counts, as `weight_sum` must be); `out_of_domain` counts the stations beyond
--max-distance-km per frame; lead 0 of the model chain equals lead 0 of analysis persistence through the same chain.
Plausibility as `WARN` lines that do not fail the run: lead-0 RMSE above --lead0-rmse-max (3 K), station-mean 2t not
correlated with elevation (r > -0.5: the data columns may not belong to the coordinates), more than 0.1 % of pairs
with |error| > 15 K, gross observations (2t outside [233.15, 315.15] K, 2d outside [223.15, 308.15] K, 2d > 2t +
0.5 K). The weighted framework state is written to --output (default `validate-<config stem>-points.nc`).

`ANEMOI_INFERENCE_NUM_CHUNKS` is set to 8 unless the environment already sets it (needed at 1 km resolution to avoid
running out of GPU memory in the mapper).
"""

import argparse
import datetime
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
from anemoi.evaluation.sources.anemoi_inference import create_input_state  # noqa: E402
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
points_group = parser.add_argument_group("station mode (point targets, Python-only in the package)")
points_group.add_argument("--stations", metavar="ZARR", help="Gridded station dataset; switches to station mode.")
points_group.add_argument("--station-file", metavar="CSV", help="Station file in dataset order (latitude, longitude).")
points_group.add_argument("--elevation-column", default="elevation", help="Station file column with the elevation (m).")
points_group.add_argument(
    "--obs-variable",
    action="append",
    default=[],
    metavar="RAW=NAME[:offset=X][:scale=Y]",
    help="Dataset variable, its result name and conversion, repeatable, e.g. tre200s0=2t:offset=273.15.",
)
points_group.add_argument("--lapse-rate", type=float, nargs="?", const=0.0065, help="Chain LapseRate (K/m) on 2t.")
points_group.add_argument("--max-distance-km", type=float, default=5.0)
points_group.add_argument("--lead0-rmse-max", type=float, default=3.0, help="WARN above this lead-0 RMSE.")
args = parser.parse_args()


def log(*items):
    print("VALIDATE:", *items, flush=True)


# Plausibility bounds of the station mode's WARN lines, in the result units (K).
GROSS = {"2t": (233.15, 315.15), "2d": (223.15, 308.15)}


def validate_points(config) -> bool:
    """Station mode: the config's checkpoint against a gridded station dataset through Nearest (+ LapseRate).

    One rollout per init time (comma-separated `--init`), its frames fed to two framework evaluations, uniform
    weights and seeded non-dyadic array weights on the fixed station set (`begin`/`add_frame`/`finish`, as `run()`
    does), and to a numpy float64 reference: own KD-tree gather, own lapse correction with `z / g` read from the
    model's dataset, own read and conversion of the observations. Checks, PASS or FAIL: the chain's mapped frames
    equal the numpy ones bit for bit; the sums equal `math.fsum` references (counts exact under uniform weights, the
    other sums to --rtol, default 1e-13 here); `weight_sum` is the finite-pair count; `out_of_domain` is the number of
    stations beyond --max-distance-km; lead 0 of the model chain equals lead 0 of analysis persistence through the
    same chain. Plausibility, WARN only: lead-0 RMSE, station-mean 2t against elevation, |error| > 15 K, gross values.
    """
    import math

    from anemoi.utils.dates import frequency_to_string
    from scipy.spatial import cKDTree

    from anemoi.evaluation.evaluate import PhaseTimer
    from anemoi.evaluation.operators import Chain
    from anemoi.evaluation.operators import DatasetConstants
    from anemoi.evaluation.operators import LapseRate
    from anemoi.evaluation.operators import Nearest

    rtol = args.rtol if any(a.startswith("--rtol") for a in sys.argv) else 1e-13
    conversions = {}
    for item in args.obs_variable:
        raw, _, rest = item.partition("=")
        name, *options = rest.split(":")
        conversions[raw] = {"name": name, **{k: float(v) for k, v in (o.split("=") for o in options)}}
    if not conversions:
        parser.error("--stations needs at least one --obs-variable")
    if args.lapse_rate is not None and not args.station_file:
        parser.error("--lapse-rate needs the station elevations of --station-file")
    variables = [spec["name"] for spec in conversions.values()]
    inits = [as_datetime(text) for text in args.init.split(",")]
    forecast = ae.InferenceForecastSource(**config.forecast.anemoi_inference.model_dump())
    lead = frequency_to_timedelta(args.lead) if args.lead else args.steps * forecast.timestep if args.steps else None
    lead = lead or config.lead_time
    frequency = frequency_to_string(forecast.timestep)
    attributes = {"elevation": args.elevation_column} if args.station_file else None
    targets = ae.StationDatasetTargets(
        ae.DatasetTargets(args.stations, frequency=frequency, select=list(conversions)),
        conversions,
        stations=args.station_file,
        attributes=attributes,
    )
    points = targets.fixed_points
    steps = [Nearest(max_distance_km=args.max_distance_km)]
    if args.lapse_rate is not None:
        steps.append(LapseRate("2t", rate=args.lapse_rate))
    chain = Chain(steps)
    rng = np.random.default_rng(0)
    point_weights = rng.uniform(0.5, 1.5, len(points))  # non-dyadic, so the summation order shows
    regions = {"all": "all", "half": rng.random(len(points)) < 0.5}
    common = dict(metrics=["bias", "rmse", "mae"], variables=variables, bins="none", include_lead_zero=True)
    common["regions"] = regions
    runs = {
        "uniform": ae.Evaluation(forecast, targets, inits, lead, observation_operator=chain, **common),
        "weighted": ae.Evaluation(
            forecast, targets, inits, lead, observation_operator=chain, weights=point_weights, **common
        ),
    }
    analysis = ae.DatasetTargets.from_forecast(forecast)
    persistence = ae.Evaluation(
        ae.PersistenceForecastSource(analysis, forecast.timestep),
        targets,
        inits,
        lead,
        observation_operator=chain,
        constants=DatasetConstants(ae.DatasetTargets.from_forecast(forecast)) if args.lapse_rate is not None else None,
        device=forecast.device,
        **common,
    )
    log(
        f"station mode: {len(points)} stations from {args.stations}, variables {conversions}, chain "
        f"{[type(step).__name__ for step in steps]}, inits {[str(i) for i in inits]}, lead {lead}, rtol {rtol}"
    )

    # reference side: own tree, own orography read, own observation read and conversion
    grid = forecast.grid

    def xyz(lat, lon):
        lat, lon = np.deg2rad(lat), np.deg2rad(lon)
        return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)

    chord, nodes = cKDTree(xyz(grid.latitudes, grid.longitudes)).query(xyz(points.latitudes, points.longitudes))
    distance_km = 2 * 6371.0 * np.arcsin(np.minimum(chord / 2, 1))
    inside = distance_km <= args.max_distance_km
    nodes = np.where(inside, nodes, 0)
    model_ds = open_dataset(*forecast.dataset_args_kwargs()[0], **forecast.dataset_args_kwargs()[1])
    gi = slice(None) if forecast.grid_indices is None else forecast.grid_indices
    elevation = points.attributes.get("elevation")
    dh = None
    if (
        args.lapse_rate is not None
    ):  # the orography as the default ForecastConstants serves it, z * (1 / g) at the first init
        row = model_ds.dates.astype("datetime64[us]").tolist().index(min(inits))
        z = np.asarray(model_ds[row][model_ds.name_to_index["z"], 0], dtype=np.float32)[gi]
        dh = elevation - np.where(inside, z.astype(np.float64)[nodes] * (1 / 9.80665), np.nan)
    obs_ds = open_dataset(args.stations, frequency=frequency, select=list(conversions))
    obs_index = {date: i for i, date in enumerate(obs_ds.dates.astype("datetime64[us]").tolist())}
    obs_columns = [obs_ds.name_to_index[raw] for raw in conversions]
    scale = np.array([spec.get("scale", 1.0) for spec in conversions.values()])[:, None]
    offset = np.array([spec.get("offset", 0.0) for spec in conversions.values()])[:, None]

    def observed(valid_time):
        raw = np.asarray(obs_ds[obs_index[valid_time]], dtype=np.float32)[obs_columns, 0]
        return (raw.astype(np.float64) * scale + offset).astype(np.float32)

    def reference_frame(data):  # (M, V_model, N) float32 -> (M, V, P) float32
        mapped = data[:, [frame_variables.index(v) for v in variables]][:, :, nodes]
        mapped[:, :, ~inside] = np.nan
        if args.lapse_rate is not None and "2t" in variables:
            j = variables.index("2t")
            mapped[:, j] = (mapped[:, j].astype(np.float64) - args.lapse_rate * dh).astype(np.float32)
        return mapped

    device = forecast.device
    states = {name: evaluation.begin() for name, evaluation in runs.items()}
    bound = runs["uniform"].observation_operator  # a plain accessor: bound here because begin() binds
    assert bound.bound
    plan = bound.prepare(points)
    frame_variables = runs["uniform"].forecast_variables
    leads = runs["uniform"].lead_times
    fc = np.full((len(inits), len(leads), forecast.members, len(variables), len(points)), np.nan, np.float32)
    obs = np.full((len(inits), len(leads), len(variables), len(points)), np.nan, np.float32)
    exact_frames = total_frames = 0
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i, init in enumerate(inits):
            timers = {name: PhaseTimer(device) for name in runs}
            initial = forecast.initial_frame(init, frame_variables, device)
            for frame in itertools.chain([initial], forecast.frames(init, lead, frame_variables, device)):
                for name, evaluation in runs.items():
                    evaluation.add_frame(states[name], frame, timers[name])
                j = leads.index(frame.lead_time)
                fc[i, j] = reference_frame(frame.data.cpu().numpy())
                obs[i, j] = observed(frame.valid_time)
                out = bound.apply([frame], plan, frame.valid_time)
                mapped = out.data.cpu().numpy()[:, [out.variables.index(v) for v in variables]]
                exact_frames += int(np.array_equal(mapped, fc[i, j], equal_nan=True))
                total_frames += 1
            for timer in timers.values():
                timer.mark("model")
    log(
        f"{total_frames} frames in {time.perf_counter() - t0:.1f}s, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
    )
    for name, evaluation in runs.items():
        evaluation.finish(states[name], inits)
    persistence_state = persistence.run()
    out = (
        Path(args.output)
        if args.output
        else Path(config.output.path).parent / f"validate-{Path(args.config).stem}-points.nc"
    )
    write(states["weighted"].to_xarray(runs["weighted"].metrics), out)
    log(f"weighted framework state written to {out}")

    ok = True

    def check(label, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        log(f"{'PASS' if good else 'FAIL'} {label}{': ' + detail if detail else ''}")

    check(
        "mapped frames equal the numpy gather/lapse loop bit for bit",
        exact_frames == total_frames,
        f"{exact_frames}/{total_frames} frames",
    )
    error = fc.astype(np.float64).mean(2) - obs  # ensemble mean, float64
    median_error = np.median(fc.astype(np.float64), 2) - obs  # mae scores the ensemble median
    valid = np.isfinite(fc).all(2) & np.isfinite(obs)
    masks = {"all": np.ones(len(points), bool), "half": regions["half"]}
    for name, weight in (("uniform", np.ones(len(points))), ("weighted", point_weights)):
        state = states[name].cpu()
        worst = {}
        weights_exact = True
        for j in range(len(leads)):
            for v in range(len(variables)):
                for r, region in enumerate(state.regions):
                    ok_pairs = valid[:, j, v] & masks[region]
                    e, w = error[:, j, v][ok_pairs], np.broadcast_to(weight, ok_pairs.shape)[ok_pairs]
                    ref = {
                        "error": math.fsum(w * e),
                        "squared_error": math.fsum(w * e**2),
                        "absolute_median_error": math.fsum(w * np.abs(median_error[:, j, v][ok_pairs])),
                    }
                    absolute = math.fsum(w * np.abs(e))
                    got_w, ref_w = float(state.weights[j, 0, v, r]), math.fsum(w)
                    if name == "uniform":
                        weights_exact &= got_w == ref_w
                    else:
                        ref["weights"] = ref_w
                    for key, value in ref.items():
                        got = got_w if key == "weights" else float(state.sums[key][j, 0, v, r])
                        # relative to the sum of |terms|, the scale of a summation error: the signed error sum can
                        # cancel to near zero, where a difference relative to the sum itself means nothing
                        scale = absolute if key == "error" else abs(value)
                        worst[key] = max(worst.get(key, 0.0), abs(got - value) / max(scale, 1e-300))
        detail = ", ".join(f"{key} {rel:.2e}" for key, rel in worst.items())
        if name == "uniform":
            check("uniform weights: weight sums exact (finite pair counts)", weights_exact)
        check(f"{name} weights: sums against math.fsum within {rtol:g}", all(r <= rtol for r in worst.values()), detail)
        written = state.to_xarray(runs[name].metrics)
        if name == "uniform":
            count = np.array(
                [
                    [
                        [float((valid[:, j, v] & masks[region]).sum()) for region in state.regions]
                        for v in range(len(variables))
                    ]
                    for j in range(len(leads))
                ]
            )
            check(
                "weight_sum equals the finite pair count",
                np.array_equal(written["weight_sum"].sel(bin="all").values, count),
            )
        expected_out = int((~inside).sum()) * state.n_init[:, 0].numpy()
        check(
            f"{name}: out_of_domain is the stations beyond {args.max_distance_km} km per frame",
            np.array_equal(state.out_of_domain[:, 0].numpy(), expected_out),
            f"{int((~inside).sum())} stations",
        )
    uniform, baseline = states["uniform"].cpu(), persistence_state.cpu()
    same = all(torch.equal(uniform.sums[k][0], baseline.sums[k][0]) for k in uniform.sums)
    check(
        "lead 0 of the model chain equals lead 0 of persistence through the same chain",
        same and torch.equal(uniform.weights[0], baseline.weights[0]),
    )

    # plausibility, WARN only
    def warn(label, bad, detail):
        log(f"{'WARN' if bad else 'OK'} plausibility {label}: {detail}")

    lead0 = leads.index(datetime.timedelta(0))
    for v, variable in enumerate(variables):
        e0 = error[:, lead0, v][valid[:, lead0, v]]
        rmse0 = float(np.sqrt(np.mean(e0**2))) if e0.size else float("nan")
        warn(
            f"{variable} lead-0 RMSE",
            not rmse0 <= args.lead0_rmse_max,
            f"{rmse0:.2f} (bound {args.lead0_rmse_max}, analysis against the observations)",
        )
        if elevation is not None and variable == "2t":
            with np.errstate(invalid="ignore"):
                station_mean = np.nanmean(obs[:, :, v].reshape(-1, len(points)), 0)
            good = np.isfinite(station_mean) & np.isfinite(elevation)
            r = float(np.corrcoef(elevation[good], station_mean[good])[0, 1])
            warn(
                f"{variable} station mean against elevation",
                not r < -0.5,
                f"r {r:+.3f} over {int(good.sum())} stations (a station network has a strong negative relation; "
                "near zero means the data columns do not belong to the coordinates)",
            )
        e = error[:, :, v][valid[:, :, v]]
        big = int((np.abs(e) > 15).sum())
        warn(f"{variable} |error| > 15 K", big > 0.001 * e.size, f"{big} of {e.size} pairs")
        if variable in GROSS:
            low, high = GROSS[variable]
            values = obs[:, :, v][np.isfinite(obs[:, :, v])]
            gross = int(((values < low) | (values > high)).sum())
            warn(f"{variable} gross observations", gross > 0, f"{gross} of {values.size} outside [{low}, {high}] K")
    if "2t" in variables and "2d" in variables:
        t, d = obs[:, :, variables.index("2t")], obs[:, :, variables.index("2d")]
        above = int(np.nansum(d > t + 0.5))
        warn("2d above 2t + 0.5 K", above > 0, f"{above} values")
    log("PASS" if ok else "FAIL", f"(station mode, rtol {rtol})")
    targets.close()
    forecast.close()
    return ok


config = load_config(args.config)
if args.stations:
    sys.exit(0 if validate_points(config) else 1)
block = config.forecast.anemoi_inference.model_dump()
init = as_datetime(args.init)
forecast = ae.InferenceForecastSource(**block)
targets = ae.DatasetTargets.from_forecast(forecast)
downscaler = forecast.temporal_downscaling
if downscaler:
    if args.lead_zero:
        parser.error("a temporal downscaler has no lead 0: it is the window's left bound")
    if args.lead and frequency_to_timedelta(args.lead) != forecast.window:
        parser.error(f"a temporal downscaler's lead is its window {forecast.window}, got --lead {args.lead}")
    lead = forecast.window
elif args.lead:
    lead = frequency_to_timedelta(args.lead)
elif args.steps:
    lead = args.steps * forecast.timestep
else:
    lead = config.lead_time
members_list = args.members or [forecast.members]
name = forecast.dataset_name
runner = forecast.runner
device = forecast.device
calls_of = [] if downscaler else [call for call, _ in forecast.rollout_schedule(lead)]
log(
    f"checkpoint {forecast.checkpoint}: timestep {forecast.timestep}, multi_step_input {forecast.multi_step_input}, "
    f"multi_step_output {forecast.multi_step_output}, "
    + (
        f"temporal downscaler with window {forecast.window}"
        if downscaler
        else f"input offsets {[str(t) for t in forecast.input_offsets]}, "
        f"output offsets {[str(t) for t in forecast.output_offsets]}, rollout shift {forecast.output_horizon}"
    )
    + f", lead {lead}, targets prefetch {targets.lookahead}"
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
    state0 = forecast._initial_state(name, init)["fields"]
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


class CapturedOutput:
    """An anemoi-inference output keeping the evaluated variables of every written state in memory."""

    def __init__(self):
        self.states = []

    def open(self, state):
        pass

    def write_initial_state(self, state):
        pass

    def write_state(self, state):
        fields = state["fields"]
        if all(v in fields for v in variables):
            self.states.append(
                (
                    state["step"],
                    state["date"],
                    np.stack([np.asarray(fields[v], dtype=np.float64)[gi] for v in variables]),
                )
            )

    def close(self):
        pass


def execute_window(members):
    """The downscaler's window from `init` through `runner.execute()`, member by member: (step, date, arrays)."""
    saved = dict(runner.outputs)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    per_member, times = [], []
    try:
        for member in range(members):
            capture = CapturedOutput()
            runner.outputs[name] = capture
            runner.reference_date, runner.lead_time = init, forecast.window
            torch.manual_seed(member_seed(forecast.seed, init, member, 0))
            t0 = time.perf_counter()
            with quiet_inference():
                runner.execute()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            per_member.append([state for state in capture.states if state[0] > datetime.timedelta(0)])
    finally:
        runner.outputs.clear()
        runner.outputs.update(saved)
    peak = torch.cuda.max_memory_allocated() / 2**30
    log(
        f"M={members}: execute() per member {[f'{t:.2f}' for t in times]} s (input reading included), "
        f"peak GPU {peak:.2f} GiB"
    )
    first = per_member[0]
    assert all([s for s, _, _ in states] == [s for s, _, _ in first] for states in per_member), "members out of step"
    return [(step, date, np.stack([states[i][2] for states in per_member])) for i, (step, date, _) in enumerate(first)]


for members in members_list:
    if downscaler:
        steps = execute_window(members)
        got_steps = [s for s, _, _ in steps]
        expected_steps = evaluation.forecast.lead_times(lead)
        assert got_steps == expected_steps, (got_steps, expected_steps)
        fields_by_members[members] = steps
        continue
    lagged = runner.checkpoint.multi_dataset_metadata[name].lagged
    prognostic = create_input_state(runner.prognostics_inputs[name], init, lagged)
    constant = create_input_state(runner.constant_forcings_inputs[name], init, lagged)
    dynamic = create_input_state(runner.dynamic_forcings_inputs[name], init, lagged)
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
                call = calls_of[step_index] if step_index < len(calls_of) else None
                first = call is not None and (step_index == 0 or calls_of[step_index - 1] != call)
                outputs = []
                for member, generator in enumerate(generators):
                    if first:
                        torch.manual_seed(member_seed(forecast.seed, init, member, call))
                    outputs.append(next(generator, None))
                if any(o is None for o in outputs):
                    assert all(o is None for o in outputs), "members out of step"
                    break
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                next_times.append((step_index, t1 - t0, first))
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
    log(f"M={members}: runner steps {[str(s) for s in got_steps]}")
    assert got_steps == expected_steps, (got_steps, expected_steps)
    fields_by_members[members] = steps


def trapezoid(y, x):
    """Trapezoid area under `y(x)`, spelled out because `np.trapezoid` needs numpy 2."""
    return float(((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0).sum())


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
    mean, median = arrays.mean(0), np.median(arrays, 0)
    finite = np.isfinite(arrays).all(0) & np.isfinite(y)
    for region, mask in masks.items():
        for j, variable in enumerate(variables):
            ok = finite[j] & mask
            w = weights[ok]
            e = mean[j][ok] - y[j][ok]
            ref = {
                "rmse": np.sqrt((w * e**2).sum() / w.sum()),
                "mae": (w * np.abs(median[j][ok] - y[j][ok])).sum() / w.sum(),
                "bias": (w * e).sum() / w.sum(),
            }
            ref["error_std"] = np.sqrt((w * (e - ref["bias"]) ** 2).sum() / w.sum())
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
                ref["spread_skill"] = ref["spread"] / np.sqrt(ref["rmse"] ** 2 - ref["spread"] ** 2 / M)
            if c is not None:
                okc = ok & np.isfinite(c[j])
                wc = weights[okc]
                fa, ta = mean[j][okc] - c[j][okc], y[j][okc] - c[j][okc]
                ref["acc"] = (wc * fa * ta).sum() / np.sqrt((wc * fa * fa).sum() * (wc * ta * ta).sum())
                ref["activity_ratio"] = np.sqrt((wc * fa * fa).sum() / (wc * ta * ta).sum())
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
                    hit_rate, false_rate = hits / (hits + misses), alarms / (alarms + negatives)
                    logs = np.log([false_rate, hit_rate, 1 - false_rate, 1 - hit_rate])
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
                        "pofd": false_rate,
                        "sedi": (logs[0] - logs[1] - logs[2] + logs[3]) / logs.sum(),
                        "odds_ratio": hits * negatives / (misses * alarms),
                        "orss": (hits * negatives - misses * alarms) / (hits * negatives + misses * alarms),
                        "f1": 2 * hits / (2 * hits + misses + alarms),
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
                # the ROC decisions "count >= k", from the nodes rather than from the level sums
                decisions = [(count >= k).astype(np.float64) for k in range(M + 1)]
                with np.errstate(divide="ignore", invalid="ignore"):  # no event or no non-event is 0/0
                    roc_h = np.array([(w * d * observed).sum() / (w * observed).sum() for d in decisions])
                    roc_f = np.array([(w * d * (1 - observed)).sum() / (w * (1 - observed)).sum() for d in decisions])
                levels = {
                    "reliability": o_k,
                    "forecast_frequency": n / total,
                    "brier_reliability": np.nansum(np.where(n > 0, n * (p_k - o_k) ** 2, 0.0)) / total,
                    "brier_resolution": np.nansum(np.where(n > 0, n * (o_k - obar_k) ** 2, 0.0)) / total,
                    "roc_pod": roc_h,
                    "roc_pofd": roc_f,
                    "roc_area": trapezoid(np.r_[0.0, roc_h[::-1]], np.r_[0.0, roc_f[::-1]]),
                }
                for metric in level_metrics:
                    if metric.label != label:
                        continue
                    if metric.kind == "rev":  # the expenses of the M decisions k >= 1 against climate and perfect
                        s, value = (w * observed).sum() / w.sum(), []
                        for a in metric.cost_loss:
                            expenses = [(w * (a * d + (1 - d) * observed)).sum() / w.sum() for d in decisions[1:]]
                            with np.errstate(divide="ignore", invalid="ignore"):
                                value.append(max((min(a, s) - x) / (min(a, s) - s * a) for x in expenses))
                        ref[metric.name] = np.array(value)
                    else:
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
