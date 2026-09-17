# Changelog

## Unreleased

### Features

* Categorical scores (`pod`, `far`, `csi`, `ets`, `frequency_bias`, `hss`, `pss`) and probabilistic
  scores (`brier`, `bss`, `event_frequency`) at explicit per-variable thresholds with a user-given
  label, for example `{csi: {label: heavy, thresholds: {tp: 0.005}}}`, which writes `csi_heavy` and
  the sums it derives from. A label names one event, so every metric carrying it must give the same
  threshold map. The contingency scores binarise the ensemble mean and so score it as a point forecast:
  at a rare threshold the mean crosses far less often than a single member does, so they are not
  comparable between a deterministic checkpoint and its ensemble sibling. What compares the two is `brier`
  and `bss`, and the reliability diagram at one member: at one member the Brier score is
  `miss + false_alarm` exactly elementwise, hence equal on the weighted means up to float64 rounding.
* A variable that a label's threshold map does not name reads as NaN in that label's metrics and
  sums, rather than as a count of zero; a threshold naming a variable the run does not have is an
  error at construction.
* Two metrics whose statistics collide in name but not in their parameters are now refused at
  construction, and so are two metrics with the same name, which `metrics: [rmse, rmse]` used to
  produce silently.
* The rank histogram of an ensemble, `metrics: [rank_histogram]`, written as one `rank_histogram`
  variable with an extra `rank` axis of `M + 1` values, and `outlier_fraction`, the fraction of
  targets that fell outside the ensemble range, whose calibrated value is `2 / (M + 1)`. Tied members
  are spread deterministically, `1 / (ties + 1)` into each bin the target could occupy, and no seed is
  involved. On a field that is exactly zero at most nodes, such as precipitation, that removes the
  artificial first-bin spike an exact tie would otherwise produce; a spike that remains is a real bias
  (the model is never exactly dry where the target is), and the reliability diagram at threshold 0 is
  the tool for that question.
* The reliability diagram and the Brier decomposition of a threshold label: `reliability` (the observed
  event frequency at each forecast probability), `forecast_frequency` (how often each probability was
  issued, the sample under the curve) and the three components `brier_reliability`, `brier_resolution`
  and `brier_uncertainty`, which add up to the Brier score, `BS = REL - RES + UNC`, to the float64
  rounding of the sums. The first two carry an extra `probability` axis. The bins are the `M + 1`
  probability levels an `M`-member ensemble can produce, so no bin edges are configured and nothing
  depends on a binning convention; the family works at one member too, where the diagram has the two
  points of a contingency table.
* Because the stored level sums know their forecast probability exactly, any coarser reliability diagram
  can be computed from a result file afterwards, with no new statistics and no re-run.
* A config in which two metrics put different thresholds behind one label is refused with
  `one label, one threshold map`, and a malformed metric spec, such as `{crps: {alpah: 0.5}}`, is now a
  config validation error naming the accepted keywords instead of a traceback. Statistics compare by
  value, so two metrics asking for the same statistic share one stored sum.
* A metric may now learn the ensemble size of the run, and may declare an extra output dimension;
  both are how the rank histogram works and neither changes the stored state, which is still one
  `(lead_time, bin, variable, region)` sum per statistic.

## 0.1.0 — 2026-09-16

First release of anemoi-evaluation: in-memory evaluation of anemoi forecasts in the spirit of
WeatherBench-X. Forecasts are produced on the fly by anemoi-inference and scored on the GPU
step by step against targets from an anemoi-datasets zarr; forecast fields never touch disk and
the output is one small netcdf of aggregated statistics.

### Features

* Statistics accumulated as float64 sums per (lead time, time bin, variable, region), metrics
  derived from the means: RMSE, MAE, bias, member RMSE and MAE, CRPS for any `alpha` (standard
  and fair), spread, spread-skill, ACC against a climatology.
* Ensembles as lockstep rollouts of one runner, reseeded per member and model call; the computed
  forcings are shared across members and init times through a host cache (`forcings_cache_bytes`).
* Multi-step-output models scored per output time; optional lead-0 scoring from the initial
  state.
* Sharding over init times (`--shard i/n`, Slurm job steps and job arrays); `merge` validates
  that the inputs are the complete `0..n-1` shards of one run, or merges an incomplete set with
  `--partial`.
* Target prefetch on a worker thread with an optional decoded-row cache.
* `--dry-run` plans a run without loading the model and, with `--step-time`, estimates model
  time, wall time and GPU-hours.
* YAML configuration validated with pydantic: `base:` merges a config on top of another one,
  `from_checkpoint` stands for the `open_dataset` arguments the checkpoint records. A Python API
  exposes the same objects.
* Small source protocols for forecasts, targets and climatologies; persistence baseline,
  in-memory climatology and fakes included.
* Result files record the resolved config, package versions, checkpoint ids, init times, GPU
  model, inference chunk counts, forcings cache counters and timings.
* Processor chunks default to 1 at inference when `ANEMOI_INFERENCE_NUM_CHUNKS` is set and
  `ANEMOI_INFERENCE_NUM_CHUNKS_PROCESSOR` is not: chunking the processor costs time and saves no
  memory.

### Validation

Validated on 1 km stretched-grid GraphTransformer checkpoints (1.7M nodes; deterministic,
ensemble and hourly multi-step-output) on one A100 against numpy float64 references, with
sharded runs bit-exact against unsharded ones. See `docs/benchmarks.md`.
