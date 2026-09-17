# Changelog

## 0.3.0 — 2026-09-17

Checkpoints trained on multiple datasets, scored as one evaluation per dataset that share the
model rollout; single-dataset runs are unchanged.

### Breaking

* `config.checkpoint_dataset_arguments` returns one `open_dataset` argument set per dataset.
* `Evaluation.from_config` returns a `MultiEvaluation` for a checkpoint trained on multiple datasets.
* `output.path` must not contain `{dataset}` on a single-dataset run.

### Features

* Checkpoints trained on multiple datasets are scored: one evaluation per dataset, sharing one
  runner and one rollout, each with its own targets, variables, weights, regions, climatology and
  result file. Single-dataset runs are unchanged.
* `targets`, `variables`, `weights`, `regions` and `climatology` accept a `datasets:` mapping
  keyed by the checkpoint's dataset names, and `from_checkpoint:` resolves per dataset.
* `output.path` takes a `{dataset}` placeholder, required by a multi-dataset run; `merge` refuses
  results of different datasets.
* Checkpoints whose datasets disagree on the timing, and downscaling checkpoints whose model does
  not decode every dataset, are refused with a message naming the problem.
* A top-level `datasets:` key scores a subset of a multi-dataset checkpoint's datasets; the model
  still predicts every dataset, the skipped ones simply get no targets, no aggregator and no
  result file. Selecting one dataset gives an ordinary single-dataset run, whose result names the
  dataset it scored.

### Fixes

* `tools/inspect_results.py` compares the zero lead time with a unit, as numpy 2.5 requires.

## 0.2.0 — 2026-09-17

Categorical, calibration and reliability scores, all stored as additive per-element sums, so
sharding, `merge` and the existing metrics are unchanged. Verified on real deterministic and
ensemble checkpoints against numpy float64 references.

### Features

* Categorical scores (`pod`, `far`, `csi`, `ets`, `frequency_bias`, `hss`, `pss`) and probabilistic
  scores (`brier`, `bss`, `event_frequency`) at per-variable thresholds with a user-given label,
  for example `{csi: {label: heavy, thresholds: {tp: 0.005}}}`, written as `csi_heavy`. The
  contingency scores binarise the ensemble mean; `brier` and `bss` are what compares a
  deterministic checkpoint with an ensemble one.
* Variables a threshold map does not name read as NaN for that label; a variable the run does not
  have is an error at construction.
* `rank_histogram`, on an extra `rank` axis, and `outlier_fraction`. Ties are spread
  deterministically over the bins the target could occupy, with no seed.
* The reliability diagram of a label: `reliability` and `forecast_frequency` on an extra
  `probability` axis, and `brier_reliability`, `brier_resolution`, `brier_uncertainty`, which add
  up to the Brier score. The bins are the `M + 1` probability levels an `M`-member ensemble can
  produce, so there are no bin edges to configure; coarser diagrams can be derived from a result
  file.
* Refused at config load: duplicate metric names, one label with different thresholds, and
  malformed metric specs (previously a traceback).
* A metric may learn the run's ensemble size and declare an extra output dimension; the stored
  state is unchanged.

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
