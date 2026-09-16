# Changelog

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
