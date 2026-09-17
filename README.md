# anemoi-evaluation

Forecast evaluation for [anemoi](https://github.com/ecmwf/anemoi-core) models, in the spirit of
[WeatherBench-X](https://github.com/google-research/weatherbenchX). Forecasts are produced *on
the fly* by `anemoi-inference` from a checkpoint and scored on the GPU step by step against
targets read from an `anemoi-datasets` zarr. Forecast fields are never written to disk: the only
output is one small netcdf of aggregated statistics and the metrics derived from them.

**Status:** prototype, alpha (version 0.2.0), not on PyPI. Config keys, the Python API and the
contents of the result file may still change between releases.

## Features

* **Statistics and metrics** in the WeatherBench-X sense: per-element statistics (errors,
  ensemble variance, kernel-CRPS primitives, anomaly products) are accumulated as float64 sums
  per (lead time, time bin, variable, region); metrics are derived from the means. Included:
  RMSE, MAE, bias, member RMSE and MAE, CRPS for any `alpha` (standard and fair), spread,
  spread-skill, ACC against a climatology, the categorical and probabilistic scores at
  user-given thresholds (POD, FAR, CSI, ETS, frequency bias, HSS, PSS, Brier score and skill
  score, base rate), the rank histogram of an ensemble with its outlier fraction, and the
  reliability diagram of a threshold with the Brier decomposition `BS = REL - RES + UNC`.
* **Ensembles** as lockstep rollouts of the same runner, reseeded per member and per model call.
* **Multi-step-output models** (several lead times per forward pass) are scored per output time;
  lead 0 can be scored from the initial state.
* **Weights, regions and time bins**: node weights from the model graph, a file, spherical
  Voronoi areas or uniform; regions from a graph attribute, a bounding box, a sub-grid of a
  cutout or a file; bins per season, month or init time, of the init or the valid time.
* **Multi-dataset checkpoints**: a checkpoint trained on multiple datasets is scored as one
  evaluation per dataset sharing one rollout, each with its own grid, variables, targets, weights,
  regions and result file.
* **Sharding** over init times (`--shard i/n`, or automatically from Slurm job steps and job
  arrays) with exact merging of the partial results.
* **Bring your own sources**: forecast, target and climatology sources are small protocols; a
  persistence baseline, an in-memory climatology and in-memory fakes are included.
* **Low I/O**: targets are prefetched on a worker thread, with an optional decoded-row cache,
  while the model steps.
* **Reproducible runs**: the result file records the resolved config, the package versions, the
  checkpoint ids, the init times, the GPU model and the run's timings.

## Installation

The package is not on PyPI. Install it from a checkout, into the environment that has the
anemoi stack:

```bash
git clone https://github.com/MeteoSwiss/anemoi-evaluation.git
pip install -e ./anemoi-evaluation
```

Python 3.11 to 3.13. The dependencies are `anemoi-datasets`, `anemoi-graphs`,
`anemoi-inference`, `anemoi-transform`, `anemoi-utils`, `torch`, `xarray`, `netcdf4`, `numpy`,
`pydantic` and `pyyaml`. Installing gives the `anemoi-evaluation` command.

## Quick start

An evaluation is a YAML config. This one scores a checkpoint over eight init times, five days
ahead, against the dataset the checkpoint was trained on, read over 2024:

```yaml
# config.yaml
forecast:
  anemoi_inference:
    checkpoint: /path/to/inference-last.ckpt
    input:
      dataset:
        from_checkpoint: true      # the open_dataset arguments the checkpoint records
        start: 2024
        end: 2024
    device: cuda
    members: 1                     # >1 for an ensemble checkpoint

lead_time: 120h
init_times: {start: 2024-01-02T00, end: 2024-01-05T12, frequency: 12h}
variables: [2t, 10u, t_850, z_500]
metrics: [rmse, mae, bias]
regions: {global: all}
bins: {time: season, by: init_time}
output:
  path: results/run-{shard}.nc
```

Plan it without loading the model, then run it:

```bash
anemoi-evaluation run config.yaml --dry-run                   # resolved config, dates, memory, plan
anemoi-evaluation run config.yaml --dry-run --step-time 3.15  # the same, with a time and GPU-hour estimate
anemoi-evaluation run config.yaml                             # writes results/run-0.nc
```

The result is an xarray-readable netcdf, with the metrics on
`(lead_time, bin, variable, region)`:

```python
import xarray as xr

results = xr.open_dataset("results/run-0.nc")
print(results["rmse"].sel(bin="all", region="global", variable="2t"))
```

To spread the run over four processes, shard it and merge the parts. `--shard i/n` is implicit
under `srun` and `sbatch --array`:

```bash
anemoi-evaluation run config.yaml --shard 0/4     # ... 1/4, 2/4, 3/4
anemoi-evaluation merge results/run-*.nc -o results/run.nc
```

The same objects are available from Python:

```python
import anemoi.evaluation as ae

forecast = ae.InferenceForecastSource(
    checkpoint="/path/to/inference-last.ckpt",
    input={"dataset": {"use_original_paths": True}},
    device="cuda",
    members=4, seed=0,                     # members=1 for a deterministic checkpoint
)
targets = ae.DatasetTargets.from_forecast(forecast)   # same zarr as the forecast input

evaluation = ae.Evaluation(
    forecast=forecast,
    targets=targets,
    init_times=ae.init_times("2024-01-02T00", "2024-01-05T12", "12h"),
    lead_time="120h",
    variables=["2t", "10u", "t_850", "z_500", "tp"],
    metrics=[ae.metrics.RMSE(), ae.metrics.Bias(), ae.metrics.CRPS(alpha=1.0), ae.metrics.Spread()],
    weights=forecast.graph_node_attribute("area_weight"),
    regions={"global": ae.regions.all(forecast.grid),
             "lam": forecast.graph_node_attribute("cutout_mask").astype(bool)},
    bins={"time": "season", "by": "init_time"},
)
state = evaluation.run()
state.to_xarray(evaluation.metrics).to_netcdf("results.nc")
```

## Documentation

* [User guide](docs/user-guide.md): configuration, sources, ensembles, regions and weights,
  sharded runs, troubleshooting
* [Reference](docs/reference.md): CLI, config keys, Python API, result-file layout
* Design: [requirements](docs/design/requirements.md) (what and why) and
  [architecture](docs/design/architecture.md) (how)
* [Benchmarks](docs/benchmarks.md): measured step times, I/O, sharding and the agreement with
  numpy float64
* [Contributing](docs/contributing.md): development setup, tests, validation tools, conventions
* [Tools](tools/README.md): the validation, comparison and climatology scripts
* [Changelog](CHANGELOG.md)

## Repository layout

```text
src/anemoi/evaluation/
    __main__.py          command line: run and merge
    config.py            YAML config model (pydantic), base: inheritance, from_checkpoint
    evaluate.py          the driver: sources in, aggregation state out
    frame.py             Grid and Frame, the data model shared by sources and the aggregator
    statistics.py        per-element statistics in float64
    aggregation.py       weighted, masked, binned sums over nodes and init times
    metrics.py           metrics as functions of mean statistics
    binning.py           time bins: season, month, init time; of the init or the valid time
    weights.py           node weights
    regions.py           region masks
    output.py            xarray output, netcdf round trip, merging
    sources/base.py      source protocols and base classes
    sources/anemoi_inference.py   forecasts from an anemoi-inference runner
    sources/anemoi_dataset.py     targets from an anemoi-datasets zarr, prefetched
    sources/persistence.py        persistence baseline
    sources/climatology.py        climatologies for the anomaly statistics
    sources/fake.py               in-memory synthetic sources
tests/                   CPU-only test suite on the in-memory fakes
tools/                   validation, comparison and climatology scripts (not part of the package)
docs/                    user guide, reference, design documents, benchmarks, contributing
.github/workflows/ci.yml CI: pre-commit, tests on Python 3.11 and 3.13
```

## License

BSD 3-Clause License, see [LICENSE](LICENSE). Copyright (c) 2026, MeteoSwiss.

## Acknowledgements

Built on the [anemoi](https://github.com/ecmwf/anemoi-core) packages of ECMWF and its partners.
The statistics-and-metrics design follows
[WeatherBench-X](https://github.com/google-research/weatherbenchX).
