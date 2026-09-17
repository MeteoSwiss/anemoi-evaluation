# User guide

How to score anemoi forecasts with `anemoi-evaluation`: configs, runs, ensembles, sharding and
results. Exhaustive listings of the CLI, the config keys, the Python API and the result file are
in the [reference](reference.md); the reasons behind the design are in the
[architecture](design/architecture.md).

- [Concepts](#concepts)
- [Installation and prerequisites](#installation-and-prerequisites)
- [Writing an evaluation config](#writing-an-evaluation-config)
- [Running an evaluation](#running-an-evaluation)
- [Dry runs and cost estimates](#dry-runs-and-cost-estimates)
- [Ensembles](#ensembles)
- [Multi-step-output models and lead 0](#multi-step-output-models-and-lead-0)
- [Regions, weights and time bins](#regions-weights-and-time-bins)
- [Anomaly correlation with a climatology](#anomaly-correlation-with-a-climatology)
- [Categorical scores at thresholds](#categorical-scores-at-thresholds)
- [Rank histogram](#rank-histogram)
- [Reliability diagram](#reliability-diagram)
- [Persistence and the other included sources](#persistence-and-the-other-included-sources)
- [Sharding and merging](#sharding-and-merging)
- [Reading the results](#reading-the-results)
- [Bringing your own source](#bringing-your-own-source)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

## Concepts

- An **init time** is the date a forecast starts from; a run scores a list of them, given as a
  range (`start`, `end`, `frequency`) or as explicit dates. The **lead times** are `k * timestep`
  up to `lead_time`, where `timestep` comes from the forecast source (the checkpoint, or the
  `timestep` of a persistence baseline), so `lead_time` must be a positive multiple of it.
- A **frame** is one forecast step: float32 `(members, variables, nodes)` at one valid time.
  Every frame is paired with the target field at the same valid time.
- A **statistic** is computed per element (variable, node) of a frame and accumulated as a
  weighted float64 sum. A **metric** is a function of the *means* of those statistics: `rmse` is
  the square root of the mean squared error, `crps` the mean `skill` minus a member-dependent
  multiple of the mean `pairs`. Only the sums are stored, so results add
  ([reference](reference.md#statistics-and-metrics)). A statistic may be parametrised: a threshold
  score carries its label in the name of both the metric and the sums, as in `csi_heavy` and
  `state_sum_hit_heavy`.
- **Weights** are one float per node and **regions** are boolean node masks, so a result is a
  weighted mean over the nodes of a region; regions may overlap. A **time bin** is the second
  aggregation axis (season, month, date or a single bin); lead time is always its own axis.
- A **source** produces forecasts, targets or a climatology. Sources are small protocols, so a
  run mixes an anemoi-inference checkpoint, a persistence baseline or your own code.
- Nodes where any member or the target is non-finite are excluded from every statistic *and* from
  the weight sum of that frame, so a metric is always a mean over what was finite.
- Fields are compared in the units the runner yields them in, against targets from a zarr in the
  same units, so `tp` is scored as the per-step quantity the model was trained on: nothing is
  accumulated over lead time.

## Installation and prerequisites

Python 3.11 to 3.13; clone the repository and install in editable mode:

```bash
git clone https://github.com/MeteoSwiss/anemoi-evaluation
cd anemoi-evaluation
pip install -e ".[tests]"
pytest tests        # CPU only, a few seconds
```

The dependencies are anemoi-datasets, anemoi-graphs, anemoi-inference, anemoi-transform,
anemoi-utils, numpy, torch, xarray, netcdf4, pydantic and pyyaml. To score a checkpoint you also
need a **GPU** (about 16 GiB for one member of a 1 km stretched-grid model, see
[benchmarks](benchmarks.md#3-model-step-time-and-gpu-memory)), a **single-dataset
anemoi-inference checkpoint** and an **anemoi-datasets zarr** for the targets, by default the
dataset the checkpoint reads its inputs from. The device defaults to the runner's, else `cuda`
when available, else CPU. The persistence baseline and `merge` need neither a GPU nor a
checkpoint.

## Writing an evaluation config

A config is YAML, validated with pydantic. Unknown keys are rejected everywhere except in the two
pass-through blocks (`forecast.anemoi_inference` and `targets.anemoi_dataset`).

### A minimal config

```yaml
forecast:
  anemoi_inference:
    checkpoint: /path/to/inference-last.ckpt
    device: cuda
    input:
      dataset:
        use_original_paths: true

lead_time: 120h
init_times:
  start: 2024-01-02T00
  end: 2024-01-05T12
  frequency: 12h

output:
  path: results/det-120h.nc
```

`forecast`, `lead_time` and `init_times` are required, plus `output` to run from the command line.
Everything else has a default: targets from the forecast's own dataset, the variables the two
sources share, metrics `rmse`, `mae` and `bias`, the checkpoint's `area_weight` as node weights,
one region `global`, seasonal bins by init time, missing targets skipped, no lead 0. Every key,
its type and its default are in the [reference](reference.md#top-level-evaluationconfig).

`forecast.anemoi_inference` consumes `members`, `seed`, `quiet` and `forcings_cache_bytes`; every
other key goes to the anemoi-inference run configuration (`checkpoint`, `device`, `input`, `env`,
...), except `date` and a non-`none` `output`, which are refused because init times come from the
evaluation and forecast fields never leave memory. `targets.anemoi_dataset` consumes `prefetch`
and `cache_bytes`; every other key goes to `anemoi.datasets.open_dataset`, and an empty block
means the forecast source's own dataset.

### A fuller config

```yaml
forecast:
  anemoi_inference:
    checkpoint: /path/to/inference-last.ckpt
    device: cuda
    input:
      dataset:                  # the arguments the checkpoint records, over 2024 only
        from_checkpoint: true
        start: 2024
        end: 2024
    env:
      ANEMOI_INFERENCE_NUM_CHUNKS: 8
    members: 4
    seed: 0

targets:
  anemoi_dataset: {from_checkpoint: true, start: 2024, end: 2024, cache_bytes: 2GiB}

lead_time: 120h
init_times:
  dates: [2024-01-02T00, 2024-01-02T12, 2024-01-03T00]

variables: [2t, 10u, 10v, t_850, z_500, tp]
metrics: [rmse, mae, bias, crps, fair_crps, spread, spread_skill, acc]
weights: {graph_attribute: area_weight}
regions:
  global: all
  lam: {graph_attribute: cutout_mask}
bins: {time: month, by: valid_time}
climatology: {file: climatology.nc}
include_lead_zero: true
output:
  path: results/ens-120h-{shard}.nc
```

### Targets from another dataset

`targets.anemoi_dataset` takes the `open_dataset` arguments of any single-member dataset whose
grid matches the model's to within 1e-5 degrees:

```yaml
targets:
  anemoi_dataset:
    dataset: /scratch/analysis-2024.zarr
    start: 2024
    end: 2024
    prefetch: 4
```

### Reusing configs: `base` and `from_checkpoint`

`base: <path>` merges this file on top of another one. The path is relative to the file naming it,
that file may name a base of its own, and mappings merge key by key while scalars and lists
replace. A campaign's variants are then a few lines each:

```yaml
# ens-120h.yaml
base: common.yaml
forecast:
  anemoi_inference:
    members: 4
metrics: [rmse, crps, fair_crps, spread, spread_skill]
output: {path: results/ens-120h.nc}
```

Two things the merge cannot do. It cannot **drop** an inherited key: a base's `climatology` or
`variables` stays unless the leaf overrides it with something else. And it cannot **switch a block
to its other form**, because the forms merge into an invalid mixture: `init_times: {dates: [...]}`
on top of `{start, end, frequency}` is neither `InitTimesRange` nor `InitTimesList`, and
`forecast: {persistence: ...}` on top of `{anemoi_inference: ...}` has two sources where one is
allowed. The same holds for `weights`; keep those blocks in the leaf configs.

`from_checkpoint` inside a dataset block stands for the `open_dataset` arguments the checkpoint
records, with the rest of the block on top. `true` means the checkpoint of the `anemoi_inference`
forecast, a string is a path to any checkpoint (how a persistence baseline reads the model's
dataset without a model), `false` disables it. Nothing is filtered out, so without an override the
block also inherits the training period's `start` and `end`. Both `base` and `from_checkpoint`
resolve on the raw mapping before validation, so what `--dry-run`, `to_config()` and the result
file's `config` attr show is the resolved config, and a run can be reproduced from its own result
file.

## Running an evaluation

From the command line:

```bash
anemoi-evaluation run config.yaml
```

It logs, per init time, the wall time of each phase and the number of model calls, then a total
(abbreviated here):

```
INFO ...evaluate: 2024-01-02T00:00:00: model 60.12s, target 0.05s, statistics 0.03s, total 60.20s (20 model calls)
...
INFO ...evaluate: 8 init times: model 480.96s, target 0.40s, statistics 0.24s, total 481.60s, peak GPU memory 15.89 GiB
```

`model` is the time spent producing frames, `target` the main thread's *wait* for targets (small
when prefetch hides the reads), `statistics` the aggregation; a further line reports the rows the
target source read and its cache hits. The result is one netcdf at `output.path`.

The same run from Python:

```python
import anemoi.evaluation as ae

evaluation = ae.Evaluation.from_config("config.yaml")
state = evaluation.run()                    # or evaluation.run(some_init_times)
evaluation.close()
state.to_xarray(evaluation.metrics).to_netcdf("results.nc")
```

The sources can also be built directly: `InferenceForecastSource`, `DatasetTargets.from_forecast`
and `Evaluation(...)` take the same arguments as the config keys, see the
[reference](reference.md#anemoievaluationevaluate) and the README's quick start. Weights and
regions then accept either a spec (a dict, as in the YAML) or a ready `(N,)` array; specs are
resolved lazily, so `plan()` never loads a model graph, while arrays cannot be serialised and
leave the result file without a `config` attr. `Evaluation.from_config(config, forecast=...,
targets=..., climatology=...)` overrides the configured sources with objects you already have, so
one loaded model can serve several evaluations. For a custom loop, `evaluation.pairs()` yields
`(frame, target)` pairs without aggregating.

## Dry runs and cost estimates

```bash
anemoi-evaluation run config.yaml --dry-run
```

resolves the config and the sources, checks that every date a rollout needs exists in the dataset,
and prints the plan as YAML without loading the model: the source descriptions, the init and lead
times, the model calls, the variables, the metrics and their statistics, the weights, regions and
bins, the frame counts and a `bytes` block sized from the grid. It is the cheapest way to catch a
wrong date range, a variable the sources do not share or a lead time that is not a multiple of the
timestep.

`--step-time <seconds>` adds a cost estimate, in seconds of model time per model call per member
([benchmarks](benchmarks.md#3-model-step-time-and-gpu-memory) reports measured values per
checkpoint). For the minimal config above (8 init times, 120 h in 6 h steps, one member, so 20
model calls per init time):

```bash
anemoi-evaluation run config.yaml --dry-run --step-time 3.15
```

```
INFO ...: dry run: about 0.14 GPU-hours of model time, about 10 minutes 9 seconds; model loaded: False
```

The estimate is `model calls x members x step time` plus 105 s of process startup per shard, so
the same run with `--shard 0/4` reports about 3 minutes 51 seconds per shard over 4 shards. It
counts neither target reads (assumed hidden by prefetch) nor statistics. `--step-time` without
`--dry-run` is an error.

## Ensembles

Set `members` in the inference block. Members are lockstep rollouts of one runner: one input
state, `members` generators advanced together, so a frame carries all members at once.

```yaml
forecast:
  anemoi_inference:
    checkpoint: /path/to/ensemble.ckpt
    members: 4
    seed: 0
metrics: [rmse, crps, fair_crps, spread, spread_skill, member_rmse, member_mae]
```

The global RNG is reseeded before every member's model call with a hash of
`(seed, init time, member, call)`, so a member depends on neither the ensemble size nor the other
members: member 2 of a 4-member run is bit-identical to member 2 of an 8-member run with the same
seed.

`crps`, `fair_crps`, `spread`, `spread_skill`, `rank_histogram` and `outlier_fraction` need at
least two members and raise when the source has fewer. Five choices worth making deliberately:

| metric | meaning |
|---|---|
| `crps` / `fair_crps` | kernel CRPS at `alpha = 0` / `alpha = 1`; the fair version is the unbiased one for a finite ensemble and is always the smaller |
| `{crps: {alpha: 0.5}}` | any `alpha` in `[0, 1]`; the result variable is named after it, `crps_0p5` |
| `member_rmse`, `member_mae` | the error of a randomly drawn member in expectation, never below `rmse` / `mae`, which are computed on the ensemble mean |
| `rank_histogram`, `outlier_fraction` | whether the ensemble is calibrated rather than how accurate it is, see [Rank histogram](#rank-histogram) |
| `reliability`, `forecast_frequency` | whether the forecast probabilities of a threshold mean what they say, see [Reliability diagram](#reliability-diagram); unlike the rank histogram they work at one member too |

The full list is in the [reference](reference.md#metrics). A checkpoint with no `noise_injector`
logs a warning and produces identical members, so `spread` is zero and `crps` equals `mae`.

Model time and memory both grow linearly with members: four lockstep members cost four times one,
and an extra member costs about 1.45 GiB on a 6-hourly 2-in/1-out 1 km checkpoint, about 6 GiB on
an hourly 7-in/6-out one ([benchmarks](benchmarks.md#3-model-step-time-and-gpu-memory)).

## Multi-step-output models and lead 0

A model whose checkpoint has `multi_step_output > 1` produces several output times per forward
pass. Nothing has to be configured: the source reports `frames_per_pass`, each output time is
scored as its own frame, and the target prefetch depth is raised to match. The one thing to watch
is that `lead_time` should be a multiple of the model's **output horizon**
(`timestep * multi_step_output`), not only of the timestep: a 1-hourly 6-output model asked for
`lead_time: 9h` makes two calls, computes 12 h and discards the last three frames. That is legal,
logs a warning, and the dry-run plan says so in its `note` and in `computed_lead_time`.

`include_lead_zero: true` adds a lead-0 frame from the initial state, before any model call. It
works for sources that declare `supports_lead_zero` (the inference and persistence sources do;
asking for it elsewhere raises `cannot produce lead-0 frames`). For the inference source the
lead-0 frame is the init-time slice of every requested variable present in the input state, and
**NaN for the rest**: diagnostic variables such as `tp` have no analysis. Those NaN nodes are
excluded, so a diagnostic variable at lead 0 has weight sum 0 and a NaN metric, while prognostic
variables score exactly 0 when the targets come from the input dataset. That exact zero is a
useful end-to-end check; `tools/inspect_results.py` verifies it.

## Regions, weights and time bins

Weights are a `(N,)` float array and a region a `(N,)` boolean mask; the config maps specs onto
them, and only relative weights matter since every result is a weighted mean.

```yaml
weights: {graph_attribute: area_weight}   # or spherical_voronoi, uniform, file
regions:
  global: all
  lam: {graph_attribute: cutout_mask}     # a boolean node attribute of the training graph
  alps: {bbox: {north: 48.0, west: 5.5, south: 45.5, east: 11.0}}
  lam_grid: {grid: 0}                     # sub-grid 0 of a cutout or join target dataset
```

Every spec is in the reference, for [weights](reference.md#weights) and [regions](reference.md#regions).

On a stretched grid the training `area_weight` gives the LAM nodes tiny weights and the global
nodes large ones. That is the physically correct global mean, and it means a LAM-only number
cannot come from different weights: it comes from a LAM region, such as the `cutout_mask` graph
attribute above or `{grid: i}` on a cutout target dataset. There is no land/sea mask spec: use
`file:` with a `.npy` mask. `{graph_attribute: ...}` resolves only when the run starts, so
`--dry-run` does not catch a wrong attribute name.

Bins are the second aggregation axis:

```yaml
bins: {time: season, by: init_time}
```

`time` is `season` (DJF, MAM, JJA, SON), `month` (`01` to `12`), `init_time` (one bin per date) or
`none` (a single bin `all`); `by` is `init_time` or `valid_time`. Whatever the binning, the result
file also carries a derived `all` bin, summed over the stored bins *before* the division, so
`bin="all"` is the number over the whole run.

## Anomaly correlation with a climatology

`acc` is the anomaly correlation of the ensemble mean against a climatology, pooled over nodes and
init times within a bin: `mean(fa * ta) / sqrt(mean(fa^2) * mean(ta^2))`, where `fa` and `ta` are
the forecast and target anomalies. Without a `climatology` block it fails with `the anomaly
metrics need a climatology source`.

### Building a climatology

`tools/make_climatology.py` builds one from the target dataset of an existing config:

```bash
python tools/make_climatology.py config.yaml --start 2024-01-01 --end 2024-01-31T18 \
    --key hour_of_day --prefetch 4 -o climatology.nc
```

It reads every dataset date in `[start, end]` once, accumulates float64 sums per key, writes
float32 means and asserts that nothing is non-finite. No GPU is needed, but it reads one zarr row
per date: a month of six-hourly dates takes a few minutes off Lustre. The config must have an
`anemoi_inference` forecast with an `input.dataset` block (that is the dataset it reads) and must
list `variables` explicitly. `--key` selects what a valid time maps to: `hour_of_day` (the
default), `month`, `day_of_year` or `constant`; a run fails if it reaches a key the file does not
have, for example an hourly model against an hour-of-day climatology built from six-hourly dates.
The same from Python:

```python
climatology = ae.ArrayClimatology.from_targets(targets, dates, variables, key="hour_of_day")
climatology.to_netcdf("climatology.nc")
```

### Using it

```yaml
metrics: [rmse, acc]
climatology: {file: climatology.nc}
```

- The climatology must be on the same grid as the forecast and cover the evaluated variables.
- Nodes where the climatology is non-finite contribute zero to the three anomaly sums while the
  shared weights still count them, so only the ACC ratio is meaningful, not the raw means of the
  anomaly statistics. Such nodes are logged and recorded in `climatology_nonfinite_nodes`.
- Per-season or per-init ACC comes from the binning (each bin pooled), not from averaging per-init
  correlations. The two are not the same number.
- A device copy per (key, variable set) is kept, about 0.12-0.23 GiB for a 232 MiB climatology.

## Categorical scores at thresholds

`pod`, `far`, `csi`, `ets`, `frequency_bias`, `hss` and `pss` score a binarised event, `brier`, `bss`
and `event_frequency` score its probability. Each one takes a **label** and a per-variable **threshold**
map, and is named after both:

```yaml
metrics:
  - rmse
  - {csi:             {label: heavy, thresholds: {tp: 0.005}}}   # example numbers, not defaults
  - {pod:             {label: heavy, thresholds: {tp: 0.005}}}
  - {brier:           {label: heavy, thresholds: {tp: 0.005}}}
  - {bss:             {label: heavy, thresholds: {tp: 0.005}}}
  - {event_frequency: {label: heavy, thresholds: {tp: 0.005}}}
  - {pod:             {label: frost, thresholds: {2t: 273.15}}}
```

That writes `csi_heavy`, `pod_heavy` and the rest as data variables and
`state_sum_hit_heavy`, `state_sum_miss_heavy`, `state_sum_false_alarm_heavy`,
`state_sum_brier_heavy` and `state_sum_event_frequency_heavy` as the raw sums. Metrics that share a
label share those sums, so the five metrics above cost five sums, not nine. The thresholds must be
the same for one label: two metrics with the same label and different numbers are refused at config load,
and at construction from Python, as are two metrics that resolve to the same name.

The event is `value > threshold`, strictly, for the target and for the ensemble mean of the forecast;
the Brier score uses the fraction of members above the threshold. Only exceedances can be configured, so a
"below" event such as frost is scored as its complement: `pod_frost` above reads as the detection of
"not frost", and the frost cells are the mirror image of the stored ones.

Picking a threshold takes three facts. The unit is whatever the training dataset stores, which is the
`units` entry of the variable's metadata in the zarr (`ds.attrs["variables_metadata"]`), typically metres
for `tp` and kelvin for `2t`. `tp` is the **per-step** quantity (see the concepts above), so `0.005` is
5 mm per 6 h on a 6-hourly model and 5 mm per hour on an hourly one, the same number is a different event
on the two, and nothing accumulates over lead time. And the base rate of a candidate threshold costs no
GPU: run the `persistence` forecast source over the period with
`{event_frequency: {label: heavy, thresholds: {tp: 0.005}}}` and read `event_frequency_heavy`, which is the
observed frequency of the event and nothing to do with the model. There are no default thresholds.

A vector event such as "wind above 10 m/s" is not expressible: thresholding `10u` and `10v` separately is
not a wind speed, and the package builds no derived variables (see the [requirements](design/requirements.md)),
so the dataset must carry the scalar variable itself.

Write a small threshold as `0.001` or `1.0e-3`: PyYAML follows YAML 1.1 and reads `1e-3` as the *string*
`'1e-3'`, which is refused with `threshold for 'tp' must be a finite number`.

A variable that a label's map does not name is **NaN** in that label's metrics and sums, at every lead
time, bin and region. A column of NaN therefore means "no threshold for this variable", not "zero". A
threshold naming a variable the run does not evaluate is an error instead, since it is almost always a
typo. Read `event_frequency_<label>` first: it is the base rate, and it tells you whether the
threshold was scorable at all. Where nothing was observed and nothing forecast, `pod`, `far`, `csi`,
`ets` and `frequency_bias` are 0/0 NaN, which is the honest answer, not a bug.

All ten work for a deterministic checkpoint. What compares the two model kinds is `brier` and `bss`, and the
reliability diagram at one member: they all read the fraction of members above the threshold, and at one
member the Brier score collapses to `miss + false_alarm` exactly at every element, and so to within float64
rounding once summed.

The contingency scores do not, and this is the one trap of the family. For an ensemble they binarise the
**ensemble mean**, so they score it as a point forecast. The mean of a skewed field has less spread than a
member, so it crosses a rare threshold far less often than a single member of the same calibrated ensemble
does. A calibrated ensemble therefore reads as badly under-forecasting heavy events, and `pod_heavy` or
`frequency_bias_heavy` must not be compared between a deterministic checkpoint and its ensemble sibling. What you usually want instead is the contingency table
at a chosen forecast probability, which the reliability level sums already contain (see
[Reliability diagram](#reliability-diagram)) and which a later release can derive with no new statistic.

Note that `bss` is a skill score against the base rate of its own cell, so it does not compare across lead
times, regions or bins; on a rare event with `bins: {time: init_time}` most dates have a base rate of 0 or
1 and read NaN or minus infinity, so the pooled `all` bin is the one to read.

## Rank histogram

The rank histogram says whether an ensemble is calibrated: how often the target fell below every
member, between the first and the second, and so on up to above every member. It needs at least two
members and no configuration at all.

```yaml
metrics: [rmse, rank_histogram, outlier_fraction]
```

The file gets `rank_histogram` on `(lead_time, bin, variable, region, rank)`, with a `rank`
coordinate of the `M + 1` values `0..M`, and `outlier_fraction` as one number per
`(lead_time, bin, variable, region)`:

```python
results["rank_histogram"].sel(bin="all", region="global", variable="2t").isel(lead_time=-1).values
results["outlier_fraction"].sel(bin="all", region="global", variable="2t").values
```

How to read it. A **flat** histogram, every bin at `1 / (M + 1)`, is a calibrated ensemble. A **U**
shape, both end bins high, means the target lands outside the ensemble too often: the ensemble is
under-dispersed. A **dome** means it is over-dispersed. A **slope** is a bias, downwards when the
forecasts are too high. `outlier_fraction` is the sum of the two end bins and is the one number worth
quoting: a calibrated `M`-member ensemble gives `2 / (M + 1)` (0.222 at 8 members, 0.038 at 51), more
means under-dispersed, less over-dispersed.

Ties are spread, not broken. When `ties` members equal the target exactly, each of the `ties + 1` bins
the target could occupy gets `1 / (ties + 1)`, which is what random tie-breaking would give in
expectation, with no seed and no noise. This matters for a field that is exactly zero at most nodes: a
naive rank would put the mass of every exact tie in the first bin and read as a catastrophic high bias,
and spreading removes that artefact.

It removes the artefact only. A `tp` histogram with a bin-0 spike left over is saying something real,
usually that the model is never exactly dry where the target is: a dry target below every (slightly
positive) member is not a tie at all, it is a bias, and no tie rule touches it. Read the rank histogram of
a bounded variable together with the reliability diagram at the bound, `thresholds: {tp: 0.0}` for
precipitation occurrence, which answers the dry/wet question directly.

Two more things to know. `M` comes from the run, so no number goes in the config, and a result file
written at one ensemble size cannot be merged with one written at another. And at lead 0 every member
equals every other one, so the target is either tied with all of them, which spreads the mass evenly and
gives a flat histogram (the case when the targets come from the model's input dataset), or outside all of
them, which puts the whole mass in bin 0 or bin `M`. Either way it says nothing about calibration.

## Reliability diagram

The rank histogram asks whether the ensemble is calibrated as a whole; the reliability diagram asks
whether the probability it gives one **event** means what it says. Of all the cases where the
ensemble said 30 %, how often did the event happen? On the diagonal is perfect.

```yaml
metrics:
  - {reliability: {label: heavy, thresholds: {tp: 0.005}}}
  - {forecast_frequency: {label: heavy, thresholds: {tp: 0.005}}}
  - {brier_reliability: {label: heavy, thresholds: {tp: 0.005}}}
  - {brier_resolution: {label: heavy, thresholds: {tp: 0.005}}}
  - {brier_uncertainty: {label: heavy, thresholds: {tp: 0.005}}}
```

The label and the threshold map are the ones every threshold score uses, and a label names one event:
every metric carrying `heavy` must give the same map, in this family and in the categorical one.

The file gets `reliability_heavy` and `forecast_frequency_heavy` on
`(lead_time, bin, variable, region, probability)`, with a `probability` coordinate of the `M + 1`
values `k / M`, and the three components as one number per `(lead_time, bin, variable, region)`:

```python
cell = {"bin": "all", "region": "global", "variable": "tp"}
curve = results["reliability_heavy"].sel(**cell).isel(lead_time=-1)      # observed frequency
sample = results["forecast_frequency_heavy"].sel(**cell).isel(lead_time=-1)   # how often each was issued
weight = sample * results["weight_sum"].sel(**cell).isel(lead_time=-1)   # absolute sample weight
```

How to read it. The diagonal is perfect. A curve **flatter** than the diagonal is over-confident: it
says 90 % and is right 70 % of the time. A curve **steeper** than the diagonal is under-confident. A
curve **above** the diagonal everywhere means the event is under-forecast. And a point with almost no
`forecast_frequency` under it means nothing at all, which is exactly why the second metric is there:
read the curve and the histogram together or not at all.

The decomposition. `brier_reliability` (`REL`) is the vertical distance from the diagonal, weighted by
the sample of each level; small is good. `brier_resolution` (`RES`) is how far the levels separate
events from non-events; large is good. `brier_uncertainty` (`UNC`) is the climatological difficulty of
the threshold in that cell and has nothing to do with the model. Together they are the Brier score,
`BS = REL - RES + UNC`, to the float64 rounding of the sums, so they say where a Brier difference
between two runs comes from.

Four things to know. The bins are the `M + 1` ensemble probability levels, so no number goes in the
config and a run at another ensemble size is a different set of bins. A level no forecast fell into
reads NaN in `reliability` and 0 in `forecast_frequency`. At lead 0 every member equals every other
one, so only the two end levels are populated and the diagram says nothing about calibration. And at
one member the levels are `{0, 1}`: the diagram has two points and is the contingency table in
disguise, which is what lets it, `brier` and `bss` put a deterministic checkpoint and its ensemble sibling
on one axis.

## Persistence and the other included sources

The persistence baseline repeats the target at init time at every lead. It needs no model and no
GPU, and it needs an **explicit** `targets` block, because there is no forecast source to borrow
a dataset from:

```yaml
forecast:
  persistence:
    timestep: 6h
    members: 1
targets:
  anemoi_dataset:
    from_checkpoint: /path/to/inference-last.ckpt   # the model's dataset, without the model
    start: 2024
    end: 2024
lead_time: 120h
init_times: {start: 2024-01-02T00, end: 2024-01-05T12, frequency: 12h}
output: {path: results/persistence-120h.nc}
```

`timestep` sets the lead-time grid, so use the model's timestep to get comparable lead times. Init
times whose target is missing are warned about at construction and skipped at run time. Also
included, mainly for tests and notebooks: `ArrayTargets`, `FakeForecastSource` and
`ArrayClimatology`, all exported from `anemoi.evaluation`.

## Sharding and merging

A run splits over init times: shard `i` of `n` evaluates `init_times[i::n]` and writes its own
file, one process per shard, no communication; the files are summed afterwards.

```bash
anemoi-evaluation run config.yaml --shard 0/4     # with output.path: results/run-{shard}.nc
```

With more than one shard `output.path` must contain a `{shard}` placeholder or the run refuses to
start; `{shards}` (the count) is substituted too but does not satisfy that requirement. Without
`--shard`, the CLI combines the Slurm job-array and job-step variables, so job arrays and
job steps compose and a plain run outside Slurm is shard 0 of 1 (the formula is in the
[reference](reference.md#sharding)). `--shard 0/1` disables the automatic sharding explicitly.

Four tasks on one node, one GPU each:

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --time=01:00:00

srun anemoi-evaluation run config.yaml       # output.path must hold {shard}, for the glob below
anemoi-evaluation merge results/run-{0..3}.nc -o results/run.nc
```

As a job array instead (`#SBATCH --array=0-3`, one task per job), the same command works
unchanged. One trap there: do not overwrite `CUDA_VISIBLE_DEVICES` with `SLURM_LOCALID` in a job
wrapper. Slurm may co-schedule the array tasks on one node with one device each, and
`SLURM_LOCALID` is 0 in every one-task job, so the tasks collide on GPU 0
([benchmarks](benchmarks.md#9-operational-findings)).

```bash
anemoi-evaluation merge results/run-*.nc -o results/run.nc
anemoi-evaluation merge results/run-{0..10}of12.nc --partial -o partial.nc   # 11 of 12 shards
anemoi-evaluation merge partial.nc results/run-11of12.nc -o results/run.nc   # completed later
```

`merge` sums the raw state, so the merged file is the run as if it had been one process: timings
and counters summed, `peak_gpu_memory_bytes` the maximum, `init_times` the union, `config` a JSON
object `{"merge": [...]}` of the inputs' configs, `merged_from` the paths. It refuses anything
that is not a whole run: a missing shard, a duplicate, or inputs that disagree on `n`. A set that
is all untagged (states from the Python API, already-merged files) skips that check; any mix of
tagged and untagged inputs raises. Differing configs among the shards are only warned about.
`--partial` merges an incomplete set and records what it holds as `shard: 0,2/3`, which keeps the
file mergeable with the shards it lacks. Sums are bit-exact against an unsharded run when every
bin is filled by a single shard, which is the case for `init_time` bins, and agree to float64
rounding otherwise. The same from Python:

```python
state = ae.load_state("results/run-0.nc")                  # the raw state of one shard
merged = ae.merge(["results/run-0.nc", "results/run-2.nc"], partial=True)
```

## Reading the results

One netcdf holds both the metrics and the raw sums they came from.
```python
import xarray as xr

ds = xr.open_dataset("results/run.nc", decode_timedelta=True)
print(ds.attrs["checkpoint"], ds.attrs["init_times"], ds.attrs["members"])
ds["rmse"].sel(bin="all", region="global", variable="2t")
```

Data variables:

- one per metric, on `(lead_time, bin, variable, region)`;
- `n_init` on `(lead_time, bin)` and `weight_sum` on all four dims;
- the raw state: `state_sum_<statistic>`, `state_weights` and `state_n_init`, on `state_bin`.

The coordinates and the full list of attributes (checkpoint identity, package versions, timings,
counters, peak memory, shard tag) are in the [reference](reference.md#netcdf-layout).

Because the sums are stored, any metric of those statistics can be recomputed after the fact and
subsets of bins can be re-aggregated:

```python
import numpy as np

rmse = np.sqrt(ds["state_sum_squared_error"] / ds["state_weights"])          # per stored bin
djf = ds.sel(state_bin=["DJF", "MAM"])
rmse_djf_mam = np.sqrt(djf["state_sum_squared_error"].sum("state_bin") / djf["state_weights"].sum("state_bin"))
```

Sum first, divide last: averaging the per-bin metrics is a different number. `weight_sum` is what
each metric was divided by; divided by `n_init` it is the weight of one init time, and its
shortfall against the maximum over lead times and variables is the fraction of the region excluded
as non-finite.

Comparing two runs:

```bash
python tools/inspect_results.py results/run.nc      # mechanics and headline numbers
python tools/compare_results.py a.nc b.nc           # element by element, --leads/--stats common
```

`inspect_results.py` checks weight sums, the excluded fraction per lead, the lead-0 zeros and the
inequalities that must hold (`mae <= rmse`, `fair_crps <= crps`, `member_rmse >= rmse`, `acc` in
`[-1, 1]`). `compare_results.py` reports every raw sum and metric as bit-exact or by its maximum
relative difference and exits non-zero on a structural difference; `--leads common --stats common`
compares files that differ in lead times or statistics ([`tools/README.md`](../tools/README.md)).
To compare two *models*, score both with the same config, init times, regions and weights and
compare the metrics directly; the attrs carry each checkpoint's identity.

## Bringing your own source

A source is a protocol, not a base class hierarchy. A forecast source needs `grid`, `variables`,
`members`, `lead_times()` and `frames()`; a target or climatology source needs `grid`, `variables`
and `frame()`. Subclass `ForecastSourceBase`, `TargetSourceBase` or `ClimatologySourceBase` to
inherit defaults for everything optional. A climatological forecast baseline:

```python
import datetime
from collections.abc import Iterator

from anemoi.evaluation import ForecastSourceBase, Frame
from anemoi.evaluation.sources.base import lead_time_steps


class ClimatologyForecast(ForecastSourceBase):
    """Forecasts the climatology at each valid time."""

    # __init__ sets self.climatology, self.grid, self.variables, self.timestep and self.members

    def lead_times(self, lead_time: datetime.timedelta) -> list[datetime.timedelta]:
        return lead_time_steps(lead_time, self.timestep)

    def frames(self, init_time, lead_time, variables, device) -> Iterator[Frame]:
        for lead in self.lead_times(lead_time):
            valid_time = init_time + lead
            field = self.climatology.frame(valid_time, variables, device)
            data = field[None].repeat(self.members, 1, 1).float()
            yield Frame(init_time, lead, valid_time, list(variables), data)
```

It is then passed to `ae.Evaluation(forecast, targets, init_times, "120h", ["rmse", "bias"])` like
any other source. The contract, in short: `frames()` yields strictly increasing lead times, each `Frame.data`
float32 `(members, variables, nodes)` on the requested device and owned by the caller; `Frame`
validates itself (`valid_time == init_time + lead_time`, float32, one column per name); a target
source raises `MissingTargetError` when it cannot deliver a valid time, which the driver skips or
re-raises according to `on_missing_target`; and the forecast, target and climatology grids must
agree to within 1e-5 degrees. The optional hooks (lead 0, a model graph, several frames per model
call, an early date check, the dry-run description, the result attrs, serialisation) are listed
with their defaults under [source protocols](reference.md#source-protocols). Sources built in
Python go to `Evaluation(...)`, or to `Evaluation.from_config(config, forecast=..., targets=...)`
to keep the rest of a YAML config.

## Performance

The [benchmarks](benchmarks.md) have the measured numbers and their testbeds; the knobs are:

- **Target prefetch.** `targets.anemoi_dataset.prefetch`, `null` by default, which means
  `max(2, frames per model call)`. It cut a 48 h deterministic run by 16 % and a 12 h multi-step
  run by 15 %; raising it further helps only when the run is read-bound, and costs some model time
  to contention with the worker thread. `prefetch: 0` reads synchronously on the main thread.
- **Target cache.** `cache_bytes` (default 0, `"2GiB"` style strings accepted) keeps decoded rows
  in a host LRU. On that multi-step run it served 18 of the 48 rows from memory and took the total
  from 101.4 s to 91.5 s; it changes no number, only the number of rows read
  ([benchmarks](benchmarks.md#5-prefetch-and-cache)).
- **`ANEMOI_INFERENCE_NUM_CHUNKS`.** Set it in the `env` block of the inference config (8 is the
  value used at 1 km). It bounds the decoder's per-edge memory: 1 mapper chunk needed 71 GiB where
  8 needed 16 GiB. Different mapper chunk counts give *different numbers* (a reduction order,
  amplified by the rollout), so keep one value across a campaign.
- **`ANEMOI_INFERENCE_NUM_CHUNKS_PROCESSOR`.** Added as `1` when `ANEMOI_INFERENCE_NUM_CHUNKS` is
  set (in the block or the environment) and this one is not; a value given anywhere is left alone.
  Chunking the processor costs up to 46 % of every forward and saves no memory. Both are read by
  anemoi-models at import time, so set them in the `env` block or before the process starts.
- **`forcings_cache_bytes`.** A host LRU of computed forcings arrays, 1 GiB by default, shared
  across lockstep members and init times; it took a 4-member step from 8.9 s to 8.2 s. 0 disables.
- **Members** cost model time and memory linearly. **Sharding** is nearly free: four tasks with
  prefetch on one node scored eight 120 h forecasts in 2:13 against 10:01 unsharded, bit-exact
  ([benchmarks](benchmarks.md#6-sharding)). A **cold node** costs about 90 s of startup per
  process. **`numcodecs.blosc.use_threads = True`** takes a worker read from 0.83 s to 0.63 s per
  row, but it is process-global, so the package never sets it.

Use `--dry-run --step-time` to size a job before submitting it, and `tools/ab_prefetch.py` to
measure prefetch and cache variants in one process.

## Troubleshooting

Config and sources:

| message | meaning |
|---|---|
| `from_checkpoint at <key> is only resolved in ...` | it was used in a nested block; move it to the top of the dataset block |
| `checkpoint ... only a single mapping can be reused` | the recorded `open_dataset` arguments are not one mapping; spell the dataset out |
| `target datasets must have one member` | the target zarr has an ensemble dimension; select a member in the `open_dataset` arguments |

Dates, grids, variables and merging:

| message | meaning |
|---|---|
| `init time T: dates [...] are not in the <kind> dataset` | a rollout reads the lagged input window and the forcings dates of every model call but the last; widen the dataset range or move the init times. `--dry-run` catches this without a GPU |
| `source cannot deliver variables [...]` | a name is missing from the forecast outputs or the target dataset; omit `variables` to get the intersection |
| `grids differ in size`, `grid coordinates differ by more than 1e-05 degrees` | the targets or the climatology are not on the model grid; with a cutout checkpoint take the targets from the forecast, which also applies `grid_indices` |
| `a forecast source with a model graph is needed for the node attribute ...` | use `spherical_voronoi`, `uniform` or a `file:` mask instead |
| `graph nodes 'data' have no attribute '...'` | the training graph does not carry it; the error lists the ones it does |
| `the metrics need at least 2 members, the forecast source has 1` | `crps`, `fair_crps`, `spread` and `spread_skill` need an ensemble; the rank histogram reports it itself |
| `rank_histogram needs at least 2 members, the run has 1` | `rank_histogram` and `outlier_fraction` are ensemble metrics; drop them for a deterministic checkpoint |
| `rank_histogram is already bound to 4 members, cannot rebind to 8` | the same metric instance was reused for a run with another ensemble size; build a fresh one |
| `reliability_heavy is already bound to 4 members, cannot rebind to 8` | the same for the reliability family, which is bound to the run's ensemble size in the same way |
| `reliability_heavy was not bound to the run's ensemble size` | a metric built by hand was used without `metrics.build(..., members=M)` or `bind_members` |
| `the thresholds of ... name variables the run does not have: [...]` | a threshold names a variable the run does not evaluate; fix the name or add it to `variables` |
| `... both need a statistic named ... with different parameters` | two statistics of one name disagree on their parameters; give the metrics different labels |
| `label 'warm' is used with different thresholds by csi_warm ({...}) and reliability_warm ({...}); one label, one threshold map` | a label names one event, so every metric carrying it must give the same threshold map, across every family; give the two events different labels |
| `duplicate metric names: ... appears 2 times` | the same metric, or the same kind and label, is listed twice |
| `N of the M shards of the run were given, missing ...` | rerun the shard, or merge with `--partial` |
| `cannot merge states that share init times`, `duplicate shards of n` | the same result would be counted twice |
| `cannot merge states with different coordinates or statistics` | the inputs came from different configs |

Behaviour that is not an error:

- **Members are identical and `spread` is 0**: the checkpoint has no noise injector (a warning is
  logged when the source is built).
- **NaN metrics at lead 0 for some variables**: diagnostic variables have no analysis to score
  against; prognostic variables should be exactly 0.
- **A low `n_init`**: targets were missing and skipped. Set `on_missing_target: raise` to find out,
  or read `n_init` per lead in the result file.
- **GPU out of memory**: lower `members`, raise `ANEMOI_INFERENCE_NUM_CHUNKS`, or evaluate fewer
  `variables` (the statistics temporaries are proportional to them).
