# Reference

Exact behaviour and interfaces of `anemoi-evaluation` 0.3.0. Task-oriented instructions are in the
[user guide](user-guide.md); the reasoning behind the design is in
[design/architecture.md](design/architecture.md) and [design/requirements.md](design/requirements.md).

- [Command-line interface](#command-line-interface)
- [Configuration](#configuration)
- [Statistics and metrics](#statistics-and-metrics)
- [Aggregation and the results file](#aggregation-and-the-results-file)
- [Python API](#python-api)
- [Sharding](#sharding)
- [Seeding and reproducibility](#seeding-and-reproducibility)
- [Limitations and undefined behaviour](#limitations-and-undefined-behaviour)

## Command-line interface

The console script `anemoi-evaluation` (entry point `anemoi.evaluation.__main__:main`) has two
subcommands. A subcommand is required.

```
anemoi-evaluation run <config.yaml> [--shard i/n] [--dry-run] [--step-time seconds]
anemoi-evaluation merge <results.nc>... -o <merged.nc> [--partial]
```

### `run`

Runs an evaluation from a YAML config and writes the results netcdf.

| Argument | Type | Default | Semantics |
|---|---|---|---|
| `config` | path, positional, required | n/a | YAML config, loaded by `load_config` (`base:` chain and `from_checkpoint:` blocks resolved first). |
| `--shard i/n` | string `i/n` | Slurm auto-detection, else `0/1` | Evaluate init times `i`, `i+n`, `i+2n`, ... and write `output.path` with `{shard}` / `{shards}` filled in. See [Sharding](#sharding). |
| `--dry-run` | flag | off | Resolve the config and the sources, check the dates, print the plan as YAML on stdout, and exit without loading a model or running it. |
| `--step-time seconds` | float | unset | Seconds of model time per model call per member; adds the `time` block to the plan. Only valid with `--dry-run`. `docs/benchmarks.md` section 3 reports measured values; one call of a multi-step-output model produces several frames. |

Order of operations in `run`:

1. `--step-time` without `--dry-run` raises `SystemExit("--step-time is only used by --dry-run")`.
2. The config is loaded, the shard is resolved, and without `--dry-run` a config with no `output.path`
   raises `SystemExit("the config needs output.path to run from the command line")`.
3. `logging.basicConfig` is called with `config.log_level` and the format
   `%(asctime)s %(levelname)s %(name)s: %(message)s`.
4. `Evaluation.from_config` builds the sources; `evaluation.shard(index, count)` selects the init times.
   An empty shard logs a warning and still writes an empty state.
5. With `--dry-run`, the plan is printed with `yaml.safe_dump(..., sort_keys=False, width=120)`, byte
   counts and `_s` durations rendered as `1 GiB (1073741824)` and `1 minute 30 seconds (90.0)`, and the
   `shard` block (`index`, `count`, `output`) appended. Otherwise the run executes, the sources are
   closed in a `finally`, `state.attrs["shard"]` is set to `"i/n"` (also for an unsharded run, where it
   is `"0/1"`), and the dataset is written to the formatted path, creating parent directories. A
   multi-dataset run writes one file per dataset, `{dataset}` filled in with the dataset's name.

### `merge`

Sums the aggregation states of several result files and writes one file.

| Argument | Type | Default | Semantics |
|---|---|---|---|
| `inputs` | one or more paths, positional | n/a | Result netcdf files. |
| `-o`, `--output` | path, required | n/a | Merged result netcdf. |
| `--partial` | flag | off | Allow an incomplete set of shards; the merged file records the shards it holds in its `shard` attr. Without it the inputs must be either the complete `0..n-1` shards of one run or all untagged. |

The metrics are rebuilt from the `metrics` attr of the merged state (a JSON list of metric specs), so the
merged file carries the same metric variables as its inputs. See
[merging](#merging) for the validation rules.

### Exit behaviour

- Success: exit status 0. Only the dry-run plan goes to stdout; log lines go to stderr, where
  `logging.basicConfig` sends them.
- Bad command line (unknown flag, missing subcommand, missing `-o`): argparse exits with status 2. This
  applies to both subcommands.
- `SystemExit` with a message: the two `run` checks above, exit status 1. `merge` has no such check.
- Everything else (a config validation error, a missing dataset date, an invalid shard set for `merge`)
  propagates as an exception with a traceback and a non-zero exit status. Nothing is written in that case.

### Environment variables

| Variable | Read by | Effect |
|---|---|---|
| `SLURM_STEP_NUM_TASKS` | `resolve_shard` | Tasks per job step; the per-step shard count. Default 1. |
| `SLURM_PROCID` | `resolve_shard` | Task rank within the step. Default 0. |
| `SLURM_ARRAY_TASK_ID` | `resolve_shard` | Job-array task id. Default 0. |
| `SLURM_ARRAY_TASK_MIN` | `resolve_shard` | Lowest job-array id, subtracted from the task id. Default 0. |
| `SLURM_ARRAY_TASK_COUNT` | `resolve_shard` | Number of job-array tasks. Default 1. |
| `ANEMOI_INFERENCE_NUM_CHUNKS` | `inference_env` | If set in the process environment or in the config's `env` block, the inference source adds `ANEMOI_INFERENCE_NUM_CHUNKS_PROCESSOR=1` to the `env` block unless a processor count is already set somewhere. |
| `ANEMOI_INFERENCE_NUM_CHUNKS_PROCESSOR` | `inference_env` | If set anywhere, it is left alone. |

`SLURM_NTASKS` is deliberately **not** read: a batch script sees it with `SLURM_PROCID=0` and would
silently evaluate one shard. No other environment variable is read by the package; the tools in
`tools/` set `ANEMOI_INFERENCE_NUM_CHUNKS` themselves (see [tools/README.md](../tools/README.md)).

## Configuration

The config is a YAML mapping validated by the pydantic model `EvaluationConfig`. All models except the
two pass-through blocks forbid unknown keys (`extra="forbid"`), so a typo is an error.

### Loading

`load_config(source)` accepts a path, a dict, or an existing `EvaluationConfig`. Two rewrites happen
before validation:

- **`base: <path>`**: the config is merged on top of the config the `base` key names, recursively. A
  relative `base` resolves against the directory of the file that names it. Mappings merge key by key in
  the base's order; anything else (lists, scalars) replaces. Cycles and missing files raise. The `base`
  key itself is removed. The merge has no delete syntax, so two things it deliberately cannot do are
  **drop** an inherited key and **switch a block to its other form**: `init_times: {dates: ...}` merged
  onto a base's `{start, end, frequency}` keeps all four keys and validates as neither `InitTimesRange`
  nor `InitTimesList`, and the same applies to `forecast` (both source keys present fails the
  exactly-one check) and to `weights`. Such blocks belong in the leaf config.
- **`from_checkpoint`**: only inside `forecast.anemoi_inference.input.dataset` and
  `targets.anemoi_dataset`. `true` means the checkpoint of the `anemoi_inference` forecast, a string is a
  checkpoint path, `false` removes the key and changes nothing. The checkpoint's recorded `open_dataset`
  arguments (with the original paths) replace the key, and the rest of the block is merged on top, so a
  local `start`/`end` wins. Each dataset must record a single mapping argument; a multi-dataset checkpoint
  resolves per dataset (see Multi-dataset checkpoints). A `from_checkpoint` anywhere else raises.

`EvaluationConfig` is a serialisation of the `Evaluation` constructor arguments;
`Evaluation.from_config(config)` builds the object and `Evaluation.to_config()` returns the dict again.

### Top level (`EvaluationConfig`)

| Key | Type | Default | Semantics |
|---|---|---|---|
| `base` | path | absent | Not a model field: it is consumed and removed before validation. See [Loading](#loading). |
| `forecast` | `ForecastConfig` | required | Forecast source; exactly one of `anemoi_inference`, `persistence`. |
| `targets` | `TargetsConfig`, `{datasets: {name: TargetsConfig}}` or null | null | Target source. Null (or a block with no `open_dataset` keys) means the forecast source's own dataset. See [Multi-dataset checkpoints](#multi-dataset-checkpoints). |
| `datasets` | list of str or null | null | The datasets of a multi-dataset checkpoint to score; null means every one of them. An unknown name, an empty list, or a per-dataset `datasets:` mapping that names a dataset the run does not score raises at build time. See [Multi-dataset checkpoints](#multi-dataset-checkpoints). |
| `lead_time` | duration | required | Longest lead time to score. Must be positive; strings such as `24h`, `1d`, `6h` are accepted. Must be a multiple of the forecast timestep. Serialised back as a frequency string. |
| `init_times` | `InitTimesRange` or `InitTimesList` | required | See below. |
| `variables` | list of str, `{datasets: {name: list of str}}` or null | null | Variables to score. Null means the variables shared by the two sources, in the forecast source's order (`[name for name in forecast.variables if name in targets.variables]`). A requested name that either source cannot deliver raises; an empty result raises `no variables to evaluate`. |
| `metrics` | list of str or `{name: kwargs}` | `["rmse", "mae", "bias"]` | See [metrics](#metrics). Validated at config load. |
| `weights` | `WeightsConfig`, `{datasets: {name: WeightsConfig}}` or null | null | Node weights. Null means `{graph_attribute: area_weight}` when the forecast source has a model graph, else `{uniform: {}}`. |
| `regions` | map name to `RegionConfig`, or `{datasets: {name: map}}` | `{"global": "all"}` | Named node masks; they may overlap. An empty mapping falls back to the default. |
| `bins` | `BinsConfig` | `{time: season, by: init_time}` | Time binning. |
| `climatology` | `ClimatologyConfig`, `{datasets: {name: ClimatologyConfig}}` or null | null | Required by `acc`. |
| `output` | `{path: str}` or null | null | Result file path; required by `anemoi-evaluation run`. `{shard}` and `{shards}` are filled in; `{dataset}` is required by, and only allowed for, a run that scores multiple datasets. |
| `on_missing_target` | `"skip"` or `"raise"` | `"skip"` | What to do when the target for a valid time is absent. `skip` logs a warning and drops that frame. |
| `include_lead_zero` | bool | `false` | Prepend a lead-0 frame from the initial state. Raises at construction when the forecast source has `supports_lead_zero = False`. A `MissingTargetError` from `initial_frame` for one init time is caught and logged as `no lead-0 frame for init <date>`; that init time is scored without its lead-0 frame. |
| `log_level` | str | `"INFO"` | Level passed to `logging.basicConfig` by the CLI. |

There is no top-level `device` key. The device is resolved as: the explicit `device` argument of
`Evaluation`, then the forecast source's device (the inference block's `device`), then `cuda` when it is
available, then `cpu`.

### Init times

Exactly one of the two forms.

| Model | Key | Type | Default | Semantics |
|---|---|---|---|---|
| `InitTimesRange` | `start` | datetime | required | Parsed with `anemoi.utils.dates.as_datetime`. |
| | `end` | datetime | required | Inclusive. Must not be before `start`. |
| | `frequency` | duration | required | Must be positive; `6h`, `1d`, ... |
| `InitTimesList` | `dates` | list of datetime | required, at least 1 | Explicit dates, used in the order given. |

### Lead times

Lead times are not configured directly. The forecast source turns `lead_time` into the list it yields:
both included sources return `k * timestep` for `k = 1 .. lead_time / timestep`, and raise unless
`lead_time` is a positive multiple of the timestep. With `include_lead_zero: true` a lead of `0` is
prepended. A multi-step-output model computes `ceil(lead_time / output_horizon)` calls, where
`output_horizon = timestep * multi_step_output`; when `lead_time` is not a multiple of the horizon the
last call computes past `lead_time` and the extra outputs are computed and discarded, with a warning.

### Forecast sources

`forecast:` takes exactly one key.

#### `anemoi_inference` (`AnemoiInferenceConfig`, `extra="allow"`)

| Key | Type | Default | Semantics |
|---|---|---|---|
| `members` | int >= 1 | 1 | Lockstep rollouts per init time. |
| `seed` | int | 0 | Base seed, see [seeding](#seeding-and-reproducibility). |
| `quiet` | bool | true | Raise the `anemoi.inference` and `anemoi.utils.timer` loggers to WARNING while the runner runs. |
| `forcings_cache_bytes` | int >= 0 | `2**30` (1 GiB) | Host LRU budget for computed forcings arrays. `0` disables the cache. |

Every other key goes to `anemoi.inference.config.run.RunConfiguration` (`checkpoint`, `device`, `input`,
`env`, ...). The block is loaded as `{"verbosity": 0, **block, "output": "none"}`: `verbosity: 0` is only
a default and the block can override it, while `output` is forced. Constraints: `date` must not be set
(init times come from the evaluation), `output` must be absent or `none`, and the checkpoint's datasets
must agree on the timing and all be decoded (see Multi-dataset checkpoints).

#### `persistence` (`PersistenceConfig`)

| Key | Type | Default | Semantics |
|---|---|---|---|
| `timestep` | duration | required | Step between the lead times it yields. Must be positive. |
| `members` | int >= 1 | 1 | Identical copies of the persisted field. |

Persistence needs an explicit `targets:` block; it persists the target at init time at every lead. An
init time whose target is missing produces no frames (warning), and `check_init_times` warns about such
init times up front.

### Target source

`targets:` takes exactly one key, `anemoi_dataset` (`AnemoiDatasetConfig`, `extra="allow"`).

| Key | Type | Default | Semantics |
|---|---|---|---|
| `prefetch` | int >= 0 or null | null | Rows read ahead on the worker thread. Null means 2, raised to the forecast's frames per model call by `prefetch_hint`. `0` disables the worker: rows are read synchronously in the calling thread. |
| `cache_bytes` | int >= 0 | 0 | Host LRU of decoded, variable-selected rows. Strings such as `2GiB` are parsed with `anemoi.utils.humanize.human_to_bytes`. `0` disables the cache; a row larger than the budget is never stored. |

Every other key goes to `anemoi.datasets.open_dataset`. With no other key (or no `targets:` block at
all), the targets are opened from the arguments of the forecast source's prognostics input, so forecast
and targets read the same zarr. The dataset must have exactly one member (`shape[2] == 1`).

### Multi-dataset checkpoints

A checkpoint trained on multiple datasets is scored as one evaluation per dataset, sharing one runner and
one rollout. `targets`, `variables`, `weights`, `regions` and `climatology` then take either their usual
single form, applied to every dataset, or a `datasets:` mapping keyed by the checkpoint's dataset names:

```yaml
weights: {uniform: {}}                      # one block, every dataset
regions:
  datasets:
    era5: {global: all}
    cerra: {alps: {bbox: {north: 48, west: 5, south: 45, east: 11}}}
```

The mapping must name every dataset the run scores and no other, which is anemoi-inference's rule for
its own per-dataset config entries; an incomplete or unknown set raises at build time. `from_checkpoint:`
resolves per dataset: the runner's `input` block becomes one entry per dataset name (anemoi-inference's
form) and the `targets` block becomes the `datasets:` form.

The top-level `datasets:` key scores a subset; the default is every dataset of the checkpoint. The model
still predicts every dataset (the runner runs every decoder and needs an input state for each one), so the
inputs, the forcings, the rollout and the init-time checks of FR-7 are unchanged for the skipped ones: they
only lose their targets, weights, regions, climatology, aggregator and result file. The per-dataset
`datasets:` mappings must then name exactly the selected datasets. When exactly one dataset is selected the
run is an ordinary single-dataset one: `Evaluation.from_config` returns an `Evaluation`, `output.path` must
not contain `{dataset}`, and the result has the single-dataset layout plus the `dataset` attr, which names
the dataset scored. `describe()` and `plan()` of such a run report `datasets_scored` and
`datasets_predicted_only`, and the plan's `predicted_only_state_bytes` is the forecast state the runner
holds for the datasets that are not scored.

`Evaluation.from_config` then returns a `MultiEvaluation`, whose `run()` returns one `AggregationState`
per dataset and whose `metrics` and `variables` are mappings keyed by dataset. Each dataset is written to
its own file through the `{dataset}` placeholder of `output.path`, with the same layout as any other
result plus a `dataset` attribute. Merging is per dataset: `merge` refuses inputs whose `dataset` attrs
differ, so a sharded multi-dataset run is merged once per dataset.

Refused at construction: a checkpoint whose datasets disagree on the timestep, the input and output steps
or the rollout shift, and a downscaling checkpoint whose model does not decode every dataset
(anemoi-inference cannot run one either).

The default weights (`{graph_attribute: area_weight}`) are read from each dataset's own node set of the
model graph. When `graph_data` is a mapping it must be keyed by the dataset names — a key that names no
dataset of a multi-dataset checkpoint raises rather than serving another dataset's graph. The node set is
the one asked for; only the default name `data` falls back to the dataset's name, and only when the graph
has no `data` node set. A node set the graph does not have raises with the ones it does.

### Weights

`weights:` is a single-key mapping.

| Spec | Field type | Result |
|---|---|---|
| `graph_attribute: <name>` | str | Per-node attribute of the forecast model's graph (`area_weight`, ...), shape-checked against the grid. |
| `spherical_voronoi: {}` | mapping (null accepted) | Spherical Voronoi cell areas from `anemoi.graphs.nodes.attributes.SphericalAreaWeights`. |
| `uniform: {}` | mapping (null accepted) | 1.0 for every node. |
| `file: <path>` | str | `(N,)` array from a `.npy` file. |

The result is cast to float64 and must have one value per node. Weights are resolved lazily, on first
use of `Evaluation.weights`, so `--dry-run` never loads a model graph.

`graph_attribute` has two distinct failure modes, with different messages: `weights.py` raises
`a forecast source with a model graph is needed for the node attribute '<name>'` when there is no forecast
source at all, and `ForecastSourceBase.graph_node_attribute` raises
`<Class> has no model graph to read the node attribute '<name>' from` when the source has no graph
(`has_graph = False`). `InferenceForecastSource` raises a third message naming the available attributes
when the graph has no such attribute.

### Regions

`regions:` maps a name to a spec. The masks may overlap; every region is scored independently. An empty
mapping falls back to `{global: all}`.

| Spec | Field type | Result |
|---|---|---|
| `all` | the literal string | Every node. |
| `bbox: {north, west, south, east}` | four floats | `anemoi.transform.spatial.cropping_mask`; longitude wrap is handled there. |
| `graph_attribute: <name>` | str | A model-graph node attribute cast to bool (`cutout_mask`, ...). |
| `grid: <i>` | int | Nodes of sub-grid `i` of a cutout or join target dataset, from the per-source counts in `ds.grids`, reduced by `grid_indices`. Raises without a target source that has sub-grids. |
| `file: <path>` | str | `(N,)` array from a `.npy` file, cast to bool. |

### Time bins (`BinsConfig`)

| Key | Type | Default | Values |
|---|---|---|---|
| `time` | enum | `season` | `season`, `month`, `init_time`, `none`. |
| `by` | enum | `init_time` | `init_time`, `valid_time`. |

Bin coordinates per `time` value:

| Value | Bin coordinates |
|---|---|
| `season` | `DJF`, `MAM`, `JJA`, `SON` (DJF is months 12, 1, 2). |
| `month` | `01` .. `12`. |
| `init_time` | The sorted distinct dates, ISO formatted: the init times when `by: init_time`, every `init + lead` when `by: valid_time`. A frame whose date is not one of them raises. |
| `none` | A single bin `all`. |

`by` selects which attribute of the frame is binned. A pooled `all` bin is added to the results file on
top of the stored bins (see [results file](#aggregation-and-the-results-file)).

### Climatology

`climatology: {file: <path>}`: a netcdf written by `ArrayClimatology.to_netcdf`. Needed by `acc` and by
any other user of the anomaly statistics; an `acc` metric without it raises at construction.

## Statistics and metrics

The framework accumulates per-element **statistics** and derives **metrics** from their weighted means.
A statistic maps an `(M, V, N)` float32 prediction and a `(1, V, N)` target to a `(V, N)` float64 array,
where `M` is the ensemble size, `V` the variables and `N` the nodes.

### Element masking and accumulated sums

For each frame, a node is *valid* for a variable when every member and the target are finite there:
`valid[v, n] = isfinite(pred[:, v, n]).all() and isfinite(target[0, v, n])`. Invalid elements contribute
0 to every statistic sum **and** 0 to the weight sum, so they leave the means untouched. The mask is
shared by all statistics of a frame, so every metric of a run covers exactly the same elements.

With node weights `w` (float64, shape `(N,)`) and region masks `mask` (bool, `(N, R)`), write
`W = w[:, None] * mask`. For every frame, for statistic `s`:

```
sums[s][lead, bin, v, r] += (valid[v] * s[v]) @ W[:, r]
weights[lead, bin, v, r]  += valid[v].astype(float64) @ W[:, r]
n_init[lead, bin]         += 1
```

`means[s] = sums[s] / weights`, which is NaN where nothing was accumulated. Summation is float64
throughout; the reduction over members is also float64 (the ensemble mean accumulates in float64 and the
ensemble statistics work one variable at a time), so a large offset with a small spread does not cancel.

### Statistics

Per variable and node, with `m` a member, `y` the target, `e` the ensemble mean and `M` the ensemble size.

| Name | Value | Min members | `aux` |
|---|---|---|---|
| `error` | `e - y` | 1 | n/a |
| `squared_error` | `(e - y)^2` | 1 | n/a |
| `absolute_error` | `abs(e - y)` | 1 | n/a |
| `ensemble_variance` | `var_m(m)`, unbiased | 2 | n/a |
| `skill` | `mean_m abs(m - y)` | 1 | n/a |
| `pairs` | `sum_{i<j} abs(m_i - m_j)` | 2 | n/a |
| `member_squared_error` | `mean_m (m - y)^2` | 1 | n/a |
| `anomaly_product` | `fa * ta` | 1 | `climatology` |
| `forecast_anomaly_squared` | `fa^2` | 1 | `climatology` |
| `target_anomaly_squared` | `ta^2` | 1 | `climatology` |
| `hit_<label>` | `f * o` | 1 | n/a |
| `miss_<label>` | `(1 - f) * o` | 1 | n/a |
| `false_alarm_<label>` | `f * (1 - o)` | 1 | n/a |
| `brier_<label>` | `(p - o)^2` | 1 | n/a |
| `event_frequency_<label>` | `o` | 1 | n/a |
| `rank_bin_<k>` | `1{below <= k <= below + ties} / (ties + 1)` | 2 | n/a |
| `reliability_count_<label>_<k>` | `1{c == k}`, `c` the member exceedance count | 1 | n/a |
| `reliability_event_<label>_<k>` | `1{c == k} * o` | 1 | n/a |

`ensemble_variance` uses `ddof = 1`. `pairs` is computed from the centred members as
`sum_k (2k - M - 1) * sorted(m - y)_k` for `k = 1..M`: the weights sum to zero, so centring on the target
does not change the value, and the sorted form avoids both the `(M, M)` pairwise tensor and the
cancellation of differences of large numbers. `member_squared_error` is identically
`(e - y)^2 + (M-1)/M * ensemble_variance`, and equals `squared_error` at `M = 1`.

`fa = e - c` and `ta = y - c` are the forecast and target anomalies from the climatology `c` at the
frame's valid time, computed in float64 one variable at a time. The three anomaly statistics return zero
where `c` is non-finite. A statistic that names an `aux` field the aggregator was not given raises.

The aggregator puts the ensemble mean into `aux` once per frame (`ensemble_mean`), and, when a
climatology is present, the anomalies and the finite mask (`forecast_anomaly`, `target_anomaly`,
`climatology_finite`), so every statistic of a frame reuses them.

`aux` is a fresh mapping per frame, handed by reference to every statistic of that frame, so a statistic
may also **stash a per-frame intermediate in it** under a name of its own for the other statistics to
reuse, as the rank bins do with `rank_below` and `rank_ties`. Such a field must be read through a helper
that recomputes it when it is absent (`statistics.rank_counts`); and it must not be declared in
`Statistic.aux`, which names what the aggregator must be given.

The same rule holds for a statistic called from Python: `compute(pred, target, aux)` gets whatever the
caller passes, or an empty mapping when it passes nothing, and the mapping it gets **belongs to that one
frame**. It may gain keys, so reusing it for a second frame returns the first frame's stashed values with
no error. Pass a fresh mapping per frame, as the aggregator does.

#### Thresholds

The five threshold statistics of the categorical family score one binarised event (the reliability levels
have their own section below). With `t` the threshold of the variable, `y` the
target, `e` the ensemble mean and `x_m` the members:

- `o = 1{y > t}` is the observed event, `f = 1{e > t}` the binarised deterministic-equivalent forecast and
  `p = #{m : x_m > t} / M` the ensemble exceedance fraction. At `M = 1`, `p == f`.
- The comparison is **strict `>`**, so a node exactly at the threshold is a non-event and a `tp` threshold of
  `0.0` means "any precipitation". Both sides are compared in float64: the threshold is a float64 value from
  the config, and the field is cast one variable at a time.
- Thresholds are given per variable, with a label naming the set:
  `{csi: {label: heavy, thresholds: {tp: 0.005}}}`. Forecast and target are in the same units on the same
  grid, so one number binarises both.
- A variable the map does not name is **NaN** in `o`, `f` and `p`, hence NaN in the statistic, in its sum, in
  its mean and in every metric derived from it, at every lead time, bin and region. A file therefore reads as
  "not defined for this variable" rather than as a count of zero.
- A threshold naming a variable the run does not have is an error at `Evaluation` construction
  (`the thresholds of ... name variables the run does not have`).
- `correct_negative = (1 - f) * (1 - o)` is not stored: the four cells sum to 1 elementwise, and the shared
  validity mask drives both the statistic sums and the weight sum, so `cn = 1 - h - m - fa` holds on the
  weighted means too, up to the float64 rounding of the division.
- At `M = 1`, `brier = (f - o)^2`, which is 1 exactly on a miss or a false alarm and 0 otherwise, so
  `brier == miss + false_alarm` elementwise, exactly. The stored sums group those terms differently
  (`(m + fa) @ W` against `m @ W + fa @ W`), so on the sums and the means the two agree up to the float64
  rounding of the summation, not bit for bit. That is what puts a deterministic checkpoint and its ensemble
  sibling on one axis.

#### The rank histogram

The rank histogram records where the target falls among the sorted members. For one element, with `below`
the number of members strictly under the target and `ties` the number exactly equal to it:

- the rank is `below` when nothing is tied, and the `M + 1` statistics `rank_bin_0 ... rank_bin_M` are the
  indicators of the rank being `k`;
- with ties, the rank under uniform random tie-breaking is uniform over `below .. below + ties`, and the
  stored value is that expectation, `1{below <= k <= below + ties} / (ties + 1)`. The `ties + 1` bins the
  target could occupy each receive `1 / (ties + 1)` and every other bin receives 0. This is deterministic:
  no seed is involved, and the rule adds no device dependence of its own: a re-run and a differently
  sharded run give the same sums, while the members themselves still depend on the GPU model.
- the `M + 1` values sum to 1 per element, exactly when `ties + 1` is a power of two and to a few ulp
  otherwise, hence to float64 rounding on the weighted means. Bin `0` means the target sat below every
  member, bin `M` that it sat above every one.

The comparisons are exact in the frame's own dtype, since widening float32 to float64 is exact and order
preserving, so nothing is cast before them; only the counts accumulate in float64. Unlike the threshold
statistics, where the threshold itself is a float64 config value, the rank bins need no widening at all.
The two counts are the same for all `M + 1` bins of a frame, so they are computed once and shared through
`aux`: the cost is two comparison passes over the members per variable per frame at any `M`.

A node with a non-finite member or target lands in some finite bin (a NaN member in bin 0, a `-inf` member
counted as below), which the aggregator's shared validity mask replaces with 0 before the weighted sum, so
no NaN ever enters a rank sum.

`M` comes from the **run**, not from the spec: `rank_histogram` and `outlier_fraction` carry no
parameters, and a metric that needs the ensemble size learns it through `bind_members` (see
[metrics](#metrics)). At lead 0 every member equals every other one, so the target is either tied with all of
them (the mass spread evenly, a flat histogram, the case when the targets come from the model's input
dataset) or outside all of them (the whole mass in bin `0` or bin `M`); either way it says nothing about
calibration.

#### The reliability levels

An `M`-member forecast probability is the exceedance fraction `p = c / M`, where `c` is the integer number
of members above the threshold, so it takes only the `M + 1` values `k / M`. Those levels **are** the bins
of the reliability diagram: nothing is configured, there is no binning convention to document, and two runs
at different `M` cannot put the same physical forecast in different bins.

Two sums are stored per level and label, `reliability_count_<label>_<k>` and
`reliability_event_<label>_<k>`, whose weighted sums over a cell are the weight `n_k` that landed in level
`k` and the weight `h_k` of the observed events within it. A third sum, the mean forecast probability of
the bin, is redundant here: within a level `p` is the constant `k / M`, so it is `(k / M) n_k` exactly.

The test is `c == k` on the **integer count**, never `p == k / M` on the fraction. The count is a float64
whole number far below `2^53`, so the equality is exact by construction at any ensemble size. All `2 (M + 1)`
statistics of a label need the same count and the same observed event, so they compute them once per frame
and share them through `aux` under `exceedance_count_<label>` and `exceedance_event_<label>`
(`statistics.exceedance_counts`). The label alone is the key because one label carries one threshold map
across a whole config, which `metrics.build` enforces.

From the level sums, with `W = sum_k n_k`, `obar = (sum_k h_k) / W` and `o_k = h_k / n_k`:

```
REL = (1 / W) sum_k n_k (p_k - o_k)^2     the reliability component, small is good
RES = (1 / W) sum_k n_k (o_k - obar)^2    the resolution component, large is good
UNC = obar (1 - obar)                     the uncertainty, a property of the observations alone
```

`BS = REL - RES + UNC` is an algebraic identity of those sums, not an approximation: the derivation needs
only `o` binary and `p` constant within a level, and both hold exactly here. It is an identity in exact
arithmetic, so on the stored float64 sums it holds to the rounding of the summation, never bit for bit; the
suite asserts it to `rtol=1e-12`. `REL` and `RES` take `obar` from the level sums themselves, so each metric is
self-contained; `brier_uncertainty` takes it from `event_frequency_<label>`, one sum it shares with `bss`,
and the two agree to the same rounding, asserted to `rtol=1e-12`. The same grouping argument applies to `sum_k n_k`, which equals `state_weights`
to float64 rounding.

A level no forecast ever fell into has `n_k = 0`, so `o_k` is `0/0` NaN and `forecast_frequency` is 0; the
NaN does not leak into `REL` or `RES`, whose term is replaced by zero where the weight is zero. The test is
on the weight, which is NaN rather than zero for a variable the threshold map does not name, so that
variable stays NaN in every metric of the family.

Because `p_k = k / M` is known analytically, **a coarser diagram is a sum of levels** and needs no new
statistics: for a group `B` of levels, `n_B = sum_{k in B} n_k`, `h_B = sum_{k in B} h_k`, the observed
frequency is `h_B / n_B` and the mean forecast probability `sum_{k in B} n_k (k / M) / n_B`, all exact from
the file. The reverse is impossible, which is why the levels are what is stored. Note that a `REL` computed
on coarse bins is not the `REL` of the levels: it is smaller, because the spread of `p` within a bin is
absorbed into the bin mean. The stored scalars are the level-exact ones.

### Metrics

Metrics are functions of the **weighted means** of their statistics, evaluated after all summation. They
are therefore pooled over nodes and init times of a bin, not averages of per-init scores.

| Name | Spec | Statistics | Metric from the means | Min members |
|---|---|---|---|---|
| `bias` | `bias` | `error` | `mean(error)` | 1 |
| `mae` | `mae` | `absolute_error` | `mean(absolute_error)` | 1 |
| `rmse` | `rmse` | `squared_error` | `sqrt(mean(squared_error))` | 1 |
| `crps` | `crps` or `{crps: {alpha: a}}` | `skill`, `pairs` | `mean(skill) - c(alpha, M) * mean(pairs)` | 2 |
| `fair_crps` | `fair_crps` | `skill`, `pairs` | the same with `alpha = 1` | 2 |
| `spread` | `spread` | `ensemble_variance` | `sqrt(mean(ensemble_variance))` | 2 |
| `spread_skill` | `spread_skill` | `ensemble_variance`, `squared_error` | `sqrt(mean(ensemble_variance) / mean(squared_error))` | 2 |
| `member_rmse` | `member_rmse` | `member_squared_error` | `sqrt(mean(member_squared_error))` | 1 |
| `member_mae` | `member_mae` | `skill` | `mean(skill)` | 1 |
| `acc` | `acc` | the three anomaly statistics | `mean(ap) / sqrt(mean(fas) * mean(tas))` | 1 |
| `pod_<label>` | `{pod: {label: L, thresholds: {...}}}` | `hit`, `miss` | `h / (h + m)` | 1 |
| `far_<label>` | `{far: {...}}` | `hit`, `false_alarm` | `fa / (h + fa)` | 1 |
| `csi_<label>` | `{csi: {...}}` | `hit`, `miss`, `false_alarm` | `h / (h + m + fa)` | 1 |
| `ets_<label>` | `{ets: {...}}` | `hit`, `miss`, `false_alarm` | `(h - hr) / (h + m + fa - hr)` | 1 |
| `frequency_bias_<label>` | `{frequency_bias: {...}}` | `hit`, `miss`, `false_alarm` | `(h + fa) / (h + m)` | 1 |
| `hss_<label>` | `{hss: {...}}` | `hit`, `miss`, `false_alarm` | `2 (h cn - m fa) / ((h + m)(m + cn) + (h + fa)(fa + cn))` | 1 |
| `pss_<label>` | `{pss: {...}}` | `hit`, `miss`, `false_alarm` | `h / (h + m) - fa / (fa + cn)` | 1 |
| `brier_<label>` | `{brier: {...}}` | `brier` | `b` | 1 |
| `bss_<label>` | `{bss: {...}}` | `brier`, `event_frequency` | `1 - b / (obar (1 - obar))` | 1 |
| `event_frequency_<label>` | `{event_frequency: {...}}` | `event_frequency` | `obar` | 1 |
| `rank_histogram` | `rank_histogram` | `rank_bin_0..rank_bin_M` | the `M + 1` means, stacked on a `rank` axis | 2 |
| `outlier_fraction` | `outlier_fraction` | `rank_bin_0`, `rank_bin_M` | `mean(rank_bin_0) + mean(rank_bin_M)` | 2 |
| `reliability_<label>` | `{reliability: {label: L, thresholds: {...}}}` | `reliability_count_L_0..M`, `reliability_event_L_0..M` | `o_k = h_k / n_k`, stacked on a `probability` axis | 1 |
| `forecast_frequency_<label>` | `{forecast_frequency: {...}}` | `reliability_count_L_0..M` | `n_k / sum_j n_j`, on the same axis | 1 |
| `brier_reliability_<label>` | `{brier_reliability: {...}}` | the `2 (M + 1)` level sums | `REL` | 1 |
| `brier_resolution_<label>` | `{brier_resolution: {...}}` | the `2 (M + 1)` level sums | `RES` | 1 |
| `brier_uncertainty_<label>` | `{brier_uncertainty: {...}}` | `event_frequency` | `obar (1 - obar)` | 1 |

`c(alpha, M) = alpha / (M * (M - 1)) + (1 - alpha) / M^2`, with `M` the ensemble size of the run.
`alpha = 0` is the standard kernel CRPS, `alpha = 1` the fair CRPS; `alpha` must be in `[0, 1]`.
A `crps` with an `alpha` other than 0 or 1 is named `crps_<alpha>` with the decimal point written `p`
(`{crps: {alpha: 0.5}}` becomes `crps_0p5`) and serialises back as `{crps: {alpha: 0.5}}`.

`acc` is the anomaly correlation of the **ensemble mean**, pooled over the nodes and init times of a
bin. Per-init or per-season values come from the `init_time` or `season` binning, not from averaging
per-init correlations. Nodes where the climatology is non-finite contribute zero to all three anomaly
sums while the shared weight sum still counts them, so only the ratio is meaningful: the raw means of
the three anomaly statistics are not.

Ensemble conventions: `bias`, `mae`, `rmse`, `spread_skill` and `acc` score the **ensemble mean**;
`member_rmse` and `member_mae` score a randomly drawn member in expectation; `member_rmse` is never
below `rmse`. A deterministic model is `M = 1`, where `member_squared_error` equals `squared_error` and
the metrics needing 2 members are rejected at construction (`the metrics need at least N members`).

For an ensemble, the seven contingency scores (`pod`, `far`, `csi`, `ets`, `frequency_bias`, `hss`, `pss`)
and the cells they come from binarise the **ensemble mean** and therefore score it as a point forecast.
That is not the same event as the deterministic sibling's: the mean of a skewed variable has less spread
than a member, so at a rare threshold it exceeds far less often and `frequency_bias` and `pod` fall well
below a single member's. `brier` and `bss`, and the reliability diagram at one member, are what compares
the two model kinds directly, since they use the exceedance fraction. The contingency table at a chosen probability level is derivable from the
reliability level sums of the same label (`n_k`, `h_k`), and is a natural follow-up rather than something
the file lacks.

The ten threshold metrics are functions of the weighted means of the threshold statistics of the same label:
`h = mean(hit_L)`, `m = mean(miss_L)`, `fa = mean(false_alarm_L)`, `cn = 1 - h - m - fa`, `b = mean(brier_L)`
and `obar = mean(event_frequency_L)`, each pooled over the nodes and init times of a cell.
`hr = (h + fa) * (h + m)` is the random-hit rate, which needs no division because the four cells are
normalised to 1 (see [thresholds](#thresholds)). All ten need only **one member**, so they work for a
deterministic checkpoint.

Their names are `<kind>_<label>`, for example `csi_heavy`, and the sums they store are
`state_sum_hit_heavy` and friends. Metrics of the same label share their statistics, so `csi_heavy` and
`pod_heavy` together store three sums, not five. Two metrics that resolve to the same name are refused
(`duplicate metric names`), as are two metrics whose statistics share a name with different thresholds
(`both need a statistic named ... with different parameters`). A label names one event, so every metric
carrying that label must give the same threshold map; a config in which two of them disagree is refused at
config load, and at construction from Python, before any data is read (`one label, one threshold map`), and
that holds across the families, `csi_warm` against `reliability_warm` as much as `csi_warm` against `pod_warm`.
`brier_<label>` and `event_frequency_<label>`
are each both a metric and a statistic: the metric is the plain data variable, the sum is written as
`state_sum_brier_<label>` and `state_sum_event_frequency_<label>`.

`bss` is a skill score against **the base rate of its own cell**, so it is not comparable across cells: a
different lead time, region or bin is a different reference. Its denominator vanishes where `obar` is 0 or 1,
giving NaN when `b` is 0 and minus infinity otherwise. `pod`, `far`, `csi`, `ets` and `frequency_bias` are
0/0 NaN in a cell where nothing was observed and nothing was forecast, which is the honest reading of a
threshold too rare for the sample; `event_frequency_<label>` in the same file is how a reader sees why.
`pss` is close to `pod` for rare events, since `fa / (fa + cn)` is tiny when the correct negatives dominate.

`rank_histogram` is not one number per `(lead_time, bin, variable, region)`: it carries an extra `rank`
axis of `M + 1` values, see the [NetCDF layout](#netcdf-layout). `reliability_<label>` and
`forecast_frequency_<label>` are the other metrics carrying an extra axis, the `probability` one; the three
`brier_*` components are ordinary scalars. The five need only one member, unlike the rank histogram: at
`M = 1` the diagram has the two points of a contingency table. `outlier_fraction`
is the fraction of targets that fell outside the ensemble range, `2 / (M + 1)` for a calibrated ensemble,
and stores only the two end bins when it is asked for alone.

Missing targets never enter the sums: with `on_missing_target: skip` the frame is dropped entirely, so
its `n_init` is not incremented either. With `raise` the `MissingTargetError` propagates.

## Aggregation and the results file

### `AggregationState`

The state holds float64 sums over the four axes `(lead_time, bin, variable, region)`:

| Attribute | Type | Meaning |
|---|---|---|
| `lead_times` | list of timedelta | First axis. |
| `bins` | list of str | Second axis, the binning's coordinates. |
| `variables` | list of str | Third axis. |
| `regions` | list of str | Fourth axis. |
| `sums` | dict name to float64 tensor `(L, B, V, R)` | One per statistic, always on these four axes; a metric with an extra output axis is built from several statistics at output time. |
| `weights` | float64 tensor `(L, B, V, R)` | Weight sum of the valid elements. |
| `n_init` | int64 tensor `(L, B)` | Frames added. |
| `members` | int | Ensemble size of the frames. |
| `variable_coords` | dict name to list | Per-variable labels (`param`, `level`). |
| `init_times` | list of datetime | Init times whose frames were added. |
| `attrs` | dict | Result-file attributes. |

Shapes and dtypes are validated in `__post_init__`.

### NetCDF layout

`to_xarray(state, metrics, attrs=None)` builds the dataset that `write()` stores with the `netcdf4`
engine.

Dimensions and coordinates:

| Coordinate | Dimension | Dtype | Values |
|---|---|---|---|
| `lead_time` | `lead_time` | `timedelta64[ns]` | The scored lead times, including 0 when `include_lead_zero`. |
| `bin` | `bin` | str | The stored bins plus a derived `all`. When the binning is `none` the only bin is already `all` and nothing is appended. |
| `state_bin` | `state_bin` | str | The stored bins only. |
| `variable` | `variable` | str | Variable names. |
| `region` | `region` | str | Region names. |
| `init_time` | `init_time` | `datetime64[ns]` | Init times that contributed. |
| `param` | `variable` | str | Parameter of each variable, `""` when unknown. |
| `level` | `variable` | float | Level of each variable, NaN when unknown. |
| `rank` | `rank` | int64 | `0..M`, present only when a metric declares it (`rank_histogram`). |
| `probability` | `probability` | float64 | `k / M` for `k` in `0..M`, present only when a metric declares it (`reliability`, `forecast_frequency`). |

Data variables:

| Variable | Dimensions | Content |
|---|---|---|
| `<metric>` | `(lead_time, bin, variable, region)` plus any extra dimension the metric declares | One per configured metric, named by `Metric.name`, derived from the means including the pooled `all` bin. `rank_histogram` is on `(lead_time, bin, variable, region, rank)`, `reliability_<label>` and `forecast_frequency_<label>` on `(lead_time, bin, variable, region, probability)`. |
| `n_init` | `(lead_time, bin)` | Frames, with the `all` bin summed. |
| `weight_sum` | `(lead_time, bin, variable, region)` | Weight sum of the valid elements, with the `all` bin summed. |
| `state_sum_<statistic>` | `(lead_time, state_bin, variable, region)` | The raw float64 sums, one per statistic. |
| `state_weights` | `(lead_time, state_bin, variable, region)` | The raw weight sums. |
| `state_n_init` | `(lead_time, state_bin)` | The raw frame counts. |

The `all` bin is the elementwise sum over the stored bins, so its metrics are computed from the pooled
sums, never by averaging per-bin metrics. `state_*` variables are the exact state and are what
`load_state` reads back; `merge` only ever touches them.

Attributes. Identity of the result:

| Attr | Content |
|---|---|
| `members` | Ensemble size, as an int. |
| `dataset` | The dataset of the checkpoint this result scores; written whenever the checkpoint has multiple datasets, including when the run scored only one of them. Absent for a checkpoint with a single dataset. |
| `metrics` | JSON list of the metric specs, used by `merge` to rebuild them. |
| `bin_kind`, `bin_by` | The binning rule. |
| `init_times` | Compressed date list of the init times of this run. |
| `lead_time` | Frequency string. |
| `shard` | `i/n` of the run, or `i,j,k/n` after a partial merge. Dropped by a complete merge. |
| `merged_from` | JSON list of the input paths, written by `merge`. |

Provenance:

| Attr | Content |
|---|---|
| `checkpoint`, `checkpoint_uuid`, `checkpoint_run_id` | Checkpoint identity, from the inference source. |
| `gpu`, `inference_chunks_processor`, `inference_chunks_mapper` | Recorded once the model is loaded; part of a run's numerical identity. |
| `forecast` | `persistence`, from the persistence source. |
| `config` | The run's config as YAML; omitted when the objects cannot be serialised. |
| `package_versions` | JSON of the installed `anemoi-evaluation`, `anemoi-inference`, `anemoi-models`, `anemoi-graphs`, `anemoi-datasets` and `torch` versions (null when absent). |
| `climatology`, `climatology_nonfinite_nodes` | JSON of `climatology.describe()` and of the non-finite node count per variable. |

Timing, in seconds:

| Attr | Content |
|---|---|
| `time_model_s` | Wall time in the forecast source, CUDA-synchronised. |
| `time_target_s` | The main thread's wait for targets. |
| `time_statistics_s` | Wall time in the aggregator. |
| `time_total_s` | Wall time of the init-time loop. |
| `time_target_read_s` | Seconds the target source's reading thread spent. |

In a multi-dataset run the datasets share one rollout, so `time_model_s` is that rollout's whole time and
every dataset's result file carries the same value, not a share of it — the same convention `plan()` uses
for its estimates. `time_target_s` and `time_statistics_s` are the dataset's own, and `time_total_s` is
the sum of the three rather than the wall clock of the init-time loop, which would also be the same number
in every file. The prefetch calls and the lead-0 frames are charged to no phase.

Counters:

| Attr | Content |
|---|---|
| `model_calls` | Model calls of the run. |
| `target_reads`, `target_cache_hits` | Rows read and cache hits of the target source. |
| `target_prefetch` | Effective prefetch depth. |
| `peak_gpu_memory_bytes` | `torch.cuda.max_memory_allocated`, on CUDA only. |
| `forcings_computed`, `forcings_hits`, `forcings_bypassed` | Forcings cache counters of the inference source. |

### Merging

Two states merge by elementwise addition of `sums`, `weights` and `n_init`. No division happens until the
metrics are derived at output time, so a merge of shards adds exactly the same per-frame summands as the
unsharded run, only regrouped: the unsharded run accumulates its init times sequentially, the merge adds
each shard's partial sum in input order. Float64 addition is not associative, so the guarantee is:

- `n_init` is int64 and always identical.
- A cell (`lead_time`, bin, variable, region) whose frames all come from **one** shard is bit-exact: the
  summands reach it in the same relative order either way. This holds for every bin of an `init_time`
  binning by init time, since one bin is one init time, and for other binnings only when the shard
  assignment happens to send a bin's init times to a single shard.
- A cell fed by several shards agrees with the unsharded run to float64 rounding, not necessarily bit for
  bit.
- The derived `all` bin inherits this: `_with_total` sums over the bin axis in the same order in both
  cases, so it is bit-exact exactly when its per-bin sums are.

`benchmarks.md` section 6 observed bit-exact results for a 4-shard, 8-init-time run against the
unsharded control; that is a measurement of one configuration, not a guarantee.

`merge` refuses:

- different coordinates (`lead_times`, `bins`, `variables`, `regions`) or different statistic sets;
- different `members`;
- overlapping `init_times` (the same init time counted twice);
- inputs whose `dataset` attrs are not all the same (a multi-dataset run writes one file per dataset per
  shard, and each dataset merges on its own); results of a single-dataset checkpoint carry no `dataset` attr;
- a set of `shard` attrs that is neither all absent nor exactly the `0..n-1` of one run: a missing shard,
  a duplicate, disagreeing shard counts, a mix of tagged and untagged inputs, or a malformed tag. Inputs
  that all carry no tag (states built from the Python API, already-merged files) skip the check.
  `--partial` / `partial=True` allows a missing shard and records the ones present as `i,j,k/n`, which a
  later merge can complete.

Attributes of the merged state come from the first input, except: the timing attrs and the counters are
summed, `peak_gpu_memory_bytes` is the maximum, `init_times` is the compressed union, `config` becomes
`{"merge": [...]}` of every input's config, `merged_from` lists the input paths, and `shard` is dropped
when the set is complete. Inputs carrying different `config` attrs produce a warning, not an error.

## Python API

`anemoi.evaluation.__all__` holds 26 names and is the public surface. Five of them are modules
(`metrics`, `statistics`, `binning`, `weights`, `regions`), whose contents are public too. Every name not
listed in this section is internal and may change without notice.

### Public surface

#### `Evaluation`

```python
Evaluation(forecast, targets, init_times, lead_time, metrics, variables=None, weights=None,
           regions=None, bins=None, device=None, on_missing_target="skip",
           include_lead_zero=False, climatology=None)
```

The object behind the config. Construction validates the arguments, checks the grids for compatibility,
resolves the variables and calls `forecast.check_init_times(init_times, lead_time)`; it loads no model.
Weights and regions given as specs are resolved on first use.

| Method | Returns |
|---|---|
| `run(init_times=None)` | `AggregationState` with attrs and timings; all init times by default. |
| `pairs(init_times=None)` | `Iterator[tuple[Frame, torch.Tensor]]` for a custom loop, without aggregation; runs in inference mode and restores the caller's mode. |
| `plan(init_times=None, step_time=None, shards=1)` | `dict` describing a run, loading no model. |
| `time_estimate(step_time, n_init, shards=1)` | `dict` of model and wall seconds. |
| `shard(index, count)` | `list[datetime]`, `init_times[index::count]`. |
| `model_calls_per_init()` | `tuple[int, timedelta]`: calls per init time and the lead time they compute. |
| `variable_coords()` | `dict[str, list]`: `param` and `level` per variable. |
| `attrs(init_times=None, timing=None)` | `dict` of result-file attributes. |
| `to_config()` | `dict`, YAML-ready; raises when weights or regions were given as arrays. |
| `close()` | `None`; releases the forecast, target and climatology sources. |
| `Evaluation.from_config(config, forecast=None, targets=None, climatology=None)` | classmethod; a path, a dict or an `EvaluationConfig`, with the source arguments overriding the configured ones. Returns a `MultiEvaluation` when the run scores multiple datasets, and an `Evaluation` when the config's `datasets:` key selects one; `targets` and `climatology` may then be `{name: source}`. |

Cached properties, resolved on first use: `weights -> np.ndarray`, `regions -> dict[str, np.ndarray]`,
`aggregator -> Aggregator`.

`begin()`, `frames(init_time)`, `add_frame(state, frame, timer)`, `charge(...)` and `finish(...)` are the
halves of `run()`, for a driver that feeds the frames itself; `MultiEvaluation` uses them.

| Attribute | Type | Content |
|---|---|---|
| `forecast`, `targets` | source objects | The two sources. |
| `climatology` | source or `None` | The climatology source. |
| `metrics` | `list[Metric]` | The resolved metrics. Load-bearing: `state.to_xarray(evaluation.metrics)`. |
| `variables` | `list[str]` | The resolved variables. |
| `init_times` | `list[datetime]` | All init times of the evaluation. |
| `lead_time` | `timedelta` | The configured maximum. |
| `lead_times` | `list[timedelta]` | The scored lead times, with a leading `0` when `include_lead_zero`. |
| `binning` | `Binning` | The bin rule. |
| `device` | `torch.device` | The resolved device. |
| `on_missing_target`, `include_lead_zero` | str, bool | As configured. |
| `timing`, `model_calls`, `peak_memory_bytes`, `climatology_nonfinite` | dict, int, int or `None`, dict | Set by `run()`. |

#### `MultiEvaluation`

```python
MultiEvaluation(evaluations, forecast=None)
```

One `Evaluation` per dataset of a multi-dataset checkpoint, sharing one runner and one rollout;
`Evaluation.from_config` builds it. `forecast` is the shared `InferenceForecastSource`, or `None` when
each evaluation runs on its own (a per-dataset persistence baseline).

| Method or attribute | Returns |
|---|---|
| `run(init_times=None)` | `dict[str, AggregationState]`, one per dataset, each with a `dataset` attr. |
| `pairs(init_times=None)` | `Iterator[tuple[str, Frame, torch.Tensor]]`. |
| `plan(init_times=None, step_time=None, shards=1)` | `dict` with one plan per dataset under `datasets`, the names scored under `scored` and, when the run scores a subset, the others under `predicted_only`. |
| `shard(index, count)` | `list[datetime]`, the same for every dataset. |
| `attrs(init_times=None)` | `dict[str, dict]`, the result attrs of each dataset. |
| `to_config()`, `close()` | As `Evaluation`; `close()` also releases the shared runner. |
| `evaluations` | `dict[str, Evaluation]`. |
| `dataset_names` | `list[str]`, in the checkpoint's order. |
| `metrics`, `variables` | `dict[str, ...]` keyed by dataset. |
| `init_times`, `lead_time`, `lead_times`, `device` | Shared by every dataset. |

#### `DatasetForecast`

One dataset of a multi-dataset `InferenceForecastSource`, as an ordinary single-dataset forecast source:
`grid`, `variables`, `typed_variables` and `grid_indices` are its own, `frames()` raises (the rollout is
shared, and the parent's `multi_frames()` drives it), and everything else is delegated to the parent.
`InferenceForecastSource.datasets` maps each dataset name to one. A view whose `solo` flag is set is the
only dataset the run scores: it then drives the rollout itself through `frames()` and owns the runner, so
closing it closes the parent.

#### The `plan()` dictionary

Diagnostic output for `--dry-run`. Its keys are not an interface and carry no compatibility promise.

| Key | Content |
|---|---|
| `forecast`, `targets` | The sources' `describe()`. |
| `device` | The device as a string. |
| `init_times` | `{count, dates}`, the dates compressed. |
| `lead_times` | `{count, first, last}` as frequency strings. |
| `model_calls` | Calls per init time times the number of init times. |
| `computed_lead_time` | The lead time those calls actually compute. |
| `include_lead_zero` | The flag. |
| `variables` | `{names, param, level}`. |
| `metrics`, `statistics` | The metric specs, and the sorted names of the statistics they need. |
| `weights` | The weight spec, or `"array"` when given as an array. |
| `regions` | Name to spec, or `"array"`. |
| `bins` | `{time, by, count}`. |
| `on_missing_target` | The setting. |
| `climatology` | The climatology's `describe()`; present only with one. |
| `frames` | `{total, with_targets, distinct_valid_times}`. |
| `bytes` | `{target_to_device, target_rows_read, cache_for_all_rows, frame, aggregator_weights, statistics_temporaries}`, plus `checkpoint_bytes` and `per_member_state_bytes` from an inference source and `climatology` when one is configured. |
| `time` | Present only with `step_time`: `{step_time, model_per_init_s, shard: {model_s, wall_s}, run: {gpu_hours, wall_per_shard_s}, startup_s, note}`. |
| `note` | Present only when the last model call computes past `lead_time`. |
| `shard` | `{index, count, output}`, added by the CLI, not by `plan()`. |

#### Data model

- `Grid(latitudes, longitudes)`, frozen dataclass; coordinates in degrees, flattened to float64 `(N,)`.
  `n`, `latlons_rad() -> (N, 2)` radians, `check_shape(array, what)`,
  `check_compatible(other, tolerance=1e-5)`.
- `Frame(init_time, lead_time, valid_time, variables, data)`, frozen dataclass, `members` property.
  `data` must be float32 `(M, V, N)` with one column per variable, and
  `valid_time == init_time + lead_time`. The lead time is not checked for sign.

#### Aggregation

- `Aggregator(weights, masks, statistics, binning, device=None)`: `n_regions`,
  `new_state(lead_times, variables, regions, members, variable_coords=None) -> AggregationState`,
  `add(state, frame, target, aux=None) -> None`.
- `AggregationState`: the dataclass of the [state table](#aggregationstate), plus
  `AggregationState.zeros(...)` (classmethod), `coords`, `lead_index(lead_time)`, `means()`,
  `merge(other)`, `to(device)`, `cpu()` and `to_xarray(metrics, attrs=None)`.

#### Output

- `to_xarray(state, metrics, attrs=None) -> xr.Dataset`.
- `load_state(path) -> AggregationState`: the raw state back from a result file, attrs kept.
- `merge(items, *, partial=False) -> AggregationState`: states or paths.

#### Metrics, statistics, binning, weights, regions

- `metrics`: the `Metric` base class (`name`, `statistics`, `spec`, `min_members`, `output_dim`,
  `bind_members(members)`, `from_means(means, members)`); `Bias`, `MAE`, `RMSE`, `CRPS(alpha=0.0)`,
  `FairCRPS`, `Spread`, `SpreadSkill`, `MemberRMSE`, `MemberMAE`, `ACC`; the threshold metrics
  `ThresholdMetric(label=None, thresholds=None)` and its subclasses `POD`, `FAR`, `CSI`, `ETS`,
  `FrequencyBias`, `HSS`, `PSS`, `BrierScore`, `BSS`, `EventFrequency` and `BrierUncertainty`; the
  `LabelledMetric` mixin they take their label, name and spec from; `EnsembleSizeMetric` and its
  subclasses `RankHistogram` and `OutlierFraction`, which gain their statistics in `bind_members`;
  `ReliabilityMetric`, which depends on both the variables and the ensemble size, and its subclasses
  `ReliabilityDiagram`, `ForecastFrequency`, `BrierReliability` and `BrierResolution`; `REGISTRY`, the
  name-to-class mapping a new metric registers in; `from_spec(spec)`,
  `build(specs, variables=None, members=None)` (the constructor the driver and the config use: it refuses
  duplicate names and conflicting statistics, binds the metrics to the ensemble size and the statistics to
  the variables), `unique_statistics(metrics)`, `required_aux(metrics)`, `min_members(metrics)`.
- `statistics`: the `Statistic` base class (`name`, `min_members`, `aux`, `parameters`, `__eq__`,
  `bind_variables(variables)`, `compute(pred, target, aux=None)`) and the statistics of the
  [statistics table](#statistics), the threshold ones through
  `ThresholdStatistic(label=None, thresholds=None)` with the subclasses `Hit`, `Miss`, `FalseAlarm`,
  `Brier` and `EventFrequency`, the rank bins through `RankBin(k, members)` and the reliability levels
  through `ReliabilityLevel(label=None, thresholds=None, *, k, members)` with the subclasses
  `ReliabilityCount` and `ReliabilityEvent`;
  `crps_coefficient(alpha, members)`; `ensemble_mean(pred, aux=None)`.
  `per_variable(pred, target, function)`, `anomalies(mean, target, climatology)`,
  `climatology_anomalies(pred, target, aux)`, `exceedance(field, thresholds)`,
  `exceedance_count(pred, thresholds)`, `exceedance_fraction(pred, thresholds)`,
  `exceedance_counts(pred, target, thresholds, label, aux)`, `rank_counts(pred, target, aux)` and
  `validate_thresholds(label, thresholds)` are the helpers for writing a custom `Statistic`.
- `binning`: the `Binning` protocol (`kind`, `by`, `coords`, `index(frame)`); `SeasonBinning`,
  `MonthBinning`, `InitTimeBinning(init_times, lead_times=(), by="init_time")` and `NoBinning`, each
  taking `by="init_time"`; `build(kind="season", by="init_time", init_times=(), lead_times=())`;
  `season_of(date)` and the constants `SEASONS`, `MONTHS`, `KINDS`, `BY`.
- `weights`: `uniform(grid)`, `spherical_voronoi(grid)`, `from_file(path, grid)`,
  `graph_node_attribute(forecast, name, grid)`, each returning `(N,)` float64.
- `regions`: `all(grid)`, `bbox(grid, north, west, south, east)`, `from_file(path, grid)`, each returning
  `(N,)` bool, and `stack(regions, n) -> (names, (N, R) bool)`.

#### Sources

- `InferenceForecastSource(members=1, seed=0, quiet=True, forcings_cache_bytes=2**30, **run_config)`:
  runs an anemoi-inference checkpoint. `has_graph = True`, `supports_lead_zero = True`,
  `frames_per_pass = multi_step_output`. Members are lockstep rollouts of one runner over one shared
  initial state. Beyond the protocol: `dataset_args_kwargs()`, `model_loaded`, and the attributes
  `timestep`, `multi_step_input`, `multi_step_output`, `output_horizon`, `checkpoint`, `checkpoint_uuid`,
  `checkpoint_run_id`, `dataset_name`. `initial_frame` returns the init-time slice of the prognostic
  variables and NaN for the others, identical for every member.
- `PersistenceForecastSource(targets, timestep, members=1)`: the target at init time at every lead;
  `supports_lead_zero = True`.
- `DatasetTargets(*args, prefetch=None, cache_bytes=0, **kwargs)`: targets from
  `anemoi.datasets.open_dataset`, one row per valid time, read ahead on a single worker thread that is
  then the only thread touching the dataset, with an optional host LRU of decoded rows. Classmethods
  `from_dataset(dataset, args=(), kwargs=None, grid_indices=None, prefetch=None, cache_bytes=0)` and
  `from_forecast(forecast, prefetch=None, cache_bytes=0)`. `grid_indices` reduces the grid as the
  checkpoint does.
- `ArrayClimatology(grid, variables, fields, key="hour_of_day", counts=None, attrs=None, path=None)`: one
  `(V, N)` float32 field per key, `key` one of `hour_of_day`, `month`, `day_of_year`, `constant`.
  `key_of(valid_time)`, `nonfinite_nodes(variables)`,
  `from_targets(targets, dates, variables, key="hour_of_day", device="cpu")` (float64 accumulation,
  missing dates skipped), `to_netcdf(path)`, `from_netcdf(path)`. `frame()` returns a cached device
  tensor shared between calls: read it, do not write to it. The netcdf holds
  `climatology(key, variable, values)` with `latitude`/`longitude` on `values` and the attrs `key_kind`
  and `counts`. `to_config()` raises unless the object came from, or was written to, a file.
- `ArrayTargets(grid, variables, fields)` and `FakeForecastSource(targets, timestep, members=1,
  drift=0.0, spread=0.0, frames_per_pass=1, device=None)`: in-memory synthetic sources for tests and
  notebooks.

#### Other

`init_times(start, end, frequency) -> list[datetime]` and `__version__` (`0+unknown` from an
uninstalled source tree).

### Source protocols and base classes

Defined in `anemoi.evaluation.sources.base`. `MissingTargetError` is a `LookupError`.

`ForecastSource`: attributes `grid: Grid`, `variables: list[str]`, `members: int`,
`device: torch.device | str | None`, `grid_indices`, `has_graph: bool`, `frames_per_pass: int`,
`supports_lead_zero: bool`, and:

```python
lead_times(lead_time: timedelta) -> list[timedelta]
initial_frame(init_time: datetime, variables: list[str], device: torch.device) -> Frame
frames(init_time: datetime, lead_time: timedelta, variables: list[str], device: torch.device) -> Iterator[Frame]
check_init_times(init_times: list[datetime], lead_time: timedelta) -> None
graph_node_attribute(name: str, nodes: str = "data") -> np.ndarray
variable_info(name: str) -> dict
provenance() -> dict
stats -> dict            # property
describe() -> dict       # must not load a model
to_config() -> dict
close() -> None
```

`frames` yields strictly increasing lead times and the caller owns each `data`.

`TargetSource`: attributes `grid`, `variables`, and:

```python
frame(valid_time: datetime, variables: list[str], device: torch.device) -> torch.Tensor  # (1, V, N) float32
available(valid_time: datetime) -> bool
prefetch(valid_times: list[datetime], variables: list[str]) -> None
prefetch_hint(frames_per_pass: int) -> None
stats -> dict            # property: reads, read_seconds, cache_hits
close() -> None
grid_mask(i: int) -> np.ndarray
variable_info(name: str) -> dict
to_config() -> dict
describe() -> dict
```

`ClimatologySource`: attributes `grid`, `variables`, and:

```python
frame(valid_time: datetime, variables: list[str], device: torch.device) -> torch.Tensor  # (V, N) float32
nonfinite_nodes(variables: list[str]) -> dict[str, int]
describe() -> dict
to_config() -> dict
close() -> None
```

`ForecastSourceBase`, `TargetSourceBase` and `ClimatologySourceBase` implement the optional capabilities
with defaults: no device preference, no graph, no date check, no sub-grids, no lead-0 frame,
`frames_per_pass = 1`, empty `stats` and `provenance`, unknown `variable_info`, and `to_config()` raising.
Their `describe()` reports the class name, the node count and the variable count, and the forecast base
adds `members`. Helpers: `lead_time_steps(lead_time, timestep)`, `variable_info(variable)`,
`select_variables(requested, available)`.

### Internals (no compatibility promise)

Useful when reading the code or writing a tool against it, but not part of the public surface.

- `config.load_config(source) -> EvaluationConfig` and the pydantic models `EvaluationConfig`,
  `ForecastConfig`, `AnemoiInferenceConfig`, `PersistenceConfig`, `TargetsConfig`, `AnemoiDatasetConfig`,
  `InitTimesRange`, `InitTimesList`, `BinsConfig`, `ClimatologyConfig`, `OutputConfig`, `Bbox`: the
  validated form of the [configuration](#configuration), with the aliases `InitTimesConfig`,
  `WeightsConfig`, `RegionConfig`.
- `config.resolve_base`, `config.resolve_from_checkpoint`, `config.checkpoint_dataset_arguments`: the two
  pre-validation rewrites and the checkpoint lookup behind them.
- `config.build_weights(spec, grid, forecast=None)`, `config.build_region(spec, grid, forecast=None,
  targets=None)`: a spec to an array.
- The spec models `GraphAttributeSpec`, `SphericalVoronoiSpec`, `UniformSpec`, `FileSpec`, `BboxSpec`,
  `GridSpec`: one per weight and region spec.
- `output.write(dataset, path)`: netcdf4 writer used by the CLI, creating parent directories.
- `evaluate.package_versions(packages=PACKAGES)`: the versions recorded in a result.
- `evaluate.PhaseTimer(device=None)`: per-phase wall time with a CUDA synchronisation at each `mark`.
- `evaluate.PHASES`, `evaluate.PACKAGES`: the phase and package name tuples.
- `evaluate.STARTUP_SECONDS`: 105.0, the per-shard process startup added to every `--step-time` estimate.
- `sources.anemoi_inference.member_seed(seed, init_time, member=0, step=0)`: the per-member seed, see
  [seeding](#seeding-and-reproducibility).
- `sources.anemoi_inference.inference_env(run_config, environ=os.environ)`: the `env` block with the
  processor-chunk default added.
- `sources.anemoi_inference.ForcingsCache(cache_bytes)` and `SharedForcings(provider, cache)`: the
  forcings cache and the provider wrapper that reads through it.

## Sharding

Shard `i` of `n` evaluates `init_times[i::n]`: the partition depends on nothing but the resolved
init-time list, so it is deterministic, and it covers every init time exactly once. Each shard writes its
own file; `anemoi-evaluation merge` sums them. Worked examples are in the
[user guide](user-guide.md).

`resolve_shard` determines `(index, count)`:

1. With `--shard i/n`, `i` and `n` are taken literally. A malformed value raises
   `--shard must be of the form i/n`.
2. Otherwise, from the environment:
   `tasks = SLURM_STEP_NUM_TASKS or 1`,
   `array = (SLURM_ARRAY_TASK_ID or 0) - (SLURM_ARRAY_TASK_MIN or 0)`,
   `count = (SLURM_ARRAY_TASK_COUNT or 1) * tasks`,
   `index = array * tasks + (SLURM_PROCID or 0)`.
   Outside Slurm this gives `0/1`; a job array of 3 with 4 tasks each gives 12 shards. `--shard 0/1`
   disables the auto-detection.
3. `0 <= index < count` is enforced in both cases.

`output.path` is formatted with `{shard}` and `{shards}`; with more than one shard a `{shard}`
placeholder is required. Every run tags its state with the `shard` attr, including an unsharded `0/1`
run. For what the merge guarantees numerically, see [merging](#merging).

## Seeding and reproducibility

The inference source reseeds torch's global RNG before every member's model call:

```
call = step_index // frames_per_pass
torch.manual_seed(member_seed(seed, init_time, member, step=call))
member_seed(seed, init_time, member, step)
    = blake2b(f"{seed}|{init_time.isoformat()}|{member}|{step}", digest_size=8) >> 1   # a 63-bit int
```

The fourth argument is named `step` in the signature and is the index of the **model call**, not of the
output frame.

Consequences:

- A member's noise depends only on `(seed, init_time, member, model call)`, not on the ensemble size, on
  the other members or on how the lockstep rollouts are interleaved. Member 0 of a 4-member run is the
  same forecast as member 0 of a 50-member run, and a shard produces the members it would have produced
  unsharded.
- Members of a checkpoint without a noise injector are identical; `members > 1` on such a checkpoint logs
  a warning.
- The arithmetic identity of a run also depends on the GPU model and on the inference chunk counts, which
  is why the result file records `gpu`, `inference_chunks_processor` and `inference_chunks_mapper` along
  with the package versions and the full config.
- The rank histogram needs no seed: ties are spread deterministically rather than broken at random, so a
  re-run and a differently sharded run give the same sums.
- Shared forcings are a pure cache: the cached array is what the provider would have computed for the
  same dates on the same grid, so caching changes no number. A request on a different grid bypasses the
  cache.

## Limitations and undefined behaviour

Constraints the code enforces, with the message or behaviour:

- **Multi-dataset checkpoints must share the timing and decode every dataset.** A checkpoint whose
  datasets disagree on the timestep or the input and output steps, and a downscaling checkpoint whose
  model does not decode every dataset, are refused at construction.
- **The `tools/` scripts take a single-dataset checkpoint.** `validate.py`, `make_climatology.py` and
  `ab_prefetch.py` read `forecast.dataset_name`, the `forecast.anemoi_inference.input.dataset` block and
  a plain `regions` block, all of which are per dataset on a multi-dataset checkpoint; they raise on one.
- **One process per shard.** There is no multi-rank job and no model-parallel runner. Lockstep members
  cost GPU memory linearly.
- **Target datasets must have one member** (`shape[2] == 1`).
- **Grids must match.** `Grid.check_compatible` raises unless forecast, target and climatology grids have
  the same nodes within `1e-5` degrees.
- **`lead_time` must be a positive multiple of the forecast timestep**, otherwise `lead_time_steps`
  raises. A `lead_time` that is not a multiple of a multi-step model's `output_horizon` is allowed, warns,
  and discards the outputs beyond it.
- **Metrics needing members.** Constructing an `Evaluation` whose metrics need more members than the
  forecast source has raises. A metric whose statistics depend on the ensemble size learns it there through
  `bind_members`, called by `metrics.build` from the evaluation and from the merge CLI (where it comes from
  the file's `members` attr); it refuses a run with too few members with its own message
  (`rank_histogram needs at least 2 members, the run has 1`) and refuses to be rebound to a different size
  (`already bound to 4 members, cannot rebind to 8`).
- **The lead-0 rank histogram says nothing about calibration**: every member equals every other one there, so
  the target is either tied with all of them (the mass spread evenly, a flat histogram, the case when the
  targets come from the model's input dataset) or outside all of them (the whole mass in bin `0` or bin `M`).
- **The lead-0 reliability diagram is two points by construction**: every member equals every other one, so
  only the levels 0 and `M` are populated and the interior of the curve is NaN. Like the lead-0 rank
  histogram, that is a property of the frames, not a calibration statement.
- **The level sums grow as `2 (M + 1)` per label.** At `M = 8` and three labels that is 54 netcdf variables
  and a few MB; at `M = 51` it is 312 variables and a result file of about 90 MB. Those figures are computed
  from the state shape and checked once by hand on the fakes, not in the test suite; the per-frame GPU cost of
  that many statistics at 1.7M nodes has not been measured. A large ensemble with many labels is worth a thought. Any
  coarser diagram can be computed from the stored levels afterwards.
- **A label should not begin with a metric kind.** `{brier: {label: reliability_heavy, ...}}` produces the
  name `brier_reliability_heavy`, which collides with `{brier_reliability: {label: heavy, ...}}`; the
  collision is refused loudly (`duplicate metric names`), and the tools split such names longest kind first,
  but the confusion is avoidable.
- **`acc` needs a climatology**; `AnomalyProduct` and the other anomaly statistics raise when
  `climatology` is not in `aux`.
- **A climatology key that a valid time maps to must exist**, otherwise `ArrayClimatology.frame` raises.
  Non-finite climatology nodes are excluded from the anomaly statistics only, are counted in
  `climatology_nonfinite_nodes`, and produce a warning at the start of a run.
- **`include_lead_zero` needs `supports_lead_zero`**, otherwise construction raises. A per-init-time
  `MissingTargetError` from `initial_frame` is caught and logged, and only that lead-0 frame is dropped.
  For an inference checkpoint, diagnostic variables are NaN at lead 0 and therefore excluded there by the
  finiteness mask; `spread_skill` at lead 0 is NaN (0/0) for a perfect initial state.
- **Thresholds.** A threshold naming a variable the run does not have raises at `Evaluation` construction,
  where the variables are known; a run variable the map omits is not an error but reads as NaN in every
  threshold statistic and metric. A label must be non-empty and match `[A-Za-z0-9_]+`, a threshold value must
  be a finite number (NaN and infinity are refused) and an empty `thresholds` mapping is an error; all three
  are checked at config load. A `Metric` instance passed into two `Evaluation` objects with different
  variable orders raises rather than re-binding, since the first evaluation would otherwise score the wrong
  columns.
- **Duplicate metric names** are refused at construction, including `metrics: [rmse, rmse]`, and so are two
  metrics whose statistics share a name but not their parameters, and a label used with two different
  threshold maps (`one label, one threshold map`), which is refused before any binding.
- **Threshold metrics are legitimately NaN** where their cell is degenerate: `pod`, `far`, `csi`, `ets` and
  `frequency_bias` where nothing was observed and nothing forecast, `bss` where the base rate is 0 or 1 (NaN
  or minus infinity, depending on the Brier score).
- **Regions.** An empty `regions` mapping falls back to `{global: all}` rather than raising; regions may
  overlap and no check is made that they partition the grid. `regions.stack({}, n)`, called directly,
  raises `at least one region is required`.
- **Serialisation.** `to_config()` raises for weights or regions given as arrays, for a target source
  opened with positional arguments other than a single mapping, for an in-memory climatology, and for any
  source that does not implement it (`ArrayTargets`, `FakeForecastSource`). `Evaluation.attrs` then simply
  omits the `config` attr.
- **`merge` rules** are strict by design; see [merging](#merging). Merging results produced by different
  configs is possible (the shard tags allow it) but only warns.
- **Empty shards** are legal: a warning is logged and a state of zeros is written, whose means are NaN.
- **`NotImplementedError`** is raised by the abstract hooks `_Spec.build`, `Statistic._compute` and
  `Metric.from_means`; a subclass must override them.
- **Numerical note.** `means()` divides by the weight sum without guarding it, so an element that never
  accumulated is NaN (`0/0`) rather than an error, and a bin with no frames is NaN throughout.

Python 3.11 to 3.13. Performance numbers are in [benchmarks.md](benchmarks.md), the validation and
inspection scripts in [tools/README.md](../tools/README.md).
