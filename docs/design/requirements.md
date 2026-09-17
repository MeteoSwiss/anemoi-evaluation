# Requirements

What anemoi-evaluation must do, and why. The obligations below are the contract the package is
built and validated against at v0.2.0. How they are met is in
[`architecture.md`](architecture.md); measured evidence is in [`../benchmarks.md`](../benchmarks.md);
the surface a user types is in [`../user-guide.md`](../user-guide.md) and
[`../reference.md`](../reference.md).

- [1. Introduction](#1-introduction)
- [2. Stakeholders and users](#2-stakeholders-and-users)
- [3. Functional requirements](#3-functional-requirements)
- [4. Non-functional requirements](#4-non-functional-requirements)
- [5. Out of scope](#5-out-of-scope)
- [6. Verification](#6-verification)

## 1. Introduction

### 1.1 Purpose

anemoi-evaluation scores anemoi forecast models. A run takes a checkpoint, a set of init times
and a lead time, produces the forecasts in memory with `anemoi-inference`, compares each step
against the anemoi dataset the model was initialised from, and reduces everything to one small
netcdf of aggregated statistics. The alternative, writing forecast fields to disk and scoring them
afterwards, costs terabytes of I/O at the grid sizes anemoi models are trained on.

This document states the obligations that follow, loosely in the ISO/IEC/IEEE 29148 spirit and
kept light: numbered requirements, MUST / SHOULD / MAY, a rationale where one is needed.

### 1.2 Scope

In scope at v0.2.0: deterministic and ensemble forecasts from a single-dataset anemoi checkpoint,
gridded targets from an anemoi dataset on the same grid, deterministic and ensemble metrics,
weighting, regions, time bins, a persistence baseline, a climatology for anomaly scores, a YAML
configuration and CLI, and sharding with exact merging. Section 5 lists what is deliberately
outside. The package itself is device-agnostic, but the anemoi checkpoints it is validated
against do not run on CPU, so an end-to-end exercise needs a GPU.

### 1.3 Audience

Maintainers, reviewers of the numbers, and users who need to know what the package guarantees
before they trust a score. Users who only want to run it should read the user guide.

### 1.4 Definitions

| term | meaning |
|---|---|
| frame | one forecast step: `(M members, V variables, N nodes)` float32, with its init, lead and valid times |
| statistic | a per-node, per-variable quantity computed from a frame and its target |
| metric | a function of the *means* of statistics, for example RMSE from mean squared error |
| state | the `AggregationState`: the float64 sums, weights and counts of a run |
| model call | one forward pass of one member, yielding one frame per output step |

### 1.5 Conventions

MUST is an obligation, SHOULD a strong preference that may be traded away with a recorded reason,
MAY an option. IDs are stable: a dropped requirement keeps its ID, marked withdrawn. A requirement
not fully met at v0.2.0 carries a **Status** line.

## 2. Stakeholders and users

Model developers compare checkpoints and training variants and need scores that differ only
because the models differ; verification scientists need the definitions, the weighting and the
pooling stated and the primitives kept so a metric can be re-derived; HPC operators need a run to
fit a job array with predictable GPU hours and no inter-process communication. Behind them are the
maintainers, who need the anemoi coupling narrow enough to survive upstream change, and the
tooling that reads result files.

## 3. Functional requirements

### 3.1 Forecast production

**FR-1** The package MUST produce forecasts in memory by running an `anemoi-inference` runner on
a checkpoint, and MUST NOT write forecast fields to disk at any point.

**FR-2** A run MUST support ensemble forecasts of `M >= 1` members, and every frame MUST carry a
member axis whatever `M` is, since the ensemble statistics need all members of a step at once.

**FR-3** Each member's noise MUST be determined by a seed that is a function of the configured
seed, the init time, the member index and the model call index only. It MUST NOT depend on the
ensemble size, on the other members, or on how member rollouts are interleaved.

**FR-4** The package MUST support models that emit several lead times per forward pass, scoring
each output time as its own frame. When the requested lead time is not a multiple of the model's
output horizon the package MUST warn, compute up to the next multiple and discard the extra
outputs.

**FR-5** The package SHOULD be able to score lead 0 from the model's initial state, without an
extra model call or read, excluding the variables absent from that state (diagnostics).
*Rationale: with targets from the same dataset a lead-0 error of exactly zero is a cheap
end-to-end check of the whole pipeline.*

**FR-6** The package MUST provide a persistence forecast source (the target at init time repeated
at every lead) as a baseline and as a model-free end-to-end check.

**FR-7** Before a run, the package MUST refuse init times whose rollout would read dates absent
from, or listed as missing in, the datasets the runner reads.
*Rationale: a failure 40 minutes into a job array is expensive, and this check needs no model.*

### 3.2 Targets

**FR-8** Targets MUST be read from an anemoi dataset at the frame's valid time, on the forecast
source's grid, in the model's units, with no regridding and no unit conversion; a source on a
different grid (beyond 1e-5 degrees per node) MUST be refused at construction.
*Rationale: the comparison is then the one the model was trained against, with no interpolation
error the package cannot account for.*

**FR-9** The default target dataset MUST cover the evaluation period rather than the period the
model was trained on, without the user having to spell the dataset out.
*Rationale: the arguments recorded in a checkpoint carry the training period's `start` and `end`,
which would silently exclude the dates being evaluated.*

**FR-10** A target that cannot be delivered MUST be handled by a configured policy: skip the
frame with a warning (default) or raise. A skipped frame MUST contribute nothing: no weight, no
count, no sum.

**FR-11** A target dataset with more than one ensemble member MUST be refused: initial conditions
are deterministic and members come from the model's noise, as in training.

### 3.3 Statistics and metrics

**FR-12** The package MUST store only primitive statistics and MUST derive every metric from
their means, so that a result file supports re-deriving any metric its primitives allow and no
parameter (the CRPS `alpha`) is baked into a file.

**FR-13** The catalogue MUST cover the deterministic scores (bias, MAE, RMSE), the ensemble
scores (member MAE and RMSE, CRPS for any `alpha` in [0, 1] including the standard and fair
cases, spread, spread-skill), the anomaly correlation, the categorical and probabilistic scores at
user-given thresholds, the rank histogram of an ensemble, and the reliability diagram and Brier
decomposition of a threshold, each available per lead time, bin, variable and region. The exact names and formulas are in [`../reference.md`](../reference.md).

**FR-14** A metric needing more members than the run has MUST fail at construction, not mid-run.

**FR-15** Anomaly correlation MUST be pooled over nodes *and* over the init times of a bin;
per-init or per-season values come from the binning, not from averaging per-init correlations,
which is a different quantity.

**FR-16** The package MUST be able to build a climatology from a target source, store it and read
it back, so that an anomaly score needs no external product.
**Status: met for per-key means only** (see [`../reference.md`](../reference.md)); there is no
builder for richer climatologies, and none upstream.

**FR-17** Nodes where the climatology is not finite MUST leave the anomaly statistics only and
MUST NOT change the shared weights of the frame; the count per variable MUST be recorded.
*Rationale: only the ACC ratio is then meaningful, and it is weight-invariant; excluding those
nodes everywhere would make every other metric depend on the climatology.*

**FR-36** A threshold score MUST be configured as a per-variable mapping with a user-given label, MUST
binarise the forecast and the target with a strict `>` in float64, MUST yield NaN for the variables the
mapping omits and MUST fail at construction for a variable the run does not have. A statistic name that
means two different statistics in one run MUST be an error at construction. At one member the Brier
score MUST equal `miss + false_alarm` exactly elementwise, hence on the weighted means up to float64
rounding, so that a deterministic and an ensemble run are comparable. *Verified by the unit suite.*

**FR-37** The rank histogram MUST spread tied members deterministically, putting `1 / (ties + 1)` into
each of the `ties + 1` bins the target could occupy, so that it is the expectation of uniform random
tie-breaking and needs no seed. The `M + 1` stored values MUST sum to 1 per element to float64 rounding,
and their weighted means MUST therefore sum to 1 to float64 rounding. A metric whose statistics depend on
the ensemble size MUST learn it at construction through `bind_members` and MUST refuse to be rebound to a
different size. *Verified by the unit suite.*

**FR-38** For each threshold label, the system MUST be able to report the observed event frequency and the
forecast-probability distribution at each of the `M + 1` ensemble probability levels, and the Brier
decomposition `BS = REL - RES + UNC` as three scalars, from area-weighted sums that merge by addition. The
decomposition MUST agree with the stored Brier score to float64 rounding of the weighted sums.
*Verified by the unit suite.*

### 3.4 Aggregation

**FR-18** A run MUST accumulate its results into a single state that is additive (FR-31),
resolved by lead time, time bin, variable and region, and MUST retain alongside it the weights
and the frame counts needed to interpret a mean and to spot an under-filled cell.

**FR-19** A node MUST be excluded from *every* statistic of a frame when any member or the target
is non-finite there, and the same exclusion MUST drive the weight sum, so a mean is always over
the same set of nodes.

**FR-20** Node weights MUST be configurable, including from the checkpoint's training graph, and
MUST default to the training area weights when the forecast source has a graph, so that a score
uses exactly the weights training used. The spec vocabulary is in
[`../reference.md`](../reference.md).

**FR-21** Regions MUST be configurable as arbitrary per-node masks, including geographic boxes,
graph attributes and the sub-grids of a cutout dataset, and MAY overlap. At least one region MUST
be configured, the default being one global region.

**FR-22** Time binning MUST offer at least seasonal, monthly, per-date and no binning, keyed on
either the init time or the valid time of a frame. Lead time MUST always be an axis and never a
bin, and the total over bins MUST be derived on output rather than stored. Per-date bins keyed by
valid time MUST be built from the full init-time list in every shard, or the bin coordinates will
not match at merge time.

### 3.5 Interfaces

**FR-23** The Python API MUST be the primary interface: sources, aggregator, statistics, metrics
and the driver MUST be usable directly, including a mode that hands out the frame and target pairs
for a custom loop.

**FR-24** A YAML configuration MUST be a validated serialisation of the same objects, MUST reject
unknown keys outside the documented pass-through blocks, and MUST round-trip with the Python
objects.

**FR-25** A configuration MUST be able to extend another one, a chain deep, so that the variants
of a campaign are a few lines each. What the rest of the package sees (dry run, serialised config,
the result file's config attribute) MUST be the fully resolved configuration, so that a run stays
reproducible from its own result file.

**FR-26** A dataset block MUST be able to stand for the `open_dataset` arguments a checkpoint
records, with the block's own keys on top.
*Rationale: anemoi-inference falls back to the recorded arguments only when given nothing at all,
so a single override discards them; evaluating outside the training period needs both.*

**FR-27** The CLI MUST offer `run` (with shard selection, dry run and a step-time hint) and
`merge` (with a partial mode), and MUST be a thin layer over the Python API.

**FR-28** A dry run MUST describe what a run would do without loading a model: the sources, the
init and lead times, the model calls, the metrics and statistics, the weight and region specs,
the bins, how many frames have targets, and the bytes the run moves.

**FR-29** Given a measured step time, the dry run SHOULD estimate model and wall seconds for a
shard, and GPU hours and per-shard wall time for the run. The hint describes the machine, not the
evaluation: it MUST change no result and MUST NOT be a configuration key.

### 3.6 Output, sharding and merging

**FR-30** A run MUST write one netcdf holding the metrics, the raw aggregation state, the
coordinates, and provenance attributes identifying the checkpoint, the configuration, the package
versions, the counters and the timings. The raw state MUST be readable back from that file.

**FR-31** Result files and in-memory states MUST merge by summation, since everything the state
holds is a sum, into a state that satisfies NFR-4.

**FR-32** Merging MUST refuse inputs that share an init time, and MUST refuse a set of shard tags
that is not exactly the `0..n-1` of one run (missing, duplicate, mixed counts, mixed tagged and
untagged). A partial mode MUST allow an incomplete set and record which shards the output holds.
*Rationale: an incomplete merge is otherwise a well-formed file whose only symptom is a low
count.*

**FR-33** A run MUST be shardable over init times, explicitly and implicitly from Slurm job-step
and job-array variables, so that job steps and job arrays compose. With more than one shard the
output path MUST carry a shard placeholder or the run MUST refuse to start.

### 3.7 Extensibility

**FR-34** A forecast, target or climatology source that is not shipped with the package MUST be
usable without modifying the driver, and MUST be able to opt out of the capabilities it cannot
provide rather than implement them.

**FR-35** A new statistic or metric MUST be addable without modifying the aggregator or the
output, and MUST be able to declare the per-frame inputs it needs and the smallest ensemble it
accepts, so that a run fails at construction rather than mid-flight (FR-14).

## 4. Non-functional requirements

### 4.1 Exactness and reproducibility

**NFR-1** No reduction may lose accuracy to cancellation: the statistics, the reduction over
nodes, the weight sums and the reduction over members MUST all hold NFR-3 for variables with a
large offset and a small spread. Measured, `msl` at 1e5 Pa with 0.2 Pa of spread lost 6.4e-5
relative in a float32 `var(0)` ([`../benchmarks.md`](../benchmarks.md) section 8).

**NFR-2** The package MUST NOT sample or approximate any reduction: no metric may be computed
from a subset of nodes, members or frames.

**NFR-3** Metrics MUST agree with a naive numpy float64 computation of the same forecast to
better than 1e-5 relative.
*Evidence: 7e-16 to 4e-13 across all metrics and all four testbed checkpoints (benchmarks 8).*

**NFR-4** Sums MUST be bit-exact across shards when every bin is filled by a single shard, and
MUST otherwise agree to float64 rounding.
*Evidence: every sharded form bit-exact against the unsharded control, including a job array over
three nodes (benchmarks 6).*

**NFR-5** The package MUST record everything that is part of a run's numerical identity:
checkpoint path, uuid and training run id, package versions, resolved configuration, ensemble
size, inference chunk counts and GPU model. The mapper chunk count changes a reduction order and
device noise depends on the GPU's SM count, so two runs differing in either are not comparable bit
for bit.

**NFR-6** Bit-level reproducibility of ensemble members MUST be understood as per GPU model;
shards that must merge bit-exactly MUST run on one GPU model.
**Status: a constraint, not a defect.** Device `randn` maps generator counters to elements
through the launch grid, hence through the SM count (benchmarks 10).

**NFR-7** The package MUST NOT change global state that would alter the model under test or the
process around it (precision knobs, compression-thread switches, module-level mutable state), and
two evaluations in one process MUST NOT interfere.

**NFR-8** A source MUST hand the driver tensors it owns, never a view into a buffer it will
overwrite, except where the contract says the result is read-only and shared (the climatology).

### 4.2 Performance

**NFR-9** Scoring MUST cost a small fraction of the model step at production sizes.
*Evidence: 2.5 ms per frame at M = 1 with 9 variables and 3 regions on 1.69M nodes, 30 to 60 ms
at M = 4 (benchmarks 7), against model steps of 2.7 to 12 s (benchmarks 3).*

**NFR-10** Target reads SHOULD be hidden behind the model step.
**Status: partially met.** Met where a row read is shorter than a model step: the target phase
falls from 17.5 % of a deterministic run to 0.1 % (benchmarks 5). An hourly model reading an
interpolated dataset stays read-bound at every prefetch depth; the levers are the row cache, the
process-global blosc thread switch, or a differently chunked evaluation dataset.

**NFR-11** Quantities that depend only on the dates and the grid MUST be computed once and shared
across lockstep members and init times.
*Evidence: the forcings cache took a 4-member step from 8.9 s to 8.2 s, 59 hits against 13
computations (benchmarks 10).*

**NFR-12** The package MUST NOT spend model time on settings that buy nothing: an inference
default it sets on the user's behalf MUST be justified by a measurement, and an explicit user
value MUST be left alone.
*Rationale: the processor chunk count, where 8 costs 1.85 times the forward of 1 for the same
peak memory and bit-identical fields (benchmarks 10).*

**NFR-13** GPU memory MUST be linear in the ensemble size with a documented per-member cost, so a
user can size a job.
*Evidence: 1.45 to 1.55 GiB per member for a 6-hourly 2-in/1-out model at 1.69M nodes, about
6 GiB for an hourly 7-in/6-out one, linear to M = 24 (benchmarks 3 and 10).*

### 4.3 Scalability

**NFR-14** A run MUST scale by evaluating disjoint subsets of the init times independently, with
no communication between them, no shared filesystem state and no distributed torch anywhere. Such
a job tolerates a lost task and recombines exactly.

**NFR-15** A result file MUST be small and its size independent of the grid, up to the bin and
init-time coordinates. Memory MUST NOT grow with the number of init times or lead times.

### 4.4 Portability

**NFR-16** The package MUST run on Python 3.11 to 3.13, on CPU or one CUDA device, with no MPI
and no launcher beyond the one the user already uses. A checkpoint MUST fit on one device; there
is no model-parallel or multi-rank runner (section 5).

**NFR-17** The package MUST NOT modify the anemoi packages and MUST work around what it finds in
them. Private coupling to anemoi-inference MUST be minimal and isolated in named helpers.
**Status: met with two couplings**, both named in [`architecture.md`](architecture.md) section 8.

### 4.5 I/O

**NFR-18** Exactly three things MUST touch the filesystem during a run: the checkpoint, the
target rows, and the result netcdf.

**NFR-19** Target reading MUST cost at most one dataset row per valid time, and MUST NOT need a
lock on the dataset handle.
*Rationale: anemoi datasets are chunked one date at a time across every variable, so a row costs
the same whatever subset is asked for; reading per variable would multiply that cost.*

**NFR-20** Read-ahead MUST NOT be defeated by a forecast source that delivers several frames per
model call, since those frames all need their targets at once.
*Evidence: on a 6-output model, matching the depth to the model call cut the run by 14.5 %
(benchmarks 5).*

### 4.6 Maintainability and observability

**NFR-21** The unit test suite MUST run on CPU in seconds and MUST NOT need a checkpoint, a GPU
or a dataset; anything that needs a checkpoint belongs in
[`tools/`](../../tools/README.md). *Status: 38 tests at v0.1.0, 72 with the categorical, rank histogram and reliability scores.*

**NFR-22** Numerical claims MUST be checked against an independent reference: the aggregator
against a naive numpy float64 computation, CRPS against the naive pairwise formula, the ensemble
variance against `numpy.var(ddof=1)`, and a whole run against a direct runner loop.

**NFR-23** Configuration errors MUST be reported at configuration time or at construction time,
naming the offending key or date, rather than during a run.

**NFR-24** A run MUST report, per phase (model, target, statistics, total), wall time that
includes the device work it charges for, plus read and cache counters, model calls and peak device
memory, in the log and in the result attributes. The runner's own per-step timer does not
synchronise and under-reports.

**NFR-25** Result files MUST carry enough provenance to rebuild the run: the resolved
configuration and the versions of the packages behind it.
**Status: partially met.** The recorded versions are installed distribution versions, which for
editable installs are the strings computed when the environment was built, not the checked-out
commit.

## 5. Out of scope

Deliberate non-goals at v0.2.0. Each is a decision, not an oversight.

* **Spectra, and any statistic that is not a per-node reduction.**
* **Observations as targets.** Targets are gridded anemoi datasets on the model's grid.
* **Regridding and unit conversion** (FR-8).
* **Plots and reports.** A result file is the product; presentation is downstream tooling.
* **Distributed torch.** Data-parallel evaluation over init times was designed and deferred: it
  is the job array without the files, and its only gain is one file instead of n. Model-parallel
  evaluation was deferred too: anemoi-inference's parallel runner destroys its process group after
  every `run()`, and no testbed checkpoint needs more than one GPU.
* **A batched-member runner.** Rejected: batching raises peak memory rather than lowering it, its
  only gain is per-step Python overhead, which is nil at 1.7M nodes, and it is incompatible with a
  sharded model. Should the ensemble memory of NFR-13 ever bind, the fix is the opposite one,
  sequential members with a pinned host buffer ([`architecture.md`](architecture.md) section 9).
* **Multi-dataset checkpoints**, refused at construction, and **comparing several checkpoints in
  one run**: one checkpoint per run, comparison is offline from result files.
* **Neighbourhood scores** (FSS and its relatives). They are not per-node reductions, and on an
  unstructured grid a neighbourhood is a spatial query the package has no index for.
* **Quantile, climatological or otherwise per-node thresholds.** Thresholds are fixed numbers in the
  model's units, which is the operational question (warning levels are absolute); a per-node threshold
  field needs a per-node quantile climatology, which the climatology source does not build (FR-16).
  Only exceedances are configurable, so a below-threshold event is read as the complement of its cells.
* **Sampling uncertainty.** No confidence intervals and no bootstrap: over the 60 to 80 init times of a
  campaign a rare-event score has wide and correlated uncertainty, and quantifying it is a separate
  problem from accumulating the sums. The fractional tie spreading of the rank histogram also makes its
  bins non-multinomial, so the usual flatness test would be conservative rather than exact.

## 6. Verification

The functional requirements are verified by the unit suite (naive numpy references, fake sources,
the config forms and the merge checks) and, where they need a real checkpoint, by the scripts in
[`tools/`](../../tools/README.md): `validate.py` for FR-1 to FR-11 and NFR-1 to NFR-3,
`compare_results.py` for NFR-4, `ab_prefetch.py` for NFR-10 and NFR-20, and the run attributes
for the rest of section 4.2. The measurements behind every *Evidence* line above, and the
testbeds they were taken on, are in [`../benchmarks.md`](../benchmarks.md).
