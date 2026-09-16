# Benchmarks

Every measurement behind the package, taken on 2026-09-02 and 2026-09-03 on balfrin during the
development of 0.1.0; section 10 covers the settings that became the defaults. Each table names
the date and the package state it was measured on. Package commit hashes refer to the development
history before the first release, which is not part of this repository; the anemoi pins are
upstream commits.

- [1. Environment](#1-environment)
- [2. Testbeds](#2-testbeds)
- [3. Model step time and GPU memory](#3-model-step-time-and-gpu-memory)
- [4. Target I/O](#4-target-io)
- [5. Prefetch and cache](#5-prefetch-and-cache)
- [6. Sharding](#6-sharding)
- [7. Whole-run wall clock](#7-whole-run-wall-clock)
- [8. Agreement with numpy float64](#8-agreement-with-numpy-float64)
- [9. Operational findings](#9-operational-findings)
- [10. Chunking and shared forcings](#10-chunking-and-shared-forcings)

## 1. Environment

| item | value |
|---|---|
| machine | balfrin (CSCS Alps, Cray Shasta), 4 x A100 40 GB per node, GPUs in exclusive-process mode |
| partition | `debug` (30 min maximum), one job at a time |
| Python stack | Python 3.12, torch 2.10.0+cu126, numpy 2.4, xarray 2026.4, zarr 2.18 (zarr 2 API), numcodecs 0.15.1, netCDF4 |
| anemoi pins | anemoi-inference `d92aaf7`, anemoi-core `4288d10`, anemoi-datasets `2068e4d` (M1 and the checkpoint generation ran on anemoi-inference `b97fb6e` and anemoi-core `ace6eb6`) |
| reported package versions | anemoi-models 0.16.0.post64, anemoi-graphs 0.9.4.post92, anemoi-training 0.14.0.post64, anemoi-inference 0.11.3.dev, anemoi-datasets 0.5.43.dev1 |
| environment | `ANEMOI_INFERENCE_NUM_CHUNKS=8` in every GPU job (needed at 1 km resolution), `OMP_NUM_THREADS=16` per task |

The version strings above come from `importlib.metadata`; they were computed by setuptools-scm
when the venv was built and do not follow the submodule checkouts, so the pins are the truth.
The same limitation applies to the `package_versions` attr of every result file.

## 2. Testbeds

Four checkpoints, all on the same 1,688,744-node stretched grid (1,147,980 LAM at 1 km plus
540,764 global N320 nodes), 80 dataset variables and 67 model output variables; 9 evaluated
variables for the deterministic configs and 7 for the ensemble ones.

| name | model | parameters | timestep | steps in / out |
|---|---|---|---|---|
| det | deterministic GraphTransformer, dummy | 246,112,323 | 6 h | 2 / 1 |
| ens | same plus a noise injector, dummy | 246,378,863 | 6 h | 2 / 1 |
| ms | deterministic, hourly multi-step output, dummy | 247,264,658 | 1 h | 7 / 6 |
| real ensemble | graph-energy-score ensemble training | 246,378,863 | 6 h | 2 / 1 |

Three of them are *dummies*: unmodified varda training configs run for 40 optimizer steps from a
random initialisation with the current package versions, so that the inference checkpoints
unpickle on the current stack. Their forecasts are meaningless; what they validate is the
pipeline, the timing, the memory and the exactness. They exist because the MeteoSwiss production
checkpoints were pickled with an older anemoi-models and no longer load.

| name | how it was made |
|---|---|
| det | `varda-forecaster-det-sgm` for 40 steps, checkpointing on, model compilation off |
| ens | `varda-forecaster-ens-sgm` for 40 steps, same overrides, with a `NoiseConditioning` noise injector |
| ms | `varda-forecaster-det-sgm` for 40 steps with `timestep=1H`, `multistep_input=7`, `multistep_output=6`, `data.frequency=1h` |
| real ensemble | a genuine ensemble training with a graph-energy-score loss, same architecture as `ens`, taken from a training run on the current package versions |

The `ms` checkpoint needed one trick. Its cutout mixes an hourly limited-area dataset with a
six-hourly global one, and `adjust: all` snaps both to the coarser frequency, so anemoi-datasets
refuses `frequency: 1h` on the result. Opening the *global element* with
`interpolate_frequency: 1h`, which interpolates linearly in time between its six-hourly rows,
makes both elements hourly before the adjustment; the recorded dataset arguments carry that
option, so the checkpoint reproduces its own hourly dataset at inference time.

Two caveats apply to every number taken from these testbeds. The dummies' training period covers
the evaluation dates, so every score is in-sample and, being 40-step models, they drift; only the
real ensemble checkpoint produces physically meaningful ensemble statistics. And none of these
checkpoints runs on CPU: the triton graph attention has no CPU fallback, so every shake-out needs
a GPU.

## 3. Model step time and GPU memory

Step times are CUDA-synchronised. "Model call" is one forward pass of one member; for `ms` one
call produces six output frames. Peak memory is `torch.cuda.max_memory_allocated`.

Direct `runner.run()` loops through `tools/validate.py`, 2026-09-03 (det and dummy ens at
`1b8c2a8`, the rest at `010a2e9`):

| checkpoint | members | first call | steady call | peak GPU |
|---|---|---|---|---|
| det | 1 | 3.27 s | 2.65-2.67 s | 15.78 GiB |
| ens (dummy) | 1 | 3.68-3.70 s | 3.05-3.07 s | 15.78 GiB |
| ens (dummy) | 4 | 14.67-14.78 s | 12.15-12.21 s | 20.02 GiB |
| real ensemble | 1 | 3.72 s | 3.09 s | 15.89 GiB |
| real ensemble | 4 | 14.64 s | 12.14 s | 20.14 GiB |
| ms | 1 | 5.13 s | 3.65 s | 26.37 GiB |

The first call of a rollout includes building the input tensor. On `ms` the five `next()` calls
that only hand out an already-computed output time take at most 48.5 ms. Lockstep members cost
exactly M times the single-member time at every M measured: 4 x 3.05 s = 12.2 s.

Inside the framework, at 48 h and 4 init times (2026-09-03, det and ens at `5d73409`, ACC run
and real ensemble at `010a2e9`):

| run | members | per-step model time | peak GPU |
|---|---|---|---|
| det-48h | 1 | 3.0 s | 15.89 GiB |
| det-120h | 1 | 2.75 s | 15.89 GiB |
| ens-48h (dummy) | 4 | 12.6 s (4 x 3.1-3.3 s) | 20.24 GiB |
| ens-real-48h | 4 | 12.6 s (4 x ~3.15 s) | 20.36 GiB |
| ms-12h | 1 | 3.7 s per 6-output call | 26.49 GiB |
| det-48h with an hour-of-day climatology | 1 | 3.0 s | 16.12 GiB |

Per-member increment and the memory model (2026-09-03): one extra lockstep member costs
1.45 GiB on the 6-hourly 2-in/1-out checkpoints — the `(1, 2, N, 79)` float32 input tensor
(1.07 GiB) plus the `(N, 67)` `y_pred` held across the yield (0.42 GiB) — and about 6.0 GiB on
`ms`, whose per-member state carries seven input slices and six output times. The base is
about 15.9 GiB for det/ens and about 20.5 GiB for `ms`, so

    peak ~ base + 1.45 GiB x (M - 1) + M x frame + 3 x statistics temporaries

predicts 20.2 GiB at M = 4 with 9 variables (measured 20.24), about 26 GiB at M = 8 and about
34 GiB at M = 12. At 67 variables add roughly 0.4 GiB per member and 2.7 GiB of temporaries,
which puts the 40 GB ceiling near M = 10 for det/ens and near M = 3 for `ms`. A 232 MiB
device climatology adds 0.12-0.23 GiB.

## 4. Target I/O

The det/ens dataset is a 6-hourly cutout with 80 variables: one row `ds[i]` is
80 x 1,688,744 x 4 B = 540 MB (515.4 MiB) decompressed, and the 9-variable selection out of it
is 61 MB and takes 7 ms. Because the zarr chunks span every variable, the read costs the same
whatever subset is requested. Login node, 2026-09-03:

| reader | cold | warm (page cache) |
|---|---|---|
| main thread | 1.0-1.2 s | 0.63 s |
| one worker thread | 1.21 s | 0.80 s |
| worker while the main thread spins in pure Python | | 0.94 s |
| worker while the main thread runs numpy matmuls | | 0.81 s |
| worker with `numcodecs.blosc.use_threads = True` | | 0.64 s |

Off the main thread numcodecs 0.15.1 decodes with the single-threaded blosc context API, which
is the 0.17 s difference; `use_threads` is process-global and the package never sets it.

In the GPU runs the same read costs 0.75-0.86 s per row on the worker, and 0.83 s per row with
four shard processes reading concurrently on one node.

The `ms` dataset is the same cutout at hourly frequency, with the global element opened with
`interpolate_frequency: 1h`, so a row at an hour that is not a multiple of 6 costs one LAM row
plus two N320 rows:

| where | seconds per row |
|---|---|
| login node, cold | 1.39-1.61 s |
| worker, cold compute node | 1.51 s |
| worker, warm | 1.25-1.36 s |
| worker, warm, `numcodecs.blosc.use_threads = True` | 1.11 s |
| main thread, warm | 1.19 s |

Two consequences, both measured: the runner's own reading of the 7-row input window for one
init time takes 18.8 s on a cold node (about 14.7 s warm, 8.7 s on the login node), more than
the two model calls of a 12 h forecast; and building an hour-of-day climatology from 124
six-hourly January rows took 162.7 s, 1.3 s per row on a cold login node.

## 5. Prefetch and cache

In-process A/B through `tools/ab_prefetch.py`: one model load, a warm-up that reads every
valid time once and runs one model step, then the whole config once per variant with a fresh
target source. Variant `p<depth>[c<GiB>][t]`, `t` = blosc threads. `time_target_s` is the main
thread's wait for targets, `time_target_read_s` the seconds the worker spent reading.
Every variant's aggregation state was bit-exact against the first variant's in all three
tables: prefetch, cache and blosc threads change no number.

det-48h, 4 init times x 8 steps, 32 target frames, M = 1, warm node, `5b6391e`, 2026-09-03:

| variant | model s | target wait s | worker read s | reads / hits | total s | delta | peak GiB |
|---|---|---|---|---|---|---|---|
| p0 (synchronous) | 95.32 | 20.25 | 20.09 (main thread) | 32 / 0 | 115.66 | - | 15.89 |
| p2 | 96.92 | 0.16 | 27.55 | 32 / 0 | 97.17 | -16.0 % | 15.89 |
| p2c2 (2 GiB cache) | 96.72 | 0.14 | 13.37 | 14 / 18 | 96.96 | -16.2 % | 15.89 |
| p0 (bracket) | 94.88 | 20.31 | 20.15 (main thread) | 32 / 0 | 115.27 | - | 15.89 |

The target phase falls from 17.5 % of the total to 0.1 % of it, at the cost of 1.6 s of model
time (0.05 s per step of GIL and CPU contention with the worker). The cache halves the rows
read but changes nothing on the wall clock once the reads are hidden.

ens-48h, 2 init times x 8 steps, 16 target frames, M = 4, same commit and day:

| variant | model s | target wait s | worker read s | reads / hits | total s | delta | peak GiB |
|---|---|---|---|---|---|---|---|
| p0 (synchronous) | 201.43 | 10.03 | 9.97 (main thread) | 16 / 0 | 211.82 | - | 20.24 |
| p2 | 202.45 | 0.06 | 13.79 | 16 / 0 | 202.88 | -4.2 % | 20.24 |

At four members the model step is four times longer, so the target phase was only 4.7 % of the
total to begin with; it goes to 0.03 %.

ms-12h, 4 init times x 12 h (8 model calls), 48 target frames, M = 1, warm node, `b34bef3`,
2026-09-03:

| variant | model s | target wait s | worker read s | reads / hits | total s | delta | peak GiB |
|---|---|---|---|---|---|---|---|
| p2 (the old default) | 82.65 | 35.81 | 60.07 | 48 / 0 | 118.60 | - | 26.49 |
| p6 (= frames per pass, the default) | 88.24 | 12.99 | 65.45 | 48 / 0 | 101.36 | -14.5 % | 26.49 |
| p6c2 (p6 + 2 GiB cache) | 87.96 | 3.36 | 44.62 | 30 / 18 | 91.45 | -22.9 % | 26.49 |
| p12 (every lead of an init time) | 94.29 | 0.22 | 71.35 | 48 / 0 | 94.65 | -20.2 % | 26.49 |
| p6t (p6 + blosc threads) | 89.38 | 7.65 | 53.42 | 48 / 0 | 97.16 | -18.1 % | 26.49 |

`ms` is read-bound at every depth: 48 rows x 1.25 s = 60 s of target reads against a model
phase of about 83 s of which about 45 s are the runner's own input reads, so the GPU computes
for roughly 30 s of a 95-120 s run. Depth 12 hides everything but costs 11.6 s (+14 %) of
model time to worker contention; the 2 GiB cache buys more (30 of 48 rows read) because the
four init times overlap in valid time.

## 6. Sharding

det-120h: 8 init times x 20 steps, 9 variables, 3 regions, M = 1, 2026-09-03. A, B and C ran
at `1f053e7` (sharding, no prefetch), F at `477c60c` (sharding plus the default prefetch 2).
"Merge vs A" is `tools/compare_results.py` on the merged file against the unsharded control.

| form | tasks x GPUs | node(s) | startup | per-shard total s | max shard | job wall | merge vs A |
|---|---|---|---|---|---|---|---|
| A, 1 shard (control) | 1 x 1 | one, warm | 26 s | 555.4 | 555.4 | 10:01 | - |
| B, 4 tasks on one node | 4 x 4 | one, warm | 10 s | 139.8-141.7 | 141.7 | 2:39 (3.8x) | bit-exact |
| C, job array of 4 | 4 x (1 x 1) | three, cold | 89 s | 155.7-167.6 | 167.6 | 5:09 (1.9x) | bit-exact |
| F, 4 tasks + prefetch | 4 x 4 | one, warm | 11 s | 115.0-116.2 | 116.2 | 2:13 (4.5x) | bit-exact |

Four processes sharing one node do not slow each other down: 68.5 s per init time in B against
68.7 s in A, and the sum of the four shards' loop time is 563.0 s against the control's 555.4 s
(1.4 % overhead). C is slower only because its nodes were cold: about 12 s of extra model time
and 2 s of statistics are first-call overheads paid once per process, and its target reads were
cold Lustre reads. The headline is F: eight 120 h forecasts on a 1.69M-node grid scored in
2:13 of wall clock on one node, against 10:01 unsharded and unprefetched, bit-exact.

## 7. Whole-run wall clock

Framework totals from the result attrs, one GPU unless stated, 2026-09-03. `target` is the
main thread's wait; the worker's read seconds are given where they matter.

| run | package state | model s | target s | statistics s | total s | peak GiB | job wall |
|---|---|---|---|---|---|---|---|
| det-48h, 4 inits | `5d73409` | 99.45 | 27.18 | 0.13 | 126.76 | 15.89 | 2:23 |
| det-48h with lead 0 and ACC, 4 inits | `010a2e9` | 100.76 | 0.19 | 0.34 | 101.29 | 16.12 | 1:57 |
| ens-48h (dummy), 2 inits, M = 4 | `5d73409` | 205.33 | 10.06 | 0.41 | 215.80 | 20.24 | 3:51 |
| ens-real-48h, 2 inits, M = 4 | `010a2e9` | 205.46 | 0.06 | 0.44 | 205.97 | 20.36 | 3:41 |
| ms-12h with lead 0, 4 inits | `010a2e9` | 94.30 | 13.16 | 0.22 | 107.68 | 26.49 | 2:11 |
| det-120h, 8 inits, 4 shards | `477c60c` | 114.6-115.8 | 0.20-0.22 | 0.16 | 115.0-116.2 | 15.89 | 2:13 |
| persistence-48h, 4 inits, login node CPU | `477c60c` | \- | \- | \- | 22.6 (whole run) | \- | 22.6 s |

The persistence run has no model: its 4 init times x 9 leads needed 15 distinct rows, read in
16.8 s on the worker with 21 cache hits from a 2 GiB cache, and cost 9.2 s for the first init
time and 2.7 s for each of the others.

Statistics cost 2.5 ms per frame at M = 1 with 9 variables and 3 regions on the 1.69M-node
grid, and 30-60 ms per frame at M = 4 with 7 variables (the CRPS sort dominates). The first
init time of every process pays 4-14 s of first-call overheads (triton kernels, cuBLAS
handles, the first target read); on a cold node add about 90 s of startup for imports and
reading the checkpoint metadata off Lustre.

## 8. Agreement with numpy float64

`tools/validate.py` runs the framework and a direct `runner.run()` loop over the same init
time with the same seeding, reduces the direct loop's fields in numpy float64, and reports the
maximum relative difference over every variable, region and lead time. The threshold is 1e-5;
every run below passed. All 2026-09-03.

| run | package state | agreement |
|---|---|---|
| det, 2 steps | `5d73409` | rmse 6.8e-16, mae 1.1e-15, bias 3.8e-15 |
| det with ACC and lead 0, 2 steps | `010a2e9` | rmse 6.8e-16, mae 1.1e-15, bias 3.8e-15, acc 3.3e-15 |
| ens (dummy), 2 steps, M = 4 | `5d73409` | rmse 1.7e-15, mae 1.0e-15, bias 5.3e-15, crps 9.0e-16, fair_crps 1.0e-15, spread 1.4e-14, spread_skill 1.5e-14 |
| ens (dummy), 2 steps, M = 4 | `010a2e9` | the same to every printed digit, plus member_mae 7.8e-16, member_rmse 1.1e-15 |
| real ensemble, 2 steps, M = 4 | `010a2e9` | rmse 2.4e-15, mae 8.3e-16, bias 2.0e-14, member_mae 8.1e-16, member_rmse 8.2e-16, crps 1.4e-15, fair_crps 1.9e-15, spread 1.2e-15, spread_skill 1.2e-15 |
| ms, lead 9 h (2 model calls) | `60ce805` | rmse 1.9e-15, mae 1.9e-15, bias 4.1e-13 |

Lead 0, checked on `ms` and on the det ACC run: rmse, mae and bias are exactly 0.0 for the
eight prognostic variables and NaN for the diagnostic `tp`, whose excluded weight fraction is
exactly 1.

The float64 reduction is what buys this. Before it (M1, statistics computed in float32 on the
GPU) the same comparison gave rmse 1.6e-8, mae 1.8e-8, bias 1.8e-7, crps 1.8e-7,
fair_crps 2.4e-7 but spread and spread_skill 6.4e-5, a failure at 1e-5. Every mismatch was a
large-offset variable with a tiny spread: `msl` at 1e5 Pa with 0.2 Pa of spread lost 6.4e-5
relative in a float32 `var(0)`, and 2t over the LAM with 0.002 K of spread lost 1.1e-5. The
CRPS `pairs` term has the same exposure, masked in that run by `skill` being much larger.
Commits `56b1b96` and `1b8c2a8` moved the member reductions to float64 one variable at a time
and the differences fell to the 1e-15 above.

Two more exactness checks, both 2026-09-03:

* Every sharded run merged bit-exactly against the unsharded control (section 6), including
  the job-array form spread over three nodes and four processes.
* Adding lead-0 scoring, the anomaly statistics and a climatology left every pre-existing sum
  and metric bit-exact on the shared lead times (`tools/compare_results.py --leads common
  --stats common`, `010a2e9` against `5b6391e`).

## 9. Operational findings

* **Respect Slurm's `CUDA_VISIBLE_DEVICES` in job arrays.** A first attempt at the job-array
  form failed: Slurm co-scheduled all four single-GPU array tasks on one node and gave each
  job its own `CUDA_VISIBLE_DEVICES`, but the job wrapper overwrote it with `SLURM_LOCALID`,
  which is 0 in every one-task job. Three tasks then opened GPU 0, which is in exclusive-process
  mode, and died in their first CUDA call with `CUDA-capable device(s) is/are busy or
  unavailable`. A wrapper must leave a single-device list alone, pick the `SLURM_LOCALID`-th
  entry only when several devices are exposed to every task, and fall back to the local rank
  only when the variable is unset.
* **A cold node costs about 90 s** of startup (imports and the checkpoint metadata off Lustre)
  plus about 14 s of first-call overheads per process, so short evaluations benefit from
  landing on a node that has already run one.
* **`numcodecs.blosc.use_threads = True`** takes the worker's read from 0.83-0.86 s to 0.63 s
  per row (1.25 s to 1.11 s on the hourly dataset), but it is process-global; it is a knob for
  the user to set, not something the package does.

## 10. Chunking and shared forcings

Measured on 2026-09-03 on the real ensemble checkpoint (trained from scratch on 2026-09-02 on
anemoi-core `4288d10`, run on that commit and anemoi-inference `d92aaf7`), init time
2024-01-02T00, 10 variables, 5 model calls, one process per setting because the chunk counts are
read at import time; `predict_step` timed between two CUDA synchronisations, peak memory
`max_memory_allocated` from before the checkpoint load. The cluster had been re-provisioned that
day and is mixed: `PG506-232` boards with 124 SMs and 95.15 GiB (node nid002028, used below
unless stated) and `A100-SXM4-80GB` with 108 SMs and 79.25 GiB (nid002812), both compute
capability 8.0. The forward is ~10 % faster on the former (2.83 vs 3.06 s at chunks 8/8).

Processor chunks (`ANEMOI_INFERENCE_NUM_CHUNKS_PROCESSOR`) at mapper chunks 8:

| processor chunks | 8 | 4 | 2 | 1 |
|---|---|---|---|---|
| forward, one member | 2.83 s | 2.13 s | 1.79 s | 1.53 s |
| 4-member step / 8-member step | 12.10 / 24.14 s | | | 6.90 / 13.67 s |
| peak, M = 1 / 4 / 8 | 16.0 / 20.6 / 26.8 GiB | same | same | same |
| member-0 fields vs 8, lead 30 h | reference | bit-exact | bit-exact | bit-exact |

Mapper chunks (`ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER`, `max` with the checkpoint's own 4) at
processor chunks 1:

| mapper chunks | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| forward, one member | 1.53 s | 1.53 s | 1.55 s | 1.53 s | 1.54 s | 1.57 s |
| peak, M = 1 | 71.4 GiB | 40.6 GiB | 24.6 GiB | 16.0 GiB | 14.2 GiB | 14.2 GiB |
| peak, M = 8 | out of memory (> 83 GiB) | 51.5 GiB | 35.5 GiB | 26.8 GiB | 25.0 GiB | 25.0 GiB |
| fields vs 8, lead 30 h | 7.0e-2 | 6.5e-2 | 5.5e-2 | reference | rounding | rounding |

The mapper differences are a reduction order: 2-3e-6 relative in the per-variable field means at
6 h, then amplified by the 1 km rollout (1e-3 of the range for wind, 5e-2 for `tp` at 30 h, 99 %
of nodes differing). Memory per lockstep member at 10 variables is 1.55 GiB, linear to M = 24
(49.8 GiB at processor 1 / mapper 16); 16 members peak at 39.2 GiB at mapper 8.

A seeded `torch.randn(1688744, 4, device="cuda")` gives different tensors on the two GPU models
(the first elements agree, later ones do not), and the 6 h fields of every setting on the
A100-80GB node differ from the other node's by ensemble-spread amounts while being identical
across settings within a node.

Inside the framework, on an `A100-SXM4-80GB` node, the ensemble smoke configuration of the
first campaign (2 init times x 48 h x 4 members, 7 variables, 2 regions, commit `c1b7972`,
2026-09-03), one variant per task on one node; every pair of results is bit-exact
(`tools/compare_results.py`):

| variant | model s | per 4-member step | forcings computed / hits | peak GiB |
|---|---|---|---|---|
| processor chunks 8, no forcings cache | 228.3 | 14.3 s | - | 20.36 |
| processor chunks 1 (the default now), no cache | 142.3 | 8.9 s | - | 20.36 |
| processor chunks 1, forcings cache 1 GiB (the defaults) | 131.5 | 8.2 s | 13 / 59 | 20.36 |

The same configuration ran in 225.3 s of model time in the campaign's smoke run on the
cluster's previous nodes; its result file is not bit-exact against the first variant here
(7.5e-2 relative on the ensemble metrics), the GPU-model effect above.
