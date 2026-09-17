# CLAUDE.md

See @README.md for the project overview and repository layout.

## Sources of truth

* Behaviour: `src/`, then `tests/`. Never state a flag, config key, metric or number you have not seen there.
* The documents under `docs/` and `README.md` are kept consistent with the code in the same change; measured
  numbers go in `docs/benchmarks.md` only, with their testbed.

## Invariants

* Forecast fields are never written to disk; the only output is the statistics netcdf.
* Exactness: float64 sums; a sharded run merges to the unsharded result (bit-exact per bin fed by one shard,
  float64 rounding otherwise). No sampling, no float32 reductions.
* Result-file layout (variables, coords, attrs) is an interface: changing it updates `docs/reference.md` too.
* Verify any scoring-path change (statistics, aggregation, weights, regions, binning, metrics, frames/pairs,
  source field selection) with `python tools/validate.py config.yaml --init <date> --steps 2 --members 1 4`,
  plus `tools/compare_results.py` for intended no-ops. Both need a GPU and a checkpoint: say so if you cannot run them.

## Commands and conventions

* `pytest tests -q`: CPU only, seconds, in-memory fakes; no markers, no skips, no GPU or real-data tests.
* `pip install -e ".[tests]"`, or `uv pip install --torch-backend=cpu -e ".[tests]"` (without
  `--torch-backend=cpu` the CUDA wheels fill a CI runner's disk).
* `pre-commit run --all-files`: CI's `quality` job. No licence header in source files. ruff: line length
  120, import rules (`I`), isort force-single-line (one import per name).
* Conventional-commit titles on `main` and for PRs (`feat:`, `fix:`, `perf:`, `docs:`, `chore:`, `!` for
  breaking); every user-visible change adds a `CHANGELOG.md` entry under `## Unreleased`, in the existing
  `### Breaking` / `### Features` / `### Fixes` grouping.

## Gotchas

* `setuptools-scm` versioning: a clone without tags reports `0+unknown`.
* `ANEMOI_INFERENCE_NUM_CHUNKS` without `ANEMOI_INFERENCE_NUM_CHUNKS_PROCESSOR`: the inference source adds the
  latter as 1; the GPU tools default the former to 8.
* `output.path` needs a `{shard}` placeholder whenever more than one shard is written, and a
  `{dataset}` placeholder for a run that scores multiple datasets (which it refuses on a run that scores one,
  including a subset of one selected with `datasets:`).
* Slurm sharding uses `SLURM_STEP_NUM_TASKS`, not `SLURM_NTASKS`, which would silently make a batch script 1 of n.
* `base:` and `from_checkpoint:` resolve on the raw mapping before pydantic validation; the rest sees the result.
