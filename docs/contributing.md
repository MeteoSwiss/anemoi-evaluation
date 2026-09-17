# Contributing

The repository is [MeteoSwiss/anemoi-evaluation](https://github.com/MeteoSwiss/anemoi-evaluation).
The package is a prototype: interfaces, config keys and result-file contents still change
between releases.

## Development setup

Python 3.11 to 3.13 (`requires-python = ">=3.11,<3.14"`). Clone the repository and install it
in editable mode with the `tests` extra:

```bash
git clone https://github.com/MeteoSwiss/anemoi-evaluation.git
cd anemoi-evaluation
pip install -e ".[tests]"
```

With [uv](https://docs.astral.sh/uv/), which is what CI uses:

```bash
uv venv --python 3.12
uv pip install --torch-backend=cpu -e ".[tests]"
uv run --no-sync pytest tests -q
```

`--torch-backend=cpu` applies to the whole resolution and keeps the CUDA wheels out. Without
it, torch resolves to a CUDA build: on a machine with a GPU that is what you want, in CI it
fills the runner disk (`anemoi-graphs` caps torch below 2.11, and the default index only has
CUDA builds there). There is no lock file and no dependency group; dependencies come from
`pyproject.toml` only.

The clone must have the tags (`fetch-depth: 0` in CI): the version is computed by
`setuptools-scm`, and without tags the package reports `0+unknown`.

## Code style

Formatting and linting are `ruff`, configured in `pyproject.toml`:

* line length 120,
* the import rules (`I`) are enabled on top of the default set,
* isort in force-single-line mode: one `import` per name, no `from x import a, b`.

Source files carry no licence header; the licence is the repository's [`LICENSE`](../LICENSE)
(BSD 3-Clause).

The other hooks in `.pre-commit-config.yaml` check YAML and TOML syntax, leftover debugger
statements and `breakpoint()`, end-of-file newlines, trailing whitespace, large files and
merge-conflict markers, reject type comments in place of annotations, blanket `# noqa` and
`log.warn`, and format `pyproject.toml` with `pyproject-fmt`.

Run them all before pushing:

```bash
pre-commit run --all-files          # or: uvx pre-commit run --all-files
```

## Running tests

```bash
pytest tests -q
pytest tests/test_end_to_end.py -q
pytest tests -q -k crps
```

The suite runs on the CPU in a few seconds. It is built on the in-memory fakes in
`src/anemoi/evaluation/sources/fake.py` (`FakeForecastSource`, `ArrayTargets`, wired up by the
`fake_sources` fixture in `tests/conftest.py`) and on hand-written stand-ins for an anemoi
dataset and an anemoi-inference runner, so no test needs a GPU, a checkpoint or a real zarr.
Apart from `parametrize` there are no markers, and there are no skips: everything in `tests/` runs everywhere, and a test that cannot
run is a failure, not a skip. Shared helpers that are too big for `conftest.py` live in a private
module under `tests/` (`_multi_dataset.py`), which `pythonpath = ["tests"]` in `pyproject.toml`
makes importable.

| file | covers |
|---|---|
| `test_aggregation.py` | weighted, masked, binned sums against a naive reference; exact merging; the binnings |
| `test_end_to_end.py` | metrics against closed forms, lead 0, missing targets, netcdf round trip, `merge` and its shard validation, sharding and the dry run, persistence |
| `test_ensemble_statistics.py` | CRPS against a naive pairwise sum, ensemble variance, the member-error identity, float32 offsets |
| `test_categorical_scores.py` | threshold statistics against a numpy reference, the contingency identities, the metric formulas, the identity guard and the end-to-end NaN behaviour |
| `test_rank_histogram.py` | the rank bins against a numpy reference with planted ties, the sum-to-one identity, the calibrated, under-dispersed and biased shapes, member binding and the extra output axis end to end |
| `test_reliability.py` | the level statistics against a numpy reference, the decomposition identity against the stored Brier score, the calibrated, over-confident and under-confident diagram shapes, the double binding and both extra axes in one file |
| `test_climatology.py` | `ArrayClimatology` keys and its netcdf round trip, and ACC end to end |
| `test_config.py` | config forms and validation errors, the round trip, `base:` merging and cycles, `from_checkpoint` |
| `test_inference_source.py` | the inference-source pieces that need no checkpoint: chunk defaults, the forcings cache, the chunk report, graph attributes |
| `test_real_sources.py` | `DatasetTargets` reads and member seeding against a stand-in anemoi dataset |
| `test_multi_dataset.py` | the two-dataset fixture runner (`tests/_multi_dataset.py`, `tests/fixtures/multi-dataset/`), the per-dataset views, the refusals, the per-dataset config forms, and the end-to-end equality of a two-dataset run with the two single-dataset runs of the same fake |

The GPU and real-data paths are therefore not covered by `pytest`. They are covered by the
tools below, which are run by hand against a real checkpoint.

## Validation tools

[`tools/README.md`](../tools/README.md) documents five scripts that are not part of the
package. Two of them need a GPU and a real checkpoint, the other three run anywhere in
seconds.

* **`tools/validate.py` is the reference for exactness.** It scores one init time twice, once
  through the framework and once through a direct `runner.run()` loop reduced in numpy
  float64, and reports the relative difference per metric. **Run it after any change to the
  scoring path**: statistics, aggregation, weights, regions, binning, metric derivation, the
  frame/pair plumbing, or anything in the sources that decides which fields are compared.
  Cover both the deterministic and the ensemble case:

  ```bash
  python tools/validate.py config.yaml --init 2024-01-02T00 --steps 2 --members 1 4
  python tools/validate.py config.yaml --init 2024-01-02T00 --lead 9h --lead-zero
  ```

* `tools/compare_results.py` compares two result files element by element and reports each raw
  sum and metric as bit-exact or by its maximum relative difference. Use it to show that a
  change is a no-op (performance work, refactors) and to check sharded runs against unsharded
  ones.
* `tools/inspect_results.py` checks the mechanics of a result file (weights, exclusions, metric
  inequalities, lead 0) and prints its headline numbers.
* `tools/ab_prefetch.py` times prefetch depths, cache sizes and blosc settings and checks that
  the results stay bit-identical. Use it for I/O changes.
* `tools/make_climatology.py` builds the climatology netcdf the `acc` metric needs.

The GPU scripts belong on a compute node with a GPU. Both set
`ANEMOI_INFERENCE_NUM_CHUNKS` to 8 unless the environment already sets it.

Performance claims belong in [`benchmarks.md`](benchmarks.md), with the testbed and the
environment they were measured on; do not add numbers elsewhere.

## Documentation

The document set is the source of truth for behaviour, and a change that alters behaviour
updates it in the same pull request:

* [`design/requirements.md`](design/requirements.md): what the package must do, and why.
* [`design/architecture.md`](design/architecture.md): how it does it.
* [`user-guide.md`](user-guide.md): task-oriented guide.
* [`reference.md`](reference.md): CLI, config keys, Python API, result-file layout.
* [`benchmarks.md`](benchmarks.md): measured numbers.
* [`../README.md`](../README.md): overview and quick start.

In particular, the result file is an interface: any change to its variables, coordinates or
attributes is a change to `reference.md`.

## Commits and pull requests

* Commits on `main` use [Conventional Commits](https://www.conventionalcommits.org/) titles:
  `feat:`, `fix:`, `perf:`, `docs:`, `refactor:`, `chore:`, with `!` for a breaking change
  (`feat!: ...`). Pull requests are merged with such a
  title.
* Every user-visible change adds an entry to [`CHANGELOG.md`](../CHANGELOG.md) in the same
  change. The file is grouped by release, newest first, with a short paragraph of prose under
  the version heading and `### Breaking`, `### Features`, `### Fixes` sections. Entries are full sentences that say what
  changed for the user, not what was edited. Unreleased work goes under an `## Unreleased`
  heading. Internal refactors and test-only changes need no entry.
* CI (`.github/workflows/ci.yml`) runs on every pull request and on pushes to `main`: a
  `quality` job that runs `pre-commit run --all-files` on Python 3.12, and a `tests` job on
  Python 3.11 and 3.13 that installs the package with `--torch-backend=cpu`, asserts that no
  `nvidia-*` or `triton` wheel was pulled in, and runs `pytest tests -q`. Both must pass.

## Releases

The version comes from `setuptools-scm`, so a release is a tag:

1. Turn the `## Unreleased` heading in `CHANGELOG.md` into a `## x.y.z` heading with the
   release date, in the form the existing entries use, add a short paragraph summarising the
   release, and commit it as `chore(release): x.y.z`.
2. Tag that commit `vx.y.z` (annotated).

The package is not published to PyPI, and there is no release workflow in
`.github/workflows/`; releases exist as tags and changelog entries only.

## Proposing changes

There is no template and no CLA.

* Open an issue at
  [MeteoSwiss/anemoi-evaluation/issues](https://github.com/MeteoSwiss/anemoi-evaluation/issues)
  for bugs and proposals. A bug report is most useful with the config, the command line and
  the log; for a scoring question, the result file or the `tools/inspect_results.py` output.
* Send pull requests against `main` of MeteoSwiss/anemoi-evaluation. Keep them one topic each,
  with the changelog entry, the documentation update and, where the scoring path is touched,
  the `tools/validate.py` or `tools/compare_results.py` output in the description.
