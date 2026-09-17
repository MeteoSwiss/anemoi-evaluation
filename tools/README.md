# tools

Scripts used to validate and profile the framework. They are not part of the package: run them
with the repository checked out, from the package directory, with `anemoi-evaluation` installed.
Every path they touch is an argument; each script's module docstring says what it prints.

| script | what it does | GPU |
|---|---|---|
| `validate.py` | Scores one init time both through the framework and through a direct `runner.run()` loop reduced in numpy float64, and reports the relative difference per metric, threshold metrics, the rank histogram and the reliability levels included. The reference for exactness. | yes |
| `ab_prefetch.py` | Runs the same config several times with different target prefetch depths, cache sizes and blosc settings, and reports the timing of each and whether the results are bit-identical. | yes |
| `make_climatology.py` | Builds an hour-of-day (or month / day-of-year / constant) climatology netcdf from a config's target dataset, for the `acc` metric. | no |
| `inspect_results.py` | Checks the mechanics of a results file (weights, exclusions, metric inequalities, lead 0, the threshold labels and their contingency identities, the rank histogram and its sum-to-one identity, the reliability diagram and the Brier decomposition identity) and prints its headline numbers. | no |
| `compare_results.py` | Compares two results files element by element, reporting each raw sum and metric as bit-exact or by its maximum relative difference. | no |

The two GPU scripts load a real checkpoint and run it, so they belong on a compute node with a
GPU (on balfrin, `scripts/submit.sh` of the aggregation repository); the other three run
anywhere in a few seconds. `validate.py` and `ab_prefetch.py` set `ANEMOI_INFERENCE_NUM_CHUNKS`
to 8 unless the environment already sets it.

```bash
python tools/validate.py config.yaml --init 2024-01-02T00 --steps 2 --members 1 4
python tools/validate.py config.yaml --init 2024-01-02T00 --lead 9h --lead-zero
python tools/ab_prefetch.py config.yaml --variants p2 p6 p6c2 p12 p6t
python tools/make_climatology.py config.yaml --start 2024-01-01 --end 2024-01-31T18 -o climatology.nc
python tools/inspect_results.py results.nc
python tools/compare_results.py results-a.nc results-b.nc --leads common --stats common
```
