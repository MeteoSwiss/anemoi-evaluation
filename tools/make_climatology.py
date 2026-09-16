"""Build an hour-of-day (or month / day-of-year / constant) climatology netcdf for the `acc` metric.

    python tools/make_climatology.py <config.yaml> --start 2024-01-01 --end 2024-01-31T18
                                     [--key hour_of_day] [--prefetch 4] -o <climatology.nc>

No GPU needed, but one zarr row per date: a month of six-hourly dates takes a few minutes off Lustre.

Reads every dataset date in [start, end] once through `DatasetTargets` (worker prefetch) — the dataset of the config's
`forecast.anemoi_inference.input.dataset` block, for the config's `variables` — accumulates float64 sums per key and
writes float32 means in the layout `ArrayClimatology.from_netcdf` reads (key, variable, values). The file records the
window, the counts per key and the dataset arguments, and the script asserts that no value is non-finite.

Prints `CLIM:` lines: the dates and variables it will read, the counts per key, the read time and target-source
counters, the output path and size, and the field mean per key for every variable (a physical-plausibility eyeball).
"""

import argparse
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

import numpy as np  # noqa: E402
from anemoi.utils.dates import as_datetime  # noqa: E402

from anemoi.evaluation.config import load_config  # noqa: E402
from anemoi.evaluation.sources.anemoi_dataset import DatasetTargets  # noqa: E402
from anemoi.evaluation.sources.climatology import ArrayClimatology  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("--start", required=True)
parser.add_argument("--end", required=True)
parser.add_argument("--key", default="hour_of_day")
parser.add_argument("--prefetch", type=int, default=4)
parser.add_argument("-o", "--output", required=True)
args = parser.parse_args()

config = load_config(args.config)
block = config.forecast.anemoi_inference.model_dump()
targets = DatasetTargets(**block["input"]["dataset"], prefetch=args.prefetch)
start, end = as_datetime(args.start), as_datetime(args.end)
dates = [date for date in targets._dates if start <= date <= end and targets.available(date)]
variables = list(config.variables)
print(f"CLIM: {len(dates)} dates {dates[0]} .. {dates[-1]}, {len(variables)} variables, key {args.key}", flush=True)
t0 = time.perf_counter()
climatology = ArrayClimatology.from_targets(targets, dates, variables, args.key)
seconds = time.perf_counter() - t0
nonfinite = climatology.nonfinite_nodes(variables)
assert not any(nonfinite.values()), nonfinite
out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
climatology.attrs.update(window=f"{dates[0].isoformat()} .. {dates[-1].isoformat()}", built_from=str(args.config))
climatology.to_netcdf(out)
print(
    f"CLIM: counts {climatology.counts}, {seconds:.1f} s ({targets.stats}), wrote {out} ({out.stat().st_size / 2**20:.0f} MiB)",
    flush=True,
)
for name in variables:
    values = np.stack([climatology.fields[k][climatology.variables.index(name)] for k in sorted(climatology.fields)])
    print(f"CLIM: {name}: mean over keys {values.mean(axis=1)}", flush=True)
targets.close()
