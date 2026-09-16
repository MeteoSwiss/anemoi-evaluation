"""Compare two results files element by element, to show that a change left the numbers alone.

    python tools/compare_results.py <a.nc> <b.nc> [--leads common] [--stats common]

CPU only, a second or two. By default coordinates, n_init, weight sums and init times must be identical; with
`--leads common` (and `--stats common`) the comparison runs on the lead times (and statistics) both files share, so a
file with extra leads (lead 0) or extra statistics can be checked against an older one.

Prints the number of leads and the statistics compared, whether the coordinates, members, init times, `n_init` and
weights are identical, one line per raw sum and per metric saying `bit-exact` or the maximum relative difference, the
timing and counter attrs of both files, and a final `STRUCTURE OK` or `STRUCTURE DIFFERS` (also the exit status, 0 or 1).
"""

import argparse
import sys

import numpy as np
import torch
import xarray as xr

from anemoi.evaluation.output import load_state

parser = argparse.ArgumentParser()
parser.add_argument("a")
parser.add_argument("b")
parser.add_argument("--leads", choices=["all", "common"], default="all")
parser.add_argument("--stats", choices=["all", "common"], default="all")
args = parser.parse_args()
a, b = (load_state(path) for path in (args.a, args.b))
leads = [lead for lead in a.lead_times if lead in b.lead_times] if args.leads == "common" else a.lead_times
ia, ib = [a.lead_times.index(lead) for lead in leads], [b.lead_times.index(lead) for lead in leads]
names = [name for name in a.sums if name in b.sums] if args.stats == "common" else list(a.sums)
ok = a.coords[1:] == b.coords[1:] and a.members == b.members and a.init_times == b.init_times
ok &= bool(leads) and (args.leads == "common" or a.lead_times == b.lead_times)
ok &= args.stats == "common" or set(a.sums) == set(b.sums)
ok &= torch.equal(a.n_init[ia], b.n_init[ib]) and torch.equal(a.weights[ia], b.weights[ib])
print(f"leads compared: {len(leads)} of {len(a.lead_times)} / {len(b.lead_times)}; statistics: {names}")
print("coords/members/init_times/n_init/weights identical:", ok)
for name in names:
    x, y = a.sums[name][ia], b.sums[name][ib]
    exact = torch.equal(x, y)
    rel = float(((x - y).abs() / x.abs().clamp_min(1e-30)).max())
    print(f"sum {name}: {'bit-exact' if exact else f'max relative difference {rel:.3e}'}")
da, db = (xr.open_dataset(path, decode_timedelta=True).load() for path in (args.a, args.b))
lead_values = np.array(leads, dtype="timedelta64[ns]")
for name in da.data_vars:
    if name.startswith("state_") or name in ("n_init", "weight_sum") or name not in db:
        continue
    x, y = da[name].sel(lead_time=lead_values).values, db[name].sel(lead_time=lead_values).values
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.nanmax(np.abs(x - y) / np.maximum(np.abs(x), 1e-30))
    print(
        f"metric {name}: {'bit-exact' if np.array_equal(x, y, equal_nan=True) else f'max relative difference {rel:.3e}'}"
    )
print(
    "attrs a:", {k: v for k, v in a.attrs.items() if k.startswith(("time_", "peak", "shard", "target", "model_calls"))}
)
print(
    "attrs b:", {k: v for k, v in b.attrs.items() if k.startswith(("time_", "peak", "shard", "target", "model_calls"))}
)
print("STRUCTURE", "OK" if ok else "DIFFERS")
sys.exit(0 if ok else 1)
