"""Check the mechanics of a results file and print its headline numbers.

    python tools/inspect_results.py <results.nc>

CPU only, a second or two. Hard checks: n_init, weight sums consistent with the region weight totals (the implied
excluded fraction is the NaN fraction; at lead 0 diagnostic variables are excluded entirely and prognostic scores are
exactly zero when the targets come from the input dataset), mae <= rmse, no NaN metric at leads > 0, and for ensembles
spread > 0, crps > 0, fair_crps <= crps, member_rmse >= rmse, member_mae >= mae; acc within [-1, 1].

Prints the attrs, the lead times, regions and variables, n_init per lead, the per-region weight total of one init time,
the excluded-node fraction per lead, the range of every metric present, the inequality checks, the 2t / z_500 / tp
values per lead and region, and a final `MECHANICS OK` or `MECHANICS FAILED`.
"""

import argparse

import numpy as np
import xarray as xr

parser = argparse.ArgumentParser()
parser.add_argument("results", help="Results netcdf written by anemoi-evaluation.")
args = parser.parse_args()

ds = xr.open_dataset(args.results, decode_timedelta=True).load()
leads = [int(x / np.timedelta64(1, "h")) for x in ds["lead_time"].values]
positive = ds["lead_time"].values > np.timedelta64(0)
season = "all"  # the derived bin
print("file:", args.results)
print("attrs:", {k: v for k, v in ds.attrs.items() if k not in ("config",)})
print("lead hours:", leads, "regions:", list(ds["region"].values), "variables:", list(ds["variable"].values))
n_init = ds["n_init"].sel(bin=season).values
print("n_init per lead:", n_init.tolist())
per_init = ds["weight_sum"].sel(bin=season) / ds["n_init"].sel(bin=season)
full = per_init.max(dim=("lead_time", "variable"))  # per region: the total weight when every node is finite
print("region weight totals per init time:", {str(r): float(full.sel(region=r)) for r in ds["region"].values})
ok = True
for i, lead in enumerate(leads):
    frac = 1 - per_init.isel(lead_time=i) / full
    worst = float(frac.max())
    if worst > 1e-12:
        where = frac.where(frac == frac.max(), drop=True)
        note = " (expected at lead 0: no analysis for diagnostic variables)" if lead == 0 else ""
        print(
            f"  lead {lead:3d}h: excluded (NaN) weight fraction max {worst:.3e} at",
            [str(v) for v in where["variable"].values],
            [str(r) for r in where["region"].values],
            note,
        )
    else:
        print(f"  lead {lead:3d}h: no excluded nodes (weight_sum == n_init * total for every variable/region)")
if 0 in leads:
    zero = ds.sel(bin=season).isel(lead_time=leads.index(0))
    finite = np.isfinite(zero["rmse"].values)
    worst = float(np.nanmax(np.abs(zero["rmse"].values))) if finite.any() else float("nan")
    nan_vars = [str(v) for v in zero["variable"].values[~finite.all(axis=1)]]
    print(
        f"lead 0: max |rmse| over finite entries {worst:.3e} (must be 0 when targets == inputs); NaN variables {nan_vars}"
    )
    ok &= worst == 0.0
for name in ("rmse", "mae", "bias", "crps", "fair_crps", "spread", "spread_skill", "member_rmse", "member_mae", "acc"):
    if name not in ds:
        continue
    a = ds[name].sel(bin=season)
    bad = int(np.isnan(a.values[positive]).sum())
    print(f"{name}: NaN entries at leads > 0: {bad}; min {float(a.min()):.4g} max {float(a.max()):.4g}")
    ok &= bad == 0
if "mae" in ds:
    viol = float((ds["mae"] - ds["rmse"]).sel(bin=season).max())
    print("max(mae - rmse):", f"{viol:.3e}", "(must be <= 0)")
    ok &= viol <= 1e-9
if "member_rmse" in ds:
    viol = float((ds["rmse"] - ds["member_rmse"]).sel(bin=season).max())
    print("max(rmse - member_rmse):", f"{viol:.3e}", "(must be <= 0)")
    ok &= viol <= 1e-9
if "member_mae" in ds:
    viol = float((ds["mae"] - ds["member_mae"]).sel(bin=season).max())
    print("max(mae - member_mae):", f"{viol:.3e}", "(must be <= 0)")
    ok &= viol <= 1e-9
if "acc" in ds:
    a = ds["acc"].sel(bin=season).values[positive]
    print(f"acc: min {np.nanmin(a):.4f} max {np.nanmax(a):.4f} (within [-1, 1])")
    ok &= bool(np.nanmin(a) >= -1 - 1e-12 and np.nanmax(a) <= 1 + 1e-12)
if "spread" in ds:
    sp = ds["spread"].sel(bin=season).values[positive]
    cr = ds["crps"].sel(bin=season).values[positive]
    print("min spread (leads > 0):", float(sp.min()), "min crps:", float(cr.min()))
    print(
        "max(fair_crps - crps):", f"{float((ds['fair_crps'] - ds['crps']).sel(bin=season).max()):.3e}", "(must be <= 0)"
    )
    ok &= float(sp.min()) > 0 and float(cr.min()) > 0
    ok &= float((ds["fair_crps"] - ds["crps"]).sel(bin=season).max()) <= 1e-9
for var in ("2t", "z_500", "tp"):
    if var not in ds["variable"].values:
        continue
    for name in ("rmse", "spread", "fair_crps", "member_rmse", "acc"):
        if name in ds:
            line = ds[name].sel(bin=season, variable=var)
            print(
                f"{name} {var}:",
                {
                    r: [f"{float(line.sel(region=r).isel(lead_time=i)):.4g}" for i in range(len(leads))]
                    for r in ds["region"].values
                },
            )
print("MECHANICS", "OK" if ok else "FAILED")
