"""Check the mechanics of a results file and print its headline numbers.

    python tools/inspect_results.py <results.nc>

CPU only, a second or two. Hard checks: n_init, weight sums consistent with the region weight totals (the implied
excluded fraction is the NaN fraction; at lead 0 diagnostic variables are excluded entirely and prognostic scores are
exactly zero when the targets come from the input dataset), mae <= rmse, no NaN metric at leads > 0, and for ensembles
spread > 0, crps > 0, fair_crps <= crps, member_rmse >= rmse, member_mae >= mae; acc within [-1, 1]. For every
threshold label found in the file: the score ranges (pod, far, csi, brier and the base rate within [0, 1], ets within
[-1/3, 1], hss and pss within [-1, 1], bss <= 1, frequency_bias >= 0), csi <= pod, csi <= 1 - far, the four-cell
identity hit + miss + false_alarm <= state_weights, and, at one member, brier == miss + false_alarm to 1e-12
relative (the identity is exact per element, the sums group the terms differently). For a rank histogram: every bin
within [0, 1], the bins summing to 1 to 1e-12 relative, `outlier_fraction` equal to the two end bins, the `rank`
coordinate matching the file's `members` attr, and the raw sums adding up to `state_weights` to 1e-9 relative.
For every reliability label: the `probability` coordinate matching `k / M`, the diagram and the
forecast-probability distribution within [0, 1], that distribution summing to 1 to 1e-12 relative, the raw level
sums adding up to `state_weights` to 1e-9 relative and NaN for the variables without a threshold, no NaN
component at leads > 0, `brier_reliability` and `brier_resolution` non-negative, `brier_uncertainty` within
[0, 0.25], and `brier_reliability - brier_resolution + brier_uncertainty == brier` to 1e-12 relative.

Prints the attrs, the lead times, regions and variables, n_init per lead, the per-region weight total of one init time,
the excluded-node fraction per lead, the range of every metric present, the inequality checks, the 2t / z_500 / tp
values per lead and region, the threshold labels with their scored variables, their base rates and the variables they
have no threshold for, the rank histogram at the last lead with its outlier fraction against the calibrated
`2 / (M + 1)`, the reliability curve of 2t / z_500 / tp at the last lead next to the diagonal with the three
Brier components, and a final `MECHANICS OK` or `MECHANICS FAILED`.
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
THRESHOLD_KINDS = ("pod", "far", "csi", "ets", "frequency_bias", "hss", "pss", "brier", "bss", "event_frequency")
BOUNDS = {
    "pod": (0.0, 1.0),
    "far": (0.0, 1.0),
    "csi": (0.0, 1.0),
    "brier": (0.0, 1.0),
    "event_frequency": (0.0, 1.0),
    "ets": (-1 / 3, 1.0),
    "hss": (-1.0, 1.0),
    "pss": (-1.0, 1.0),
    "bss": (-np.inf, 1.0),
    "frequency_bias": (0.0, np.inf),
}
RELIABILITY_KINDS = (
    "reliability",
    "forecast_frequency",
    "brier_reliability",
    "brier_resolution",
    "brier_uncertainty",
)
KINDS = tuple(sorted(THRESHOLD_KINDS + RELIABILITY_KINDS, key=len, reverse=True))


def split_kind(name):
    """The (kind, label) of a metric name, longest kind first so `brier_reliability_x` is not `brier` on label
    `reliability_x`. This is a heuristic on a flat name space: a label that itself begins with a metric kind can
    still be read wrong, so do not write one."""
    for kind in KINDS:
        if name.startswith(kind + "_"):
            return kind, name[len(kind) + 1 :]
    return None, None


split = [split_kind(name) for name in ds.data_vars]
labels = sorted({label for kind, label in split if kind in THRESHOLD_KINDS})
reliability_labels = sorted({label for kind, label in split if kind in RELIABILITY_KINDS})


def extremes(values):
    """Smallest and largest of the entries that are not NaN, or NaN when there are none."""
    known = values[~np.isnan(values)]
    return (float(known.min()), float(known.max())) if known.size else (float("nan"), float("nan"))


for label in labels:
    present = {kind: ds[f"{kind}_{label}"].sel(bin=season) for kind in THRESHOLD_KINDS if f"{kind}_{label}" in ds}
    variables = [str(v) for v in ds["variable"].values]
    # A variable the threshold map does not name is NaN in the raw sums; a variable with a threshold but no observed
    # event is finite there while its `pod` is all NaN, so the sums are the discriminator when the file has them.
    # Only the populated cells count: a bin no frame fell into holds the exact zeros of the empty state, not NaN.
    sums = [f"state_sum_{kind}_{label}" for kind in ("event_frequency", "hit")]
    sample = next((ds[name].values for name in sums if name in ds), None)
    if sample is None:
        sample = next(iter(present.values())).values  # (lead_time, variable, region)
        scored = ~np.isnan(sample).all(axis=(0, 2))
    else:
        populated = ds["state_weights"].values > 0
        scored = (np.isfinite(sample) & populated).any(axis=(0, 1, 3))  # (lead_time, state_bin, variable, region)
    print(f"threshold label {label!r}: scored variables {[v for v, s in zip(variables, scored) if s]}")
    print("  no threshold (all NaN):", [v for v, s in zip(variables, scored) if not s])
    if not scored.any():
        ok = False
        continue
    for kind, values in present.items():
        a = values.values[positive][:, scored, :]
        low, high = BOUNDS[kind]
        bad = int(np.isnan(a).sum())
        smallest, largest = extremes(a)
        print(f"  {kind}_{label}: NaN entries at leads > 0: {bad}; min {smallest:.4g} max {largest:.4g}")
        ok &= bool(np.isnan(smallest) or (smallest >= low - 1e-9 and largest <= high + 1e-9))
    if "csi" in present and "pod" in present:
        viol = extremes((present["csi"] - present["pod"]).values[positive][:, scored, :])[1]
        print(f"  max(csi - pod): {viol:.3e} (must be <= 0)")
        ok &= bool(np.isnan(viol) or viol <= 1e-9)
    if "csi" in present and "far" in present:
        viol = extremes((present["csi"] + present["far"]).values[positive][:, scored, :])[1] - 1.0
        print(f"  max(csi - (1 - far)): {viol:.3e} (must be <= 0)")
        ok &= bool(np.isnan(viol) or viol <= 1e-9)
    if "event_frequency" in present:
        rate = present["event_frequency"].values[positive][:, scored, :]
        print(
            "  base rate per scored variable:",
            {
                v: "{:.3g} to {:.3g}".format(*extremes(rate[:, j]))
                for j, v in enumerate([v for v, s in zip(variables, scored) if s])
            },
        )
    cells = [f"state_sum_{kind}_{label}" for kind in ("hit", "miss", "false_alarm")]
    if all(name in ds for name in cells):
        total = sum(ds[name] for name in cells).values[:, :, scored, :]
        weights = ds["state_weights"].values[:, :, scored, :]
        excess = extremes(np.where(weights > 0, total - weights, 0.0) / np.where(weights > 0, weights, 1.0))[1]
        print(f"  max((hit + miss + false_alarm) / weights - 1): {excess:.3e} (must be <= 0)")
        ok &= bool(np.isnan(excess) or excess <= 1e-9)
    brier, misses = f"state_sum_brier_{label}", [f"state_sum_{kind}_{label}" for kind in ("miss", "false_alarm")]
    if int(ds.attrs["members"]) == 1 and brier in ds and all(name in ds for name in misses):
        left, right = ds[brier].values, sum(ds[name] for name in misses).values
        agree = bool(np.allclose(left, right, rtol=1e-12, atol=0.0, equal_nan=True))
        print(f"  brier == miss + false_alarm at one member (to 1e-12): {agree} (must be True)")
        ok &= agree
if "rank_histogram" in ds:
    # The rank histogram carries an extra `rank` axis, so the four-axis idioms above do not apply to it. Lead 0 is
    # excluded: every member equals every other one there, so the target is either tied with all of them (flat, when
    # the targets come from the input dataset) or outside all of them (the two end bins), neither a calibration fact.
    members = int(ds.attrs["members"])
    h = ds["rank_histogram"].sel(bin=season).values[positive]  # (lead_time, variable, region, rank)
    shape_ok = h.shape[-1] == members + 1 and ds["rank"].values.tolist() == list(range(members + 1))
    print(f"rank histogram: {h.shape[-1]} bins for {members} members, rank coordinate 0..M: {shape_ok} (must be True)")
    ok &= shape_ok
    finite = np.isfinite(h).all(axis=-1)
    bad = int((~finite).sum())
    print(f"  NaN entries at leads > 0: {bad} of {finite.size} (must be 0)")
    ok &= bad == 0
    values = h[finite]
    if values.size:
        smallest, largest = float(values.min()), float(values.max())
        print(f"  bins: min {smallest:.4g} max {largest:.4g} (within [0, 1])")
        ok &= bool(smallest >= -1e-9 and largest <= 1 + 1e-9)
        deviation = float(np.abs(values.sum(-1) - 1.0).max())
        print(f"  max |sum of the bins - 1|: {deviation:.3e} (must be <= 1e-12 relative)")
        ok &= bool(np.allclose(values.sum(-1), 1.0, rtol=1e-12, atol=0.0))
        if "outlier_fraction" in ds:
            outliers = ds["outlier_fraction"].sel(bin=season).values[positive][finite]
            ends = values[..., 0] + values[..., -1]
            agree = bool(np.allclose(outliers, ends, rtol=1e-12, atol=0.0))
            print(f"  outlier_fraction == first + last bin (to 1e-12): {agree} (must be True)")
            ok &= agree
            calibrated = 2.0 / (members + 1)
            print(
                f"  outlier fraction {float(outliers.min()):.4g} to {float(outliers.max()):.4g}, "
                f"calibrated {calibrated:.4g}, ratio {float(outliers.mean()) / calibrated:.3g} "
                "(above 1 under-dispersed, below 1 over-dispersed)"
            )
    sums = [f"state_sum_rank_bin_{k}" for k in range(members + 1)]
    if all(name in ds for name in sums):
        total = sum(ds[name] for name in sums).values
        weights = ds["state_weights"].values
        rel = np.abs(np.where(weights > 0, total - weights, 0.0)) / np.where(weights > 0, weights, 1.0)
        worst = float(np.nanmax(rel))
        print(f"  max |sum of the raw rank sums / state_weights - 1|: {worst:.3e} (must be <= 1e-9)")
        ok &= bool(np.isnan(worst) or worst <= 1e-9)
    for var in ("2t", "z_500", "tp"):
        if var in ds["variable"].values:
            last = ds["rank_histogram"].sel(bin=season, variable=var).isel(lead_time=-1)
            print(
                f"  {var} at lead {leads[-1]}h:",
                {str(r): [f"{v:.3g}" for v in last.sel(region=r).values] for r in ds["region"].values},
            )
for label in reliability_labels:
    # The diagram carries an extra `probability` axis, so the four-axis idioms of the threshold block do not apply.
    # Lead 0 is excluded: every member equals every other one there, so only the two end levels are populated.
    members = int(ds.attrs["members"])
    levels = np.arange(members + 1) / members
    present = {kind: ds[f"{kind}_{label}"].sel(bin=season) for kind in RELIABILITY_KINDS if f"{kind}_{label}" in ds}
    counts = [f"state_sum_reliability_count_{label}_{k}" for k in range(members + 1)]
    # The raw sums are the discriminator, as in the threshold block: the level sums when the label has a level metric,
    # otherwise the base rate, which `brier_uncertainty` alone still stores. Only the populated cells count.
    sums = [counts[0], f"state_sum_event_frequency_{label}"]
    sample = next((ds[name].values for name in sums if name in ds), None)
    if sample is None:
        values = next(iter(present.values())).values  # (lead_time, variable, region) plus a probability axis or not
        scored = ~np.isnan(values).all(axis=tuple(axis for axis in range(values.ndim) if axis != 1))
    else:
        scored = (np.isfinite(sample) & (ds["state_weights"].values > 0)).any(axis=(0, 1, 3))
    variables = [str(v) for v in ds["variable"].values]
    print(f"reliability label {label!r}: scored variables {[v for v, s in zip(variables, scored) if s]}")
    print("  no threshold (all NaN):", [v for v, s in zip(variables, scored) if not s])
    if not scored.any():
        ok = False
        continue
    for kind in ("reliability", "forecast_frequency"):
        if kind not in present:
            continue
        values = present[kind].values[positive][:, scored, :, :]  # (lead_time, variable, region, probability)
        shape_ok = values.shape[-1] == members + 1 and np.array_equal(ds["probability"].values, levels)
        print(f"  {kind}_{label}: {values.shape[-1]} levels for {members} members, probability k/M: {shape_ok}")
        ok &= shape_ok
        finite = values[np.isfinite(values)]
        if finite.size:
            smallest, largest = float(finite.min()), float(finite.max())
            print(f"    min {smallest:.4g} max {largest:.4g} (within [0, 1])")
            ok &= bool(smallest >= -1e-9 and largest <= 1 + 1e-9)
        if kind == "forecast_frequency":
            rows = values[np.isfinite(values).all(axis=-1)]
            if rows.size:
                deviation = float(np.abs(rows.sum(-1) - 1.0).max())
                print(f"    max |sum over the levels - 1|: {deviation:.3e} (must be <= 1e-12 relative)")
                ok &= bool(np.allclose(rows.sum(-1), 1.0, rtol=1e-12, atol=0.0))
    for kind in ("brier_reliability", "brier_resolution", "brier_uncertainty"):
        if kind not in present:
            continue
        values = present[kind].values[positive][:, scored, :]
        bad = int(np.isnan(values).sum())
        low, high = (0.0, np.inf) if kind != "brier_uncertainty" else (0.0, 0.25)
        smallest, largest = extremes(values)
        print(f"  {kind}_{label}: NaN entries at leads > 0: {bad}; min {smallest:.4g} max {largest:.4g}")
        ok &= bad == 0
        ok &= bool(np.isnan(smallest) or (smallest >= low - 1e-9 and largest <= high + 1e-9))
    if all(name in ds for name in counts):
        total = sum(ds[name] for name in counts).values[:, :, scored, :]
        weights = ds["state_weights"].values[:, :, scored, :]
        rel = np.abs(np.where(weights > 0, total - weights, 0.0)) / np.where(weights > 0, weights, 1.0)
        worst = float(np.nanmax(rel))
        print(f"  max |sum of the raw level sums / state_weights - 1|: {worst:.3e} (must be <= 1e-9)")
        ok &= bool(np.isnan(worst) or worst <= 1e-9)
        # Only the populated cells: a bin no frame fell into holds the exact zeros of the empty state, not NaN.
        missing = sum(ds[name] for name in counts).values[:, :, ~scored, :]
        unscored = np.isnan(missing[(ds["state_weights"].values > 0)[:, :, ~scored, :]])
        print(f"  levels NaN for the {int((~scored).sum())} variables without a threshold: {bool(unscored.all())}")
        ok &= bool(unscored.all())
    components = [f"brier_{kind}_{label}" for kind in ("reliability", "resolution", "uncertainty")]
    if f"brier_{label}" in ds and all(name in ds for name in components):
        left = (ds[components[0]] - ds[components[1]] + ds[components[2]]).sel(bin=season).values[positive]
        right = ds[f"brier_{label}"].sel(bin=season).values[positive]
        agree = bool(np.allclose(left[:, scored], right[:, scored], rtol=1e-12, atol=0.0, equal_nan=True))
        print(f"  brier_reliability - brier_resolution + brier_uncertainty == brier (to 1e-12): {agree}")
        ok &= agree
    for var in ("2t", "z_500", "tp"):
        if var not in ds["variable"].values or "reliability" not in present:
            continue
        last = present["reliability"].sel(variable=var).isel(lead_time=-1)
        for region in ds["region"].values:
            curve = last.sel(region=region).values
            print(f"  {var} {region} at lead {leads[-1]}h: observed {[f'{v:.3g}' for v in curve]}")
            print(f"    forecast {[f'{v:.3g}' for v in levels]}")
        for kind in ("brier_reliability", "brier_resolution", "brier_uncertainty"):
            if kind in present:
                line = present[kind].sel(variable=var).isel(lead_time=-1)
                print(f"  {kind} {var}:", {str(r): f"{float(line.sel(region=r)):.4g}" for r in ds["region"].values})
print("MECHANICS", "OK" if ok else "FAILED")
