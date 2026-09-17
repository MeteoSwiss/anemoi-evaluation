# Two-dataset test fixture

A small, self-contained stand-in for a checkpoint trained on two datasets: the global ERA5 n320
analysis and the MeteoSwiss 1 km LAM `realch1`, both at a 6 h timestep. No zarr is opened at test
time.

| dataset | source zarr | variables | grid points | kinds |
|---|---|---|---|---|
| `era5` | `aifs-ea-an-oper-0001-mars-n320-1979-2024-6h-v1-for-single-v2.zarr` | 16 | 288 | 9 forcings, 1 diagnostic |
| `realch1` | `mch-realch1-fdb-1km-2005-2025-1h-pl13-v1.0.zarr` | 15 | 176 | 8 forcings, 1 diagnostic |

`realch1` is opened with `rename:` to the ECMWF parameter names, so the two variable namespaces
collide (`2t`, `10u`, `10v`, `msl`, `t_850`, `q_850`, `tp`
appear in both) -- which is the point of the fixture.

Contents:

* `metadata.json` -- the checkpoint metadata as anemoi-training writes it for a multi-dataset run:
  `dataset[name]` (real variables, `variables_metadata`, arguments, dates), `data_indices[name]`
  (data and model sides) and `metadata_inference` with `dataset_names` and, per dataset,
  `timesteps`, `data_indices`, `variable_types` and `shapes`. `config.model.encoders/decoders`
  route every dataset through its own encoder and decoder (a forecaster, not a downscaler).
* `arrays.npz` -- per dataset `<name>.latitudes`, `<name>.longitudes` (real, `float64`, for the
  retained grid points), `<name>.grid_points` (the indices into the full grid) and the real
  per-variable statistics `<name>.mean`, `.stdev`, `.minimum`, `.maximum`.

Dates: 2020-01-01T00:00:00 to 2020-01-05T00:00:00 every 6h.

## Provenance and licence

The grid coordinates and the per-variable statistics are the real ones, derived from the two
datasets named above: the ERA5 n320 analysis dataset and the MeteoSwiss `realch1` dataset. ERA5 is
Copernicus data (Copernicus Climate Change Service information, ECMWF); the `realch1` statistics are
derived data of a MeteoSwiss dataset. No field values are committed and no test opens a zarr; the
datasets are named by their zarr basenames as provenance. The script that generated the fixture
lives outside this repository: it reads the metadata, the statistics and the coordinates of the two
zarrs, nothing else, and is deterministic, so regenerating an unchanged fixture produces
byte-identical files.

## Deviations from a real checkpoint

The metadata is faithful to what anemoi-training writes for a two-dataset forecaster, except:

* `dataset[name]` keeps only the keys anemoi-inference reads; `specific`, `sources` and the
  provenance blocks are dropped.
* `config` holds only the `data`, `training`, `dataloader` and `model` blocks inference reads.
* `shapes.variables` is computed as `2 * input columns + 12`; the real value
  depends on the training graph's node attributes, which the fixture does not carry. Nothing in
  anemoi-inference reads it.
* The grids are 288 and 176 points sampled with
  `np.linspace` from the real grids, so they are not a contiguous region of either dataset.
