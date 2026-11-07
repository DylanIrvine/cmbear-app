# CMBEAR app

A Streamlit web app that estimates diffuse groundwater recharge for Australian bores with the
chloride mass balance (CMB) method, and reads the matching values from the national random forest
recharge grids of Lee et al. (2024).

The user supplies a spreadsheet with four columns (**ID, Lat, Lon, Clgw** in mg/L; optional
**Clgw_sd**). For each bore the app

1. finds the 0.05 degree grid cell and extracts rainfall, PET, aridity, runoff coefficient and
   chloride deposition statistics (mean, 5th, 95th, sd, skew);
2. computes the regular CMB recharge, R = 100 D / Cl_gw, with 5th and 95th percentile bounds;
3. computes the stochastic CMBEAR2 recharge with a runoff term, R = 100 D (1 - alpha RC) / Cl_gw,
   over 1000 realisations and reports the 5th, 50th, 95th percentiles and the mean;
4. reads the random forest recharge (R5, R50, R95) and recharge/rainfall (RP5, RP50, RP95) grids;
5. offers CSV and Excel downloads and four sets of plots: a bore map over the national grid with an
   Australia inset, CMB vs RF, recharge vs climate, and the per-bore realisation histogram.

Results reproduce `CMBEAR2_v1.0.py` to float32 precision for the same inputs (see
`tests/` and the Method tab in the app for the documented differences).

## Repository layout

```
app.py                  Streamlit user interface
cmbear/                 Calculation package (no Streamlit dependency)
  grids.py              Grid archive loading and pixel lookup
  inputs.py             Spreadsheet parsing, column aliases, QA
  cmb.py                Regular and stochastic CMB, RF extraction, gap filling
  plotting.py           Bore map (plotly)
data/grids.npz          15 national grids, float32, compressed (13.6 MB)
scripts/build_grids.py  Rebuilds data/grids.npz from the Lee et al. (2024) ASCII grids
examples/               Example input file
tests/                  pytest regression tests
```

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 and upload `examples/example_input.csv`.

Tests: `python -m pytest tests`

## Rebuilding the grid archive

`data/grids.npz` is derived from the ASCII grids distributed with Lee et al. (2024)
(HydroShare: https://www.hydroshare.org/resource/5e7b8bfcc1514680902f8ff43cc254b8/). To rebuild:

```bash
python scripts/build_grids.py --lee2024 "path/to/Lee2024"
```

The script checks that all grids share one geometry (841 x 681 cells, 0.05 degrees,
lower-left corner 111.975 E, -44.025 S, NODATA -999) and stores that geometry in the archive.

## Input file

| Column   | Required | Meaning                                              |
|----------|----------|------------------------------------------------------|
| ID       | yes      | Bore or sample identifier. Repeat IDs are averaged.  |
| Lat      | yes      | Latitude, decimal degrees (negative in Australia)    |
| Lon      | yes      | Longitude, decimal degrees                           |
| Clgw     | yes      | Groundwater chloride, mg/L                           |
| Clgw_sd  | no       | Standard deviation of Clgw to use instead of the CV  |

Column names are matched case-insensitively and common variants (latitude, longitude, chloride,
bore_id, ...) are accepted. Censored values such as `<1` are dropped.

## Settings

- **Cl_gw coefficient of variation** (default 0.37): applied to every bore without its own SD.
  0.37 is the national value Lee et al. (2024) derived from bores with more than 10 samples.
- **Per-bore SD**: a `Clgw_sd` column always takes priority; the SD of repeat samples is used only
  when a bore has at least the chosen minimum number of samples (default 10).
- **Realisations**, **alpha range** and **seeds** reproduce the original defaults (1000, U(0.33, 0.66),
  seeds 1 / 3100 / 2).
- **Cells without climate or RC data**: 1853 land cells (coastal fringe and islands, identified by having
  chloride deposition) have no rainfall, PET, aridity or runoff coefficient, and 6400 land cells have no
  random forest value. By default the app averages the valid cells in a 3 x 3 window around the bore,
  widening one ring at a time up to a maximum radius (default 2 cells, a 5 x 5 window of about 11 km),
  and records the window used in `fill_radius_climate_km` / `fill_radius_rf_km` and in `status`. If nothing
  is found within the maximum radius, RC falls back to an assumed value (default 0) and climate is left
  blank. Alternatives: always assume the RC value, or leave such bores without a stochastic result.

## Differences from CMBEAR2_v1.0.py from Lee et al. (2024)

- Minor correction on the chloride deposition files (that did not affect the random forest analyses)
  from the original Lee et al. (2024) version.
- The runoff coefficient NODATA value (-999) is treated as missing. The original did not replace it,
  and 209 bores in the published point output carry RC = -999 with recharge in the tens of thousands
  of mm/y. These bores are absent from the random forest training set, so the maps are unaffected.
- Bores outside the grid or on NODATA cells are kept and flagged in `status` instead of dropped, and
  coastal cells without climate/RC are filled as described under Settings.
- Points exactly on a 0.05 degree cell boundary are assigned deterministically to the northern or
  western cell. The original's result on such points depended on accumulated floating point error.

## Deploying to Streamlit Community Cloud

1. Push this repository to GitHub (public).
2. Sign in at https://share.streamlit.io with GitHub, choose **Create app**, select the repository,
   branch `main` and `app.py` as the entry point.
3. Deploy. The app reads `requirements.txt` and `.streamlit/config.toml` automatically; no secrets
   are needed.

## References

Irvine, D.J., Cartwright, I. (2022). CMBEAR: Python-based recharge estimator using the chloride mass
balance method in Australia. *Groundwater*, 60(3), 418-425. https://doi.org/10.1111/gwat.13161

Lee, S., Irvine, D.J., Duvert, C., Rau, G.C., Cartwright, I. (2024). A high-resolution map of diffuse
groundwater recharge rates for Australia. *Hydrology and Earth System Sciences*, 28, 1771-1790.
https://doi.org/10.5194/hess-28-1771-2024

Wilkins, A., Crosbie, R., Louth-Robins, T., Davies, P., Raiber, M., Dawes, W., Gao, L. (2022) Australian
gridded chloride deposition-rate dataset. *Data in Brief*, 42, 108189. 
https://doi.org/10.1016/j.dib.2022.1081892352-3409

## Citation and licence

If you use this app, please consider citing the papers above. Code licence: MIT License.
The gridded data retain the licence of their original sources; see Lee et al. (2024).
