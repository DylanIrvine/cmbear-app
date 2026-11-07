"""Chloride mass balance recharge calculations.

Two estimates are produced for every bore, following CMBEAR (Irvine and
Cartwright, 2022) and CMBEAR2 (Lee et al., 2024):

1. Regular CMB:        R = 100 * D / Cl_gw
   with D the mean (and 5th/95th percentile) chloride deposition (kg/ha/y)
   and Cl_gw the groundwater chloride (mg/L); R in mm/y.

2. Stochastic CMB with a runoff coefficient term:
       R = 100 * D * (1 - alpha * RC) / Cl_gw
   evaluated for n realisations in which
       D       ~ Pearson type III (mean, sd, skew from the deposition grids)
       Cl_gw   ~ |Normal(Cl_gw, CV * Cl_gw)|  (or the SD of repeat samples when 3 or more exist)
       alpha   ~ Uniform(0.33, 0.66)
   and the 5th, 50th, 95th percentiles and mean are reported.

The random streams reproduce CMBEAR2_v1.0.py exactly: the same three fixed
seeds are used and, because the original re-seeded identically for every bore,
one set of standard deviates is shared across bores. This lets the whole
calculation be vectorised while giving identical numbers to the original loop.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .grids import GridStack


@dataclass
class CMBSettings:
    n_realisations: int = 1000
    cv_clgw: float = 0.37            # blanket CV of groundwater chloride (Lee et al. 2024 national value 0.366)
    min_samples_for_sd: int = 3      # use the SD of repeat samples when a bore has at least this many; else the CV
    alpha_low: float = 0.33
    alpha_high: float = 0.66
    seed_deposition: int = 1
    seed_clgw: int = 3100
    seed_alpha: int = 2
    percentiles: tuple = (5, 50, 95)
    keep_realisations: bool = True   # return the full realisation matrix for plotting
    # Handling of land cells that have chloride deposition but no climate/RC data (coastal fringe,
    # islands; 1853 cells) and cells with no random forest value (6400 land cells).
    fill_method: str = "neighbours"  # "neighbours": mean of valid cells in a 3x3 window, widening one ring at a
                                     #   time up to fill_max_cells; "assume_rc": leave climate blank and use
                                     #   fill_rc_value for RC; "none": leave the bore without a stochastic result
    fill_max_cells: int = 2          # largest window radius in cells (2 = 5 x 5 cells, about 11 km)
    fill_rc_value: float = 0.0       # RC used by "assume_rc" (and by "neighbours" when the window finds nothing)
    fill_rf: bool = True             # also fill RF values by the same neighbourhood mean


def pearson3_deviates(mean, sd, skew, z):
    """Pearson type III replicates via the frequency-factor (Kirby/Wilson-Hilferty) approximation.

    mean, sd, skew: arrays of shape (n_bores, 1); z: standard normal deviates (1, n).
    Mirrors generate_pearsonIII() in CMBEAR2_v1.0.py, including the absolute value.
    """
    sd = np.maximum(sd, 0.001)
    skew = np.where(skew == 0, 0.001, skew)
    ky = (2.0 / skew) * ((((z - skew / 6.0) * (skew / 6.0) + 1.0) ** 3) - 1.0)
    return np.abs(mean + ky * sd)


def _deviates(settings: CMBSettings):
    """The three fixed random streams used by CMBEAR2 (legacy RandomState)."""
    n = settings.n_realisations
    rs = np.random.RandomState(settings.seed_deposition)
    z = rs.normal(size=n)
    rs = np.random.RandomState(settings.seed_clgw)
    eps = rs.standard_normal(n)          # normal(mean, sd) == mean + sd * standard_normal
    rs = np.random.RandomState(settings.seed_alpha)
    alpha = rs.uniform(settings.alpha_low, settings.alpha_high, n)
    return z[None, :], eps[None, :], alpha[None, :]


def run_cmbear(bores: pd.DataFrame, grids: GridStack, settings: CMBSettings | None = None):
    """Run the CMB calculations and extract random forest values for each bore.

    bores: output of cmbear.inputs.parse_input (columns ID, Lat, Lon, Clgw, Clgw_sd_samples, n_samples).
    Returns (results DataFrame, realisations ndarray or None).
    """
    s = settings or CMBSettings()
    b = bores.reset_index(drop=True)
    nb = len(b)
    lat = b["Lat"].to_numpy(float)
    lon = b["Lon"].to_numpy(float)
    clgw = b["Clgw"].to_numpy(float)

    row, col = grids.rowcol(lat, lon)
    plat, plon = grids.pixel_centre(row, col)
    g = {k: grids.sample(k, row, col) for k in grids.arrays}
    km_per_cell = grids.res * 111.0

    # ---- fill missing climate/RC and RF values -----------------------------
    clim_keys = ["rain", "pet", "aridity", "rc"]
    rf_keys = ["rf_r5", "rf_r50", "rf_r95", "rf_rp5", "rf_rp50", "rf_rp95"]
    on_land = np.isfinite(g["cld_mean"])
    clim_missing = on_land & ~np.isfinite(g["rc"])
    fill_note = np.full(nb, "", dtype=object)
    fill_r_clim = np.zeros(nb, dtype=float); fill_r_clim[~on_land] = np.nan
    if s.fill_method == "neighbours":
        idx = np.where(clim_missing)[0]
        if len(idx):
            vals, rad = grids.neighbourhood_mean(clim_keys, row[idx], col[idx], s.fill_max_cells)
            found = rad > 0
            for k in clim_keys:
                g[k][idx[found]] = vals[k][found]
            fill_r_clim[idx[found]] = rad[found] * km_per_cell
            fill_note[idx[found]] = [f"climate/RC = mean of cells within {r} cell(s)" for r in rad[found]]
            g["rc"][idx[~found]] = s.fill_rc_value
            fill_note[idx[~found]] = f"no climate within {s.fill_max_cells} cells; RC assumed {s.fill_rc_value:g}"
            fill_r_clim[idx[~found]] = np.nan
    elif s.fill_method == "assume_rc":
        g["rc"] = np.where(clim_missing, s.fill_rc_value, g["rc"])
        fill_note[clim_missing] = f"RC assumed {s.fill_rc_value:g}; climate unavailable"
        fill_r_clim[clim_missing] = np.nan
    else:
        fill_note[clim_missing] = "no climate/RC data; not filled"
        fill_r_clim[clim_missing] = np.nan

    rf_missing = on_land & ~np.isfinite(g["rf_r50"])
    fill_r_rf = np.zeros(nb, dtype=float); fill_r_rf[~on_land | rf_missing] = np.nan
    if s.fill_rf:
        idx = np.where(rf_missing)[0]
        if len(idx):
            vals, rad = grids.neighbourhood_mean(rf_keys, row[idx], col[idx], s.fill_max_cells)
            found = rad > 0
            for k in rf_keys:
                g[k][idx[found]] = vals[k][found]
            fill_r_rf[idx[found]] = rad[found] * km_per_cell
            add = [f"RF = mean of cells within {r} cell(s)" for r in rad[found]]
            fill_note[idx[found]] = [a if n == "" else f"{n}; {a}" for n, a in zip(fill_note[idx[found]], add)]

    # ---- regular CMB -------------------------------------------------------
    r_mean = 100.0 * g["cld_mean"] / clgw
    r_5 = 100.0 * g["cld_5th"] / clgw
    r_95 = 100.0 * g["cld_95th"] / clgw

    # ---- stochastic CMB with runoff coefficient ---------------------------
    z, eps, alpha = _deviates(s)
    d_mean, d_sd, d_skew = (g[k][:, None] for k in ("cld_mean", "cld_sd", "cld_skew"))
    dep = pearson3_deviates(d_mean, d_sd, d_skew, z)                    # (nb, n)

    def _col(name):
        return pd.to_numeric(b[name], errors="coerce").to_numpy(float) if name in b else np.full(nb, np.nan)
    samp_sd, nsamp = _col("Clgw_sd_samples"), _col("n_samples")
    # SD of repeat samples when there are enough of them, otherwise the blanket CV
    has_samp = np.isfinite(samp_sd) & (samp_sd > 0) & (nsamp >= s.min_samples_for_sd)
    sd_used = np.where(has_samp, samp_sd, clgw * s.cv_clgw)
    sd_source = np.where(has_samp, "repeat samples", "blanket CV").astype(object)
    cv_used = sd_used / clgw
    clgw_dist = np.abs(clgw[:, None] + sd_used[:, None] * eps)         # (nb, n)
    rc = g["rc"][:, None]
    real = 100.0 * dep * (1.0 - alpha * rc) / clgw_dist                 # (nb, n)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN rows for bores off the grid
        pct = np.nanpercentile(real, s.percentiles, axis=1)      # (3, nb)
        rrc_mean = np.nanmean(real, axis=1)
    rrc_5, rrc_50, rrc_95 = pct[0], pct[1], pct[2]

    # ---- status flags ------------------------------------------------------
    status = np.full(nb, "ok", dtype=object)
    filled = fill_note != ""
    status[filled] = np.char.add("ok (", np.char.add(fill_note[filled].astype(str), ")"))
    no_rf = ~np.isfinite(g["rf_r50"]) & on_land
    status[no_rf & (status == "ok")] = "ok (no RF value at pixel)"
    status[on_land & ~np.isfinite(g["rc"])] = "no climate/RC data at pixel (not filled)"
    status[~on_land] = "no CMB input data at pixel (ocean or outside data coverage)"
    status[row < 0] = "outside grid extent"

    res = pd.DataFrame({
        "ID": b["ID"], "Lat": lat, "Lon": lon, "n_samples": b["n_samples"],
        "Clgw_mg_L": clgw, "Clgw_sd_samples_mg_L": samp_sd, "Clgw_sd_used_mg_L": sd_used,
        "CV_used": cv_used, "Clgw_sd_source": sd_source,
        "pixel_row": row, "pixel_col": col, "pixel_lat": plat, "pixel_lon": plon,
        "Rain_mm_y": g["rain"], "PET_mm_y": g["pet"], "Aridity": g["aridity"], "RC": g["rc"],
        "ClDep_mean_kg_ha_y": g["cld_mean"], "ClDep_5th_kg_ha_y": g["cld_5th"],
        "ClDep_95th_kg_ha_y": g["cld_95th"], "ClDep_sd_kg_ha_y": g["cld_sd"], "ClDep_skew": g["cld_skew"],
        "R_CMB_mean_mm_y": r_mean, "R_CMB_5_mm_y": r_5, "R_CMB_95_mm_y": r_95,
        "RP_CMB": r_mean / g["rain"],
        "R_RC_5_mm_y": rrc_5, "R_RC_50_mm_y": rrc_50, "R_RC_mean_mm_y": rrc_mean, "R_RC_95_mm_y": rrc_95,
        "RP_RC_50": rrc_50 / g["rain"],
        "RF_R5_mm_y": g["rf_r5"], "RF_R50_mm_y": g["rf_r50"], "RF_R95_mm_y": g["rf_r95"],
        "RF_RP5": g["rf_rp5"], "RF_RP50": g["rf_rp50"], "RF_RP95": g["rf_rp95"],
        "fill_radius_climate_km": fill_r_clim, "fill_radius_rf_km": fill_r_rf,
        "status": status,
    })
    return res, (real.astype(np.float32) if s.keep_realisations else None)


COLUMN_NOTES = {
    "Clgw_mg_L": "Groundwater chloride used (mean of repeat samples if any)",
    "Clgw_sd_samples_mg_L": "Standard deviation of repeat samples at the bore (blank for single samples)",
    "Clgw_sd_used_mg_L": "Standard deviation used for the Cl_gw distribution",
    "CV_used": "Coefficient of variation actually applied to Cl_gw",
    "Clgw_sd_source": "Where the SD came from: repeat samples (3 or more) or the blanket CV",
    "pixel_row/pixel_col": "Indices of the 0.05 degree grid cell the bore fell in",
    "Rain_mm_y, PET_mm_y, Aridity, RC": "Climate and runoff coefficient at the pixel; Aridity is P/PET (lower is drier)",
    "ClDep_*": "Chloride deposition statistics at the pixel (kg/ha/y)",
    "R_CMB_mean/5/95_mm_y": "Regular CMB recharge from mean/5th/95th percentile deposition (no runoff term)",
    "RP_CMB": "R_CMB_mean divided by rainfall",
    "R_RC_5/50/mean/95_mm_y": "Stochastic CMB with runoff coefficient: percentiles and mean of the realisations",
    "RP_RC_50": "R_RC_50 divided by rainfall",
    "RF_R5/R50/R95_mm_y": "Random forest recharge (Lee et al. 2024) at the pixel",
    "RF_RP5/RP50/RP95": "Random forest recharge/rainfall ratio at the pixel",
    "fill_radius_climate_km": "0 if climate/RC came from the bore's own cell; half-width of the averaging window if filled; blank if unavailable",
    "fill_radius_rf_km": "0 if RF values came from the bore's own cell; half-width of the averaging window if filled; blank if unavailable",
    "status": "ok, ok with a note on filled values, or why a bore has no result",
}
