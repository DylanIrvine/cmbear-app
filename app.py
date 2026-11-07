"""CMBEAR online: Chloride mass balance Estimator of Australian Recharge."""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cmbear import CMBSettings, load_grids, parse_input, run_cmbear
from cmbear.cmb import COLUMN_NOTES
from cmbear.inputs import read_table
from cmbear.plotting import bore_map

LOGO_DIR = Path(__file__).resolve().parent / "logo"
BADGE, FAVICON = LOGO_DIR / "cmbear_badge.png", LOGO_DIR / "favicon.png"

st.set_page_config(page_title="CMBEAR online", page_icon=str(FAVICON) if FAVICON.exists() else ":droplet:")

# One hue per series, fixed assignment. Backgrounds and fonts follow the Streamlit theme.
C_CMB, C_RF, C_LINE = "#4a90c2", "#e0812b", "#9a9a9a"
PLOT_LAYOUT = dict(font=dict(size=13), margin=dict(l=60, r=20, t=50, b=55))
FILL_LABELS = {"Mean of neighbouring cells": "neighbours", "Assume RC value, leave climate blank": "assume_rc",
               "Leave blank": "none"}


@st.cache_resource(show_spinner="Loading national grids...")
def grids():
    return load_grids()


@st.cache_data(show_spinner=False)
def template_csv() -> bytes:
    return pd.DataFrame({
        "ID": ["RN012345", "RN012346", "Bore_A"],
        "Lat": [-12.4512, -14.6300, -23.7000],
        "Lon": [130.9820, 132.2000, 133.8700],
        "Clgw": [12.5, 48.0, 310.0],
        "Clgw_sd": [np.nan, 9.5, np.nan],
    }).to_csv(index=False).encode()


def results_xlsx(res, settings, msgs, real, include_real) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        res.to_excel(xw, sheet_name="results", index=False)
        pd.DataFrame({"column": list(COLUMN_NOTES), "meaning": list(COLUMN_NOTES.values())}) \
            .to_excel(xw, sheet_name="column_notes", index=False)
        meta = {**settings.__dict__, "run_time_utc": datetime.utcnow().isoformat(timespec="seconds"),
                "grid_source": "Lee et al. (2024) 0.05 degree grids", "messages": " | ".join(msgs)}
        pd.DataFrame({"setting": list(meta), "value": [str(v) for v in meta.values()]}) \
            .to_excel(xw, sheet_name="settings", index=False)
        if include_real and real is not None:
            pd.DataFrame(real.T, columns=res["ID"]).to_excel(xw, sheet_name="realisations_mm_y", index=False)
    return buf.getvalue()


def marker(color):
    return dict(color=color, size=8, opacity=0.8, line=dict(width=1, color="rgba(255,255,255,0.6)"))


# --------------------------------------------------------------------------- #
# header
# --------------------------------------------------------------------------- #
head_text, head_logo = st.columns([3, 1], vertical_alignment="center")
with head_text:
    st.title("CMBEAR online")
    st.subheader("Chloride Mass Balance Estimator Australian Recharge")
    st.write("Dylan Irvine and Stephen Lee, Charles Darwin University.\n\n")
with head_logo:
    if BADGE.exists():
        st.image(str(BADGE), width="stretch")

st.markdown(
    "Upload a spreadsheet of bores with groundwater chloride concentrations and obtain recharge estimates "
    "from the chloride mass balance (CMB) method. For each bore the app extracts chloride deposition, rainfall, "
    "PET, aridity and runoff coefficient from national 0.05 degree grids, calculates the regular CMB recharge "
    "and the stochastic CMBEAR2 recharge with a runoff term, and reads the matching values from the random "
    "forest recharge model of Lee et al. (2024).\n\n"
    "Note: CMBEAR online only works for Australian bores.\n\n"
    "Input: a CSV or Excel file with columns ID, Lat, Lon and Clgw (mg/L); optional Clgw_sd."
)

with st.expander("**References**"):
    st.write(
        "Crosbie, R.S., Rachakonda, P.K. (2021) Constraining probabilistic chloride mass-balance recharge estimates"
        "using baseflow and remotely sensed evapotranspiration: the Cambrian Limestone Aquifer in northern Australia."
        "Hydrogeology Journal, 29, 1399–1419. https://doi.org/10.1007/s10040-021-02323-1/"
        "Irvine, D.J., Cartwright, I. (2022). CMBEAR: Python-based recharge estimator using the chloride mass "
        "balance method in Australia. Groundwater, 60(3), 418-425. https://doi.org/10.1111/gwat.13161\n\n"
        "Lee, S., Irvine, D.J., Duvert, C., Rau, G.C., Cartwright, I. (2024). A high-resolution map of diffuse "
        "groundwater recharge rates for Australia. Hydrology and Earth System Sciences, 28, 1771-1790. "
        "https://doi.org/10.5194/hess-28-1771-2024\n\n"
        "Wilkins, A., Crosbie, R., Louth-Robins, T., Davies, P. Raiber, M., Dawes, W., Gao, L. (2022)"
        "Australian gridded chloride deposition-rate dataset. Data In Brief, 42, 108189."
        "10.1016/j.dib.2022.108189/ \n\n"
        "Data and original scripts: https://www.hydroshare.org/resource/5e7b8bfcc1514680902f8ff43cc254b8/"
    )

# --------------------------------------------------------------------------- #
# 1. bore data
# --------------------------------------------------------------------------- #
st.header("1. Bore data")
st.subheader("Requires the headers: ID,	Lat,	Lon,	Clgw,	Clgw_sd")
col_up, col_tpl = st.columns([3, 1], vertical_alignment="bottom")
upload = col_up.file_uploader("Bore spreadsheet (CSV or Excel)", type=["csv", "xlsx", "xls"])
col_tpl.download_button("Download template", template_csv(), "cmbear_template.csv", "text/csv", width="stretch")

bores, msgs = None, []
if upload is not None:
    try:
        bores, msgs = parse_input(read_table(upload, upload.name))
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read the input file: {e}")
    else:
        for m in msgs:
            st.caption(m)
        with st.expander("Preview parsed input"):
            st.dataframe(bores, width="stretch", height=240)

# --------------------------------------------------------------------------- #
# 2. settings
# --------------------------------------------------------------------------- #
st.header("2. Settings")
c1, c2, c3 = st.columns(3)
cv = c1.slider("Cl_gw coefficient of variation", 0.05, 1.00, 0.37, 0.01,
               help="Blanket CV applied to groundwater chloride. 0.37 is the national value derived from bores "
                    "with more than 10 samples in Lee et al. (2024).")
n_real = c2.select_slider("Realisations per bore", [250, 500, 1000, 2000, 5000], value=1000)
a_lo, a_hi = c3.slider("Runoff factor alpha range", 0.0, 1.0, (0.33, 0.66), 0.01,
                       help="Fraction of the runoff coefficient that removes chloride. Lee et al. used U(0.33, 0.66).")
c4, c5 = st.columns(2)
use_sd = c4.checkbox("Use per-bore SD where available", value=True,
                     help="A user-supplied Clgw_sd column always takes priority. Otherwise the SD of repeat samples "
                          "is used if the bore has at least the number of samples set alongside.")
min_n = c5.number_input("Minimum repeat samples to use their SD", 2, 100, 10, 1)

with st.expander("Cells without climate or RC data"):
    st.caption("1853 coastal and island cells have chloride deposition but no rainfall, PET, aridity or runoff "
               "coefficient; 6400 land cells have no random forest value.")
    fill_method = st.radio("Handling", list(FILL_LABELS), horizontal=True,
                           help="Averages the valid cells in a 3 x 3 window around the bore; if none, the window grows "
                                "one ring at a time up to the maximum. The window size used is recorded.")
    f1, f2, f3 = st.columns(3)
    fill_max = f1.slider("Maximum window radius (cells)", 1, 10, 2, 1, help="2 cells is a 5 x 5 window, about 11 km across.")
    fill_rc = f2.number_input("Assumed RC when no data", 0.0, 1.0, 0.0, 0.01, format="%.2f")
    fill_rf = f3.checkbox("Fill RF values too", value=True)

with st.expander("Advanced"):
    s1, s2, s3 = st.columns(3)
    seed_d = s1.number_input("Seed (deposition)", value=1, step=1)
    seed_c = s2.number_input("Seed (Cl_gw)", value=3100, step=1)
    seed_a = s3.number_input("Seed (alpha)", value=2, step=1)
    keep_real = st.checkbox("Keep realisations (needed for the histogram and Excel export)", value=True)

settings = CMBSettings(n_realisations=int(n_real), cv_clgw=cv, use_bore_sd=use_sd, min_samples_for_sd=int(min_n),
                       alpha_low=a_lo, alpha_high=a_hi, seed_deposition=int(seed_d), seed_clgw=int(seed_c),
                       seed_alpha=int(seed_a), keep_realisations=keep_real, fill_method=FILL_LABELS[fill_method],
                       fill_max_cells=int(fill_max), fill_rc_value=float(fill_rc), fill_rf=fill_rf)

# --------------------------------------------------------------------------- #
# 3. run
# --------------------------------------------------------------------------- #
st.header("3. Run")
if bores is None:
    st.info("Upload a bore spreadsheet in section 1 to enable the run.")
if st.button("Calculate recharge", type="primary", disabled=bores is None):
    res, real = run_cmbear(bores, grids(), settings)
    st.session_state["run"] = dict(res=res, real=real, settings=settings, msgs=msgs, file=upload.name)

run = st.session_state.get("run")
if run is not None:
    res, real, run_settings = run["res"], run["real"], run["settings"]
    if run_settings != settings or run["file"] != (upload.name if upload else None):
        st.caption("Settings or input have changed since the last run. Press Calculate recharge to update.")

    bad = res["status"].str.startswith(("outside", "no CMB", "no climate"))
    if bad.any():
        st.warning(f"{int(bad.sum())} bore(s) have no CMB result (outside the grid, over NODATA such as ocean, "
                   "or on a coastal cell left unfilled). See the status column.")
    filled = res["status"].str.contains("mean of cells|assumed", regex=True)
    if filled.any():
        st.info(f"{int(filled.sum())} bore(s) sit on cells without climate/RC or RF data and were filled as set "
                "in section 2. See status and the fill_radius columns.")

    st.subheader("Results")
    st.dataframe(res, width="stretch", height=360)
    ok_all = res[res["status"].str.startswith("ok")]
    if len(ok_all):
        summ = ok_all[["R_CMB_mean_mm_y", "R_RC_50_mm_y", "RF_R50_mm_y", "Rain_mm_y", "RP_RC_50", "RF_RP50"]] \
            .describe(percentiles=[0.05, 0.5, 0.95]).T
        with st.expander("Summary statistics"):
            st.dataframe(summ.apply(lambda c: c.map(lambda v: f"{v:.3g}")), width="stretch")

    # ----------------------------------------------------------------------- #
    # 4. downloads
    # ----------------------------------------------------------------------- #
    st.header("4. Downloads")
    d1, d2, d3 = st.columns([1, 1, 1], vertical_alignment="center")
    d1.download_button("Results (CSV)", res.to_csv(index=False).encode(), "cmbear_results.csv", "text/csv",
                       width="stretch")
    inc = d3.checkbox("Include realisations in Excel", value=False,
                      help="Adds a sheet with every realisation (bores x realisations). Can be large.")
    d2.download_button("Results (Excel)", results_xlsx(res, run_settings, run["msgs"], real, inc),
                       "cmbear_results.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")

    # ----------------------------------------------------------------------- #
    # 5. plots
    # ----------------------------------------------------------------------- #
    st.header("5. Plots")
    ok = ok_all.reset_index()

    st.subheader("Bore map")
    m1, m2, m3 = st.columns([2, 2, 1], vertical_alignment="bottom")
    cvar = m1.selectbox("Colour bores by", ["R_RC_50_mm_y", "R_CMB_mean_mm_y", "RF_R50_mm_y", "RP_RC_50",
                                             "Clgw_mg_L", "Rain_mm_y", "fill_radius_climate_km"])
    bgopt = m2.selectbox("Background", ["Random forest R50", "Rainfall", "Land only"])
    logc = m3.checkbox("Log colour", value=cvar not in ("fill_radius_climate_km", "Rain_mm_y"))
    bgkey = {"Random forest R50": "rf_r50", "Rainfall": "rain", "Land only": None}[bgopt]
    with_coords = res[np.isfinite(res["Lat"]) & np.isfinite(res["Lon"])]
    if len(with_coords):
        st.plotly_chart(bore_map(with_coords, grids(), cvar, cvar, bgkey, logc), width="stretch")
        st.caption("Main panel: bores over the chosen grid (greys, darker is higher). Inset: Australia with the "
                   "main panel's extent outlined. Bores without a result are still plotted; hover for status.")

    st.subheader("Stochastic CMB versus random forest recharge")
    logax = st.checkbox("Log axes", value=True, key="log1")
    d = ok[ok["RF_R50_mm_y"].notna()]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["RF_R50_mm_y"], y=d["R_RC_50_mm_y"], mode="markers", name="Bores (R50, bars 5th to 95th)",
        marker=marker(C_CMB),
        error_y=dict(type="data", symmetric=False, array=d["R_RC_95_mm_y"] - d["R_RC_50_mm_y"],
                     arrayminus=d["R_RC_50_mm_y"] - d["R_RC_5_mm_y"], color=C_CMB, thickness=1, width=0),
        error_x=dict(type="data", symmetric=False, array=d["RF_R95_mm_y"] - d["RF_R50_mm_y"],
                     arrayminus=d["RF_R50_mm_y"] - d["RF_R5_mm_y"], color=C_RF, thickness=1, width=0),
        customdata=np.stack([d["ID"], d["Clgw_mg_L"], d["Rain_mm_y"]], axis=1),
        hovertemplate="<b>%{customdata[0]}</b><br>CMB R50 %{y:.1f} mm/y<br>RF R50 %{x:.1f} mm/y"
                      "<br>Cl_gw %{customdata[1]:.1f} mg/L<br>Rain %{customdata[2]:.0f} mm/y<extra></extra>"))
    if len(d):
        lo = max(1e-3, float(np.nanmin([d["RF_R5_mm_y"].min(), d["R_RC_5_mm_y"].min()])) * 0.8)
        hi = float(np.nanmax([d["RF_R95_mm_y"].max(), d["R_RC_95_mm_y"].max()])) * 1.2
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="1:1",
                                 line=dict(color=C_LINE, width=1.5, dash="dash")))
    fig.update_layout(**PLOT_LAYOUT, xaxis_title="Random forest median recharge (mm/y)",
                      yaxis_title="Stochastic CMB median recharge (mm/y)", height=520,
                      legend=dict(orientation="h", y=1.08))
    if logax:
        fig.update_xaxes(type="log"); fig.update_yaxes(type="log")
    st.plotly_chart(fig, width="stretch")
    st.caption("Vertical bars: 5th to 95th percentile of the CMB realisations at the bore. Horizontal bars: "
               "5th to 95th percentile random forest grids at the same pixel. The RF model was trained on CMB "
               "estimates of this kind, so agreement is expected on average but not bore by bore.")

    st.subheader("Recharge against climate")
    which = st.radio("Recharge estimate", ["Stochastic CMB (R_RC_50)", "Regular CMB (R_CMB_mean)", "Random forest (RF_R50)"],
                     horizontal=True)
    ycol = {"Stochastic CMB (R_RC_50)": "R_RC_50_mm_y", "Regular CMB (R_CMB_mean)": "R_CMB_mean_mm_y",
            "Random forest (RF_R50)": "RF_R50_mm_y"}[which]
    f1 = go.Figure(go.Scatter(x=ok["Rain_mm_y"], y=ok[ycol], mode="markers", name=which, marker=marker(C_CMB),
                              text=ok["ID"], hovertemplate="<b>%{text}</b><br>P %{x:.0f} mm/y<br>R %{y:.2f} mm/y<extra></extra>"))
    f1.update_layout(**PLOT_LAYOUT, xaxis_title="Mean annual rainfall (mm/y)", yaxis_title="Recharge (mm/y)",
                     yaxis_type="log", height=420, title="Recharge versus rainfall")
    f2 = go.Figure(go.Scatter(x=ok["Aridity"], y=ok[ycol] / ok["Rain_mm_y"], mode="markers", name=which,
                              marker=marker(C_CMB), text=ok["ID"],
                              hovertemplate="<b>%{text}</b><br>PET/P %{x:.2f}<br>R/P %{y:.3f}<extra></extra>"))
    f2.update_layout(**PLOT_LAYOUT, xaxis_title="Aridity index (PET/P)", yaxis_title="Recharge / rainfall",
                     yaxis_type="log", xaxis_type="log", height=420, title="Recharge ratio versus aridity")
    st.plotly_chart(f1, width="stretch")
    st.plotly_chart(f2, width="stretch")

    st.subheader("Recharge distribution for one bore")
    if real is None:
        st.info("Enable 'Keep realisations' under Advanced to see per-bore distributions.")
    elif len(ok):
        bid = st.selectbox("Bore", ok["ID"].tolist())
        idx = int(ok.loc[ok["ID"] == bid, "index"].iloc[0])
        r = res.iloc[idx]
        vals = real[idx]; vals = vals[np.isfinite(vals)]
        fh = go.Figure(go.Histogram(x=vals, nbinsx=60, name="CMB realisations",
                                    marker=dict(color=C_CMB, line=dict(color="rgba(255,255,255,0.5)", width=1))))
        for p, lab in ((r["R_RC_5_mm_y"], "5th"), (r["R_RC_50_mm_y"], "50th"), (r["R_RC_95_mm_y"], "95th")):
            fh.add_vline(x=p, line=dict(color=C_LINE, width=1.5, dash="dot"),
                         annotation_text=f"{lab} {p:.1f}", annotation_position="top")
        if np.isfinite(r["RF_R50_mm_y"]):
            fh.add_vline(x=r["RF_R50_mm_y"], line=dict(color=C_RF, width=2),
                         annotation_text=f"RF R50 {r['RF_R50_mm_y']:.1f}", annotation_position="top right")
        fh.update_layout(**PLOT_LAYOUT, xaxis_title="Recharge (mm/y)", yaxis_title="Count", height=420,
                         title=f"{bid}: Cl_gw {r['Clgw_mg_L']:.1f} mg/L, P {r['Rain_mm_y']:.0f} mm/y, "
                               f"RC {r['RC']:.3f}, deposition {r['ClDep_mean_kg_ha_y']:.1f} kg/ha/y")
        st.plotly_chart(fh, width="stretch")

# --------------------------------------------------------------------------- #
# method and notes
# --------------------------------------------------------------------------- #
st.header("Method and notes")
with st.expander("What the app calculates, and how it differs from the original script"):
    st.markdown(r"""
For each bore the app locates the 0.05 degree grid cell containing it and extracts long-term mean rainfall,
PET, aridity, the mean annual runoff coefficient (RC) and the chloride deposition statistics from the grids
of Lee et al. (2024). Two recharge estimates follow.

**Regular chloride mass balance** (Irvine and Cartwright, 2022):

$R = \dfrac{100\,D}{Cl_{gw}}$

with $D$ the mean chloride deposition (kg/ha/y), $Cl_{gw}$ groundwater chloride (mg/L) and $R$ in mm/y.
The 5th and 95th percentile deposition grids give `R_CMB_5` and `R_CMB_95`.

**Stochastic CMB with a runoff term** (CMBEAR2, Lee et al., 2024):

$R = \dfrac{100\,D\,(1 - \alpha\,RC)}{Cl_{gw}}$

evaluated for *n* realisations where $D$ follows a Pearson type III distribution (mean, sd, skew grids),
$Cl_{gw}$ follows $|N(Cl_{gw},\,CV \cdot Cl_{gw})|$ and $\alpha \sim U(0.33, 0.66)$. The 5th, 50th and 95th
percentiles and the mean of the realisations are reported (`R_RC_*`).

**Random forest values** (`RF_*`) are read directly from the published national grids at the same pixel:
recharge (R5, R50, R95, mm/y) and recharge divided by rainfall (RP5, RP50, RP95).

**Differences from the original CMBEAR2_v1.0.py script**

- Input is reduced to ID, Lat, Lon, Clgw (optional Clgw_sd). Repeat samples per ID are averaged.
- The blanket CV of groundwater chloride is a user setting (default 0.37) rather than being derived from
  bores with more than 10 samples in the input file. 0.37 reproduces the national analysis.
- The regular CMB upper bound now uses the 95th percentile deposition grid. The original read the 5th
  percentile file for both bounds, so its `Recharge 95%` column equalled `Recharge 5%`.
- The runoff coefficient NODATA value (-999) is treated as missing. The original did not replace it, so a
  small number of bores on NODATA cells received physically meaningless recharge values.
- Bores outside the grid or on NODATA cells are kept and flagged in `status` rather than dropped.
- 1853 land cells (coastal fringe and islands) have chloride deposition but no climate or RC data, and
  6400 land cells have no RF value. The app either averages the valid cells in a 3 x 3 window around the
  bore, widening the window one ring at a time up to a chosen maximum and recording the window used, or
  uses an assumed RC with climate left blank, as chosen in section 2.
- Points lying exactly on a cell boundary are assigned to the northern/western cell deterministically.
  Otherwise pixel selection and random streams are identical, and results match the original to float32
  precision.
""")
with st.expander("Output columns"):
    st.table(pd.DataFrame({"column": list(COLUMN_NOTES), "meaning": list(COLUMN_NOTES.values())}))
