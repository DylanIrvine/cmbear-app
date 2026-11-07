"""CMBEAR online: Chloride mass balance Estimator of Australian Recharge."""
from __future__ import annotations

import io
import re
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

# Default plot colours (users can override with their own hex codes under Plot colours).
DEFAULT_COLOURS = {"primary": "#4a90c2", "secondary": "#e0812b", "reference": "#9a9a9a"}
PLOT_LAYOUT = dict(font=dict(size=13), margin=dict(l=60, r=20, t=50, b=70))
ATTRIBUTION = "Produced using CMBEAR online - Dylan Irvine"
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


def brand(fig):
    """Small, light attribution line a few pixels above the figure's bottom edge, outside the plot area,
    so it sits in the margin and can be cropped off. Position is in pixels so it is independent of height."""
    b = (fig.layout.margin.b if fig.layout.margin and fig.layout.margin.b is not None else 70)
    fig.add_annotation(text=ATTRIBUTION, xref="paper", yref="paper", x=1.0, y=0, xanchor="right",
                       yanchor="bottom", yshift=-(b - 4), showarrow=False,
                       font=dict(size=9, color="rgba(140,140,140,0.55)"))
    return fig


def cap_range(values, cap, log, lower_floor=1e-3):
    """Axis range [lo, hi] with hi capped at `cap`; in log10 units when log is True."""
    v = np.asarray(values, float); v = v[np.isfinite(v)]
    if log:
        v = v[v > 0]
    if v.size == 0:
        return None
    lo, hi = float(v.min()), min(float(v.max()), float(cap))
    if hi <= lo:
        hi = lo * 1.5 if log else lo + 1
    if log:
        lo = max(lo, lower_floor)
        return [np.log10(lo) - 0.1, np.log10(hi) + 0.05]
    return [0 if lo >= 0 else lo * 1.05, hi * 1.05]


def valid_hex(text, fallback):
    text = (text or "").strip()
    if not text.startswith("#"):
        text = "#" + text
    return text if re.fullmatch(r"#[0-9a-fA-F]{6}", text) else fallback


# --------------------------------------------------------------------------- #
# header
# --------------------------------------------------------------------------- #
head_text, head_logo = st.columns([3, 1], vertical_alignment="center")
with head_text:
    st.title("CMBEAR online")
    st.subheader("Chloride Mass Balance Estimator Australian Recharge")
    st.write("Dylan Irvine, Charles Darwin University.\n\n")
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
    "Input: a CSV or Excel file with columns ID, Lat, Lon and Clgw (mg/L). Repeat samples under the same ID are "
    "averaged, and where there are three or more their standard deviation is used in place of the blanket CV.\n \n"
    "Please cite the Lee et al. (2024) paper:\n\n"
    "Lee, S., Irvine, D.J., Duvert, C., Rau, G.C., Cartwright, I. (2024). A high-resolution map of diffuse "
    "groundwater recharge rates for Australia. Hydrology and Earth System Sciences, 28, 1771-1790. "
    "https://doi.org/10.5194/hess-28-1771-2024\n\n"
    "Other references to consider are in the dropdown below."
)

with st.expander("**References**"):
    st.write(
        "Stochastic CMB equations orginally from:\n\n"
        "Crosbie, R.S., Rachakonda, P.K. (2021) Constraining probabilistic chloride mass-balance recharge estimates"
        "using baseflow and remotely sensed evapotranspiration: the Cambrian Limestone Aquifer in northern Australia."
        "Hydrogeology Journal, 29, 1399–1419. https://doi.org/10.1007/s10040-021-02323-1/ \n\n"
        "First implementation of CMBEAR:\n\n"
        "Irvine, D.J., Cartwright, I. (2022). CMBEAR: Python-based recharge estimator using the chloride mass "
        "balance method in Australia. Groundwater, 60(3), 418-425. https://doi.org/10.1111/gwat.13161\n\n"
        "Implementation of CMBEAR2:\n\n"
        "Lee, S., Irvine, D.J., Duvert, C., Rau, G.C., Cartwright, I. (2024). A high-resolution map of diffuse "
        "groundwater recharge rates for Australia. Hydrology and Earth System Sciences, 28, 1771-1790. "
        "https://doi.org/10.5194/hess-28-1771-2024\n\n"
        "Chloride deposition maps (original reference):\n\n"
        "Wilkins, A., Crosbie, R., Louth-Robins, T., Davies, P. Raiber, M., Dawes, W., Gao, L. (2022)"
        "Australian gridded chloride deposition-rate dataset. Data In Brief, 42, 108189."
        "10.1016/j.dib.2022.108189/ \n\n"
    )

# --------------------------------------------------------------------------- #
# 1. bore data
# --------------------------------------------------------------------------- #
st.header("1. Bore data")
st.subheader("Requires the headers: ID, Lat, Lon, Clgw, or download the template as a starting point.")
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
st.subheader("Update any of the other optional parameters here.")
c1, c2, c3 = st.columns(3)
cv = c1.slider("Cl_gw coefficient of variation", 0.05, 1.00, 0.37, 0.01,
               help="Applied to groundwater chloride for bores with fewer than three samples. 0.37 is the national "
                    "value derived from bores with more than 10 samples in Lee et al. (2024).")
n_real = c2.select_slider("Realisations per bore", [250, 500, 1000, 2000, 5000], value=1000)
a_lo, a_hi = c3.slider("Runoff factor alpha range", 0.0, 1.0, (0.33, 0.66), 0.01,
                       help="Fraction of the runoff coefficient that removes chloride. Lee et al. used U(0.33, 0.66).")
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

settings = CMBSettings(n_realisations=int(n_real), cv_clgw=cv, alpha_low=a_lo, alpha_high=a_hi, seed_deposition=int(seed_d), seed_clgw=int(seed_c),
                       seed_alpha=int(seed_a), keep_realisations=keep_real, fill_method=FILL_LABELS[fill_method],
                       fill_max_cells=int(fill_max), fill_rc_value=float(fill_rc), fill_rf=fill_rf)

# --------------------------------------------------------------------------- #
# 3. run
# --------------------------------------------------------------------------- #
st.header("3. Run")
st.subheader("Not much to be said here. Click the button to run the analysis.")
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

    with st.expander("Plot colours"):
        st.caption("Enter hex codes (for example your organisation's colours). Primary is used for CMB results, "
                   "secondary for random forest values, reference for 1:1 lines and percentile markers.")
        pc1, pc2, pc3 = st.columns(3)
        C_CMB = valid_hex(pc1.text_input("Primary", DEFAULT_COLOURS["primary"]), DEFAULT_COLOURS["primary"])
        C_RF = valid_hex(pc2.text_input("Secondary", DEFAULT_COLOURS["secondary"]), DEFAULT_COLOURS["secondary"])
        C_LINE = valid_hex(pc3.text_input("Reference", DEFAULT_COLOURS["reference"]), DEFAULT_COLOURS["reference"])
        sw1, sw2, sw3 = st.columns(3)
        for col, c in zip((sw1, sw2, sw3), (C_CMB, C_RF, C_LINE)):
            col.markdown(f'<div style="background:{c};height:14px;border-radius:3px"></div>', unsafe_allow_html=True)

    with st.expander("Advanced plot options"):
        axis_cap = st.number_input("Maximum recharge shown on axes (mm/y)", 10.0, 100000.0, 2000.0, 100.0,
                                   help="Axes showing recharge are limited to this value. Larger estimates, usually "
                                        "from very low chloride, are clipped from view but remain in the results.")
        show_ids = st.checkbox("Label markers with bore IDs", value=False,
                               help="Adds the ID beside each marker on the map and scatter plots. Best for small "
                                    "numbers of bores; labels can overlap when many are plotted.")
    label_mode = "markers+text" if show_ids else "markers"
    label_font = dict(size=10)

    st.subheader("Recharge by bore")
    b1, b2, b3 = st.columns([2, 2, 1], vertical_alignment="bottom")
    bar_est = b1.selectbox("Estimate", ["Stochastic CMB (R_RC)", "Regular CMB (R_CMB)", "Random forest (RF)"],
                           key="bar_est")
    bar_sort = b2.selectbox("Order", ["Input order", "Recharge, descending", "Rainfall, ascending"], key="bar_sort")
    bar_log = b3.checkbox("Log y", value=False, key="bar_log")
    centre, lo_col, hi_col = {
        "Stochastic CMB (R_RC)": ("R_RC_50_mm_y", "R_RC_5_mm_y", "R_RC_95_mm_y"),
        "Regular CMB (R_CMB)": ("R_CMB_mean_mm_y", "R_CMB_5_mm_y", "R_CMB_95_mm_y"),
        "Random forest (RF)": ("RF_R50_mm_y", "RF_R5_mm_y", "RF_R95_mm_y"),
    }[bar_est]
    bd = ok.dropna(subset=[centre])
    if bar_sort == "Recharge, descending":
        bd = bd.sort_values(centre, ascending=False)
    elif bar_sort == "Rainfall, ascending":
        bd = bd.sort_values("Rain_mm_y")
    fb = go.Figure(go.Bar(
        x=bd["ID"].astype(str), y=bd[centre], name=bar_est, marker=dict(color=C_CMB, line=dict(color=C_LINE, width=0.5)),
        error_y=dict(type="data", symmetric=False, array=bd[hi_col] - bd[centre], arrayminus=bd[centre] - bd[lo_col],
                     color=C_LINE, thickness=1.2, width=4),
        customdata=np.stack([bd[lo_col], bd[hi_col], bd["Rain_mm_y"]], axis=1),
        hovertemplate="<b>%{x}</b><br>%{y:.2f} mm/y (5th %{customdata[0]:.2f}, 95th %{customdata[1]:.2f})"
                      "<br>Rain %{customdata[2]:.0f} mm/y<extra></extra>"))
    label_px = min(220, 20 + 7 * int(bd["ID"].astype(str).str.len().max() if len(bd) else 6))
    fb.update_layout(**{**PLOT_LAYOUT, "margin": dict(l=60, r=20, t=50, b=label_px + 50)},
                     xaxis_title="Bore", yaxis_title="Recharge (mm/y)", bargap=0.35, height=440 + label_px,
                     xaxis=dict(type="category", tickangle=-90, automargin=False))
    fb.update_yaxes(type="log" if bar_log else "linear", range=cap_range(bd[[lo_col, hi_col]].to_numpy().ravel(), axis_cap, bar_log))
    st.plotly_chart(brand(fb), width="stretch")
    st.caption("Bars: median (stochastic CMB, random forest) or mean (regular CMB). Whiskers: 5th to 95th percentile. "
               "Whiskers that would cross zero on a log axis are truncated by the axis.")
    if len(bd) > 80:
        st.caption(f"{len(bd)} bores plotted; labels may overlap. Use the plot's zoom, or sort and read the hover text.")

    st.subheader("Bore map")
    m1, m2, m3 = st.columns([2, 2, 1], vertical_alignment="bottom")
    cvar = m1.selectbox("Colour bores by", ["R_RC_50_mm_y", "R_CMB_mean_mm_y", "RF_R50_mm_y", "RP_RC_50",
                                             "Clgw_mg_L", "Rain_mm_y", "fill_radius_climate_km"])
    bgopt = m2.selectbox("Background", ["Random forest R50", "Rainfall", "Land only"])
    logc = m3.checkbox("Log colour", value=cvar not in ("fill_radius_climate_km", "Rain_mm_y"))
    bgkey = {"Random forest R50": "rf_r50", "Rainfall": "rain", "Land only": None}[bgopt]
    with_coords = res[np.isfinite(res["Lat"]) & np.isfinite(res["Lon"])]
    if len(with_coords):
        st.plotly_chart(brand(bore_map(with_coords, grids(), cvar, cvar, bgkey, logc, point_colour=C_CMB, show_ids=show_ids)),
                        width="stretch")
        st.caption("Main panel: bores over the chosen grid (greys, darker is higher). Inset: Australia with the "
                   "main panel's extent outlined. Bores without a result are still plotted; hover for status.")

    st.subheader("Stochastic CMB versus random forest recharge")
    logax = st.checkbox("Log axes", value=True, key="log1")
    d = ok[ok["RF_R50_mm_y"].notna()]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["RF_R50_mm_y"], y=d["R_RC_50_mm_y"], mode=label_mode, name="Bores (R50, bars 5th to 95th)",
        marker=marker(C_CMB), text=d["ID"].astype(str), textposition="top right", textfont=label_font,
        error_y=dict(type="data", symmetric=False, array=d["R_RC_95_mm_y"] - d["R_RC_50_mm_y"],
                     arrayminus=d["R_RC_50_mm_y"] - d["R_RC_5_mm_y"], color=C_CMB, thickness=1, width=0),
        error_x=dict(type="data", symmetric=False, array=d["RF_R95_mm_y"] - d["RF_R50_mm_y"],
                     arrayminus=d["RF_R50_mm_y"] - d["RF_R5_mm_y"], color=C_RF, thickness=1, width=0),
        customdata=np.stack([d["ID"], d["Clgw_mg_L"], d["Rain_mm_y"]], axis=1),
        hovertemplate="<b>%{customdata[0]}</b><br>CMB R50 %{y:.1f} mm/y<br>RF R50 %{x:.1f} mm/y"
                      "<br>Cl_gw %{customdata[1]:.1f} mg/L<br>Rain %{customdata[2]:.0f} mm/y<extra></extra>"))
    if len(d):
        lo = max(1e-3, float(np.nanmin([d["RF_R5_mm_y"].min(), d["R_RC_5_mm_y"].min()])) * 0.8)
        hi = min(axis_cap, float(np.nanmax([d["RF_R95_mm_y"].max(), d["R_RC_95_mm_y"].max()])) * 1.2)
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="1:1",
                                 line=dict(color=C_LINE, width=1.5, dash="dash")))
    fig.update_layout(**PLOT_LAYOUT, xaxis_title="Random forest median recharge (mm/y)\n (Lee et al. (2024)",
                      yaxis_title="Stochastic CMB median recharge (mm/y)", height=520,
                      legend=dict(orientation="h", y=1.08))
    both = d[["RF_R5_mm_y", "RF_R95_mm_y", "R_RC_5_mm_y", "R_RC_95_mm_y"]].to_numpy().ravel()
    rng = cap_range(both, axis_cap, logax)
    fig.update_xaxes(type="log" if logax else "linear", range=rng)
    fig.update_yaxes(type="log" if logax else "linear", range=rng)
    st.plotly_chart(brand(fig), width="stretch")
    st.caption("Vertical bars: 5th to 95th percentile of the CMB realisations at the bore. Horizontal bars: "
               "5th to 95th percentile random forest grids at the same pixel. The RF model was trained on CMB "
               "estimates of this kind, so agreement is expected on average but not bore by bore.")

    st.subheader("Recharge against climate")
    which = st.radio("Recharge estimate", ["Stochastic CMB (R_RC_50)", "Regular CMB (R_CMB_mean)", "Random forest (RF_R50)"],
                     horizontal=True)
    ycol = {"Stochastic CMB (R_RC_50)": "R_RC_50_mm_y", "Regular CMB (R_CMB_mean)": "R_CMB_mean_mm_y",
            "Random forest (RF_R50)": "RF_R50_mm_y"}[which]
    f1 = go.Figure(go.Scatter(x=ok["Rain_mm_y"], y=ok[ycol], mode=label_mode, name=which, marker=marker(C_CMB),
                              text=ok["ID"].astype(str), textposition="top right", textfont=label_font, hovertemplate="<b>%{text}</b><br>P %{x:.0f} mm/y<br>R %{y:.2f} mm/y<extra></extra>"))
    f1.update_layout(**PLOT_LAYOUT, xaxis_title="Mean annual rainfall (mm/y)", yaxis_title="Recharge (mm/y)",
                     yaxis_type="log", height=420, title="Recharge versus rainfall",
                     yaxis_range=cap_range(ok[ycol], axis_cap, True))
    f2 = go.Figure(go.Scatter(x=ok["Aridity"], y=ok[ycol] / ok["Rain_mm_y"], mode=label_mode, name=which,
                              marker=marker(C_CMB), text=ok["ID"].astype(str), textposition="top right",
                              textfont=label_font,
                              hovertemplate="<b>%{text}</b><br>PET/P %{x:.2f}<br>R/P %{y:.3f}<extra></extra>"))
    f2.update_layout(**PLOT_LAYOUT, xaxis_title="Aridity index (PET/P)", yaxis_title="Recharge / rainfall",
                     yaxis_type="log", xaxis_type="log", height=420, title="Recharge ratio versus aridity")
    st.plotly_chart(brand(f1), width="stretch")
    st.plotly_chart(brand(f2), width="stretch")

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
                         xaxis_range=cap_range(vals, axis_cap, False),
                         title=f"{bid}: Cl_gw {r['Clgw_mg_L']:.1f} mg/L, P {r['Rain_mm_y']:.0f} mm/y, "
                               f"RC {r['RC']:.3f}, deposition {r['ClDep_mean_kg_ha_y']:.1f} kg/ha/y")
        st.plotly_chart(brand(fh), width="stretch")

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

- Input is reduced to ID, Lat, Lon, Clgw. Repeat samples per ID are averaged; with three or more samples
  their standard deviation is used for the Cl_gw distribution.
- For bores with fewer than three samples the CV of groundwater chloride is a user setting (default 0.37)
  rather than being derived from bores with more than 10 samples in the input file. 0.37 reproduces the
  national analysis.
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
