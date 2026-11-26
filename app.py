"""CMBEAR online: Chloride mass balance Estimator of Australian Recharge."""
from __future__ import annotations

import io
import re
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from cmbear import CMBSettings, load_grids, parse_input, run_cmbear
from cmbear import run_counter
from cmbear.cmb import COLUMN_NOTES
from cmbear.inputs import read_table
from cmbear.maplayers import (MAX_LAYERS, MAX_UPLOAD_MB, POLY_TYPES, TOWN_TYPES, MapError, NeedsCRS, Upload,
                               layer_paths, read_towns, read_vector, towns_template_bytes)
from cmbear.plotting import BERGHUIJS_ALPHA, BERGHUIJS_BETA, BG_CMAPS, berghuijs_curve, bore_map, fit_berghuijs

LOGO_DIR = Path(__file__).resolve().parent / "logo"
# The template download and the example data are the same file.
EXAMPLE_CSV = Path(__file__).resolve().parent / "examples" / "example_input.csv"
BADGE, FAVICON = LOGO_DIR / "cmbear_badge.png", LOGO_DIR / "favicon.png"

st.set_page_config(page_title="CMBEAR online", page_icon=str(FAVICON) if FAVICON.exists() else ":droplet:")

thisyear = current_year = str(datetime.now().year)

# Default plot colours (users can override with their own hex codes under 4. Plot options in the sidebar).
DEFAULT_COLOURS = {"primary": "#0089D0", "secondary": "#F37021", "reference": "#9a9a9a"}
# Map layers: default colours of the up to three polygon or line layers (as in the Hydrochem Workbench)
LAYER_COLOURS = ["#FCB711", "#0089D0", "#CC004C"]
for _k in range(MAX_LAYERS):
    for _key, _val in ((f"opt_map_poly{_k}_edge", LAYER_COLOURS[_k]), (f"opt_map_poly{_k}_face", LAYER_COLOURS[_k]),
                       (f"opt_map_poly{_k}_alpha", 0.25), (f"opt_map_poly{_k}_width", 1.5),
                       (f"opt_map_poly{_k}_label", f"Layer {_k + 1}")):
        st.session_state.setdefault(_key, _val)
MIME = {"png": "image/png", "svg": "image/svg+xml"}
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
# White background on every plot, on screen and in downloads, whatever the app's light/dark theme.
PLOT_LAYOUT = dict(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
                   font=dict(size=13, color="#222222"), margin=dict(l=60, r=20, t=50, b=70))
DL_LAYOUT_W = 800   # pixel width figures are laid out at for downloads, before scaling to the chosen dpi
ATTRIBUTION = "Produced using CMBEAR online - Dylan Irvine"
# Colour-bar / axis titles for result columns and background grids on the bore map.
MAP_LABELS = {
    "R_RC_50_mm_y": "Stochastic CMB recharge, median (mm/y)",
    "R_CMB_mean_mm_y": "Regular CMB recharge, mean (mm/y)",
    "RF_R50_mm_y": "Random forest recharge, median (mm/y)",
    "RP_RC_50": "Recharge/rainfall, stochastic CMB median (-)",
    "Clgw_mg_L": "Groundwater chloride (mg/L)",
    "Rain_mm_y": "Rainfall (mm/y)",
    "fill_radius_climate_km": "Climate gap-fill radius (km)",
}
RECHARGE_COLS = {"R_RC_50_mm_y", "R_CMB_mean_mm_y", "RF_R50_mm_y"}
FILL_LABELS = {"Mean of neighbouring cells": "neighbours", "Assume RC value, leave climate blank": "assume_rc",
               "Leave blank": "none"}


@st.cache_resource(show_spinner="Loading national grids...")
def grids():
    return load_grids()


@st.cache_data(show_spinner=False)
def template_csv() -> bytes:
    """The template users download, which is also the example data (examples/example_input.csv)."""
    return EXAMPLE_CSV.read_bytes()


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


def log_ticks(fig, axis, values=None, rng=None):
    """Label a log axis at 1, 2, 5 per decade (decades only if that would crowd) instead of plotly's bare digits.

    axis: "x" or "y". Pass the data (values) or a log10 range (rng) to determine the span.
    """
    if rng is None:
        v = np.asarray(values, float); v = v[np.isfinite(v) & (v > 0)]
        if v.size == 0:
            return
        rng = [np.log10(v.min()), np.log10(v.max())]
    lo, hi = 10 ** rng[0], 10 ** rng[1]
    k0, k1 = int(np.floor(rng[0])), int(np.ceil(rng[1]))
    span = rng[1] - rng[0]
    mults = (1,) if span > 4 else (1, 2, 5) if span > 1.2 else (1, 1.5, 2, 3, 4, 5, 6, 8)
    vals = [m * 10.0 ** k for k in range(k0, k1 + 1) for m in mults if lo * 0.999 <= m * 10.0 ** k <= hi * 1.001]
    text = [f"{t:g}" for t in vals]
    (fig.update_xaxes if axis == "x" else fig.update_yaxes)(tickvals=vals, ticktext=text, minor=dict(ticks=""))


def valid_hex(text, fallback):
    text = (text or "").strip()
    if not text.startswith("#"):
        text = "#" + text
    return text if re.fullmatch(r"#[0-9a-fA-F]{6}", text) else fallback


def map_layers():
    """The loaded polygon or line layers with their styles, in order, for bore_map."""
    ss = st.session_state
    out = []
    for k in range(MAX_LAYERS):
        if ss.get(f"map_poly_{k}"):
            out.append(dict(paths=ss[f"map_poly_{k}"][1], edge=ss[f"opt_map_poly{k}_edge"],
                            face=ss[f"opt_map_poly{k}_face"], alpha=ss[f"opt_map_poly{k}_alpha"],
                            width=ss[f"opt_map_poly{k}_width"], label=ss[f"opt_map_poly{k}_label"] or f"Layer {k + 1}"))
    return out


def read_towns_upload(files, epsg=None):
    f = files[0]
    towns = read_towns(io.BytesIO(f.data), f.name)
    return f"{f.name} ({len(towns)} town(s))", towns


def read_layer_upload(files, epsg=None):
    gdf, info = read_vector(files, epsg=epsg)
    return info, layer_paths(gdf)


def layer_upload(slot, label, types, reader):
    """One uploaded map layer kept in session_state[slot] as (description, data). A file with no coordinate
    reference system waits in slot + '_pending' until the user gives an EPSG code. Adapted from the Hydrochem
    Workbench. Uploaded files stay in this session's memory only."""
    ss = st.session_state
    ss.setdefault(slot + "_n", 0)
    if ss.get(slot):
        st.markdown(f"**Loaded:** {ss[slot][0]}")
        if st.button("Remove", key=slot + "_remove"):
            ss.pop(slot)
            st.rerun()
        return
    pending = ss.get(slot + "_pending")
    if pending:
        st.warning(pending[1])
        epsg = st.number_input("EPSG code of the file (e.g. 28353 for GDA94 / MGA zone 53, 7853 for GDA2020 / "
                               "MGA zone 53, 4283 for GDA94 latitude and longitude)", min_value=1024,
                               max_value=99999, value=4326, step=1, key=slot + "_epsg")
        c1, c2 = st.columns(2)
        if c1.button("Use this EPSG code", key=slot + "_use"):
            try:
                ss[slot] = reader(pending[0], epsg=int(epsg))
            except MapError as exc:
                st.error(str(exc))
            else:
                ss.pop(slot + "_pending")
                st.rerun()
        if c2.button("Cancel", key=slot + "_cancel"):
            ss.pop(slot + "_pending")
            st.rerun()
        return
    multi = len(types) > 2
    up = st.file_uploader(label, type=types, accept_multiple_files=multi, key=f"{slot}_up_{ss[slot + '_n']}")
    files = up if multi else ([up] if up is not None else [])
    if files:
        kept = [Upload(f.name, f.getvalue()) for f in files]
        try:
            ss[slot] = reader(kept)
        except NeedsCRS as exc:
            ss[slot + "_pending"] = (kept, str(exc))
        except MapError as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read that file: {exc}")
            return
        ss[slot + "_n"] += 1
        st.rerun()


def export_opts(fig, fmt, width_cm, dpi):
    """Image export options for a figure: laid out DL_LAYOUT_W pixels wide (fonts keep their on-screen
    proportion), then scaled so a PNG has the pixels for the print width at the chosen dpi. A 16 cm figure
    at 400 dpi is about 2520 pixels wide. SVG is vector, so it is not scaled."""
    fmt = fmt.lower()
    scale = 1.0 if fmt == "svg" else width_cm / 2.54 * dpi / DL_LAYOUT_W
    return dict(format=fmt, width=DL_LAYOUT_W, height=int(fig.layout.height or 500), scale=scale)


@st.cache_resource(show_spinner="Checking figure export...")
def export_available() -> bool:
    """True if kaleido and a Chrome/Chromium browser are available to render figures on the server."""
    try:
        go.Figure().to_image(format="png", width=40, height=40)
    except Exception:  # noqa: BLE001
        return False
    return True


def figure_bytes(fig, opts):
    """Deferred download data: rendered only when the button is clicked."""
    return lambda: fig.to_image(**opts)


def everything_zip(items, xlsx):
    """Deferred download data for the zip: the results spreadsheet plus every figure as drawn.

    items: list of (file stem, figure, export options). Figures are rendered in one batch, which is much
    faster than one at a time because the browser is started once."""
    def build():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("cmbear_results.xlsx", xlsx)
            if items:
                with tempfile.TemporaryDirectory() as tmp:
                    paths = [Path(tmp) / f"{n}.{o['format']}" for n, _, o in items]
                    pio.write_images([f for _, f, _ in items], paths,
                                     format=[o["format"] for *_, o in items], width=[o["width"] for *_, o in items],
                                     height=[o["height"] for *_, o in items], scale=[o["scale"] for *_, o in items])
                    for p in paths:
                        z.write(p, f"figures/{p.name}")
        return buf.getvalue()
    return build


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
    "from the chloride mass balance (CMB) method. For each bore, the app extracts chloride deposition, rainfall, "
    "PET (Penman-Monteith), aridity and runoff coefficient from national 0.05 degree grids, calculates the regular CMB recharge "
    "and the stochastic CMBEAR2 recharge with a runoff term, and reads the matching values from the random "
    "forest recharge model of Lee et al. (2024).\n\n"
    
    "Input: a CSV or Excel file with columns for bore ID, latitude, longitude and groundwater chloride (mg/L), "
    "such as ID, Lat, Lon and Clgw. Repeat samples under the same ID are "
    "averaged, and where there are three or more their standard deviation is used in place of the blanket "
    "coefficient of variation (CV = standard deviation / mean). \n \n"
    
    "Please cite as:\n\n"
    f"Irvine, D.J. ({thisyear}) CMBEAR online, https://cmbear-app.streamlit.app/ \n\n"
    "Lee, S., Irvine, D.J., Duvert, C., Rau, G.C., Cartwright, I. (2024). A high-resolution map of diffuse "
    "groundwater recharge rates for Australia. Hydrology and Earth System Sciences, 28, 1771-1790. "
    "https://doi.org/10.5194/hess-28-1771-2024\n\n"
    "Other references to consider are in the dropdown below."
)

# Shared run counter. Read once per session; shows nothing if not configured.
if "run_count" not in st.session_state:
    st.session_state["run_count"] = run_counter.get_runs()
if st.session_state["run_count"] is not None:
    st.caption(f"This tool has been run {st.session_state['run_count']:,} times.")

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
        "https://doi.org/10.1016/j.dib.2022.108189/ \n\n"
    )

with st.expander("**Privacy**"):
    st.write(
        "This app does not store, log or record who uses it, uploaded files, bore locations or results. "
        "Your file is processed in memory for your session only and is discarded when you close or reload "
        "the page. The only thing recorded is the total number of runs, a single number with no "
        "information about the run attached.\n\n"
        "The app is hosted on Streamlit Community Cloud (Snowflake), so your file passes through their "
        "servers while it is processed, and their own infrastructure is covered by the Streamlit privacy "
        "notice. To keep your data entirely on your own computer, contact the author about running the "
        "app locally. The source code can be read on [GitHub](https://github.com/DylanIrvine/cmbear-app)."
    )

# --------------------------------------------------------------------------- #
# sidebar: 1. bore data, 2. settings, 3. run, 4. plot options
# --------------------------------------------------------------------------- #
bores, msgs, read_error = None, [], None
with st.sidebar:
    st.header("1. Bore data")
    st.markdown("Requires columns for bore ID, latitude, longitude and groundwater chloride (mg/L). Common "
                "names are recognised (for example ID or bore_id, Lat or latitude, Lon or longitude, Clgw or "
                "Chloride). Download the template as a starting point.")
    st.download_button("Download template", template_csv(), "cmbear_template.csv", "text/csv")
    use_example = st.checkbox("Use the template as example data", value=False,
                              help="Runs the app on the template's bores so you can try it without a file of "
                                   "your own. Untick to upload your own spreadsheet.")
    if use_example:
        source_name, source = "example_input.csv", template_csv()
        st.caption("Using the example data. Press Calculate recharge under 3. Run.")
    else:
        upload = st.file_uploader("Bore spreadsheet (CSV or Excel)", type=["csv", "xlsx", "xls"])
        source_name, source = (upload.name, upload) if upload is not None else (None, None)
    if source is not None:
        try:
            bores, msgs = parse_input(read_table(source, source_name))
        except Exception as e:  # noqa: BLE001
            read_error = f"Could not read the input file: {e}"
            st.error(read_error)
        else:
            for m in msgs:
                st.caption(m)

    st.header("2. Settings")
    st.markdown("Update any of the other optional parameters here.")
    cv = st.slider("Cl_gw coefficient of variation", 0.05, 1.00, 0.37, 0.01,
                   help="Applied to groundwater chloride for bores with fewer than three samples. 0.37 is the "
                        "national value derived from bores with more than 10 samples in Lee et al. (2024).")
    n_real = st.select_slider("Realisations per bore", [250, 500, 1000, 2000, 5000], value=1000)
    a_lo, a_hi = st.slider("Runoff factor alpha range", 0.0, 1.0, (0.33, 0.66), 0.01,
                           help="Fraction of the runoff coefficient that removes chloride. Lee et al. used "
                                "U(0.33, 0.66).")
    with st.expander("Cells without climate or RC data"):
        st.caption("1853 coastal and island cells have chloride deposition but no rainfall, PET, aridity or runoff "
                   "coefficient; 6400 land cells have no random forest value.")
        fill_method = st.radio("Handling", list(FILL_LABELS),
                               help="Averages the valid cells in a 3 x 3 window around the bore; if none, the window "
                                    "grows one ring at a time up to the maximum. The window size used is recorded.")
        fill_max = st.slider("Maximum window radius (cells)", 1, 10, 2, 1,
                             help="2 cells is a 5 x 5 window, about 11 km across.")
        fill_rc = st.number_input("Assumed RC when no data", 0.0, 1.0, 0.0, 0.01, format="%.2f")
        fill_rf = st.checkbox("Fill RF values too", value=True)

    with st.expander("Advanced"):
        seed_d = st.number_input("Seed (deposition)", value=1, step=1)
        seed_c = st.number_input("Seed (Cl_gw)", value=3100, step=1)
        seed_a = st.number_input("Seed (alpha)", value=2, step=1)
        keep_real = st.checkbox("Keep realisations (needed for the histogram and Excel export)", value=True)

    settings = CMBSettings(n_realisations=int(n_real), cv_clgw=cv, alpha_low=a_lo, alpha_high=a_hi,
                           seed_deposition=int(seed_d), seed_clgw=int(seed_c), seed_alpha=int(seed_a),
                           keep_realisations=keep_real, fill_method=FILL_LABELS[fill_method],
                           fill_max_cells=int(fill_max), fill_rc_value=float(fill_rc), fill_rf=fill_rf)

    st.header("3. Run")
    st.markdown("Not much to be said here. Click the button to run the analysis.")
    if bores is None:
        st.caption("Upload a bore spreadsheet, or tick Use the template as example data, under 1. Bore data "
                   "to enable the run.")
    run_clicked = st.button("Calculate recharge", type="primary", disabled=bores is None)

    st.header("4. Plot options")
    with st.expander("Plot colours"):
        st.caption("Enter hex codes (for example your organisation's colours). Primary is used for CMB results, "
                   "secondary for random forest values, reference for 1:1 lines and percentile markers.")
        C_CMB = valid_hex(st.text_input("Primary", DEFAULT_COLOURS["primary"]), DEFAULT_COLOURS["primary"])
        C_RF = valid_hex(st.text_input("Secondary", DEFAULT_COLOURS["secondary"]), DEFAULT_COLOURS["secondary"])
        C_LINE = valid_hex(st.text_input("Reference", DEFAULT_COLOURS["reference"]), DEFAULT_COLOURS["reference"])
        st.markdown("".join(f'<div style="background:{c};height:14px;border-radius:3px;margin-bottom:4px"></div>'
                            for c in (C_CMB, C_RF, C_LINE)), unsafe_allow_html=True)
    with st.expander("Axes and labels"):
        axis_cap = st.number_input("Maximum recharge shown on axes (mm/y)", 10.0, 100000.0, 2000.0, 100.0,
                                   help="Axes showing recharge are limited to this value. Larger estimates, usually "
                                        "from very low chloride, are clipped from view but remain in the results.")
        show_ids = st.checkbox("Label markers with bore IDs", value=False,
                               help="Adds the ID beside each marker on the map and scatter plots. Best for small "
                                    "numbers of bores; labels can overlap when many are plotted.")
    with st.expander("Figure downloads"):
        st.caption("Used by the download button under each plot, the Download everything zip and the camera "
                   "icon at the top right of each plot.")
        dl_fmt = st.selectbox("Format", ["PNG", "SVG"], key="dl_fmt",
                              help="SVG is vector, so it is sharp at any size and dpi does not apply.")
        dl_cm = st.number_input("Print width (cm)", 5.0, 40.0, 16.0, 0.5, key="dl_cm",
                                help="Width the figure will be printed at. With dpi this sets the pixel size "
                                     "of PNG downloads.")
        dl_dpi = st.number_input("Resolution (dpi)", 72, 1200, 400, 50, key="dl_dpi")

if bores is None and read_error is None:
    st.info("Upload a bore spreadsheet in the sidebar (1. Bore data) to begin, or tick **Use the template as "
            "example data** there to try the app straight away.")
if bores is not None:
    with st.expander("Preview parsed input"):
        st.dataframe(bores, width="stretch", height=240)

if run_clicked:
    # Caught so that an unexpected failure is not written to the server log as a traceback,
    # which could quote values from the user's file. Nothing about the input is logged.
    try:
        res, real = run_cmbear(bores, grids(), settings)
    except Exception:  # noqa: BLE001
        st.error("The calculation failed for this file. Please check the input values and settings, "
                 "or contact the developer with a description of the problem (not the data).")
    else:
        st.session_state["run"] = dict(res=res, real=real, settings=settings, msgs=msgs, file=source_name)
        new_total = run_counter.increment_runs()
        if new_total is not None:
            st.session_state["run_count"] = new_total

run = st.session_state.get("run")
if run is not None:
    res, real, run_settings = run["res"], run["real"], run["settings"]
    if run_settings != settings or run["file"] != source_name:
        st.caption("Settings or input have changed since the last run. Press Calculate recharge to update.")

    bad = res["status"].str.startswith(("outside", "no CMB", "no climate"))
    if bad.any():
        st.warning(f"{int(bad.sum())} bore(s) have no CMB result (outside the grid, over NODATA such as ocean, "
                   "or on a coastal cell left unfilled). See the status column.")
    filled = res["status"].str.contains("mean of cells|assumed", regex=True)
    if filled.any():
        st.info(f"{int(filled.sum())} bore(s) sit on cells without climate/RC or RF data and were filled as set "
                "under 2. Settings in the sidebar. See status and the fill_radius columns.")

    st.header("Results")
    st.dataframe(res, width="stretch", height=360)
    ok_all = res[res["status"].str.startswith("ok")]
    if len(ok_all):
        summ = ok_all[["R_CMB_mean_mm_y", "R_RC_50_mm_y", "RF_R50_mm_y", "Rain_mm_y", "RP_RC_50", "RF_RP50"]] \
            .describe(percentiles=[0.05, 0.5, 0.95]).T
        with st.expander("Summary statistics"):
            st.dataframe(summ.apply(lambda c: c.map(lambda v: f"{v:.3g}")), width="stretch")

    # ----------------------------------------------------------------------- #
    # downloads
    # ----------------------------------------------------------------------- #
    st.header("Downloads")
    d1, d2, d3 = st.columns([1, 1, 1], vertical_alignment="center")
    d1.download_button("Results (CSV)", res.to_csv(index=False).encode(), "cmbear_results.csv", "text/csv",
                       width="stretch")
    inc = d3.checkbox("Include realisations in Excel", value=False,
                      help="Adds a sheet with every realisation (bores x realisations). Can be large. "
                           "Also applies to the Excel file in Download everything.")
    xlsx_bytes = results_xlsx(res, run_settings, run["msgs"], real, inc)
    d2.download_button("Results (Excel)", xlsx_bytes, "cmbear_results.xlsx", XLSX_MIME, width="stretch")

    # ----------------------------------------------------------------------- #
    # plots
    # ----------------------------------------------------------------------- #
    st.header("Plots")
    st.markdown(
        "**These plots are interactive.** Hover over a marker or bar to read its values, drag a box to zoom, "
        "double-click to reset the view, and click a legend entry to hide or show that series. The toolbar "
        "at the top right of each plot appears when the mouse is over it.\n\n"
        "The **Download figure** button under each plot saves the whole figure as it is set up here, at the "
        "format, print width and resolution under *4. Plot options* in the sidebar. It does not keep a zoom "
        "or hidden series; to save the current zoomed view use the camera icon in the plot's toolbar instead.")
    ok = ok_all.reset_index()
    label_mode = "markers+text" if show_ids else "markers"
    can_export = export_available()
    if not can_export:
        st.warning("Figure files cannot be made on this server (kaleido or Chrome/Chromium is missing), so the "
                   "Download figure buttons and the figures in Download everything are unavailable. The camera "
                   "icon in each plot's toolbar still works.")
    fig_items = []   # (file stem, figure, export options) for every figure drawn, for Download everything

    def show(fig, name, **kw):
        """Draw a figure with a Download figure button beneath it, and record it for Download everything.

        Downloads (button, zip and camera icon) are laid out DL_LAYOUT_W pixels wide and scaled to the chosen
        print width and dpi (see export_opts). The button's image is rendered only when it is clicked.
        """
        brand(fig)
        opts = export_opts(fig, dl_fmt, dl_cm, dl_dpi)
        cfg = dict(displaylogo=False, toImageButtonOptions=dict(filename=name, **opts))
        kw.setdefault("theme", None)   # Streamlit's theme would replace the white background in dark mode
        st.plotly_chart(fig, width="stretch", config=cfg, **kw)
        st.download_button(f"Download figure ({dl_fmt})", figure_bytes(fig, opts), f"{name}.{opts['format']}",
                           MIME[opts["format"]], key=f"dl_{name}", disabled=not can_export, on_click="ignore",
                           icon=":material/download:")
        fig_items.append((name, fig, opts))
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
    bar_rng = cap_range(bd[[lo_col, hi_col]].to_numpy().ravel(), axis_cap, bar_log)
    fb.update_yaxes(type="log" if bar_log else "linear", range=bar_rng)
    if bar_log and bar_rng:
        log_ticks(fb, "y", rng=bar_rng)
    show(fb, "cmbear_recharge_by_bore")
    st.caption("Bars: median (stochastic CMB, random forest) or mean (regular CMB). Whiskers: 5th to 95th percentile. "
               "Whiskers that would cross zero on a log axis are truncated by the axis.")
    if len(bd) > 80:
        st.caption(f"{len(bd)} bores plotted; labels may overlap. Use the plot's zoom, or sort and read the hover text.")

    st.subheader("Bore map")
    m1, m2, m3 = st.columns([2, 2, 1], vertical_alignment="bottom")
    cvar = m1.selectbox("Colour bores by", list(MAP_LABELS), format_func=MAP_LABELS.get)
    bgopt = m2.selectbox("Background", ["Random forest recharge, median", "Rainfall", "Land only"])
    logc = m3.checkbox("Log colour", value=cvar not in ("fill_radius_climate_km", "Rain_mm_y"))
    bgkey = {"Random forest recharge, median": "rf_r50", "Rainfall": "rain", "Land only": None}[bgopt]
    bg_label = {"rf_r50": "Random forest recharge, median (mm/y)", "rain": "Rainfall (mm/y)", None: None}[bgkey]
    m4, _ = st.columns([2, 3])
    bg_cmap = m4.selectbox("Background colour map", list(BG_CMAPS), key="map_bg_cmap", disabled=bgkey is None,
                           help="Colours for the background grid. Greys keeps the bores in front; the others show "
                                "more detail in the grid. '_r' maps are reversed so that darker is higher.")
    # One colour range for bores and background when they are the same quantity, so shades can be compared.
    shared = (bgkey == "rf_r50" and cvar in RECHARGE_COLS) or (bgkey == "rain" and cvar == "Rain_mm_y")
    with_coords = res[np.isfinite(res["Lat"]) & np.isfinite(res["Lon"])]
    towns_kept = st.session_state.get("map_towns")
    layers = map_layers()
    if len(with_coords):
        show(bore_map(with_coords, grids(), cvar, MAP_LABELS[cvar], bgkey, logc, point_colour=C_CMB,
                      show_ids=show_ids, background_label=bg_label, shared_range=shared, background_cmap=bg_cmap,
                      towns=towns_kept[1] if towns_kept else None, layers=layers),
             "cmbear_bore_map")
        st.caption("Main panel: bores over the chosen grid (darker is higher), each with its own colour bar in "
                   "discrete classes. Cells with no grid value are left white. "
                   + ("Bores and background share one colour range. " if shared else "")
                   + "Inset: Australia with the main panel's extent outlined; scale bar true at 27°S. Bores without a result are still "
                     "plotted; hover for status. Towns and polygon or line layers can be added below.")

    # Fixed expander titles: a title that changes (e.g. with a count) makes a new element, which closes it
    with st.expander("**Towns**"):
        st.write("A csv with the columns Name, Lat and Lon (decimal degrees). Towns inside the map extent are "
                 "drawn as black squares with their name.")
        st.download_button("Download the towns template", towns_template_bytes(), "cmbear_towns_template.csv",
                           "text/csv")
        layer_upload("map_towns", "Upload the towns file", TOWN_TYPES, read_towns_upload)

    with st.expander("**Polygons and lines**",
                     expanded=any(st.session_state.get(f"map_poly_{k}_pending") for k in range(MAX_LAYERS))):
        st.write(f"Up to {MAX_LAYERS} layers of polygons or lines, for example a basin boundary, rivers or roads. "
                 "Each is a shapefile as a .zip or as its separate files (.shp, .shx, .dbf and .prj together), a "
                 f"GeoPackage, GeoJSON or KML, up to {MAX_UPLOAD_MB} MB, and is converted to latitude and longitude "
                 "from whatever coordinate reference system the file states. Layer 1 is drawn first, with the "
                 "others on top, all beneath the bores. Lines take the edge colour.")
        loaded = [bool(st.session_state.get(f"map_poly_{k}")) for k in range(MAX_LAYERS)]
        for k in range(MAX_LAYERS):
            # the loaded layers, and one empty slot after them
            if not (loaded[k] or all(loaded[:k])):
                continue
            st.markdown(f"**Layer {k + 1}**")
            layer_upload(f"map_poly_{k}", f"Upload the file(s) for layer {k + 1}", POLY_TYPES, read_layer_upload)
            if loaded[k]:
                c1, c2, c3 = st.columns(3)
                c1.color_picker("Edge colour", key=f"opt_map_poly{k}_edge")
                c2.color_picker("Fill colour", key=f"opt_map_poly{k}_face")
                c3.slider("Fill opacity (alpha)", 0.0, 1.0, step=0.05, key=f"opt_map_poly{k}_alpha",
                          help="0 is no fill (outline only), 1 is solid. Not used for lines.")
                c1, c2 = st.columns(2)
                c1.slider("Edge width", 0.5, 5.0, step=0.25, key=f"opt_map_poly{k}_width")
                c2.text_input("Name in the legend", key=f"opt_map_poly{k}_label")

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
    fig.update_layout(**PLOT_LAYOUT, xaxis_title="Random forest median recharge (mm/y), Lee et al. (2024)",
                      yaxis_title="Stochastic CMB median recharge (mm/y)", height=520,
                      legend=dict(orientation="h", y=1.08))
    both = d[["RF_R5_mm_y", "RF_R95_mm_y", "R_RC_5_mm_y", "R_RC_95_mm_y"]].to_numpy().ravel()
    rng = cap_range(both, axis_cap, logax)
    fig.update_xaxes(type="log" if logax else "linear", range=rng)
    fig.update_yaxes(type="log" if logax else "linear", range=rng)
    if logax and rng:
        log_ticks(fig, "x", rng=rng); log_ticks(fig, "y", rng=rng)
    show(fig, "cmbear_cmb_vs_rf")
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
    f1_rng = cap_range(ok[ycol], axis_cap, True)
    if f1_rng:
        log_ticks(f1, "y", rng=f1_rng)
    show(f1, "cmbear_recharge_vs_rainfall")

    # Recharge fraction against aridity, laid out as Berghuijs et al. (2024) Fig. 1a
    st.subheader("Recharge fraction versus aridity")
    a1, a2, a3 = st.columns([1, 1, 2], vertical_alignment="bottom")
    rp_log = a1.checkbox("Log y", value=False, key="rp_log")
    show_global = a2.checkbox("Global curve", value=True, key="rp_global",
                              help=f"Berghuijs et al. (2022) global fit, alpha = {BERGHUIJS_ALPHA}, "
                                   f"beta = {BERGHUIJS_BETA}.")
    fit_all = a3.checkbox("Include aridity < 1 in the fit", value=False, key="rp_fitall",
                          help="Berghuijs et al. fit only sites where PET exceeds rainfall (aridity >= 1).")
    phi = (ok["PET_mm_y"] / ok["Rain_mm_y"]).to_numpy(float)
    rp = (ok[ycol] / ok["Rain_mm_y"]).to_numpy(float)
    good = np.isfinite(phi) & np.isfinite(rp) & (phi > 0) & (rp > 0)
    dry = good & (phi >= 1)
    wet = good & (phi < 1)
    f2 = go.Figure()
    for sel, name, colour in ((dry, "Aridity ≥ 1", C_CMB), (wet, "Aridity < 1", C_LINE)):
        if sel.any():
            f2.add_trace(go.Scatter(
                x=phi[sel], y=rp[sel], mode=label_mode, name=name, marker=marker(colour),
                text=ok["ID"].astype(str).to_numpy()[sel], textposition="top right", textfont=label_font,
                hovertemplate="<b>%{text}</b><br>PET/P %{x:.2f}<br>R/P %{y:.3f}<extra></extra>"))
    fsel = good if fit_all else dry
    fit = fit_berghuijs(phi[fsel], rp[fsel])
    x_lo = max(0.05, float(np.nanmin(phi[good])) * 0.9) if good.any() else 0.7
    x_hi = float(np.nanmax(phi[good])) * 1.1 if good.any() else 10
    x_lo, x_hi = min(x_lo, 0.7), max(x_hi, 10)
    xc_start = x_lo if fit_all else 1.0
    xc = np.logspace(np.log10(xc_start), np.log10(x_hi), 200)
    if fit:
        # Solid over the aridity range of the fitted bores, dashed where it is extrapolated.
        p_lo, p_hi = float(np.nanmin(phi[fsel])), float(np.nanmax(phi[fsel]))
        xin = np.logspace(np.log10(p_lo), np.log10(p_hi), 100)
        f2.add_trace(go.Scatter(x=xc, y=berghuijs_curve(xc, fit["alpha"], fit["beta"]), mode="lines",
                                name="Fit, extrapolated", line=dict(color=C_RF, width=1.2, dash="dash"),
                                hoverinfo="skip"))
        f2.add_trace(go.Scatter(x=xin, y=berghuijs_curve(xin, fit["alpha"], fit["beta"]), mode="lines",
                                name=f"Fit: α = {fit['alpha']:.3f}, β = {fit['beta']:.2f} (n = {fit['n']})",
                                line=dict(color=C_RF, width=3)))
    if show_global:
        f2.add_trace(go.Scatter(x=xc, y=berghuijs_curve(xc, BERGHUIJS_ALPHA, BERGHUIJS_BETA), mode="lines",
                                name="Berghuijs et al. (2022) global fit", line=dict(color=C_LINE, width=1.5, dash="dot")))
    f2.add_vline(x=1, line=dict(color="#555555", width=1.2, dash="dash"))
    f2.add_annotation(x=0, xref="x", xshift=6, y=1, yref="paper", yanchor="top", xanchor="left", showarrow=False,
                      text="E<sub>p</sub>/P = 1", font=dict(size=11, color="#555555"))
    if not rp_log:
        f2.add_hline(y=1, line=dict(color="#555555", width=1, dash="dash"))
    f2.update_layout(**PLOT_LAYOUT, height=520, xaxis_title="Climatic aridity, PET/P",
                     yaxis_title=f"Recharge fraction, R/P ({which.split(' (')[0]})",
                     legend=dict(orientation="h", y=1.14, x=0))
    f2.update_xaxes(type="log", range=[np.log10(x_lo), np.log10(x_hi)])
    log_ticks(f2, "x", rng=[np.log10(x_lo), np.log10(x_hi)])
    if rp_log:
        f2.update_yaxes(type="log")
        log_ticks(f2, "y", values=rp[good])
    else:
        ymax = float(np.nanmax(rp[good])) if good.any() else 1
        f2.update_yaxes(range=[0, min(max(1.1, ymax * 1.05), 2.0)])
    show(f2, "cmbear_recharge_fraction_vs_aridity")
    if fit:
        msg = (f"Fitted to individual bores ({fit['n']}, {'all aridity' if fit_all else 'PET/P ≥ 1'}) by least absolute "
               f"residuals in R/P, as in Berghuijs et al. (2022): R/P = alpha / (1 + ln(1 + (PET/P)^beta)). "
               f"Mean absolute error {fit['mae']:.3f}; R2 {fit['r2']:.2f}. ")
        span = np.log10(np.nanmax(phi[dry]) / np.nanmin(phi[dry])) if dry.sum() > 1 else 0
        if fit["at_bound"] or span < 0.3:
            msg += ("The aridity range of these bores is narrow, so alpha and beta are poorly constrained "
                    "individually; treat the curve as a local description only (dashed where extrapolated). ")
        st.caption(msg)
    else:
        st.caption("Fewer than 5 bores with PET/P >= 1 and a positive recharge estimate, so no curve was fitted.")

    st.subheader("Recharge distribution for one bore")
    if real is None:
        st.info("Enable 'Keep realisations' under 2. Settings > Advanced in the sidebar to see per-bore distributions.")
    elif len(ok):
        h1, h2 = st.columns([3, 1], vertical_alignment="bottom")
        bid = h1.selectbox("Bore", ok["ID"].tolist())
        hist_log = h2.checkbox("Log x", value=True, key="hist_log",
                               help="CMB recharge distributions are strongly right-skewed because chloride is in the "
                                    "denominator; a log axis shows their shape. The top 0.5% of realisations is trimmed.")
        idx = int(ok.loc[ok["ID"] == bid, "index"].iloc[0])
        r = res.iloc[idx]
        vals = real[idx]; vals = vals[np.isfinite(vals) & (vals > 0)]
        n_total = len(vals)
        upper = min(float(np.percentile(vals, 99.5)), axis_cap)
        shown = vals[vals <= upper]
        if hist_log:
            edges = np.logspace(np.log10(max(shown.min(), 1e-3)), np.log10(upper), 50)
        else:
            edges = np.linspace(0, upper, 50)
        counts, edges = np.histogram(shown, bins=edges)
        centres = np.sqrt(edges[:-1] * edges[1:]) if hist_log else 0.5 * (edges[:-1] + edges[1:])
        fh = go.Figure(go.Bar(x=centres, y=counts, width=np.diff(edges), name="CMB realisations",
                              marker=dict(color=C_CMB, line=dict(color="rgba(255,255,255,0.5)", width=0.5)),
                              hovertemplate="%{x:.3g} mm/y: %{y} realisations<extra></extra>"))
        ymax = counts.max() if counts.size else 1
        marks = [(r["R_RC_5_mm_y"], "5th", C_LINE, "dot", 0.98), (r["R_RC_50_mm_y"], "50th", C_LINE, "dot", 0.90),
                 (r["R_RC_95_mm_y"], "95th", C_LINE, "dot", 0.82)]
        if np.isfinite(r["RF_R50_mm_y"]) and r["RF_R50_mm_y"] > 0:
            marks.append((r["RF_R50_mm_y"], "RF R50", C_RF, "solid", 0.74))
        for x, lab, colour, dash, yfrac in marks:
            if x <= 0 or x > upper:
                continue
            fh.add_shape(type="line", x0=x, x1=x, y0=0, y1=ymax * 1.02, line=dict(color=colour, width=1.5, dash=dash))
            fh.add_annotation(x=np.log10(x) if hist_log else x, y=yfrac, yref="paper", text=f"{lab} {x:.3g}",
                              showarrow=False, xanchor="left", xshift=4, font=dict(size=11, color=colour))
        fh.update_layout(**PLOT_LAYOUT, xaxis_title="Recharge (mm/y)", yaxis_title="Realisations", height=420,
                         bargap=0.05, showlegend=False,
                         title=f"{bid}: Cl_gw {r['Clgw_mg_L']:.1f} mg/L, P {r['Rain_mm_y']:.0f} mm/y, "
                               f"RC {r['RC']:.3f}, deposition {r['ClDep_mean_kg_ha_y']:.1f} kg/ha/y")
        fh.update_xaxes(type="log" if hist_log else "linear")
        if hist_log:
            log_ticks(fh, "x", values=edges)
        show(fh, f"cmbear_distribution_{re.sub(r'[^A-Za-z0-9_-]+', '_', str(bid))}")
        trimmed = n_total - len(shown)
        st.caption(f"{n_total} realisations; {trimmed} above {upper:.3g} mm/y not shown. Mean {r['R_RC_mean_mm_y']:.3g} mm/y, "
                   f"median {r['R_RC_50_mm_y']:.3g} mm/y. The gap between them reflects the right tail that comes from "
                   "low-chloride realisations.")

    # ----------------------------------------------------------------------- #
    # download everything
    # ----------------------------------------------------------------------- #
    st.header("Download everything")
    st.markdown(
        "One zip file with the results spreadsheet (as the Results (Excel) button above, including the "
        "realisations sheet if that box is ticked) and every figure on this page with its current settings "
        f"(estimate, order, axes, colours and labels), as {dl_fmt} at the print width and resolution under "
        "*4. Plot options*. Figures are made when the button is clicked, which takes a few seconds.")
    with st.expander(f"Files in the zip ({1 + len(fig_items) if can_export else 1})"):
        st.markdown("- cmbear_results.xlsx\n" + ("\n".join(f"- figures/{n}.{o['format']}" for n, _, o in fig_items)
                                                 if can_export else "- (no figures: figure export unavailable)"))
    zip_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(run["file"]).stem) or "results"
    st.download_button("Download everything (.zip)", everything_zip(fig_items if can_export else [], xlsx_bytes),
                       f"cmbear_{zip_stem}.zip", "application/zip", type="primary", on_click="ignore",
                       icon=":material/folder_zip:")

# --------------------------------------------------------------------------- #
# method and notes
# --------------------------------------------------------------------------- #
st.header("Method and notes")
with st.expander("What the app calculates, and how it differs from the original script"):
    st.markdown(r"""
For each bore the app locates the 0.05 degree grid cell containing it and extracts long-term mean rainfall,
PET, aridity, the mean annual runoff coefficient (RC) and the chloride deposition statistics from the grids
of Lee et al. (2024). Two recharge estimates follow.

**Regular chloride mass balance** (Crosbie and Rachakonda, 2021; Irvine and Cartwright, 2022):

$R = \dfrac{100\,D}{Cl_{gw}}$

with $D$ the mean chloride deposition (kg/ha/y), $Cl_{gw}$ groundwater chloride (mg/L) and $R$ in mm/y.
The 5th and 95th percentile deposition grids give `R_CMB_5` and `R_CMB_95`.

**Stochastic CMB with a runoff term** (Crosbie and Rachakonda, 2021; Lee et al., 2024):

$R = \dfrac{100\,D\,(1 - \alpha\,RC)}{Cl_{gw}}$

evaluated for *n* realisations where $D$ follows a Pearson type III distribution (mean, sd, skew grids),
$Cl_{gw}$ follows $|N(Cl_{gw},\,CV \cdot Cl_{gw})|$ and $\alpha \sim U(0.33, 0.66)$. The 5th, 50th and 95th
percentiles and the mean of the realisations are reported (`R_RC_*`).

**Random forest values** (`RF_*`) are read directly from the published national grids at the same pixel:
recharge (R5, R50, R95, mm/y) and recharge divided by rainfall (RP5, RP50, RP95).

**Differences from the original CMBEAR2_v1.0.py script from Lee et al. (2024)**

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
  uses an assumed RC with climate left blank, as chosen under 2. Settings.
- Points lying exactly on a cell boundary are assigned to the northern/western cell deterministically.
  Otherwise pixel selection and random streams are identical, and results match the original to float32
  precision.
""")
with st.expander("Output columns"):
    st.table(pd.DataFrame({"column": list(COLUMN_NOTES), "meaning": list(COLUMN_NOTES.values())}))
with st.expander("Warnings"):
    st.markdown("CMBEAR was named after the mighty Australian predator, the Drop Bear.\n\n"
                "Remember, that unlike most Australians, drop bears don't like Vegemite. "
                "To stay safe from drop bears, dab a little Vegemite behind each ear."
                )