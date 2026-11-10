"""Plotly figures used by the app. Kept separate so the calculation package stays free of plotly."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .grids import GridStack

GREYS = [[0, "#f7f7f7"], [1, "#8c8c8c"]]          # muted background ramp (land data)
DEFAULT_POINT = "#e0812b"                         # end colour of the single-hue ramp for the bore colour variable
OCEAN = "#dfe9f3"
INSET_LON, INSET_LAT = (112.0, 154.0), (-44.0, -10.0)   # Australia inset extent (degrees)
MAP_NOMINAL_W = 640       # px; figure height is set so the main panel fills the full height at this width or wider
MAP_MARGIN = dict(l=60, r=20, t=30, b=70)
MAP_X_DOMAIN = 0.72       # fraction of the plot-area width given to the main panel
MAP_ASPECT = (0.7, 1.5)   # allowed main-panel height/width (pixels); the short side is widened with grid
INSET_REF_LAT = -27.0      # latitude at which the inset's aspect ratio and scale bar are true
KM_PER_DEG_LAT = 111.32


def _downsample(arr: np.ndarray, max_n: int):
    f = max(1, int(np.ceil(max(arr.shape) / max_n)))
    return arr[::f, ::f], f


def _hex_ramp(c0: str, c1: str, n: int) -> list[str]:
    """n colours evenly spaced in RGB between hex colours c0 and c1."""
    a = np.array([int(c0[i:i + 2], 16) for i in (1, 3, 5)], float)
    b = np.array([int(c1[i:i + 2], 16) for i in (1, 3, 5)], float)
    t = np.linspace(0, 1, n) if n > 1 else np.array([1.0])
    return ["#%02x%02x%02x" % tuple(np.round(a + (b - a) * x).astype(int)) for x in t]


def _class_edges(lo: float, hi: float, log: bool) -> np.ndarray:
    """Class boundaries at round values covering [lo, hi] (log10 units when log), about 4 to 10 classes.

    Log: 1-2-5 per decade for spans up to ~3 decades, 1-3 up to ~6, otherwise decades.
    Linear: a step of 1, 2, 2.5 or 5 x 10^k giving roughly 7 classes.
    """
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return np.array([0.0, 1.0])
    if hi <= lo:
        hi = lo + (0.1 if log else max(abs(lo) * 0.1, 1e-6))
    if log:
        span = hi - lo
        mults = (1, 2, 5) if span <= 3.2 else (1, 3) if span <= 6 else (1,)
        k0, k1 = int(np.floor(lo)) - 1, int(np.ceil(hi)) + 1
        cand = np.array(sorted(np.log10(m * 10.0 ** k) for k in range(k0, k1 + 1) for m in mults))
        i0 = int(np.searchsorted(cand, lo + 1e-9, side="right")) - 1
        i1 = int(np.searchsorted(cand, hi - 1e-9, side="left"))
        return cand[max(i0, 0):i1 + 1]
    raw = (hi - lo) / 7
    mag = 10 ** np.floor(np.log10(raw))
    step = mag * min((m for m in (1, 2, 2.5, 5, 10) if m * mag >= raw))
    e0, e1 = np.floor(lo / step + 1e-9) * step, np.ceil(hi / step - 1e-9) * step
    return np.arange(e0, e1 + step / 2, step)


def _discrete(edges: np.ndarray, c0: str, c1: str):
    """Stepped plotly colourscale for the classes between edges (uneven widths allowed), and its colour range."""
    zmin, zmax = float(edges[0]), float(edges[-1])
    n = len(edges) - 1
    cols = _hex_ramp(c0, c1, n)
    pos = (edges - zmin) / (zmax - zmin)
    scale = []
    for i, c in enumerate(cols):
        scale += [[float(pos[i]), c], [float(pos[i + 1]), c]]
    return scale, zmin, zmax


def _edge_ticks(edges: np.ndarray, log: bool) -> dict:
    """Label every class boundary (thinned to every other one if there are many)."""
    e = edges if len(edges) <= 11 else edges[::2]
    text = [f"{round(10 ** v, 6):g}" if log else f"{round(v, 6):g}" for v in e]
    return dict(tickvals=[float(v) for v in e], ticktext=text, ticks="outside")


def _scale_bar(fig: go.Figure, length_km: float = 1000, n_seg: int = 4, x_start: float = 113.5,
               y_base: float = -42.6, bar_h: float = 0.75):
    """Alternating black/white scale bar in the lower left of the Australia inset (axes x2/y2).

    Lengths are converted to degrees of longitude at INSET_REF_LAT, where the inset's aspect ratio is true;
    on a lat/lon plot the scale is exact only at that latitude.
    """
    deg_per_km = 1 / (KM_PER_DEG_LAT * np.cos(np.deg2rad(INSET_REF_LAT)))
    seg = length_km / n_seg * deg_per_km
    for i in range(n_seg):
        fig.add_shape(type="rect", xref="x2", yref="y2", x0=x_start + i * seg, x1=x_start + (i + 1) * seg,
                      y0=y_base, y1=y_base + bar_h, fillcolor="black" if i % 2 == 0 else "white",
                      line=dict(color="black", width=1), opacity=1, layer="above")
    lab = dict(xref="x2", yref="y2", y=y_base + bar_h, yanchor="bottom", yshift=1, showarrow=False,
               font=dict(size=9, color="#222222"))
    fig.add_annotation(x=x_start, text="0", xanchor="center", **lab)
    fig.add_annotation(x=x_start + n_seg * seg / 2, text=f"{length_km / 2:g}", xanchor="center", **lab)
    fig.add_annotation(x=x_start + n_seg * seg, text=f"{length_km:g} km", xanchor="left", xshift=-6, **lab)


def bore_map(res: pd.DataFrame, grids: GridStack, colour_var: str, colour_label: str,
             background: str | None = "rf_r50", log_colour: bool = True, pad_frac: float = 0.15,
             point_colour: str = DEFAULT_POINT, show_ids: bool = False, background_label: str | None = None,
             shared_range: bool = False) -> go.Figure:
    """Map of bores coloured by one result column over a grey grid, with an Australia inset.

    background: key of a grid to show in greys (None for land/sea only), with its own colour bar.
    shared_range: put bores and background on one colour range (use when both are the same quantity and units,
    e.g. CMB recharge over random forest recharge), so their shades can be compared.
    log_colour applies to both the bores and, when shared or a random forest grid, the background.
    The main panel's extent is the bounding box of the bores with padding (at least 0.5 degrees).
    The figure has a white page background and is meant to be drawn with Streamlit's theme switched off.
    """
    ok = res[np.isfinite(res["Lat"]) & np.isfinite(res["Lon"])]
    lat0, lat1 = ok["Lat"].min(), ok["Lat"].max()
    lon0, lon1 = ok["Lon"].min(), ok["Lon"].max()
    pad_lat = max(0.5, (lat1 - lat0) * pad_frac)
    pad_lon = max(0.5, (lon1 - lon0) * pad_frac)
    y0, y1 = max(grids.yll, lat0 - pad_lat), min(grids.ytop, lat1 + pad_lat)
    x0, x1 = max(grids.xll, lon0 - pad_lon), min(grids.xright, lon1 + pad_lon)

    # Keep the panel's shape within MAP_ASPECT by showing more grid along the short side, so the figure height
    # (set from the aspect below) stays sensible and the inset/colour bars fit beside it.
    sr = 1 / np.cos(np.deg2rad((y0 + y1) / 2))
    asp = (y1 - y0) * sr / (x1 - x0)
    if asp < MAP_ASPECT[0]:
        need = MAP_ASPECT[0] * (x1 - x0) / sr
        ym = 0.5 * (y0 + y1)
        y0, y1 = max(grids.yll, ym - need / 2), min(grids.ytop, ym + need / 2)
    elif asp > MAP_ASPECT[1]:
        need = (y1 - y0) * sr / MAP_ASPECT[1]
        xm = 0.5 * (x0 + x1)
        x0, x1 = max(grids.xll, xm - need / 2), min(grids.xright, xm + need / 2)

    land = np.isfinite(grids["cld_mean"])
    bg_log = bool(background) and log_colour and (shared_range or background.startswith("rf_r"))
    bg = grids[background] if background else np.where(land, 1.0, np.nan)
    if bg_log:
        bg = np.log10(np.where(bg > 0, bg, np.nan))

    # crop to the main extent
    r0, c0 = grids.rowcol(np.array([y1]), np.array([x0]))
    r1, c1 = grids.rowcol(np.array([y0]), np.array([x1]))
    r0, r1, c0, c1 = int(r0[0]), int(r1[0]) + 1, int(c0[0]), int(c1[0]) + 1
    crop, f = _downsample(bg[r0:r1, c0:c1], 400)
    lats_c = grids.ytop - (np.arange(r0, r1, f) + 0.5) * grids.res
    lons_c = grids.xll + (np.arange(c0, c1, f) + 0.5) * grids.res
    # Clip the axes to the outer cell edges of the grid actually drawn, so no empty margin shows
    half = 0.5 * f * grids.res
    x0, x1 = float(lons_c[0] - half), float(lons_c[-1] + half)
    y0, y1 = float(lats_c[-1] - half), float(lats_c[0] + half)

    vals = ok[colour_var].to_numpy(float)
    cvals = np.log10(np.where(vals > 0, vals, np.nan)) if log_colour else vals
    pfin = cvals[np.isfinite(cvals)]
    p_lo, p_hi = (float(pfin.min()), float(pfin.max())) if pfin.size else (0.0, 1.0)

    finite = crop[np.isfinite(crop)]
    b_lo, b_hi = (np.percentile(finite, [2, 98]) if (background and finite.size) else (0.0, 1.0))
    if background and shared_range and finite.size and pfin.size:
        lo, hi = min(p_lo, b_lo), max(p_hi, b_hi)
        p_lo = b_lo = lo
        p_hi = b_hi = hi
    # Discrete classes with round boundaries; shared ranges get identical boundaries.
    p_edges = _class_edges(p_lo, p_hi, log_colour)
    b_edges = p_edges if (background and shared_range) else _class_edges(b_lo, b_hi, bg_log)
    p_scale, p_lo, p_hi = _discrete(p_edges, "#fbf3ea", point_colour)
    b_scale, b_lo, b_hi = _discrete(b_edges, GREYS[0][1], GREYS[1][1])

    # Colour bars sit under the inset, side by side (bores, then background), titles rotated alongside.
    frac = ((x1 - x0) * (y1 - y0)) / ((grids.xright - grids.xll) * (grids.ytop - grids.yll))
    show_inset = bool(frac < 0.5)
    # Bottom of the colour bars level with the bottom of the main panel (y = 0 of the plot area);
    # tops kept below the inset, whose bottom is never below 0.60 of the plot height.
    cb_len = 0.52 if show_inset else 1.0
    cb_common = dict(len=cb_len, y=0, yanchor="bottom", thickness=14, xanchor="left",
                     outlinecolor="#999999", outlinewidth=1, tickfont=dict(size=11, color="#222222"))
    has_bg_bar = bool(background)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=crop, x=lons_c, y=lats_c, colorscale=b_scale if background else GREYS, zmin=b_lo, zmax=b_hi, hoverinfo="skip", zsmooth=False,
        showscale=has_bg_bar,
        colorbar=dict(title=dict(text=background_label or (background or ""), side="right",
                                 font=dict(size=11, color="#222222")),
                      x=0.885, **cb_common, **_edge_ticks(b_edges, bg_log)) if has_bg_bar else None))
    fig.add_trace(go.Scatter(
        x=ok["Lon"], y=ok["Lat"], mode="markers+text" if show_ids else "markers", name="Bores",
        text=ok["ID"].astype(str), textposition="top right", textfont=dict(size=10, color="#222222"),
        marker=dict(color=cvals, colorscale=p_scale, cmin=p_lo, cmax=p_hi, size=9,
                    line=dict(width=1, color="#333333"),
                    colorbar=dict(title=dict(text=colour_label, side="right", font=dict(size=11, color="#222222")),
                                  x=0.765, **cb_common, **_edge_ticks(p_edges, log_colour))),
        customdata=np.stack([ok["ID"], vals, ok["status"]], axis=1),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]:.3g}<br>lat %{y:.4f}, lon %{x:.4f}"
                      "<br>%{customdata[2]}<extra></extra>"))

    # Australia inset with the main-panel bounding box (omitted when the panel already shows most of it)
    small, fs = _downsample(land.astype(float), 180)
    lats_s = grids.ytop - (np.arange(0, grids.nrows, fs) + 0.5) * grids.res
    lons_s = grids.xll + (np.arange(0, grids.ncols, fs) + 0.5) * grids.res
    if show_inset:
        # Ocean fill for the inset only (the main panel stays white, as no-data cells there are not always sea)
        fig.add_shape(type="rect", xref="x2 domain", yref="y2 domain", x0=0, x1=1, y0=0, y1=1,
                      fillcolor=OCEAN, opacity=1, line=dict(width=0), layer="below")
        fig.add_trace(go.Heatmap(z=small, x=lons_s, y=lats_s, colorscale=[[0, OCEAN], [1, "#bdbdbd"]],
                                 showscale=False, hoverinfo="skip", xaxis="x2", yaxis="y2"))
        fig.add_trace(go.Scatter(x=[x0, x1, x1, x0, x0], y=[y0, y0, y1, y1, y0], mode="lines",
                                 line=dict(color="#c0612b", width=2), hoverinfo="skip", showlegend=False,
                                 xaxis="x2", yaxis="y2"))
        _scale_bar(fig, length_km=1000, n_seg=4)

    mid_lat = np.deg2rad((y0 + y1) / 2)
    # Height that makes the main panel exactly fill the plot-area height at MAP_NOMINAL_W. At that width or
    # wider (the app's centred layout, and downloads at 800 px) plotly shrinks the panel sideways instead, so
    # its top meets the inset top and its bottom meets the colour-bar bottoms. Narrower screens shrink it
    # vertically and the alignment loosens.
    panel_w = MAP_X_DOMAIN * (MAP_NOMINAL_W - MAP_MARGIN["l"] - MAP_MARGIN["r"])
    panel_h = panel_w * (y1 - y0) / np.cos(mid_lat) / (x1 - x0)
    fig_h = int(np.floor(panel_h)) + MAP_MARGIN["t"] + MAP_MARGIN["b"]
    fig.update_layout(
        template="simple_white", height=fig_h, margin=MAP_MARGIN,
        paper_bgcolor="white", plot_bgcolor="white", showlegend=False,
        font=dict(color="#222222"),
        xaxis=dict(title="Longitude", range=[x0, x1], constrain="domain", domain=[0, MAP_X_DOMAIN]),
        yaxis=dict(title="Latitude", range=[y0, y1], domain=[0, 1], constrain="domain", scaleanchor="x",
                   scaleratio=1 / np.cos(mid_lat)),
        xaxis2=dict(domain=[0.76, 0.99], anchor="y2", range=list(INSET_LON), constrain="domain", showticklabels=False, ticks="",
                    showgrid=False, zeroline=False, showline=True, linecolor="#999999", mirror=True,
                    visible=show_inset),
        yaxis2=dict(domain=[0.60, 1.0], anchor="x2", range=list(INSET_LAT), constrain="domain", constraintoward="top",
                    showticklabels=False, ticks="",
                    showgrid=False, zeroline=False, showline=True, linecolor="#999999", mirror=True,
                    scaleanchor="x2", scaleratio=1 / np.cos(np.deg2rad(INSET_REF_LAT)), visible=show_inset),
    )
    return fig


# --------------------------------------------------------------------------- #
# Recharge fraction against aridity (Berghuijs et al. 2022, 2024)
# --------------------------------------------------------------------------- #
BERGHUIJS_ALPHA, BERGHUIJS_BETA = 0.72, 15.11     # global fit, Berghuijs et al. (2022) GRL, Eq. 1


def berghuijs_curve(phi, alpha: float, beta: float):
    """R/P = alpha * (1 - ln(phi^beta + 1) / (1 + ln(phi^beta + 1))) = alpha / (1 + ln(1 + phi^beta)).

    phi is the aridity index PET/P. ln(1 + phi^beta) is evaluated as logaddexp(0, beta ln phi) to avoid overflow.
    """
    phi = np.asarray(phi, float)
    L = np.logaddexp(0.0, beta * np.log(phi))
    return alpha / (1.0 + L)


def _weighted_median(x, w):
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w)
    return float(x[np.searchsorted(c, 0.5 * c[-1])])


def fit_berghuijs(phi, rp, beta_bounds=(0.1, 100.0)):
    """Fit alpha and beta of the Berghuijs et al. sigmoid to individual points by least absolute residuals
    in R/P (the criterion used by Berghuijs et al. 2022). For a given beta, the alpha minimising
    sum |rp - alpha g| is the weighted median of rp/g with weights g, so only beta needs a 1-D search
    (log-spaced grid, then a finer grid around the best value). No scipy needed.

    Returns dict(alpha, beta, n, mae, r2, at_bound) or None if fewer than 5 valid points.
    """
    phi = np.asarray(phi, float); rp = np.asarray(rp, float)
    m = np.isfinite(phi) & np.isfinite(rp) & (phi > 0) & (rp >= 0)
    phi, rp = phi[m], rp[m]
    if phi.size < 5:
        return None

    def solve(beta):
        g = berghuijs_curve(phi, 1.0, beta)
        a = _weighted_median(rp / g, g)
        return a, float(np.sum(np.abs(rp - a * g)))

    lb0, lb1 = np.log10(beta_bounds[0]), np.log10(beta_bounds[1])
    grid = np.logspace(lb0, lb1, 200)
    cost = np.array([solve(b)[1] for b in grid])
    i = int(np.argmin(cost))
    step = (lb1 - lb0) / 199
    fine = np.logspace(np.log10(grid[i]) - step, np.log10(grid[i]) + step, 101)
    fine = fine[(fine >= beta_bounds[0]) & (fine <= beta_bounds[1])]
    fc = [solve(b) for b in fine]
    j = int(np.argmin([c for _, c in fc]))
    beta = float(fine[j]); alpha = fc[j][0]
    pred = berghuijs_curve(phi, alpha, beta)
    ss_res = float(np.sum((rp - pred) ** 2)); ss_tot = float(np.sum((rp - rp.mean()) ** 2))
    return dict(alpha=alpha, beta=beta, n=int(phi.size), mae=float(np.mean(np.abs(rp - pred))),
                r2=1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
                at_bound=bool(i in (0, len(grid) - 1)))
