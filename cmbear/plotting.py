"""Plotly figures used by the app. Kept separate so the calculation package stays free of plotly."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .grids import GridStack

GREYS = [[0, "#f7f7f7"], [1, "#8c8c8c"]]          # muted background ramp (land data)
DEFAULT_POINT = "#e0812b"                         # end colour of the single-hue ramp for the bore colour variable
OCEAN = "#dfe9f3"


def _downsample(arr: np.ndarray, max_n: int):
    f = max(1, int(np.ceil(max(arr.shape) / max_n)))
    return arr[::f, ::f], f


def _colour_ticks(lo: float, hi: float, log: bool):
    """Tick values/labels for a colour bar spanning [lo, hi] (log10 units when log)."""
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return {}
    if not log:
        return {}
    k0, k1 = int(np.floor(lo)), int(np.ceil(hi))
    mults = (1,) if hi - lo > 2.5 else (1, 2, 5)
    vals = [np.log10(m * 10.0 ** k) for k in range(k0, k1 + 1) for m in mults]
    vals = [v for v in vals if lo - 1e-9 <= v <= hi + 1e-9]
    if len(vals) < 2:
        return {}
    return dict(tickvals=vals, ticktext=[f"{round(10 ** v, 6):g}" for v in vals])


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
    if p_hi <= p_lo:
        p_hi = p_lo + 1e-6
    if b_hi <= b_lo:
        b_hi = b_lo + 1e-6

    # Colour bars sit under the inset, side by side (bores, then background), titles rotated alongside.
    frac = ((x1 - x0) * (y1 - y0)) / ((grids.xright - grids.xll) * (grids.ytop - grids.yll))
    show_inset = bool(frac < 0.5)
    cb_len, cb_y = (0.50, 0.02) if show_inset else (0.92, 0.04)
    cb_common = dict(len=cb_len, y=cb_y, yanchor="bottom", thickness=14, xanchor="left",
                     outlinecolor="#999999", outlinewidth=1, tickfont=dict(size=11, color="#222222"))
    has_bg_bar = bool(background)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=crop, x=lons_c, y=lats_c, colorscale=GREYS, zmin=b_lo, zmax=b_hi, hoverinfo="skip", zsmooth=False,
        showscale=has_bg_bar,
        colorbar=dict(title=dict(text=background_label or (background or ""), side="right",
                                 font=dict(size=12, color="#222222")),
                      x=0.885, **cb_common, **_colour_ticks(b_lo, b_hi, bg_log)) if has_bg_bar else None))
    fig.add_trace(go.Scatter(
        x=ok["Lon"], y=ok["Lat"], mode="markers+text" if show_ids else "markers", name="Bores",
        text=ok["ID"].astype(str), textposition="top right", textfont=dict(size=10, color="#222222"),
        marker=dict(color=cvals, colorscale=[[0, "#fbf3ea"], [1, point_colour]], cmin=p_lo, cmax=p_hi, size=9,
                    line=dict(width=1, color="#333333"),
                    colorbar=dict(title=dict(text=colour_label, side="right", font=dict(size=12, color="#222222")),
                                  x=0.765, **cb_common, **_colour_ticks(p_lo, p_hi, log_colour))),
        customdata=np.stack([ok["ID"], vals, ok["status"]], axis=1),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]:.3g}<br>lat %{y:.4f}, lon %{x:.4f}"
                      "<br>%{customdata[2]}<extra></extra>"))

    # Australia inset with the main-panel bounding box (omitted when the panel already shows most of it)
    small, fs = _downsample(land.astype(float), 180)
    lats_s = grids.ytop - (np.arange(0, grids.nrows, fs) + 0.5) * grids.res
    lons_s = grids.xll + (np.arange(0, grids.ncols, fs) + 0.5) * grids.res
    if show_inset:
        fig.add_trace(go.Heatmap(z=small, x=lons_s, y=lats_s, colorscale=[[0, OCEAN], [1, "#bdbdbd"]],
                                 showscale=False, hoverinfo="skip", xaxis="x2", yaxis="y2"))
        fig.add_trace(go.Scatter(x=[x0, x1, x1, x0, x0], y=[y0, y0, y1, y1, y0], mode="lines",
                                 line=dict(color="#c0612b", width=2), hoverinfo="skip", showlegend=False,
                                 xaxis="x2", yaxis="y2"))

    mid_lat = np.deg2rad((y0 + y1) / 2)
    fig.update_layout(
        template="simple_white", height=620, margin=dict(l=60, r=20, t=40, b=70),
        paper_bgcolor="white", plot_bgcolor=OCEAN, showlegend=False,
        font=dict(color="#222222"),
        xaxis=dict(title="Longitude", range=[x0, x1], constrain="domain", domain=[0, 0.72]),
        yaxis=dict(title="Latitude", range=[y0, y1], scaleanchor="x", scaleratio=1 / np.cos(mid_lat)),
        xaxis2=dict(domain=[0.76, 0.99], anchor="y2", range=[grids.xll, grids.xright], showticklabels=False,
                    showgrid=False, zeroline=False, showline=True, linecolor="#999999", mirror=True,
                    visible=show_inset),
        yaxis2=dict(domain=[0.60, 0.98], anchor="x2", range=[grids.yll, grids.ytop], showticklabels=False,
                    showgrid=False, zeroline=False, showline=True, linecolor="#999999", mirror=True,
                    scaleanchor="x2", scaleratio=1 / np.cos(np.deg2rad(-27)), visible=show_inset),
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
