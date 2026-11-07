"""Plotly figures used by the app. Kept separate so the calculation package stays free of plotly."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .grids import GridStack

GREYS = [[0, "#f7f7f7"], [1, "#8c8c8c"]]          # muted background ramp (land data)
SEQ = "Oranges"                                   # single-hue ramp for the bore colour variable
OCEAN = "#dfe9f3"


def _downsample(arr: np.ndarray, max_n: int):
    f = max(1, int(np.ceil(max(arr.shape) / max_n)))
    return arr[::f, ::f], f


def bore_map(res: pd.DataFrame, grids: GridStack, colour_var: str, colour_label: str,
             background: str | None = "rf_r50", log_colour: bool = True, pad_frac: float = 0.15) -> go.Figure:
    """Map of bores coloured by one result column over a muted grid, with an Australia inset.

    background: key of a grid to show in greys (None for land/sea only).
    The main panel's extent is the bounding box of the bores with padding (at least 0.5 degrees).
    """
    ok = res[np.isfinite(res["Lat"]) & np.isfinite(res["Lon"])]
    lat0, lat1 = ok["Lat"].min(), ok["Lat"].max()
    lon0, lon1 = ok["Lon"].min(), ok["Lon"].max()
    pad_lat = max(0.5, (lat1 - lat0) * pad_frac)
    pad_lon = max(0.5, (lon1 - lon0) * pad_frac)
    y0, y1 = max(grids.yll, lat0 - pad_lat), min(grids.ytop, lat1 + pad_lat)
    x0, x1 = max(grids.xll, lon0 - pad_lon), min(grids.xright, lon1 + pad_lon)

    land = np.isfinite(grids["cld_mean"])
    bg = grids[background] if background else np.where(land, 1.0, np.nan)
    if background and log_colour and background.startswith("rf_r"):
        bg = np.log10(np.where(bg > 0, bg, np.nan))

    # crop to the main extent
    r0, c0 = grids.rowcol(np.array([y1]), np.array([x0]))
    r1, c1 = grids.rowcol(np.array([y0]), np.array([x1]))
    r0, r1, c0, c1 = int(r0[0]), int(r1[0]) + 1, int(c0[0]), int(c1[0]) + 1
    crop, f = _downsample(bg[r0:r1, c0:c1], 400)
    lats_c = grids.ytop - (np.arange(r0, r1, f) + 0.5) * grids.res
    lons_c = grids.xll + (np.arange(c0, c1, f) + 0.5) * grids.res

    fig = go.Figure()
    finite = crop[np.isfinite(crop)]
    zmin, zmax = (np.percentile(finite, [2, 98]) if (background and finite.size) else (0, 1))
    fig.add_trace(go.Heatmap(z=crop, x=lons_c, y=lats_c, colorscale=GREYS, showscale=False, zmin=zmin, zmax=zmax,
                             hoverinfo="skip", zsmooth=False))
    vals = ok[colour_var].to_numpy(float)
    cvals = np.log10(np.where(vals > 0, vals, np.nan)) if log_colour else vals
    tick = None
    if log_colour and np.isfinite(cvals).any():
        lo, hi = np.floor(np.nanmin(cvals)), np.ceil(np.nanmax(cvals))
        tv = np.arange(lo, hi + 1)
        tick = dict(tickvals=tv, ticktext=[f"{10**t:g}" for t in tv])
    fig.add_trace(go.Scatter(
        x=ok["Lon"], y=ok["Lat"], mode="markers", name="Bores",
        marker=dict(color=cvals, colorscale=SEQ, size=9, line=dict(width=1, color="#333333"),
                    colorbar=dict(title=colour_label, x=0.76, xanchor="left", len=0.5, y=0.27, **(tick or {}))),
        customdata=np.stack([ok["ID"], vals, ok["status"]], axis=1),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]:.3g}<br>lat %{y:.4f}, lon %{x:.4f}"
                      "<br>%{customdata[2]}<extra></extra>"))

    # Australia inset with the main-panel bounding box (omitted when the panel already shows most of it)
    frac = ((x1 - x0) * (y1 - y0)) / ((grids.xright - grids.xll) * (grids.ytop - grids.yll))
    show_inset = frac < 0.5
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
        template="simple_white", height=620, margin=dict(l=60, r=20, t=40, b=50),
        plot_bgcolor=OCEAN, showlegend=False,
        xaxis=dict(title="Longitude", range=[x0, x1], constrain="domain", domain=[0, 0.72]),
        yaxis=dict(title="Latitude", range=[y0, y1], scaleanchor="x", scaleratio=1 / np.cos(mid_lat)),
        xaxis2=dict(domain=[0.76, 0.99], anchor="y2", range=[grids.xll, grids.xright], showticklabels=False,
                    showgrid=False, zeroline=False, showline=True, linecolor="#999999", mirror=True),
        yaxis2=dict(domain=[0.60, 0.98], anchor="x2", range=[grids.yll, grids.ytop], showticklabels=False,
                    showgrid=False, zeroline=False, showline=True, linecolor="#999999", mirror=True,
                    scaleanchor="x2", scaleratio=1 / np.cos(np.deg2rad(-27))),
    )
    return fig
