"""Parsing and validating the user's bore spreadsheet."""
from __future__ import annotations

import io
import re

import pandas as pd

REQUIRED_COLUMNS = ["ID", "Lat", "Lon", "Clgw"]
OPTIONAL_COLUMNS = ["Clgw_sd"]

# Accepted spellings (lower-cased, non-alphanumerics stripped) for each column
ALIASES = {
    "ID":      ["id", "boreid", "bore", "borename", "site", "siteid", "name", "well", "wellid"],
    "Lat":     ["lat", "latitude", "y"],
    "Lon":     ["lon", "long", "longitude", "lng", "x"],
    "Clgw":    ["clgw", "cl", "chloride", "clmgl", "chloridemgl", "clgwmgl", "cl_gw"],
    "Clgw_sd": ["clgwsd", "clsd", "chloridesd", "sd", "std", "stdev", "clgwstd"],
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def read_table(file, filename: str | None = None) -> pd.DataFrame:
    """Read a CSV or Excel upload into a DataFrame."""
    name = (filename or getattr(file, "name", "") or "").lower()
    if name.endswith((".xlsx", ".xls", ".xlsm")):
        return pd.read_excel(file)
    if isinstance(file, (bytes, bytearray)):
        file = io.BytesIO(file)
    return pd.read_csv(file)


def parse_input(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Map user columns onto the canonical names and clean the data.

    Returns (clean_df, messages). Rows that cannot be used are dropped and a
    message explains why. Multiple rows per ID are averaged (mean chloride),
    with the number of samples kept in 'n_samples' and their standard deviation
    in 'Clgw_sd_samples'. A user-supplied 'Clgw_sd' is kept separately; which
    one is used in the stochastic calculation is decided in cmb.run_cmbear.
    """
    msgs: list[str] = []
    colmap = {}
    normed = {_norm(c): c for c in df.columns}
    for canon, aliases in ALIASES.items():
        for a in [_norm(canon)] + aliases:
            if a in normed and canon not in colmap:
                colmap[canon] = normed[a]
                break
    missing = [c for c in REQUIRED_COLUMNS if c not in colmap]
    if missing:
        raise ValueError(
            f"Could not find required column(s) {missing}. Columns present: {list(df.columns)}. "
            f"Expected names (case-insensitive): ID, Lat, Lon, Clgw"
        )
    out = pd.DataFrame({canon: df[src] for canon, src in colmap.items()})
    n0 = len(out)

    # Chloride may arrive as text, including censored values like '<1'
    cl = out["Clgw"].astype(str).str.strip()
    censored = cl.str.contains("<", na=False)
    if censored.any():
        msgs.append(f"Dropped {int(censored.sum())} row(s) with censored chloride values (containing '<').")
    out = out[~censored].copy()
    for c in ["Lat", "Lon", "Clgw"] + (["Clgw_sd"] if "Clgw_sd" in out else []):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    bad = out[["Lat", "Lon", "Clgw"]].isna().any(axis=1)
    if bad.any():
        msgs.append(f"Dropped {int(bad.sum())} row(s) with missing or non-numeric Lat, Lon or Clgw.")
    out = out[~bad]
    nonpos = out["Clgw"] <= 0
    if nonpos.any():
        msgs.append(f"Dropped {int(nonpos.sum())} row(s) with chloride <= 0.")
    out = out[~nonpos]
    out["ID"] = out["ID"].astype(str).str.strip()

    # Aggregate repeat samples per bore
    agg = {"Lat": "first", "Lon": "first", "Clgw": ["mean", "std", "count"]}
    if "Clgw_sd" in out:
        agg["Clgw_sd"] = "mean"
    g = out.groupby("ID", sort=False).agg(agg)
    g.columns = ["_".join(c).rstrip("_") if c[1] not in ("first", "mean") or c[0] == "Clgw" else c[0]
                 for c in g.columns]
    g = g.rename(columns={"Clgw_mean": "Clgw", "Clgw_std": "Clgw_sd_samples",
                          "Clgw_count": "n_samples", "Clgw_sd_mean": "Clgw_sd"})
    g = g.reset_index()
    if "Clgw_sd" not in g:
        g["Clgw_sd"] = pd.NA
    g["Clgw_sd"] = pd.to_numeric(g["Clgw_sd"], errors="coerce")
    g["Clgw_sd_samples"] = pd.to_numeric(g["Clgw_sd_samples"], errors="coerce")
    nrep = int((g["n_samples"] > 1).sum())
    if nrep:
        msgs.append(f"{nrep} bore ID(s) had repeat samples; chloride was averaged per bore.")
    msgs.insert(0, f"Read {n0} row(s); {len(g)} bore(s) retained for analysis.")
    return g[["ID", "Lat", "Lon", "Clgw", "Clgw_sd", "Clgw_sd_samples", "n_samples"]], msgs
