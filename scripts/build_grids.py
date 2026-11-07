"""
build_grids.py

Convert the ASCII (ESRI .asc style) grids supplied with Lee et al. (2024)
into a single compressed NumPy archive (data/grids.npz) that the CMBEAR app
loads at start-up.

Why: the 14 ASCII grids total ~110 MB and take tens of seconds to parse with
np.loadtxt on every cold start. Stored as float32 in a compressed .npz they are
an order of magnitude smaller, load in well under a second, and need no GDAL or
rasterio dependency.

Usage (from the repo root):
    python scripts/build_grids.py --lee2024 "I:/Projects/Apps/CMBEAR/Lee2024"

All grids are checked for identical geometry before being written. The
geometry (ncols, nrows, xllcorner, yllcorner, cellsize, nodata) is stored in
the archive so the app never hard-codes it.
"""
import argparse
import json
from pathlib import Path

import numpy as np

# key in archive -> (subfolder under Lee2024, filename)
GRIDS = {
    # CMBEAR inputs
    "rain":       ("Python scripts/CMBEAR/InputData", "rain.txt"),
    "pet":        ("Python scripts/CMBEAR/InputData", "PET.txt"),
    "aridity":    ("Python scripts/CMBEAR/InputData", "aridity.txt"),
    "rc":         ("Python scripts/CMBEAR/InputData", "RC.txt"),
    "cld_mean":   ("Python scripts/CMBEAR/InputData", "cl_deposition_mean.txt"),
    "cld_5th":    ("Python scripts/CMBEAR/InputData", "cl_deposition_5th.txt"),
    "cld_95th":   ("Python scripts/CMBEAR/InputData", "cl_deposition_95th.txt"),
    "cld_sd":     ("Python scripts/CMBEAR/InputData", "cl_deposition_sd.txt"),
    "cld_skew":   ("Python scripts/CMBEAR/InputData", "cl_deposition_skew.txt"),
    # Random forest outputs (Lee et al. 2024)
    "rf_r5":      ("Gridded map output files", "aus_0p05_r05.txt"),
    "rf_r50":     ("Gridded map output files", "aus_0p05_r50.txt"),
    "rf_r95":     ("Gridded map output files", "aus_0p05_r95.txt"),
    "rf_rp5":     ("Gridded map output files", "aus_0p05_r_p_05.txt"),
    "rf_rp50":    ("Gridded map output files", "aus_0p05_r_p_50.txt"),
    "rf_rp95":    ("Gridded map output files", "aus_0p05_r_p_95.txt"),
}

HEADER_KEYS = ["ncols", "nrows", "xllcorner", "yllcorner", "cellsize", "nodata_value"]


def read_ascii_grid(path: Path):
    """Read a 6-line-header ASCII grid. Header lines may be prefixed with '#'."""
    header = {}
    with open(path) as f:
        for _ in range(6):
            line = f.readline().lstrip("#").strip()
            key, val = line.split()
            header[key.lower()] = float(val)
    data = np.loadtxt(path, skiprows=6)
    nodata = header["nodata_value"]
    data = data.astype(np.float32)
    data[data == nodata] = np.nan
    assert data.shape == (int(header["nrows"]), int(header["ncols"])), path
    return header, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lee2024", required=True, help="Path to the Lee2024 folder")
    ap.add_argument("--out", default="data/grids.npz")
    args = ap.parse_args()

    root = Path(args.lee2024)
    arrays, geom = {}, None
    for key, (sub, fname) in GRIDS.items():
        p = root / sub / fname
        print(f"Reading {key:10s} <- {p.name}")
        header, data = read_ascii_grid(p)
        g = {k: header[k] for k in HEADER_KEYS}
        if geom is None:
            geom = g
        elif g != geom:
            raise ValueError(f"Geometry mismatch in {p}: {g} vs {geom}")
        arrays[key] = data

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, geometry=json.dumps(geom), **arrays)
    print(f"\nWrote {out} ({out.stat().st_size/1e6:.1f} MB) with {len(arrays)} grids")
    print("Geometry:", geom)


if __name__ == "__main__":
    main()
