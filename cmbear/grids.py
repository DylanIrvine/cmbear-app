"""Loading the gridded inputs and locating the pixel for each bore."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_GRID_PATH = Path(__file__).resolve().parent.parent / "data" / "grids.npz"

# Human-readable descriptions and units for every grid in the archive.
GRID_INFO = {
    "rain":     ("Long-term mean annual rainfall", "mm/y"),
    "pet":      ("Long-term mean annual potential evapotranspiration", "mm/y"),
    "aridity":  ("Aridity index (PET/P)", "-"),
    "rc":       ("Mean annual runoff coefficient", "-"),
    "cld_mean": ("Chloride deposition, mean", "kg/ha/y"),
    "cld_5th":  ("Chloride deposition, 5th percentile", "kg/ha/y"),
    "cld_95th": ("Chloride deposition, 95th percentile", "kg/ha/y"),
    "cld_sd":   ("Chloride deposition, standard deviation", "kg/ha/y"),
    "cld_skew": ("Chloride deposition, skewness", "-"),
    "rf_r5":    ("Random forest recharge, 5th percentile", "mm/y"),
    "rf_r50":   ("Random forest recharge, median", "mm/y"),
    "rf_r95":   ("Random forest recharge, 95th percentile", "mm/y"),
    "rf_rp5":   ("Random forest R/P, 5th percentile", "-"),
    "rf_rp50":  ("Random forest R/P, median", "-"),
    "rf_rp95":  ("Random forest R/P, 95th percentile", "-"),
}


@dataclass
class GridStack:
    arrays: dict[str, np.ndarray]
    ncols: int
    nrows: int
    xll: float      # x of lower-left corner of lower-left cell
    yll: float      # y of lower-left corner of lower-left cell
    res: float

    @property
    def ytop(self) -> float:
        return self.yll + self.nrows * self.res

    @property
    def xright(self) -> float:
        return self.xll + self.ncols * self.res

    def __getitem__(self, key):
        return self.arrays[key]

    def in_extent(self, lat, lon):
        lat = np.asarray(lat, float); lon = np.asarray(lon, float)
        return (lat >= self.yll) & (lat <= self.ytop) & (lon >= self.xll) & (lon <= self.xright)

    def rowcol(self, lat, lon):
        """Row/column indices of the pixel containing each (lat, lon).

        Replicates the selection logic of CMBEAR2_v1.0.py: the row is the first
        row (from the top) whose lower edge is <= lat; the column is the first
        column (from the left) whose right edge is >= lon. Points exactly on a
        cell boundary therefore go to the northern/western cell, as in the
        original. Out-of-extent points return -1.
        """
        lat = np.asarray(lat, float); lon = np.asarray(lon, float)
        # 1e-9 cell tolerance: a point exactly on a boundary belongs to the
        # northern/western cell, as the original's inequalities intend.
        tol = 1e-9
        row = np.ceil((self.ytop - lat) / self.res - tol).astype(int) - 1
        col = np.ceil((lon - self.xll) / self.res - tol).astype(int) - 1
        row = np.clip(row, 0, self.nrows - 1)
        col = np.clip(col, 0, self.ncols - 1)
        ok = self.in_extent(lat, lon)
        row = np.where(ok, row, -1)
        col = np.where(ok, col, -1)
        return row, col

    def sample(self, key, row, col):
        """Pixel values for a grid; NaN where row/col is -1."""
        arr = self.arrays[key]
        out = np.full(len(row), np.nan, dtype=float)
        ok = (row >= 0) & (col >= 0)
        out[ok] = arr[row[ok], col[ok]]
        return out

    def neighbourhood_mean(self, keys, row, col, max_radius: int):
        """Mean of valid cells in a square window around (row, col), for several grids at once.

        The window starts at radius 1 (3 x 3 cells) and grows one ring at a time until at least one
        valid cell is found in the first grid of `keys` or max_radius is reached. All grids in `keys`
        are assumed to share one validity mask (true for the climate/RC set and for the RF set).
        Returns (dict key -> values, radius_used) with NaN / 0 where nothing was found or row/col is -1.
        """
        n = len(row)
        out = {k: np.full(n, np.nan) for k in keys}
        radius = np.zeros(n, dtype=int)
        ref = self.arrays[keys[0]]
        for i in range(n):
            if row[i] < 0 or col[i] < 0:
                continue
            for r in range(1, int(max_radius) + 1):
                r0, r1 = max(0, row[i] - r), min(self.nrows, row[i] + r + 1)
                c0, c1 = max(0, col[i] - r), min(self.ncols, col[i] + r + 1)
                if np.isfinite(ref[r0:r1, c0:c1]).any():
                    for k in keys:
                        out[k][i] = np.nanmean(self.arrays[k][r0:r1, c0:c1])
                    radius[i] = r
                    break
        return out, radius

    def pixel_centre(self, row, col):
        lat = self.ytop - (row + 0.5) * self.res
        lon = self.xll + (col + 0.5) * self.res
        lat = np.where(row >= 0, lat, np.nan)
        lon = np.where(col >= 0, lon, np.nan)
        return lat, lon


def load_grids(path: str | Path = DEFAULT_GRID_PATH) -> GridStack:
    with np.load(path) as z:
        geom = json.loads(str(z["geometry"]))
        arrays = {k: z[k] for k in z.files if k != "geometry"}
    return GridStack(
        arrays=arrays,
        ncols=int(geom["ncols"]), nrows=int(geom["nrows"]),
        xll=float(geom["xllcorner"]), yll=float(geom["yllcorner"]),
        res=float(geom["cellsize"]),
    )
