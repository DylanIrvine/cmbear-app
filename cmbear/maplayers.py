"""Optional layers for the bore map: a towns table and up to three polygon or line layers.

Adapted from the Hydrochem Workbench (hydrochem_workbench/maps.py) so the two apps take the same files.
geopandas is imported only when a polygon or line layer is read.
"""
from __future__ import annotations

import io
import os
import tempfile
import zipfile

import numpy as np
import pandas as pd

MAX_LAYERS = 3              # polygon or line layers on the map
MAX_UPLOAD_MB = 20          # per layer
MAX_VERTICES = 200_000      # simplified above this
POLY_TYPES = ["zip", "shp", "shx", "dbf", "prj", "cpg", "gpkg", "geojson", "json", "kml"]
TOWN_TYPES = ["csv", "xlsx"]
_SHP_PARTS = (".shp", ".shx", ".dbf", ".prj", ".cpg")
_VECTOR = (".shp", ".gpkg", ".geojson", ".json", ".kml")
_NAME_COLS = ("name", "town", "place", "label")
_LAT_COLS = ("lat", "latitude", "y")
_LON_COLS = ("lon", "long", "longitude", "lng", "x")


class MapError(ValueError):
    pass


class NeedsCRS(MapError):
    """The file has no coordinate reference system, so the user has to give one."""


class Upload:
    """An uploaded file kept in memory (name and bytes), so it can be read again later."""

    def __init__(self, name, data):
        self.name = name
        self.data = data

    def getvalue(self):
        return self.data


# --------------------------------------------------------------------------- #
# towns
# --------------------------------------------------------------------------- #
def towns_template_bytes() -> bytes:
    example = pd.DataFrame({"Name": ["Katherine", "Mataranka", "Tennant Creek"],
                            "Lat": [-14.465, -14.923, -19.648], "Lon": [132.264, 133.066, 134.190]})
    return example.to_csv(index=False).encode()


def read_towns(file, name: str = "") -> pd.DataFrame:
    """CSV (or xlsx) with Name, Lat, Lon in decimal degrees. Returns a table with those three columns."""
    name = (name or getattr(file, "name", "") or "").lower()
    raw = pd.read_excel(file, dtype=object) if name.endswith((".xlsx", ".xls")) else pd.read_csv(file, dtype=object)
    lower = {str(c).strip().lower(): c for c in raw.columns}
    pick = lambda keys: next((lower[k] for k in keys if k in lower), None)  # noqa: E731
    name_col, lat_col, lon_col = pick(_NAME_COLS), pick(_LAT_COLS), pick(_LON_COLS)
    if name_col is None or lat_col is None or lon_col is None:
        raise MapError("The towns file needs the columns Name, Lat and Lon. Use the template.")
    out = pd.DataFrame({"Name": raw[name_col].fillna("").astype(str).str.strip(),
                        "Lat": pd.to_numeric(raw[lat_col], errors="coerce"),
                        "Lon": pd.to_numeric(raw[lon_col], errors="coerce")})
    out = out[out["Lat"].notna() & out["Lon"].notna() & (out["Name"] != "") & (out["Name"] != "nan")]
    out = out[(out["Lat"].abs() <= 90) & (out["Lon"].abs() <= 180)]
    if out.empty:
        raise MapError("No rows in the towns file have a name with a latitude and longitude in decimal degrees.")
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# polygons and lines
# --------------------------------------------------------------------------- #
def read_vector(files, epsg=None):
    """One polygon (or line) layer from the uploaded file(s), in longitude and latitude.

    files: a .zip holding a shapefile, the parts of a shapefile, or one .gpkg/.geojson/.kml.
    epsg: only used when the file does not say what its coordinates are.
    Returns (GeoDataFrame with a geometry column only, short description).
    """
    import geopandas as gpd

    files = list(files) if isinstance(files, (list, tuple)) else [files]
    if not files:
        raise MapError("No file.")
    with tempfile.TemporaryDirectory() as tmp:
        total = 0
        for f in files:
            data = f.getvalue() if hasattr(f, "getvalue") else f.read()
            fname = os.path.basename(getattr(f, "name", "upload"))
            total += len(data)
            if total > MAX_UPLOAD_MB * 1e6:
                raise MapError(f"The file is larger than {MAX_UPLOAD_MB} MB. Simplify it or cut it down first.")
            if fname.lower().endswith(".zip"):
                _unzip(data, tmp)
            else:
                with open(os.path.join(tmp, fname), "wb") as out:
                    out.write(data)
        found = sorted(n for n in os.listdir(tmp) if n.lower().endswith(_VECTOR))
        if not found:
            raise MapError("No shapefile (.shp), GeoPackage, GeoJSON or KML was found in the upload.")
        path = os.path.join(tmp, found[0])
        if path.lower().endswith(".shp"):
            stem = os.path.splitext(found[0])[0].lower()
            have = {os.path.splitext(n)[1].lower() for n in os.listdir(tmp)
                    if os.path.splitext(n)[0].lower() == stem}
            missing = [e for e in (".shx", ".dbf") if e not in have]
            if missing:
                raise MapError("A shapefile is several files. Missing: " + ", ".join(missing) +
                               ". Upload them together (with the .prj), or as one .zip.")
        try:
            gdf = gpd.read_file(path)
        except Exception as exc:  # noqa: BLE001
            raise MapError(f"Could not read {found[0]}: {exc}")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if gdf.empty:
        raise MapError(f"{found[0]} has no shapes in it.")
    if gdf.crs is None:
        if epsg is None:
            raise NeedsCRS(f"{found[0]} does not say what its coordinates are (no .prj). Enter the EPSG code.")
        try:
            gdf = gdf.set_crs(epsg=int(epsg))
        except Exception:  # noqa: BLE001
            raise MapError(f"EPSG:{epsg} is not a known coordinate reference system.")
    source_crs = gdf.crs.to_string()
    try:
        gdf = gdf.to_crs(epsg=4326)
    except Exception as exc:  # noqa: BLE001
        raise MapError(f"Could not convert {found[0]} from {source_crs} to longitude and latitude: {exc}")
    gdf = gpd.GeoDataFrame(geometry=gdf.geometry.reset_index(drop=True), crs=4326)
    b = gdf.total_bounds
    if not np.all(np.isfinite(b)) or b[0] < -180.5 or b[2] > 180.5 or b[1] < -90.5 or b[3] > 90.5:
        raise MapError(f"After conversion from {source_crs} the shapes are not on the Earth. The coordinate "
                       "reference system of the file is probably wrong.")
    if int(gdf.geometry.count_coordinates().sum()) > MAX_VERTICES:
        # drop detail that could not be seen at the size of the figure
        gdf["geometry"] = gdf.geometry.simplify(max(b[2] - b[0], b[3] - b[1]) / 5000, preserve_topology=True)
    kinds = sorted({t.replace("Multi", "") for t in gdf.geom_type.unique()})
    info = f"{found[0]}: {len(gdf)} shape(s) ({', '.join(kinds).lower()}), converted from {source_crs}"
    return gdf, info


def _unzip(data, folder):
    """Only the files a layer is made of, into one flat folder."""
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise MapError("That .zip could not be opened.")
    size = 0
    for item in z.infolist():
        name = os.path.basename(item.filename)
        if item.is_dir() or not name.lower().endswith(_SHP_PARTS + _VECTOR) or name.startswith("."):
            continue
        size += item.file_size
        if size > 10 * MAX_UPLOAD_MB * 1e6:
            raise MapError("The .zip is too large once unpacked.")
        with open(os.path.join(folder, name), "wb") as out:
            out.write(z.read(item))


def layer_paths(gdf):
    """Coordinates of a layer for plotly: (polygon x, y) and (line x, y), shapes separated by None.

    Polygon exterior and interior rings are oriented in opposite directions so that, drawn as one filled path,
    holes stay empty."""
    from shapely.geometry.polygon import orient

    px, py, lx, ly = [], [], [], []

    def add(xs, ys, coords):
        c = np.asarray(coords)
        xs.extend(c[:, 0].tolist() + [None])
        ys.extend(c[:, 1].tolist() + [None])

    for geom in gdf.geometry:
        parts = getattr(geom, "geoms", [geom])
        for g in parts:
            if g.geom_type == "Polygon":
                g = orient(g, 1.0)
                add(px, py, g.exterior.coords)
                for ring in g.interiors:
                    add(px, py, ring.coords)
            elif g.geom_type in ("LineString", "LinearRing"):
                add(lx, ly, g.coords)
    return (px, py), (lx, ly)
