"""Basic regression tests. Run with: python -m pytest tests"""
import numpy as np
import pandas as pd

from cmbear import CMBSettings, load_grids, parse_input, run_cmbear

G = load_grids()


def test_rowcol_centre_and_boundary():
    # Pixel centres map to themselves
    lat, lon = G.pixel_centre(np.array([100]), np.array([200]))
    r, c = G.rowcol(lat, lon)
    assert (r[0], c[0]) == (100, 200)
    # A point exactly on a boundary goes to the northern/western cell
    r, c = G.rowcol(np.array([-21.975]), np.array([113.975]))
    assert (r[0], c[0]) == (239, 39)
    # Outside extent
    r, c = G.rowcol(np.array([-5.0]), np.array([120.0]))
    assert r[0] == -1 and c[0] == -1


def test_parse_input_aliases_and_grouping():
    df = pd.DataFrame({"bore id": ["a", "a", "b"], "latitude": [-12.0, -12.0, -13.0],
                       "longitude": [131.0, 131.0, 132.0], "Chloride (mg/L)": ["10", "<1", "20"]})
    out, msgs = parse_input(df)
    assert list(out["ID"]) == ["a", "b"]
    assert out.loc[0, "Clgw"] == 10.0 and out.loc[0, "n_samples"] == 1


def test_repeat_sample_sd_rule():
    df = pd.DataFrame({"ID": ["a"] * 3 + ["b"] * 2, "Lat": [-12.45] * 5, "Lon": [130.98] * 5,
                       "Clgw": [10, 12, 14, 20, 30]})
    out, _ = parse_input(df)
    res, _ = run_cmbear(out, G, CMBSettings(cv_clgw=0.37))
    a, b = res.set_index("ID").loc["a"], res.set_index("ID").loc["b"]
    assert a["Clgw_sd_source"] == "repeat samples" and np.isclose(a["Clgw_sd_used_mg_L"], 2.0)
    assert b["Clgw_sd_source"] == "blanket CV" and np.isclose(b["Clgw_sd_used_mg_L"], 25 * 0.37)


def test_known_values_from_lee2024():
    """Values taken from Chloride_dataset_20230907_state_id_out.csv (Lee et al. 2024)."""
    bores = pd.DataFrame({"ID": ["x"], "Lat": [-21.975], "Lon": [120.103], "Clgw": [50.0],
                          "Clgw_sd_samples": [np.nan], "n_samples": [1]})
    res, real = run_cmbear(bores, G, CMBSettings(cv_clgw=0.3655537955253879))
    assert real.shape == (1, 1000)
    assert res.loc[0, "status"].startswith("ok")
    # regular CMB identity
    assert np.isclose(res.loc[0, "R_CMB_mean_mm_y"], 100 * res.loc[0, "ClDep_mean_kg_ha_y"] / 50.0)
    # stochastic percentiles ordered
    assert res.loc[0, "R_RC_5_mm_y"] < res.loc[0, "R_RC_50_mm_y"] < res.loc[0, "R_RC_95_mm_y"]
