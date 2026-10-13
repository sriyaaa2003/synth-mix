"""Tables from results/runs/*.json: mixing rate, added synthetic data, drift-filtered selection, model size,
and whether the drift metric predicts label damage."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import ROOT

SUBSETS = ["all", "night", "adverse_weather", "clear_day"]
LABEL_NAMES = {"all": "all test", "night": "night", "adverse_weather": "rain/snow/fog", "clear_day": "clear day"}


def load_runs(runs_dir=None) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(runs_dir or ROOT / "results" / "runs").glob("*.json")):
        r = json.loads(p.read_text())
        n = r["name"]
        row = {"name": n, "group": n.split("_")[0], "model": r["spec"]["model"].replace(".pt", ""), "seed": r["spec"]["seed"],
               "n_real": r["n_real"], "n_synth": r["n_synth"], "minutes": r["seconds"] / 60}
        for s in SUBSETS:
            row[f"map_{s}"] = r["metrics"][s]["map5095"] * 100
            row[f"map50_{s}"] = r["metrics"][s]["map50"] * 100
        rows.append(row)
    df = pd.DataFrame(rows)
    df["rho"] = df.n_synth / (df.n_real + df.n_synth)
    return df


def paired(df: pd.DataFrame, a: str, b: str, col: str) -> dict:
    """b minus a over seeds that have both; 95% t-interval."""
    x = df[df.name == a][["seed", col]].merge(df[df.name == b][["seed", col]], on="seed", suffixes=("_a", "_b"))
    return _paired(x[f"{col}_a"].to_numpy(), x[f"{col}_b"].to_numpy())


def _paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = b - a
    n = len(d)
    if n < 2:
        return {"n": n, "diff": float(d.mean()) if n else np.nan, "lo": np.nan, "hi": np.nan}
    h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return {"n": n, "diff": float(d.mean()), "lo": float(d.mean() - h), "hi": float(d.mean() + h)}


def mixing_table(df: pd.DataFrame) -> pd.DataFrame:
    a = df[(df.group == "A") & (df.model == "yolov8n")].copy()
    a["rho"] = a.rho.round(2)
    rows = []
    base = a[a.rho == 0.0]
    for rho, g in a.groupby("rho"):
        row = {"synthetic_share": rho, "runs": len(g)}
        for s in SUBSETS:
            row[f"map_{s}"] = g[f"map_{s}"].mean()
            row[f"sd_{s}"] = g[f"map_{s}"].std(ddof=1) if len(g) > 1 else np.nan
            m = g.merge(base[["seed", f"map_{s}"]], on="seed", suffixes=("", "_base"))
            d = _paired(m[f"map_{s}_base"].to_numpy(), m[f"map_{s}"].to_numpy())
            row[f"diff_{s}"], row[f"lo_{s}"], row[f"hi_{s}"] = d["diff"], d["lo"], d["hi"]
        rows.append(row)
    return pd.DataFrame(rows)


def model_table(df: pd.DataFrame) -> pd.DataFrame:
    c = df[df.group == "C"].copy()
    c["rho"] = c.rho.round(2)
    n = df[(df.group == "A") & (df.model == "yolov8n") & (df.seed < 2) & (df.rho.round(2).isin([0.0, 0.5]))].copy()
    n["rho"] = n.rho.round(2)
    both = pd.concat([n, c])
    return both.groupby(["model", "rho"])[[f"map_{s}" for s in SUBSETS]].agg(["mean", "std"]).round(2)


def drift_vs_damage(drift: pd.DataFrame, scores: list[dict]) -> pd.DataFrame:
    s = pd.DataFrame(scores)
    s["iou_loss"] = s.mean_iou_real - s.mean_iou_synth
    s["recall_loss"] = s.recall50_real - s.recall50_synth
    m = s.merge(drift, on="name")
    m["edit"] = m.name.str.split("__").str[1]
    rows = []
    for metric in ("dense_disp_px", "sparse_disp_px", "rel_rot_deg", "edge_precision"):
        ok = m[[metric, "iou_loss"]].dropna()
        rho, p = stats.spearmanr(ok[metric], ok.iou_loss)
        rows.append({"metric": metric, "spearman_with_iou_loss": rho, "p": p, "n": len(ok)})
    return pd.DataFrame(rows), m
