"""Geometry drift between every real training image and its generated weather edit (uses the dgeo metrics from the
driving-geometry project). A synthetic image keeps the labels of its source only if nothing moved."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import cv2
import pandas as pd

from .config import Config, load


def _score(args):
    real_dir, synth_dir, name = args
    from dgeo.drift.metrics import all_metrics

    src = name.split("__")[0]
    a = cv2.cvtColor(cv2.imread(f"{real_dir}/{src}.jpg"), cv2.COLOR_BGR2RGB)
    b = cv2.cvtColor(cv2.imread(f"{synth_dir}/{name}.jpg"), cv2.COLOR_BGR2RGB)
    return {"name": name, **all_metrics(a, b)}


def run(cfg: Config | None = None, workers: int = 12) -> pd.DataFrame:
    cfg = cfg or load()
    synth_dir = cfg.out / "synth" / "images"
    names = sorted(p.stem for p in synth_dir.glob("*.jpg"))
    jobs = [(str(cfg.out / "real" / "images"), str(synth_dir), n) for n in names]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_score, jobs, chunksize=20))
    df = pd.DataFrame(rows)
    df["edit"] = df.name.str.split("__").str[1]
    df.to_csv(cfg.out / "drift.csv", index=False)
    print(df.groupby("edit")[["dense_disp_px", "sparse_disp_px", "rel_rot_deg", "edge_precision"]].median().round(2))
    return df


if __name__ == "__main__":
    run()
