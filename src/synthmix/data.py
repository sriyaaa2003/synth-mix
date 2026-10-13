"""BDD100K validation split -> YOLO dataset at 640x360 with a video-level train/val/test split."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import Config, load


def run(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load()
    d = cfg["data"]
    bdd = cfg.bdd_dir
    samples = json.load(open(bdd / "samples.json"))["samples"]
    man = pd.read_csv(bdd / "manifest.csv").set_index("id")
    cls = {c: i for i, c in enumerate(d["classes"])}
    img_dir = cfg.out / "real" / "images"
    lab_dir = cfg.out / "real" / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(d["seed"])
    videos = np.array(sorted(man.video.unique()))
    rng.shuffle(videos)
    f = d["split_fracs"]
    n_tr, n_va = int(f[0] * len(videos)), int(f[1] * len(videos))
    split_of = {v: "train" for v in videos[:n_tr]} | {v: "val" for v in videos[n_tr:n_tr + n_va]} | {v: "test" for v in videos[n_tr + n_va:]}
    rows = []
    for s in samples:
        iid = Path(s["filepath"]).stem
        if iid not in man.index:
            continue
        src = bdd / "images" / f"{iid}.jpg"
        if not src.exists():
            continue
        lines = []
        for det in s["detections"]["detections"]:
            if det["label"] not in cls:
                continue
            x, y, w, h = det["bounding_box"]
            lines.append(f"{cls[det['label']]} {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}")
        im = cv2.imread(str(src))
        im = cv2.resize(im, (d["width"], d["height"]), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(img_dir / f"{iid}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (lab_dir / f"{iid}.txt").write_text("\n".join(lines))
        m = man.loc[iid]
        rows.append({"id": iid, "video": m.video, "split": split_of[m.video], "weather": m.weather, "tod": m.time_of_day,
                     "n_boxes": len(lines)})
    df = pd.DataFrame(rows)
    df.to_csv(cfg.out / "splits.csv", index=False)
    print(df.groupby("split").agg(images=("id", "size"), boxes=("n_boxes", "sum")))
    te = df[df.split == "test"]
    print("test night:", (te.tod == "night").sum(), " adverse weather:", te.weather.isin(d["adverse_weather"]).sum())
    return df


if __name__ == "__main__":
    run()
