"""Run the training grid on any machine with a GPU (Kaggle, Colab, local).

    python runner.py --data /kaggle/working/data --specs /kaggle/working/data/specs_all.json --out /kaggle/working/runs

Resumable: a run whose JSON already exists in --out is skipped. With --of N --part i the spec list is split across
N processes (one per GPU); set CUDA_VISIBLE_DEVICES per process.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

CLASSES = ["car", "truck", "bus", "pedestrian", "rider", "bicycle", "motorcycle"]
ADVERSE = ["rainy", "snowy", "foggy"]
SUBSETS = {
    "all": lambda r: True,
    "night": lambda r: r.tod == "night",
    "adverse_weather": lambda r: r.weather in ADVERSE,
    "clear_day": lambda r: r.weather in ("clear", "overcast", "partly cloudy") and r.tod == "daytime",
}


def iou(a, b):
    import numpy as np

    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ua = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ub = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (ua[:, None] + ub[None, :] - inter + 1e-9)


def score_synth(model, root: Path, imgsz: int):
    """For each synthetic image and its real source: how well do the SOURCE boxes fit what a real-only detector finds?"""
    import numpy as np

    rows = []
    for syn in sorted((root / "synth" / "images").glob("*.jpg")):
        src = syn.stem.split("__")[0]
        lf = root / "real" / "labels" / f"{src}.txt"
        if not lf.exists() or lf.stat().st_size == 0:
            continue
        gt = np.loadtxt(lf, ndmin=2)
        W, H = 640, 360
        cls = gt[:, 0].astype(int)
        box = np.stack([(gt[:, 1] - gt[:, 3] / 2) * W, (gt[:, 2] - gt[:, 4] / 2) * H,
                        (gt[:, 1] + gt[:, 3] / 2) * W, (gt[:, 2] + gt[:, 4] / 2) * H], 1)
        row = {"name": syn.stem, "n_gt": int(len(gt))}
        for tag, path in (("real", root / "real" / "images" / f"{src}.jpg"), ("synth", syn)):
            r = model.predict(str(path), imgsz=imgsz, conf=0.25, verbose=False)[0].boxes
            pb, pc = r.xyxy.cpu().numpy(), r.cls.cpu().numpy().astype(int)
            best = np.zeros(len(gt))
            for i in range(len(gt)):
                m = pc == cls[i]
                if m.any():
                    best[i] = iou(box[i:i + 1], pb[m]).max()
            row[f"mean_iou_{tag}"] = float(best.mean())
            row[f"recall50_{tag}"] = float((best >= 0.5).mean())
        rows.append(row)
    return rows


def run_spec(spec: dict, root: Path, work: Path, epochs: int, imgsz: int, batch: int, workers: int) -> dict:
    import pandas as pd
    from ultralytics import YOLO, settings

    settings.update({"raytune": False, "wandb": False, "mlflow": False, "comet": False, "clearml": False, "tensorboard": False})
    t0 = time.time()
    splits = pd.read_csv(root / "splits.csv")
    paths = lambda kind, ids: [str(root / kind / "images" / f"{i}.jpg") for i in ids]

    def yaml(tag, train, val):
        (work / f"{tag}_train.txt").write_text("\n".join(train))
        (work / f"{tag}_val.txt").write_text("\n".join(val))
        y = work / f"{tag}.yaml"
        y.write_text(f"path: {root}\ntrain: {work / (tag + '_train.txt')}\nval: {work / (tag + '_val.txt')}\n"
                     f"names: {{{', '.join(f'{i}: {n}' for i, n in enumerate(CLASSES))}}}\n")
        return str(y)

    train = paths("real", spec["train_real"]) + paths("synth", spec["train_synth"])
    val = paths("real", splits[splits.split == "val"].id.tolist())
    name = spec["name"]
    batch = 16 if "yolov8m" in spec["model"] else batch
    YOLO(spec["model"]).train(data=yaml(f"{name}_run", train, val), epochs=epochs, imgsz=imgsz, batch=batch, seed=spec["seed"],
                              workers=workers, project=str(work / "runs"), name=name, exist_ok=True, plots=False, verbose=False,
                              patience=0, amp=True, deterministic=False)
    best = YOLO(str(work / "runs" / name / "weights" / "best.pt"))
    test = splits[splits.split == "test"]
    out = {"name": name, "spec": {k: v for k, v in spec.items() if not k.startswith("train_")},
           "n_real": len(spec["train_real"]), "n_synth": len(spec["train_synth"]), "metrics": {}}
    for sub, fn in SUBSETS.items():
        ids = [r.id for r in test.itertuples() if fn(r)][: spec.get("test_limit") or None]
        m = best.val(data=yaml(f"{name}_{sub}", train[:1], paths("real", ids)), split="val", imgsz=imgsz, batch=batch,
                     plots=False, verbose=False).box
        out["metrics"][sub] = {"images": len(ids), "map5095": float(m.map), "map50": float(m.map50),
                               "precision": float(m.mp), "recall": float(m.mr)}
    if spec.get("score_synth"):
        out["synth_scores"] = score_synth(best, root, imgsz)
    out["seconds"] = time.time() - t0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--specs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/tmp/synthmix_work")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--deadline-hours", type=float, default=1e9, help="do not START a run after this many hours")
    a = ap.parse_args()
    root, out, work = Path(a.data), Path(a.out), Path(a.work)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    specs = json.loads(Path(a.specs).read_text())[a.part::a.of]
    t_start = time.time()
    for s in specs:
        f = out / f"{s['name']}.json"
        if f.exists():
            continue
        if (time.time() - t_start) / 3600 > a.deadline_hours:
            print("deadline reached, not starting", s["name"], flush=True)
            continue
        r = run_spec(s, root, work, s.get("epochs", a.epochs), a.imgsz, a.batch, a.workers)
        f.write_text(json.dumps(r))
        m = r["metrics"]
        print(f"[part {a.part}] {s['name']}: all {m['all']['map5095']:.4f} night {m['night']['map5095']:.4f} "
              f"adverse {m['adverse_weather']['map5095']:.4f} ({r['seconds'] / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
