"""Build the training runs. Within a seed the mixes are NESTED and share their randomness, so differences between
mixing rates are not caused by drawing a different set of real images.

 A  mixing rate        fixed budget of N images, share rho of them synthetic (rest real)       YOLOv8n
 A2 added synthetic    N real images plus 0 / N/3 / N extra synthetic images                    YOLOv8n
 B  which synthetic    rho = 0.5, synthetic images with the lowest / highest geometry drift,
                       chosen separately within each weather type so the weather mix is equal   YOLOv8n
 C  model size         rho in {0, 0.5} with YOLOv8s and YOLOv8m
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import Config, load


def _balanced_order(plan: pd.DataFrame, rng, key: pd.Series | None = None, ascending: bool | None = None) -> list[str]:
    """Synthetic image names ordered so that every prefix is balanced across weather types."""
    per = {}
    for e, g in plan.groupby("edit"):
        g = g.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
        if key is not None:
            g = g.assign(k=g.name_.map(key)).sort_values("k", ascending=ascending, kind="stable")
        per[e] = g.name_.tolist()
    out, i = [], 0
    while any(i < len(v) for v in per.values()):
        for e in sorted(per):
            if i < len(per[e]):
                out.append(per[e][i])
        i += 1
    return out


def build(cfg: Config | None = None, drift: pd.DataFrame | None = None) -> dict[str, list[dict]]:
    cfg = cfg or load()
    tc = cfg["train"]
    N = tc["total_images"]
    splits = pd.read_csv(cfg.out / "splits.csv")
    train_ids = sorted(splits[splits.split == "train"].id)
    plan = pd.read_csv(cfg.out / "synth_plan.csv")
    plan["name_"] = plan.id + "__" + plan.edit
    src_of = dict(zip(plan.name_, plan.id))
    groups: dict[str, list[dict]] = {"A": [], "A2": [], "B": [], "C": []}

    def spec(name, model, seed, real, synth):
        return {"name": name, "model": model, "seed": seed, "epochs": tc["epochs"], "train_real": real, "train_synth": synth}

    for seed in tc["seeds"]:
        rng = np.random.default_rng(1000 + seed)
        perm_real = list(rng.permutation(train_ids))
        syn_order = _balanced_order(plan, np.random.default_rng(2000 + seed))

        def mix(rho, order):
            n_syn = int(round(rho * N))
            syn = order[:n_syn]
            used = {src_of[s] for s in syn}
            real = [i for i in perm_real if i not in used][:N - n_syn]
            return real, syn

        for rho in (0.0, 0.25, 0.5, 0.75, 1.0):
            r, s = mix(rho, syn_order)
            groups["A"].append(spec(f"A_n_rho{int(rho * 100):03d}_s{seed}", "yolov8n.pt", seed, r, s))
        for extra in (N // 3, N):
            groups["A2"].append(spec(f"A2_n_plus{extra}_s{seed}", "yolov8n.pt", seed, perm_real[:N], syn_order[:extra]))
        if drift is not None:
            key = drift.set_index("name")["dense_disp_px"]
            for tag, asc in (("low", True), ("high", False)):
                order = _balanced_order(plan, np.random.default_rng(2000 + seed), key=key, ascending=asc)
                r, s = mix(0.5, order)
                groups["B"].append(spec(f"B_n_{tag}drift_s{seed}", "yolov8n.pt", seed, r, s))
        if seed < 2:
            for model in ("yolov8s.pt", "yolov8m.pt"):
                for rho in (0.0, 0.5):
                    r, s = mix(rho, syn_order)
                    groups["C"].append(spec(f"C_{model[6]}_rho{int(rho * 100):03d}_s{seed}", model, seed, r, s))
    return groups


def write(cfg: Config | None = None, drift: pd.DataFrame | None = None) -> None:
    cfg = cfg or load()
    for k, v in build(cfg, drift).items():
        if v:
            (cfg.out / f"specs_{k}.json").write_text(json.dumps(v))
            print(k, len(v), "runs")


if __name__ == "__main__":
    drift = pd.read_csv(load().out / "drift.csv") if (load().out / "drift.csv").exists() else None
    write(drift=drift)
