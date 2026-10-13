"""results/runs/*.json (+ data/drift.csv) -> results/results.md and figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import analysis as an
from .config import ROOT

BLUE, ORANGE, GREEN, GREY, INK = "#0072B2", "#D55E00", "#009E73", "#8a8f98", "#222222"
LABEL = {"all": "all test images", "night": "night", "adverse_weather": "rain, snow or fog", "clear_day": "clear daytime"}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color="#e6e8eb", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=9)


def fig_mixing(df, out):
    a = df[(df.group == "A") & (df.model == "yolov8n")].copy()
    a["rho"] = a.rho.round(2)
    fig, axs = plt.subplots(1, 4, figsize=(12.5, 3.4), sharex=True)
    for ax, sub in zip(axs, an.SUBSETS):
        for seed, g in a.groupby("seed"):
            ax.plot(g.sort_values("rho").rho * 100, g.sort_values("rho")[f"map_{sub}"], color=GREY, lw=0.8, alpha=0.7, marker="o", ms=3)
        m = a.groupby("rho")[f"map_{sub}"].mean()
        ax.plot(m.index * 100, m.values, color=BLUE, lw=2.2, marker="o", ms=5)
        ax.set_title(LABEL[sub], loc="left", fontsize=9.5, color=INK)
        ax.set_xlabel("synthetic share of training images (%)", fontsize=8.5)
        style(ax)
    axs[0].set_ylabel("mAP50-95 (%)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "mixing.png", dpi=170)


def fig_filter(df, out):
    a = df[(df.group == "A") & (df.model == "yolov8n") & (df.rho.round(2) == 0.5)].assign(kind="random half")
    b = df[df.group == "B"].assign(kind=lambda d: np.where(d.name.str.contains("low"), "lowest-drift half", "highest-drift half"))
    d = pd.concat([a, b])
    if b.empty:
        return
    base = df[(df.group == "A") & (df.model == "yolov8n") & (df.rho == 0)][["seed"] + [f"map_{s}" for s in an.SUBSETS]]
    fig, axs = plt.subplots(1, 4, figsize=(12.5, 3.2), sharey=False)
    cols = {"lowest-drift half": GREEN, "random half": BLUE, "highest-drift half": ORANGE}
    for ax, sub in zip(axs, an.SUBSETS):
        for i, (kind, g) in enumerate(d.groupby("kind")):
            m = g.merge(base, on="seed", suffixes=("", "_base"))
            diff = m[f"map_{sub}"] - m[f"map_{sub}_base"]
            ax.plot([cols and list(cols).index(kind)] * len(diff), diff, "o", color=cols[kind], ms=6, alpha=0.8)
            ax.plot([list(cols).index(kind)], [diff.mean()], "_", color=INK, ms=22, mew=2)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(range(3), ["lowest\ndrift", "random", "highest\ndrift"], fontsize=8.5)
        ax.set_title(LABEL[sub], loc="left", fontsize=9.5, color=INK)
        style(ax)
    axs[0].set_ylabel("mAP50-95 minus the real-only run (points)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "drift_filter.png", dpi=170)


def fig_damage(m, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    cols = {"fog": GREY, "night": BLUE, "rain": GREEN, "snow": ORANGE}
    for e, g in m.groupby("edit"):
        ax.plot(g.dense_disp_px, g.iou_loss, "o", ms=3.5, color=cols[e], alpha=0.5, label=e)
    ax.set_xscale("log")
    ax.set_xlabel("geometry drift between the real image and its edit (px, dense flow)", fontsize=9)
    ax.set_ylabel("fall in how well the original boxes fit (mean IoU)", fontsize=9)
    ax.legend(frameon=False, fontsize=8.5)
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "drift_vs_label_damage.png", dpi=170)


def run(runs_dir=None) -> None:
    res = ROOT / "results"
    out = res / "figures"
    out.mkdir(exist_ok=True, parents=True)
    df = an.load_runs(runs_dir)
    lines = [f"{len(df)} runs\n"]
    if (df.group == "A").any():
        t = an.mixing_table(df)
        lines += ["## Mixing rate (YOLOv8n, 3,000 images per run)\n", t.round(2).to_markdown(index=False), ""]
        fig_mixing(df, out)
    if (df.group == "A2").any() or (df.group == "B").any():
        base = df[(df.group == "A") & (df.model == "yolov8n") & (df.rho == 0)]
        rand50 = df[(df.group == "A") & (df.model == "yolov8n") & (df.rho.round(2) == 0.5)]
        pick = lambda pat: df[df.name.str.contains(pat)]
        pairs = [("3,000 real + 1,000 synthetic vs 3,000 real", base, pick("plus1000")),
                 ("3,000 real + 3,000 synthetic vs 3,000 real", base, pick("plus3000")),
                 ("50% mix, lowest-drift half vs random half", rand50, pick("lowdrift")),
                 ("50% mix, highest-drift half vs random half", rand50, pick("highdrift")),
                 ("50% mix, lowest-drift vs highest-drift half", pick("highdrift"), pick("lowdrift")),
                 ("50% mix, lowest-drift half vs real only", base, pick("lowdrift"))]
        rows = []
        for label, a, b in pairs:
            if len(a) and len(b):
                row = {"comparison (second minus first, mAP50-95 points)": label}
                for sub in an.SUBSETS:
                    m = a[["seed", f"map_{sub}"]].merge(b[["seed", f"map_{sub}"]], on="seed", suffixes=("_a", "_b"))
                    d = an._paired(m[f"map_{sub}_a"].to_numpy(), m[f"map_{sub}_b"].to_numpy())
                    row[an.LABEL_NAMES[sub]] = f"{d['diff']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}]"
                rows.append(row)
        lines += ["## Added synthetic data and drift-filtered selection (paired by seed, 95% interval)\n", pd.DataFrame(rows).to_markdown(index=False), ""]
    if (df.group == "B").any():
        fig_filter(df, out)
    if (df.group == "C").any():
        lines += ["## Model size\n", an.model_table(df).to_markdown(), ""]
    for p in sorted((runs_dir and Path(runs_dir) or res / "runs").glob("*.json")):
        r = json.loads(p.read_text())
        if r.get("synth_scores"):
            drift = pd.read_csv(ROOT / "data" / "drift.csv")
            tab, m = an.drift_vs_damage(drift, r["synth_scores"])
            lines += ["## Does drift predict label damage?\n", tab.round(3).to_markdown(index=False), ""]
            fig_damage(m, out)
            m.to_csv(res / "synth_label_scores.csv", index=False)
    (res / "results.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    run()
