"""After generation: give every synthetic image the labels of its source image."""
from __future__ import annotations

import shutil

from .config import load


def run() -> None:
    cfg = load()
    img = cfg.out / "synth" / "images"
    lab = cfg.out / "synth" / "labels"
    lab.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in img.glob("*.jpg"):
        shutil.copy(cfg.out / "real" / "labels" / f"{p.stem.split('__')[0]}.txt", lab / f"{p.stem}.txt")
        n += 1
    print(f"{n} synthetic images labelled")


if __name__ == "__main__":
    run()
