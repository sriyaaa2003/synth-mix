import json

import pandas as pd
import pytest

from synthmix.config import load
from synthmix.specs import build

CFG = load()
HAVE = (CFG.out / "splits.csv").exists() and (CFG.out / "synth_plan.csv").exists()
pytestmark = pytest.mark.skipif(not HAVE, reason="data not built")


@pytest.fixture(scope="module")
def groups():
    drift = pd.read_csv(CFG.out / "drift.csv") if (CFG.out / "drift.csv").exists() else None
    return build(CFG, drift)


def test_split_is_by_video_and_disjoint():
    s = pd.read_csv(CFG.out / "splits.csv")
    per_video = s.groupby("video").split.nunique()
    assert (per_video == 1).all()
    assert set(s.split) == {"train", "val", "test"}


def test_only_training_images_are_ever_used_for_training(groups):
    s = pd.read_csv(CFG.out / "splits.csv").set_index("id")
    for runs in groups.values():
        for r in runs:
            ids = set(r["train_real"]) | {n.split("__")[0] for n in r["train_synth"]}
            assert set(s.loc[sorted(ids)].split) == {"train"}


def test_fixed_budget_runs_have_the_same_size_and_no_real_copy_of_a_synthetic_source(groups):
    for r in groups["A"]:
        assert len(r["train_real"]) + len(r["train_synth"]) == CFG["train"]["total_images"]
        assert not set(r["train_real"]) & {n.split("__")[0] for n in r["train_synth"]}


def test_mixes_are_nested_within_a_seed(groups):
    by = {r["name"]: r for r in groups["A"]}
    for seed in CFG["train"]["seeds"]:
        a, b = by[f"A_n_rho025_s{seed}"], by[f"A_n_rho050_s{seed}"]
        assert set(a["train_synth"]) <= set(b["train_synth"])


def test_synthetic_selection_is_balanced_across_weather_types(groups):
    for r in groups["A"]:
        if r["train_synth"]:
            counts = pd.Series([n.split("__")[1] for n in r["train_synth"]]).value_counts()
            assert counts.max() - counts.min() <= 1


def test_drift_filter_runs_pick_low_versus_high_drift(groups):
    if not groups["B"]:
        pytest.skip("no drift file")
    drift = pd.read_csv(CFG.out / "drift.csv").set_index("name").dense_disp_px
    by = {r["name"]: r for r in groups["B"]}
    for seed in CFG["train"]["seeds"]:
        lo = drift.loc[by[f"B_n_lowdrift_s{seed}"]["train_synth"]].mean()
        hi = drift.loc[by[f"B_n_highdrift_s{seed}"]["train_synth"]].mean()
        assert lo < hi


def test_specs_are_json_serialisable(groups):
    json.dumps(groups)
