"""Tests for the real-data folds and time series.

Run with:  uv run --with pytest python -m pytest tests/test_real_data_splits.py
"""

import numpy as np
import pandas as pd
import pytest

from scripts.real_data.data import (
    FOLDS,
    MISSING,
    assign_cv_groups,
    build_series,
    check_folds,
    folds_for_rotation,
    lineage_strata,
)


def toy_hematopoiesis(n_clones=60):
    rng = np.random.default_rng()
    rows = []
    for c in range(n_clones):
        fate = "Neutrophil" if c % 2 else "Monocyte"
        for day, n in (
            ("2", rng.integers(0, 4)),
            ("4", rng.integers(0, 5)),
            ("6", rng.integers(1, 6)),
        ):
            for _ in range(n):
                state = fate if day == "6" else "undiff"
                rows.append({"day": day, "state": state, "lineage": f"clone_{c}"})
    cells = pd.DataFrame(rows)
    cells["pos"] = np.arange(len(cells))
    return cells


def toy_reprogramming(n_families=40):
    rng = np.random.default_rng()
    days = ["Day6", "Day9", "Day12", "Day15", "Day21", "Day28"]
    rows = []
    for f in range(n_families):
        fate = "Failed" if f % 4 == 0 else "Reprogrammed"
        for sub in range(rng.integers(1, 3)):
            bc13 = f"t13_{f}_{sub}"
            for day in days:
                for _ in range(rng.integers(0, 3)):
                    state = fate if day in ("Day21", "Day28") else "Others"
                    rows.append(
                        {
                            "day": day,
                            "state": state,
                            "lineage": f"bc0_{f}",
                            "barcode_day0": str(f),
                            "barcode_day3": f"{f}_{sub}",
                            "barcode_day13": bc13
                            if day in ("Day15", "Day21", "Day28")
                            else "NA",
                        }
                    )
    cells = pd.DataFrame(rows)
    cells["pos"] = np.arange(len(cells))
    return cells


@pytest.mark.parametrize(
    "dataset,make,modes,max_missing",
    [
        ("hematopoiesis", toy_hematopoiesis, ["clonal"], 1),
        ("reprogramming", toy_reprogramming, ["cumulative", "single"], 4),
    ],
)
def test_folds_do_not_share_cells_or_lineages(dataset, make, modes, max_missing):
    cells = make()
    strata = lineage_strata(cells, dataset, max_missing)
    groups = assign_cv_groups(strata, n_groups=10)
    tested = []
    for rotation in range(10):
        cells["fold"] = cells["lineage"].map(folds_for_rotation(groups, rotation, 10))
        rng = np.random.default_rng()
        series = {
            mode: {
                fold: build_series(cells, dataset, mode, fold, 50, rng, max_missing)
                for fold in FOLDS
            }
            for mode in modes
        }
        check_folds(cells, series)  # raises on any shared cell or lineage
        for mode in modes:
            for fold, s in series[mode].items():
                used = s["idx"][s["idx"] != MISSING]
                assert set(cells["fold"].to_numpy()[used]) <= {fold}
        tested.extend(series[modes[0]]["test"]["lineage"].tolist())
    # Each lineage is in the test group of one rotation only.
    tested_once = pd.Series(tested).groupby(pd.Series(tested)).size()
    eligible = set(strata.index[strata != "no_series"])
    assert set(tested_once.index) <= eligible
    lineages_per_rotation = {}
    for rotation in range(10):
        folds = folds_for_rotation(groups, rotation, 10)
        for lineage in folds.index[folds == "test"]:
            lineages_per_rotation.setdefault(lineage, []).append(rotation)
    assert all(len(r) == 1 for r in lineages_per_rotation.values())


def test_fold_check_catches_shared_lineage():
    cells = toy_hematopoiesis()
    cells["fold"] = "train"
    largest = cells["lineage"].value_counts().idxmax()
    assert (cells["lineage"] == largest).sum() >= 2
    first_cell = cells.index[cells["lineage"] == largest][0]
    cells.loc[first_cell, "fold"] = "test"  # one clone now spans two folds
    series = {
        "clonal": {
            f: build_series(
                cells, "hematopoiesis", "clonal", f, 10, np.random.default_rng()
            )
            for f in FOLDS
        }
    }
    with pytest.raises(AssertionError):
        check_folds(cells, series)


def test_early_windows_never_include_the_outcome_cell():
    from scripts.real_data.run_analyses import window_arrays

    # Series "a" ends at day index 4; series "b" ends at day index 3.
    idx = np.array(
        [[0, 1, MISSING, 2, 3, MISSING], [4, MISSING, 5, 6, MISSING, MISSING]]
    )
    s = {"idx": idx, "label": np.array([1, 0]), "lineage": np.array(["a", "b"])}
    Z = np.ones((7, 2), dtype=np.float32)
    *_, keep = window_arrays(s, Z, 4)  # days 0-3: b's outcome cell would be an input
    assert keep.tolist() == [True, False]
    *_, keep_all = window_arrays(s, Z, 6)  # the all-days upper bound keeps every series
    assert keep_all.tolist() == [True, True]
