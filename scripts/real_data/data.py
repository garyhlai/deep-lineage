"""Data preparation for the real lineage-tracing datasets, both downloaded through CoSpar:

* ``hematopoiesis``: LARRY in vitro hematopoiesis (Weinreb et al., 2020), days 2, 4, 6.
* ``reprogramming``: CellTagging fibroblast-to-iEP reprogramming (Biddy et al., 2018),
  days 6, 9, 12, 15, 21, 28.

Lineages (clones, or day-0 barcode families for reprogramming) are split into groups;
each rotation tests on one group, validates on the next and trains on the rest. A time
series combines one cell per day from a single lineage.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FOLDS = ("train", "val", "test")
MISSING = -1

DATASETS = {
    "hematopoiesis": {
        "file": "LARRY_adata_preprocessed.h5ad",
        "cospar_loader": "hematopoiesis",
        "days": ["2", "4", "6"],
        "day_labels": ["Day 2", "Day 4", "Day 6"],
        "classes": ["Monocyte", "Neutrophil"],
        "modes": ["clonal"],
        "min_cells": 1,
    },
    "reprogramming": {
        "file": "CellTagging_adata_preprocessed.h5ad",
        "cospar_loader": "reprogramming",
        "days": ["Day6", "Day9", "Day12", "Day15", "Day21", "Day28"],
        "day_labels": ["Day 6", "Day 9", "Day 12", "Day 15", "Day 21", "Day 28"],
        "classes": ["Failed", "Reprogrammed"],
        "modes": ["cumulative", "single"],
        "min_cells": 100,
    },
}


# --------------------------------------------------------------------------- loading


def load_adata(dataset: str, data_dir: str | Path):
    """Read the CoSpar h5ad file, downloading it with CoSpar if it is not present."""
    import anndata as ad

    spec = DATASETS[dataset]
    data_dir = Path(data_dir)
    path = data_dir / spec["file"]
    if not path.exists():
        import cospar as cs

        data_dir.mkdir(parents=True, exist_ok=True)
        cs.settings.data_path = str(data_dir)
        getattr(cs.datasets, spec["cospar_loader"])()
    return ad.read_h5ad(path)


def preprocess(adata, min_genes: int = 200):
    """Cell filtering and log transformation (the CoSpar files are already
    quality-filtered and depth-normalized). Genes are filtered per rotation in
    ``select_hvgs``."""
    import scanpy as sc

    adata = adata.copy()
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.log1p(adata)
    return adata


def _clean_barcode(values) -> np.ndarray:
    s = pd.Series(values).astype(str)
    return s.where(~s.isin(["nan", "None", "", "NaN"]), "NA").to_numpy()


def cell_table(adata, dataset: str) -> pd.DataFrame:
    """One row per cell: position, day, state and lineage identifiers."""
    obs = adata.obs
    cells = pd.DataFrame(
        {
            "pos": np.arange(adata.n_obs),
            "day": obs["time_info"].astype(str).to_numpy(),
            "state": obs["state_info"].astype(str).to_numpy(),
        },
        index=obs.index.astype(str),
    )
    if dataset == "hematopoiesis":
        clones = adata.obsm["X_clone"].tocsr()
        n_clones_per_cell = np.diff(clones.indptr)
        clone_id = np.full(adata.n_obs, -1, dtype=np.int64)
        single = n_clones_per_cell == 1
        clone_id[single] = clones.indices[clones.indptr[:-1][single]]
        cells["lineage"] = np.where(
            clone_id >= 0, np.char.add("clone_", clone_id.astype(str)), "NA"
        )
    else:
        for col in ("barcode_day0", "barcode_day3", "barcode_day13"):
            cells[col] = _clean_barcode(obs[col].to_numpy())
        cells["lineage"] = np.where(
            cells["barcode_day0"] != "NA", "bc0_" + cells["barcode_day0"], "NA"
        )
    return cells


# --------------------------------------------------------------------------- folds


def lineage_strata(cells: pd.DataFrame, dataset: str, max_missing: int) -> pd.Series:
    """Stratum of each lineage for the fold assignment: whether it yields any time
    series and, if so, its majority outcome. Used only to balance the folds."""
    spec = DATASETS[dataset]
    probe = cells.assign(fold="all")
    mode = spec["modes"][-1]  # "clonal", or "single" (a superset of cumulative)
    series = build_series(
        probe, dataset, mode, "all", 1, np.random.default_rng(), max_missing
    )
    eligible = set(series["lineage"].tolist())
    labelled = cells[(cells["lineage"] != "NA") & cells["state"].isin(spec["classes"])]
    if dataset == "hematopoiesis":
        labelled = labelled[labelled["day"] == spec["days"][-1]]
    majority = labelled.groupby("lineage")["state"].agg(
        lambda s: s.value_counts().idxmax()
    )
    lineages = sorted(cells.loc[cells["lineage"] != "NA", "lineage"].unique())
    return pd.Series(
        {
            lin: f"series_{majority.get(lin, 'none')}"
            if lin in eligible
            else "no_series"
            for lin in lineages
        }
    )


def assign_cv_groups(strata: pd.Series, n_groups: int) -> pd.Series:
    """Randomly split lineages into ``n_groups`` equal parts, balanced within each
    stratum. The assignment differs between calls, so all rotations of a run are built
    from a single call (see ``prepare_dataset.py``)."""
    rng = np.random.default_rng()
    groups = {}
    for stratum in sorted(strata.unique()):
        ids = np.array(sorted(strata.index[strata == stratum]))
        rng.shuffle(ids)
        offset = int(rng.integers(n_groups))
        for i, lineage in enumerate(ids):
            groups[lineage] = (i + offset) % n_groups
    return pd.Series(groups).sort_index()


def folds_for_rotation(groups: pd.Series, rotation: int, n_groups: int) -> pd.Series:
    """Rotation ``r`` tests on group r, validates on group r+1 and trains on the rest
    (80%/10%/10% for 10 groups)."""
    test, val = rotation % n_groups, (rotation + 1) % n_groups
    return groups.map(lambda g: "test" if g == test else "val" if g == val else "train")


def select_hvgs(
    adata, train_mask: np.ndarray, n_top: int = 1000, min_cells: int = 1
) -> list[str]:
    """Highly variable genes of the training cells, among genes expressed in at least
    ``min_cells`` of them."""
    import scanpy as sc

    train = adata[train_mask].copy()
    sc.pp.filter_genes(train, min_cells=min_cells)
    sc.pp.highly_variable_genes(train, n_top_genes=n_top)
    return train.var_names[train.var["highly_variable"].to_numpy()].tolist()


# --------------------------------------------------------------------------- series


def _sample_from_blocks(blocks, cap: int, rng) -> list[tuple]:
    """Sample up to ``cap`` distinct combinations from a union of Cartesian products.

    ``blocks`` is a list of lists of per-day options (cell positions or MISSING).
    Sampling is uniform over all combinations of all blocks.
    """
    sizes = np.array([np.prod([len(o) for o in b], dtype=np.int64) for b in blocks])
    total = int(sizes.sum())
    if total == 0:
        return []
    if total <= cap:
        picks = np.arange(total)
    else:
        picks = np.sort(rng.choice(total, size=cap, replace=False))
    bounds = np.cumsum(sizes)
    out = []
    for p in picks:
        b = int(np.searchsorted(bounds, p, side="right"))
        local = int(p - (bounds[b - 1] if b else 0))
        shape = [len(o) for o in blocks[b]]
        coords = np.unravel_index(local, shape)
        out.append(tuple(blocks[b][d][c] for d, c in enumerate(coords)))
    return out


def _hematopoiesis_blocks(group: pd.DataFrame, spec: dict, max_missing: int):
    """Within-clone combinations of one day-2, one day-4 and one day-6 cell. The day-6
    cell must be a neutrophil or monocyte (it gives the fate label). A day is left
    missing (and later masked) only when the clone has no cell on that day."""
    days, classes = spec["days"], spec["classes"]
    last = group[(group["day"] == days[-1]) & group["state"].isin(classes)][
        "pos"
    ].tolist()
    if not last:
        return []
    options = []
    for d in days[:-1]:
        cells_on_day = group.loc[group["day"] == d, "pos"].tolist()
        options.append(cells_on_day if cells_on_day else [MISSING])
    if sum(o == [MISSING] for o in options) > max_missing:
        return []
    return [options + [last]]


def _reprogramming_blocks(group: pd.DataFrame, spec: dict, mode: str, max_missing: int):
    """Within-lineage combinations with one cell per day.

    A day is left missing only when the lineage has no cell on that day, and at most
    ``max_missing`` days may be missing. In cumulative mode, cells from day 15 onward
    with different day-13 barcodes are not combined (cells without a detected day-13
    barcode form one group). The outcome is the state of the last cell.
    """
    days, classes = spec["days"], spec["classes"]
    by_day = [group[group["day"] == d] for d in days]
    present = [len(c) > 0 for c in by_day]
    if sum(not p for p in present) > max_missing:
        return []
    last_day = max(i for i, p in enumerate(present) if p)
    by_day[last_day] = by_day[last_day][by_day[last_day]["state"].isin(classes)]
    if len(by_day[last_day]) == 0:
        return []
    late = [
        i for i, d in enumerate(days) if d in ("Day15", "Day21", "Day28") and present[i]
    ]
    if mode == "cumulative" and late:
        bc13_values = sorted(
            set().union(*[set(by_day[i]["barcode_day13"]) for i in late])
        )
        subsets = []
        for value in bc13_values:
            sub = list(by_day)
            for i in late:
                sub[i] = by_day[i][by_day[i]["barcode_day13"] == value]
            subsets.append(sub)
    else:
        subsets = [by_day]
    blocks = []
    for sub in subsets:
        options = []
        for i, cells_on_day in enumerate(sub):
            if not present[i]:
                options.append([MISSING])
            elif len(cells_on_day) == 0:
                options = None
                break
            else:
                options.append(cells_on_day["pos"].tolist())
        if options is not None:
            blocks.append(options)
    return blocks


def build_series(
    cells: pd.DataFrame,
    dataset: str,
    mode: str,
    fold: str,
    cap_per_lineage: int,
    rng,
    max_missing: int = 1,
) -> dict:
    """Time series for one fold. Returns cell indices (MISSING for absent days), labels
    (index into the class list, taken from the last cell) and the lineage of each series."""
    spec = DATASETS[dataset]
    classes = spec["classes"]
    sub = cells[(cells["fold"] == fold) & (cells["lineage"] != "NA")]
    if mode == "cumulative":
        sub = sub[sub["barcode_day3"] != "NA"]  # cumulative series need a day-3 barcode
    state_of = cells["state"].to_numpy()
    idx, labels, lineages = [], [], []
    for lineage, group in sub.groupby("lineage", sort=True):
        if dataset == "hematopoiesis":
            blocks = _hematopoiesis_blocks(group, spec, max_missing)
        elif mode == "cumulative":  # cells of a series share the day-3 barcode
            blocks = [
                block
                for _, unit in group.groupby("barcode_day3", sort=True)
                for block in _reprogramming_blocks(unit, spec, mode, max_missing)
            ]
        else:
            blocks = _reprogramming_blocks(group, spec, mode, max_missing)
        for combo in _sample_from_blocks(blocks, cap_per_lineage, rng):
            last = [c for c in combo if c != MISSING][-1]
            idx.append(combo)
            labels.append(classes.index(state_of[last]))
            lineages.append(lineage)
    n_days = len(spec["days"])
    return {
        "idx": np.asarray(idx, dtype=np.int64).reshape(-1, n_days),
        "label": np.asarray(labels, dtype=np.int64),
        "lineage": np.asarray(lineages, dtype=object),
    }


# --------------------------------------------------------------------------- checks


def check_folds(cells: pd.DataFrame, series: dict) -> dict:
    """Raise if any lineage or cell is shared between folds; return series counts."""
    fold_per_lineage = (
        cells[cells["lineage"] != "NA"].groupby("lineage")["fold"].nunique()
    )
    if (fold_per_lineage > 1).any():
        raise AssertionError("A lineage has cells in more than one fold.")
    cell_fold = cells["fold"].to_numpy()
    report = {}
    for mode, by_fold in series.items():
        used = {}
        for fold, s in by_fold.items():
            positions = s["idx"][s["idx"] != MISSING]
            if len(positions) and not (cell_fold[positions] == fold).all():
                raise AssertionError(
                    f"{mode}/{fold}: a series uses a cell from another fold."
                )
            used[fold] = (set(positions.tolist()), set(s["lineage"].tolist()))
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            if used[a][0] & used[b][0]:
                raise AssertionError(f"{mode}: cells shared between {a} and {b}.")
            if used[a][1] & used[b][1]:
                raise AssertionError(f"{mode}: lineages shared between {a} and {b}.")
        report[mode] = {
            fold: {
                "series": len(s["label"]),
                "cells": len(used[fold][0]),
                "lineages": len(used[fold][1]),
                "class_counts": np.bincount(s["label"], minlength=2).tolist(),
            }
            for fold, s in by_fold.items()
        }
    return report


# --------------------------------------------------------------------------- io


def save_dataset(path: str | Path, X, cells, series, meta) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "X": X.astype(np.float32),
        "cell_day": cells["day"].to_numpy(dtype=str),
        "cell_state": cells["state"].to_numpy(dtype=str),
        "cell_lineage": cells["lineage"].to_numpy(dtype=str),
        "cell_fold": cells["fold"].to_numpy(dtype=str),
    }
    for mode, by_fold in series.items():
        for fold, s in by_fold.items():
            for key in ("idx", "label"):
                arrays[f"{mode}__{fold}__{key}"] = s[key]
            arrays[f"{mode}__{fold}__lineage"] = s["lineage"].astype(str)
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def load_dataset(path: str | Path) -> dict:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    meta = json.loads(path.with_suffix(".json").read_text())
    series = {}
    for key in [k for k in data if k.count("__") == 2]:
        mode, fold, field = key.split("__")
        series.setdefault(mode, {}).setdefault(fold, {})[field] = data[key]
    cells = pd.DataFrame(
        {
            "day": data["cell_day"],
            "state": data["cell_state"],
            "lineage": data["cell_lineage"],
            "fold": data["cell_fold"],
        }
    )
    return {"X": data["X"], "cells": cells, "series": series, "meta": meta}
