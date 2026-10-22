"""Build the dataset files (one per rotation) for the real-data analyses.

Each call draws a new random split of the lineages into groups and builds all rotations
from it; ``summarize.py`` only pools rotations from the same build.

Example:
    uv run python -m scripts.real_data.prepare_dataset --dataset hematopoiesis
    uv run python -m scripts.real_data.prepare_dataset --dataset reprogramming
"""

import argparse
import json
import uuid

import numpy as np
import scipy.sparse as sp

from scripts.real_data.data import (
    DATASETS,
    FOLDS,
    assign_cv_groups,
    build_series,
    cell_table,
    check_folds,
    folds_for_rotation,
    lineage_strata,
    load_adata,
    preprocess,
    save_dataset,
    select_hvgs,
)

DEFAULT_MAX_MISSING = {"hematopoiesis": 1, "reprogramming": 4}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument(
        "--data_dir", default="data_cospar", help="Folder for the CoSpar h5ad files"
    )
    parser.add_argument(
        "--runs_dir",
        default="runs",
        help="Output folder; files go to <runs_dir>/<dataset> (default: runs)",
    )
    parser.add_argument(
        "--n_hvg", type=int, default=1000, help="Highly variable genes (default: 1000)"
    )
    parser.add_argument(
        "--cap_per_lineage",
        type=int,
        default=100,
        help="Maximum time series per clone (default: 100)",
    )
    parser.add_argument(
        "--max_missing",
        type=int,
        default=None,
        help="Days a series may miss; missing days are masked (default: 1 hematopoiesis, 4 reprogramming)",
    )
    parser.add_argument(
        "--min_genes",
        type=int,
        default=200,
        help="Minimum genes per cell (default: 200)",
    )
    parser.add_argument(
        "--n_groups",
        type=int,
        default=10,
        help="Lineage groups / rotations (default: 10)",
    )
    args = parser.parse_args()
    if args.n_groups < 3:
        parser.error("--n_groups must be at least 3 (test, validation and training)")
    out_dir = f"{args.runs_dir}/{args.dataset}"
    max_missing = (
        DEFAULT_MAX_MISSING[args.dataset]
        if args.max_missing is None
        else args.max_missing
    )
    run_id = uuid.uuid4().hex[:12]

    print(f"Loading {args.dataset} data...")
    adata = preprocess(load_adata(args.dataset, args.data_dir), args.min_genes)
    print(f"  After filtering: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    cells = cell_table(adata, args.dataset)
    strata = lineage_strata(cells, args.dataset, max_missing)
    print(f"  Lineages: {len(strata):,} ({strata.value_counts().to_dict()})")
    groups = assign_cv_groups(strata, args.n_groups)
    print(
        f"  Run ID {run_id}: new random lineage groups for all {args.n_groups} rotations"
    )

    rng = np.random.default_rng()
    for rotation in range(args.n_groups):
        folds = folds_for_rotation(groups, rotation, args.n_groups)
        cells["fold"] = cells["lineage"].map(folds).fillna("none")
        train_mask = (cells["fold"] == "train").to_numpy()
        hvgs = select_hvgs(
            adata, train_mask, args.n_hvg, DATASETS[args.dataset]["min_cells"]
        )
        X = adata[:, hvgs].X
        X = X.toarray() if sp.issparse(X) else np.asarray(X)

        series = {
            mode: {
                fold: build_series(
                    cells,
                    args.dataset,
                    mode,
                    fold,
                    args.cap_per_lineage,
                    rng,
                    max_missing,
                )
                for fold in FOLDS
            }
            for mode in DATASETS[args.dataset]["modes"]
        }
        report = check_folds(cells, series)
        meta = {
            "run_id": run_id,
            "dataset": args.dataset,
            "rotation": rotation,
            "n_groups": args.n_groups,
            "days": DATASETS[args.dataset]["days"],
            "classes": DATASETS[args.dataset]["classes"],
            "modes": DATASETS[args.dataset]["modes"],
            "genes": hvgs,
            "hvg_training_cells": int(train_mask.sum()),
            "hvg_min_cells": DATASETS[args.dataset]["min_cells"],
            "fold_unit": "clone"
            if args.dataset == "hematopoiesis"
            else "day-0 barcode family",
            "cap_per_lineage": args.cap_per_lineage,
            "max_missing": max_missing,
            "min_genes": args.min_genes,
            "fold_report": report,
        }
        out = f"{out_dir}/rotation_{rotation}/dataset.npz"
        save_dataset(out, X, cells, series, meta)
        counts = {
            mode: {
                f: (r["series"], r["lineages"], r["class_counts"])
                for f, r in by_fold.items()
            }
            for mode, by_fold in report.items()
            if isinstance(by_fold, dict)
        }
        print(
            f"Rotation {rotation}: (series, lineages, class counts) = {json.dumps(counts)}"
        )


if __name__ == "__main__":
    main()
