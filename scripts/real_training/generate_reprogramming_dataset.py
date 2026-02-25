"""
Unified script to generate reprogramming dataset for Deep Lineage.

Generates time series of gene expression for cellular reprogramming lineages,
with train/val/test splits for Reprogrammed vs Failed classification.

Usage:
    uv run python scripts/generate_reprogramming_dataset.py --mode cumulative
    uv run python scripts/generate_reprogramming_dataset.py --mode single

Output:
    reprogramming_dataset_cumulative.h5 or reprogramming_dataset_single.h5
"""

import argparse
import json
import os
from collections import defaultdict

import cospar as cs
import h5py
import numpy as np
import pandas as pd
import scanpy as sc

# Configuration
N_TOP_GENES = 1000
RANDOM_SEED = 17179
SAMPLES_PER_CLASS = 45000  # Will be split 15k/15k/15k
TIMEPOINTS = ["Day6", "Day9", "Day12", "Day15", "Day21", "Day28"]


def load_and_preprocess():
    """Load raw data and preprocess gene expression.

    Returns:
        adata: Preprocessed AnnData object
        df: Combined DataFrame with gene expression + metadata, 0-indexed
    """
    print("Loading reprogramming data...")
    adata = cs.datasets.reprogramming()

    print("Preprocessing...")
    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.filter_cells(adata, min_genes=1000)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=N_TOP_GENES)
    adata = adata[:, adata.var["highly_variable"]]

    print(f"After preprocessing: {adata.shape}")

    # Build combined dataframe with 0-based indexing
    gene_expr = adata.to_df()
    metadata = adata.obs[
        ["time_info", "state_info", "barcode_day0", "barcode_day3", "barcode_day13"]
    ].copy()
    metadata = metadata.astype(object).fillna("NA")  # Handle NaN barcodes

    df = pd.concat(
        [gene_expr.reset_index(drop=True), metadata.reset_index(drop=True)], axis=1
    )

    return adata, df


def build_lineage_indices(df, mode="cumulative"):
    """Build time series indices using specified barcoding mode.

    Args:
        df: DataFrame with gene expression + metadata
        mode: "cumulative" (bc0 + bc3 + bc13) or "single" (bc0 only)

    Returns:
        lineages_df: DataFrame with columns [Day6, Day9, ..., Day28, state_info]
                     Values are row indices into df, or -1 for missing
    """
    print(f"Building lineage indices (mode={mode})...")

    if mode == "single":
        # Single barcoding: group by barcode_day0 only
        barcode_groups = df[["barcode_day0"]].value_counts().reset_index()
        barcode_groups = barcode_groups[barcode_groups["barcode_day0"] != "NA"]

        # Calculate dynamic cap per group
        n_groups = len(barcode_groups)
        total_needed = SAMPLES_PER_CLASS // 3 * 2 * 2  # train + val, both classes
        cap_per_group = max(100, (total_needed * 2) // n_groups)
        print(f"Single barcoding: {n_groups} groups, cap={cap_per_group} per group")

        all_lineages = []
        for _, row in barcode_groups.iterrows():
            bc0 = row["barcode_day0"]

            mask = df["barcode_day0"] == bc0
            clone = df[mask]

            cells_by_day = defaultdict(lambda: [-1])
            for idx, cell in clone.iterrows():
                day = int(cell["time_info"][3:])
                cells_by_day[day].append(idx)

            lineages = _enumerate_lineages_single(
                clone, cells_by_day, max_lineages=cap_per_group
            )
            all_lineages.extend(lineages)
    else:
        # Cumulative barcoding: group by (barcode_day0, barcode_day3)
        barcode_groups = (
            df[["barcode_day0", "barcode_day3"]].value_counts().reset_index()
        )

        all_lineages = []
        for _, row in barcode_groups.iterrows():
            bc0, bc3 = row["barcode_day0"], row["barcode_day3"]
            if bc0 == "NA" or bc3 == "NA":
                continue

            mask = (df["barcode_day0"] == bc0) & (df["barcode_day3"] == bc3)
            clone = df[mask]

            cells_by_day = defaultdict(lambda: [-1])
            for idx, cell in clone.iterrows():
                day = int(cell["time_info"][3:])
                cells_by_day[day].append(idx)

            lineages = _enumerate_lineages_cumulative(clone, cells_by_day)
            all_lineages.extend(lineages)

    lineages_df = pd.DataFrame(
        all_lineages,
        columns=["Day6", "Day9", "Day12", "Day15", "Day21", "Day28", "state_info"],
    )

    print(f"Total lineages found: {len(lineages_df)}")
    print(f"By state: {lineages_df['state_info'].value_counts().to_dict()}")

    return lineages_df


def _count_missing(indices):
    return sum(1 for i in indices if i < 0)


def _enumerate_lineages_single(clone, cells_by_day, max_lineages=None):
    """Enumerate lineage combinations for single barcoding (bc0 only).

    No barcode matching needed - all cells with same bc0 are in same lineage.
    Early stops when max_lineages is reached.
    """
    lineages = []

    for p in cells_by_day[6]:
        if max_lineages and len(lineages) >= max_lineages:
            break
        if len(cells_by_day[6]) > 1 and p == -1:
            continue
        state = clone.loc[p, "state_info"] if p != -1 else None

        for q in cells_by_day[9]:
            if max_lineages and len(lineages) >= max_lineages:
                break
            if len(cells_by_day[9]) > 1 and q == -1:
                continue
            if q != -1:
                state = clone.loc[q, "state_info"]

            for r in cells_by_day[12]:
                if max_lineages and len(lineages) >= max_lineages:
                    break
                if len(cells_by_day[12]) > 1 and r == -1:
                    continue
                if r != -1:
                    state = clone.loc[r, "state_info"]

                for s in cells_by_day[15]:
                    if max_lineages and len(lineages) >= max_lineages:
                        break
                    if len(cells_by_day[15]) > 1 and s == -1:
                        continue
                    if s != -1:
                        state = clone.loc[s, "state_info"]

                    for t in cells_by_day[21]:
                        if max_lineages and len(lineages) >= max_lineages:
                            break
                        if len(cells_by_day[21]) > 1 and t == -1:
                            continue
                        if t != -1:
                            state = clone.loc[t, "state_info"]

                        for u in cells_by_day[28]:
                            if max_lineages and len(lineages) >= max_lineages:
                                break
                            if len(cells_by_day[28]) > 1 and u == -1:
                                continue

                            indices = [p, q, r, s, t, u]
                            if sum(indices) == -6 or _count_missing(indices) >= 2:
                                continue

                            if u != -1:
                                state = clone.loc[u, "state_info"]

                            lineages.append([p, q, r, s, t, u, state])

    return lineages


def _enumerate_lineages_cumulative(clone, cells_by_day):
    """Enumerate lineage combinations for cumulative barcoding.

    - Days 6-12: match on (barcode_day0, barcode_day3)
    - Days 15-28: additionally match on barcode_day13
    """
    lineages = []

    for p in cells_by_day[6]:
        if len(cells_by_day[6]) > 1 and p == -1:
            continue
        state = clone.loc[p, "state_info"] if p != -1 else None

        for q in cells_by_day[9]:
            if len(cells_by_day[9]) > 1 and q == -1:
                continue
            if q != -1:
                state = clone.loc[q, "state_info"]

            for r in cells_by_day[12]:
                if len(cells_by_day[12]) > 1 and r == -1:
                    continue
                if r != -1:
                    state = clone.loc[r, "state_info"]

                for s in cells_by_day[15]:
                    if len(cells_by_day[15]) > 1 and s == -1:
                        continue
                    bc13 = clone.loc[s, "barcode_day13"] if s != -1 else None
                    if s != -1:
                        state = clone.loc[s, "state_info"]

                    for t in cells_by_day[21]:
                        if len(cells_by_day[21]) > 1 and t == -1:
                            continue
                        if t != -1 and bc13 and clone.loc[t, "barcode_day13"] != bc13:
                            continue
                        if t != -1:
                            bc13 = clone.loc[t, "barcode_day13"]
                            state = clone.loc[t, "state_info"]

                        for u in cells_by_day[28]:
                            if len(cells_by_day[28]) > 1 and u == -1:
                                continue
                            if (
                                u != -1
                                and bc13
                                and clone.loc[u, "barcode_day13"] != bc13
                            ):
                                continue

                            indices = [p, q, r, s, t, u]
                            if sum(indices) == -6 or _count_missing(indices) >= 2:
                                continue

                            if u != -1:
                                state = clone.loc[u, "state_info"]

                            lineages.append([p, q, r, s, t, u, state])

    return lineages


def create_splits(lineages_df, seed=RANDOM_SEED, include_test=True):
    """Create train/val(/test) splits for two-class classification.

    Args:
        lineages_df: DataFrame with lineage indices and state_info
        seed: Random seed for reproducibility
        include_test: If True, create train/val/test. If False, only train/val.
    """
    split_type = "train/val/test" if include_test else "train/val"
    print(f"Creating {split_type} splits...")

    reprogrammed = lineages_df[lineages_df["state_info"] == "Reprogrammed"]
    failed = lineages_df[lineages_df["state_info"] == "Failed"]

    print(f"Reprogrammed lineages: {len(reprogrammed)}")
    print(f"Failed lineages: {len(failed)}")

    rng = np.random.default_rng(seed=seed)

    if include_test:
        # Full split: 15k/15k/15k per class
        n_per_class = min(SAMPLES_PER_CLASS, len(reprogrammed), len(failed))
        reprog_idx = rng.choice(len(reprogrammed), size=n_per_class, replace=False)
        failed_idx = rng.choice(len(failed), size=n_per_class, replace=False)

        split1, split2 = n_per_class // 3, 2 * n_per_class // 3

        train_df = pd.concat(
            [reprogrammed.iloc[reprog_idx[:split1]], failed.iloc[failed_idx[:split1]]],
            ignore_index=True,
        )
        val_df = pd.concat(
            [
                reprogrammed.iloc[reprog_idx[split1:split2]],
                failed.iloc[failed_idx[split1:split2]],
            ],
            ignore_index=True,
        )
        test_df = pd.concat(
            [reprogrammed.iloc[reprog_idx[split2:]], failed.iloc[failed_idx[split2:]]],
            ignore_index=True,
        )

        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df
    else:
        # Train/val only: 15k/15k per class (same sizes as cumulative)
        n_per_class = min(SAMPLES_PER_CLASS // 3 * 2, len(reprogrammed), len(failed))
        reprog_idx = rng.choice(len(reprogrammed), size=n_per_class, replace=False)
        failed_idx = rng.choice(len(failed), size=n_per_class, replace=False)

        split1 = n_per_class // 2

        train_df = pd.concat(
            [reprogrammed.iloc[reprog_idx[:split1]], failed.iloc[failed_idx[:split1]]],
            ignore_index=True,
        )
        val_df = pd.concat(
            [reprogrammed.iloc[reprog_idx[split1:]], failed.iloc[failed_idx[split1:]]],
            ignore_index=True,
        )

        print(f"Train: {len(train_df)}, Val: {len(val_df)}")
        return train_df, val_df, None


def build_time_series(indices_df, df, n_genes=N_TOP_GENES):
    """Convert lineage indices to gene expression time series arrays."""
    gene_cols = df.columns[:n_genes]  # First n_genes columns are expression

    X = []
    y = []

    for _, row in indices_df.iterrows():
        time_series = []
        for day in ["Day6", "Day9", "Day12", "Day15", "Day21", "Day28"]:
            idx = row[day]
            if idx == -1:
                time_series.append(np.zeros(n_genes, dtype=np.float32))
            else:
                time_series.append(df.iloc[idx][gene_cols].values.astype(np.float32))
        X.append(time_series)
        y.append(row["state_info"])

    X = np.array(X, dtype=np.float32)

    # One-hot encode: [Failed, Reprogrammed] -> [[1,0], [0,1]]
    y_onehot = np.array(
        [[1, 0] if label == "Failed" else [0, 1] for label in y], dtype=np.float32
    )

    return X, y_onehot


def save_dataset(
    path, X_train, y_train, X_val, y_val, X_test, y_test, gene_names, config
):
    """Save dataset to H5 file."""
    print(f"Saving to {path}...")

    with h5py.File(path, "w") as f:
        f.create_dataset("X_train", data=X_train, dtype="f")
        f.create_dataset("X_val", data=X_val, dtype="f")
        f.create_dataset("X_test", data=X_test, dtype="f")
        f.create_dataset("y_train", data=y_train, dtype="f")
        f.create_dataset("y_val", data=y_val, dtype="f")
        f.create_dataset("y_test", data=y_test, dtype="f")

        # Store gene names for interpretability
        f.create_dataset("gene_names", data=np.array(gene_names, dtype="S"))

        # Store config for reproducibility
        f.attrs["config"] = json.dumps(config)
        f.attrs["classes"] = json.dumps(["Failed", "Reprogrammed"])

    print("Done!")


def load_cumulative_test_set(path="reprogramming_dataset_cumulative.h5"):
    """Load test set from cumulative barcoding dataset."""
    print(f"Loading test set from {path}...")
    with h5py.File(path, "r") as f:
        X_test = f["X_test"][:]
        y_test = f["y_test"][:]
    return X_test, y_test


def main():
    parser = argparse.ArgumentParser(description="Generate reprogramming dataset")
    parser.add_argument(
        "--mode",
        choices=["cumulative", "single"],
        default="cumulative",
        help="Barcoding mode: cumulative (bc0+bc3+bc13) or single (bc0 only)",
    )
    args = parser.parse_args()

    output_path = f"reprogramming_dataset_{args.mode}.h5"

    # 1. Load and preprocess
    adata, df = load_and_preprocess()

    # 2. Build lineage indices
    lineages_df = build_lineage_indices(df, mode=args.mode)

    # 3. Create splits
    if args.mode == "single":
        # Single mode: use cumulative test set
        cumulative_path = "reprogramming_dataset_cumulative.h5"
        if not os.path.exists(cumulative_path):
            raise FileNotFoundError(
                f"{cumulative_path} not found. Generate cumulative dataset first."
            )
        X_test, y_test = load_cumulative_test_set(cumulative_path)

        # Generate train/val from single barcoding lineages
        train_df, val_df, _ = create_splits(lineages_df, include_test=False)
        X_train, y_train = build_time_series(train_df, df)
        X_val, y_val = build_time_series(val_df, df)
    else:
        # Cumulative mode: generate all splits
        train_df, val_df, test_df = create_splits(lineages_df, include_test=True)
        X_train, y_train = build_time_series(train_df, df)
        X_val, y_val = build_time_series(val_df, df)
        X_test, y_test = build_time_series(test_df, df)

    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    # 4. Save to H5
    config = {
        "n_top_genes": N_TOP_GENES,
        "random_seed": RANDOM_SEED,
        "log_transform": True,
        "timepoints": TIMEPOINTS,
        "barcoding_mode": args.mode,
    }
    gene_names = df.columns[:N_TOP_GENES].tolist()

    save_dataset(
        output_path,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        gene_names,
        config,
    )


if __name__ == "__main__":
    main()
