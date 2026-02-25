"""Simulate cell-level barcode dropout and filter affected trajectories."""

import argparse
import json
import numpy as np
from pathlib import Path

from deep_lineage.schema import TrajectoryList
from scripts.utils import get_run_dir


def simulate_cell_dropout(
    trajectory_list: TrajectoryList, dropout_rate: float, seed: int = 42
) -> tuple[TrajectoryList, dict]:
    """
    Mark dropout_rate% of cells as having barcode dropout.
    Remove trajectories containing any dropped-out cells.

    Args:
        trajectory_list: Input trajectories
        dropout_rate: Fraction of unique cells to mark as dropped (e.g., 0.3 = 30%)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (filtered TrajectoryList, statistics dict)
    """
    np.random.seed(seed)

    print(f"\n{'=' * 60}")
    print("Cell-Level Barcode Dropout Simulation")
    print(f"{'=' * 60}")
    print(f"Cell dropout rate: {dropout_rate * 100:.1f}%")
    print(f"Random seed: {seed}")

    # Get all unique cells from trajectories
    unique_cells_by_day = trajectory_list.get_unique_cells_by_day()

    # Flatten to get all unique cells (remove duplicates by idx)
    all_cells = []
    for day, cells in unique_cells_by_day.items():
        all_cells.extend(cells)

    cell_idx_to_cell = {cell.idx: cell for cell in all_cells}
    unique_cells = list(cell_idx_to_cell.values())

    n_unique_cells = len(unique_cells)
    n_dropout = int(n_unique_cells * dropout_rate)

    print("\nCell statistics:")
    print(f"  Total unique cells: {n_unique_cells}")
    for day, cells in sorted(unique_cells_by_day.items()):
        print(f"  {day}: {len(cells)} unique cells")

    # Randomly select cells to mark as dropped out
    if n_dropout > 0:
        dropped_cells_list = np.random.choice(
            unique_cells, size=n_dropout, replace=False
        )
        dropped_cell_ids = set(cell.idx for cell in dropped_cells_list)
    else:
        dropped_cell_ids = set()

    print(f"\nMarking {n_dropout} cells as barcode dropout ({dropout_rate * 100:.1f}%)")

    # Filter trajectories: keep only those where all cells are identifiable
    original_count = len(trajectory_list.trajectories)
    kept_trajectories = []
    dropped_count = 0

    for traj in trajectory_list.trajectories:
        # Check if any cell in this trajectory has barcode dropout
        has_dropout = any(cell.idx in dropped_cell_ids for cell in traj.cells)

        if has_dropout:
            dropped_count += 1
        else:
            kept_trajectories.append(traj)

    # Create filtered trajectory list
    new_trajectory_list = TrajectoryList(trajectories=kept_trajectories)

    # Calculate statistics
    kept_count = len(kept_trajectories)
    trajectory_loss_pct = (dropped_count / original_count) * 100
    kept_pct = (kept_count / original_count) * 100

    print("\nTrajectory filtering results:")
    print(f"  Original trajectories: {original_count}")
    print(f"  Dropped trajectories: {dropped_count} ({trajectory_loss_pct:.1f}%)")
    print(f"  Kept trajectories: {kept_count} ({kept_pct:.1f}%)")

    if kept_trajectories:
        # Count unique cells in kept trajectories
        unique_t0 = len(set(t.cells[0].idx for t in kept_trajectories))
        unique_t1 = len(set(t.cells[1].idx for t in kept_trajectories))
        unique_t2 = len(set(t.cells[2].idx for t in kept_trajectories))

        print("\nUnique cells in kept trajectories:")
        print(f"  t0: {unique_t0} cells")
        print(f"  t1: {unique_t1} cells")
        print(f"  t2: {unique_t2} cells")

    print("\nInterpretation:")
    print(
        f"  {dropout_rate * 100:.0f}% cell barcode dropout → {trajectory_loss_pct:.1f}% trajectory loss"
    )
    print(f"  Model will train on {kept_pct:.1f}% of original data")
    print(f"{'=' * 60}\n")

    # Collect statistics for later aggregation
    stats = {
        "dropout_rate": dropout_rate,
        "n_unique_cells": n_unique_cells,
        "n_dropout_cells": n_dropout,
        "original_trajectories": original_count,
        "dropped_trajectories": dropped_count,
        "kept_trajectories": kept_count,
        "trajectory_retention_pct": kept_pct,
        "trajectory_loss_pct": trajectory_loss_pct,
    }

    return new_trajectory_list, stats


def main():
    parser = argparse.ArgumentParser(
        description="Simulate cell-level barcode dropout and filter trajectories"
    )
    parser.add_argument(
        "--rate",
        type=float,
        required=True,
        help="Cell dropout rate (e.g., 0.3 for 30%% of cells with barcode dropout)",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val"],
        help="Which split to apply dropout to (train or val, never test)",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory (default: use get_run_dir())",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Validate dropout rate
    if not 0 < args.rate < 1:
        raise ValueError(f"Dropout rate must be between 0 and 1, got {args.rate}")

    # Get run directory
    run_dir = Path(args.run_dir) if args.run_dir else get_run_dir()

    # Load input trajectories
    input_file = f"trajectories_{args.split}.pkl.gz"
    input_path = run_dir / input_file

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading {args.split} trajectories from: {input_path}")
    trajectory_list = TrajectoryList.load(str(input_path))

    # Run simulation
    new_trajectory_list, stats = simulate_cell_dropout(
        trajectory_list, dropout_rate=args.rate, seed=args.seed
    )

    # Save output with naming convention: trajectories_{split}_dropout_{rate}pct.pkl.gz
    dropout_pct = int(args.rate * 100)
    output_file = f"trajectories_{args.split}_dropout_{dropout_pct}pct.pkl.gz"
    output_path = run_dir / output_file

    print(f"Saving trajectories to: {output_path}")
    new_trajectory_list.save(str(output_path))

    # Save statistics to JSON for later aggregation
    stats["split"] = args.split
    stats["seed"] = args.seed
    stats_file = f"dropout_stats_{args.split}_dropout_{dropout_pct}pct.json"
    stats_path = run_dir / stats_file

    print(f"Saving statistics to: {stats_path}")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
