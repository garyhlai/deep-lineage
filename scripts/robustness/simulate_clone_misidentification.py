"""Simulate clone misidentification by flipping final cell fate labels."""

import argparse
import json
import numpy as np
from pathlib import Path

from deep_lineage.schema import TrajectoryList
from scripts.utils import get_run_dir


def simulate_clone_misidentification(
    trajectory_list: TrajectoryList, misid_rate: float, seed: int = 42
) -> tuple[TrajectoryList, dict]:
    """
    Flip final cell labels for misid_rate% of trajectories to simulate clone calling errors.

    Args:
        trajectory_list: Input trajectories
        misid_rate: Fraction of trajectories to misidentify (e.g., 0.3 = 30%)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (modified TrajectoryList, statistics dict)
    """
    np.random.seed(seed)

    print(f"\n{'=' * 60}")
    print("Clone Misidentification Simulation")
    print(f"{'=' * 60}")
    print(f"Misidentification rate: {misid_rate * 100:.1f}%")
    print(f"Random seed: {seed}")

    trajectories = trajectory_list.trajectories
    n_trajectories = len(trajectories)
    n_misid = int(n_trajectories * misid_rate)

    print("\nTrajectory statistics:")
    print(f"  Total trajectories: {n_trajectories}")
    print(f"  Trajectories to misidentify: {n_misid} ({misid_rate * 100:.1f}%)")

    # Count original final fate distribution
    original_fate_0 = sum(1 for traj in trajectories if traj.cells[2].state == "fate_0")
    original_fate_1 = sum(1 for traj in trajectories if traj.cells[2].state == "fate_1")

    print("\nOriginal final fate distribution:")
    print(
        f"  fate_0: {original_fate_0} trajectories ({original_fate_0 / n_trajectories * 100:.1f}%)"
    )
    print(
        f"  fate_1: {original_fate_1} trajectories ({original_fate_1 / n_trajectories * 100:.1f}%)"
    )

    # Randomly select trajectories to misidentify
    if n_misid > 0:
        misidentified_trajs = np.random.choice(
            trajectories, size=n_misid, replace=False
        ).tolist()
    else:
        misidentified_trajs = []

    print(f"\nFlipping final cell labels for {n_misid} trajectories")

    # Flip only the final cell label (cells[2]) for selected trajectories
    n_fate_0_to_1 = 0
    n_fate_1_to_0 = 0

    for traj in misidentified_trajs:
        final_cell = traj.cells[2]
        if final_cell.state == "fate_0":
            final_cell.state = "fate_1"
            n_fate_0_to_1 += 1
        elif final_cell.state == "fate_1":
            final_cell.state = "fate_0"
            n_fate_1_to_0 += 1
        else:
            raise ValueError(f"Unknown final cell state: {final_cell.state}")

    print(f"  fate_0 -> fate_1: {n_fate_0_to_1} trajectories")
    print(f"  fate_1 -> fate_0: {n_fate_1_to_0} trajectories")

    # Count new final fate distribution
    new_fate_0 = sum(1 for traj in trajectories if traj.cells[2].state == "fate_0")
    new_fate_1 = sum(1 for traj in trajectories if traj.cells[2].state == "fate_1")

    print("\nNew final fate distribution:")
    print(
        f"  fate_0: {new_fate_0} trajectories ({new_fate_0 / n_trajectories * 100:.1f}%)"
    )
    print(
        f"  fate_1: {new_fate_1} trajectories ({new_fate_1 / n_trajectories * 100:.1f}%)"
    )

    print("\nInterpretation:")
    print(f"  {misid_rate * 100:.0f}% of trajectories have wrong final fate labels")
    print(f"  Classifiers will train on {misid_rate * 100:.0f}% incorrect labels")
    print("  All trajectories kept (only final labels corrupted)")
    print(f"{'=' * 60}\n")

    # Collect statistics
    stats = {
        "misid_rate": misid_rate,
        "n_trajectories": n_trajectories,
        "n_misidentified_trajectories": n_misid,
        "n_fate_0_to_1": n_fate_0_to_1,
        "n_fate_1_to_0": n_fate_1_to_0,
        "original_fate_0": original_fate_0,
        "original_fate_1": original_fate_1,
        "new_fate_0": new_fate_0,
        "new_fate_1": new_fate_1,
    }

    return trajectory_list, stats


def main():
    parser = argparse.ArgumentParser(
        description="Simulate clone misidentification by flipping final cell fate labels at trajectory level"
    )
    parser.add_argument(
        "--rate",
        type=float,
        required=True,
        help="Misidentification rate (e.g., 0.3 for 30%% of trajectories with flipped final fate labels)",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val"],
        help="Which split to apply misidentification to (train or val, never test)",
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

    # Validate misidentification rate
    if not 0 <= args.rate <= 1:
        raise ValueError(
            f"Misidentification rate must be between 0 and 1, got {args.rate}"
        )

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
    modified_trajectory_list, stats = simulate_clone_misidentification(
        trajectory_list, misid_rate=args.rate, seed=args.seed
    )

    # Save output with naming convention: trajectories_{split}_misid_{rate}pct.pkl.gz
    misid_pct = int(args.rate * 100)
    output_file = f"trajectories_{args.split}_misid_{misid_pct}pct.pkl.gz"
    output_path = run_dir / output_file

    print(f"Saving trajectories to: {output_path}")
    modified_trajectory_list.save(str(output_path))

    # Save statistics to JSON for later aggregation
    stats["split"] = args.split
    stats["seed"] = args.seed
    stats_file = f"misid_stats_{args.split}_misid_{misid_pct}pct.json"
    stats_path = run_dir / stats_file

    print(f"Saving statistics to: {stats_path}")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
