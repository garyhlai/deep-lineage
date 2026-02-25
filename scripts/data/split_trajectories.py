"""Split trajectory data into train/validation/test sets preserving group integrity."""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from typing import Tuple, Dict, List
import json

from deep_lineage.schema import TrajectoryList
from scripts.utils import get_run_dir


class TrajectoryDataSplitter:
    """Handles trajectory-aware data splitting to prevent data leakage."""

    def __init__(self, run_dir: Path, seed: int = 42):
        self.run_dir = Path(run_dir)
        self.seed = seed
        np.random.seed(seed)

    def load_trajectories(
        self, filename: str = "trajectories.pkl.gz"
    ) -> TrajectoryList:
        """Load trajectory data from run directory."""
        trajectory_path = self.run_dir / filename

        if not trajectory_path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")

        print(f"Loading trajectories from: {trajectory_path}")
        trajectories = TrajectoryList.load(str(trajectory_path))
        print(f"Loaded {len(trajectories.trajectories)} trajectories")

        return trajectories

    def analyze_trajectory_structure(self, trajectories: TrajectoryList) -> Dict:
        """Analyze the structure of trajectories for splitting insights."""
        print("\nAnalyzing trajectory structure...")

        # Group trajectories by trajectory_group_id (Day0 cell ID)
        groups = defaultdict(list)
        states_by_timepoint = defaultdict(Counter)

        for i, traj in enumerate(trajectories.trajectories):
            group_id = traj.trajectory_group_id
            groups[group_id].append(i)

            for cell in traj.cells:
                states_by_timepoint[cell.day][cell.state] += 1

        group_sizes = [len(group) for group in groups.values()]

        stats = {
            "total_trajectories": len(trajectories.trajectories),
            "total_groups": len(groups),
            "trajectories_per_group": {
                "mean": np.mean(group_sizes),
                "std": np.std(group_sizes),
                "min": np.min(group_sizes),
                "max": np.max(group_sizes),
                "distribution": dict(Counter(group_sizes)),
            },
            "states_by_timepoint": {
                tp: dict(states) for tp, states in states_by_timepoint.items()
            },
        }

        print(
            f"""   Total trajectories: {stats["total_trajectories"]}
   Total Day0 groups: {stats["total_groups"]}
   Trajectories per group: {stats["trajectories_per_group"]["mean"]:.1f} ± {stats["trajectories_per_group"]["std"]:.1f}
   Group size range: {stats["trajectories_per_group"]["min"]}-{stats["trajectories_per_group"]["max"]}"""
        )

        for timepoint in sorted(states_by_timepoint.keys()):
            states = states_by_timepoint[timepoint]
            total = sum(states.values())
            print(f"   {timepoint} states: {dict(states)} (total: {total})")

        return stats, groups

    def split_trajectory_groups(
        self,
        groups: Dict[str, List[int]],
        splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    ) -> Dict[str, List[str]]:
        """
        Split trajectory groups into train/validation/test sets.

        Args:
            groups: Dictionary mapping group_id -> list of trajectory indices
            splits: Tuple of (train_frac, val_frac, test_frac)

        Returns:
            Dictionary with 'train', 'val', 'test' keys containing group IDs
        """
        train_frac, val_frac, test_frac = splits

        if not np.isclose(sum(splits), 1.0):
            raise ValueError(f"Split fractions must sum to 1.0, got {sum(splits)}")

        print(
            f"\nSplitting trajectory groups: {train_frac:.1%}/{val_frac:.1%}/{test_frac:.1%}"
        )

        group_ids = list(groups.keys())
        np.random.shuffle(group_ids)

        n_groups = len(group_ids)
        train_end = int(train_frac * n_groups)
        val_end = train_end + int(val_frac * n_groups)

        split_groups = {
            "train": group_ids[:train_end],
            "val": group_ids[train_end:val_end],
            "test": group_ids[val_end:],
        }

        # Calculate actual splits (may differ slightly due to rounding)
        actual_splits = {
            split_name: len(split_groups[split_name]) / n_groups
            for split_name in ["train", "val", "test"]
        }

        print(
            f"""   Group splits: Train={len(split_groups["train"])}, Val={len(split_groups["val"])}, Test={len(split_groups["test"])}
   Actual fractions: {actual_splits["train"]:.3f}/{actual_splits["val"]:.3f}/{actual_splits["test"]:.3f}"""
        )

        return split_groups, actual_splits

    def create_trajectory_splits(
        self,
        trajectories: TrajectoryList,
        groups: Dict[str, List[int]],
        split_groups: Dict[str, List[str]],
    ) -> Dict[str, TrajectoryList]:
        """Create separate TrajectoryList objects for each split."""

        print("\nCreating trajectory splits...")

        splits = {}

        for split_name, group_ids in split_groups.items():
            trajectory_indices = []
            for group_id in group_ids:
                trajectory_indices.extend(groups[group_id])
            trajectory_indices.sort()

            split_trajectories = [
                trajectories.trajectories[i] for i in trajectory_indices
            ]
            splits[split_name] = TrajectoryList(trajectories=split_trajectories)

            print(
                f"   {split_name.capitalize()}: {len(split_trajectories)} trajectories from {len(group_ids)} groups"
            )

        return splits

    def validate_splits(self, splits: Dict[str, TrajectoryList]) -> Dict[str, Dict]:
        """Validate that splits maintain proper separation and class balance."""

        print("\nValidating splits...")

        validation_results = {}

        # Check for group ID overlap (should be zero)
        all_group_ids = set()
        group_ids_by_split = {}

        for split_name, trajectory_list in splits.items():
            split_group_ids = set()
            split_states = Counter()

            for traj in trajectory_list.trajectories:
                split_group_ids.add(traj.trajectory_group_id)
                # Count final states (Day14 cell types)
                final_cell = traj.cells[-1]  # Last timepoint
                split_states[final_cell.state] += 1

            group_ids_by_split[split_name] = split_group_ids

            overlap = all_group_ids & split_group_ids
            if overlap:
                print(
                    f"   WARNING: {split_name}: Found {len(overlap)} overlapping group IDs: {list(overlap)[:5]}..."
                )
            else:
                print(f"   {split_name}: No group ID overlap")

            all_group_ids.update(split_group_ids)

            total_trajectories = len(trajectory_list.trajectories)
            class_balance = {
                state: count / total_trajectories
                for state, count in split_states.items()
            }

            validation_results[split_name] = {
                "trajectories": total_trajectories,
                "groups": len(split_group_ids),
                "class_distribution": dict(split_states),
                "class_balance": class_balance,
            }

            print(
                f"   {split_name}: {dict(split_states)} (balance: {', '.join(f'{k}={v:.1%}' for k, v in class_balance.items())})"
            )

        return validation_results

    def save_splits(
        self,
        splits: Dict[str, TrajectoryList],
        validation_results: Dict[str, Dict],
        stats: Dict,
        split_info: Dict,
    ):
        """Save trajectory splits and generate report."""

        print("\nSaving trajectory splits...")

        filenames = {}

        for split_name, trajectory_list in splits.items():
            filename = f"trajectories_{split_name}.pkl.gz"
            filepath = self.run_dir / filename
            trajectory_list.save(str(filepath))
            filenames[split_name] = filename
            print(
                f"   {split_name.capitalize()}: {filepath} ({len(trajectory_list.trajectories)} trajectories)"
            )

        report = {
            "split_timestamp": pd.Timestamp.now().isoformat(),
            "random_seed": self.seed,
            "source_file": "trajectories.pkl.gz",
            "split_files": filenames,
            "split_fractions": split_info,
            "dataset_stats": stats,
            "split_validation": validation_results,
            "summary": {
                "total_trajectories": sum(
                    r["trajectories"] for r in validation_results.values()
                ),
                "total_groups": sum(r["groups"] for r in validation_results.values()),
                "splits_verified": all(
                    "overlapping" not in str(r) for r in validation_results.values()
                ),
            },
        }

        report_path = self.run_dir / "trajectory_splits_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"   Split report: {report_path}")

        print(
            f"""\nSplit Summary:
   Total trajectories: {report["summary"]["total_trajectories"]}
   Total groups: {report["summary"]["total_groups"]}
   Splits verified: {report["summary"]["splits_verified"]}"""
        )

        return report


def main():
    parser = argparse.ArgumentParser(
        description="Split trajectory dataset into train/validation/test sets"
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        help="Run directory containing trajectory data (default: use current run)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="0.8,0.1,0.1",
        help="Comma-separated split fractions for train,val,test (default: 0.8,0.1,0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="trajectories.pkl.gz",
        help="Input trajectory file name (default: trajectories.pkl.gz)",
    )

    args = parser.parse_args()

    try:
        split_fractions = [float(x.strip()) for x in args.splits.split(",")]
        if len(split_fractions) != 3:
            raise ValueError("Must provide exactly 3 split fractions")
        splits = tuple(split_fractions)
    except Exception as e:
        raise ValueError(f"Invalid split format '{args.splits}': {e}")

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = get_run_dir()

    print(
        f"""Trajectory Dataset Splitter
{"=" * 50}
Run directory: {run_dir}
Random seed: {args.seed}
Target splits: {splits[0]:.1%}/{splits[1]:.1%}/{splits[2]:.1%}"""
    )

    splitter = TrajectoryDataSplitter(run_dir, seed=args.seed)

    try:
        trajectories = splitter.load_trajectories(args.input_file)
        stats, groups = splitter.analyze_trajectory_structure(trajectories)
        split_groups, actual_splits = splitter.split_trajectory_groups(groups, splits)

        trajectory_splits = splitter.create_trajectory_splits(
            trajectories, groups, split_groups
        )
        validation_results = splitter.validate_splits(trajectory_splits)
        splitter.save_splits(
            trajectory_splits, validation_results, stats, actual_splits
        )

        print(
            """\nTrajectory splitting complete.
   Files created: trajectories_train.pkl.gz, trajectories_val.pkl.gz, trajectories_test.pkl.gz
   Report: trajectory_splits_report.json"""
        )

    except Exception as e:
        print(f"Error during trajectory splitting: {e}")
        raise


if __name__ == "__main__":
    main()
