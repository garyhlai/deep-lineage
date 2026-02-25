"""Verify trajectory split integrity and detect data leakage."""

import argparse
import json
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import Dict, Set

from deep_lineage.schema import TrajectoryList
from scripts.utils import get_run_dir


class SplitIntegrityVerifier:
    """Verifies integrity of trajectory data splits."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.splits = {}
        self.group_ids_by_split = {}

    def load_all_splits(self) -> Dict[str, TrajectoryList]:
        """Load all available trajectory splits."""
        print("Loading trajectory splits for integrity verification...")

        split_names = ["train", "val", "test"]
        loaded_splits = {}

        for split_name in split_names:
            split_file = self.run_dir / f"trajectories_{split_name}.pkl.gz"

            if split_file.exists():
                try:
                    trajectories = TrajectoryList.load(str(split_file))
                    loaded_splits[split_name] = trajectories
                    print(
                        f"   {split_name}: {len(trajectories.trajectories)} trajectories"
                    )
                except Exception as e:
                    print(f"   {split_name}: Failed to load - {e}")
            else:
                print(f"   {split_name}: File not found")

        if not loaded_splits:
            raise FileNotFoundError("No trajectory split files found in run directory")

        self.splits = loaded_splits
        return loaded_splits

    def extract_group_information(self) -> Dict[str, Dict]:
        """Extract trajectory group information from all splits."""
        print("\nExtracting trajectory group information...")

        split_info = {}

        for split_name, trajectories in self.splits.items():
            group_ids = set()
            states_count = Counter()
            trajectory_groups = {}  # group_id -> list of trajectory indices

            for i, traj in enumerate(trajectories.trajectories):
                group_id = traj.trajectory_group_id
                group_ids.add(group_id)

                if group_id not in trajectory_groups:
                    trajectory_groups[group_id] = []
                trajectory_groups[group_id].append(i)

                # Count final states (Day14 cell types)
                final_cell = traj.cells[-1]
                states_count[final_cell.state] += 1

            group_sizes = [len(trajs) for trajs in trajectory_groups.values()]

            split_info[split_name] = {
                "trajectories": len(trajectories.trajectories),
                "unique_groups": len(group_ids),
                "group_ids": group_ids,
                "group_sizes": {
                    "mean": sum(group_sizes) / len(group_sizes) if group_sizes else 0,
                    "min": min(group_sizes) if group_sizes else 0,
                    "max": max(group_sizes) if group_sizes else 0,
                    "distribution": dict(Counter(group_sizes)),
                },
                "class_distribution": dict(states_count),
                "class_balance": (
                    {
                        state: count / len(trajectories.trajectories)
                        for state, count in states_count.items()
                    }
                    if trajectories.trajectories
                    else {}
                ),
            }

            self.group_ids_by_split[split_name] = group_ids

            print(
                f"""   {split_name.capitalize()}:
      Trajectories: {split_info[split_name]["trajectories"]}
      Unique groups: {split_info[split_name]["unique_groups"]}
      Class balance: {", ".join(f"{k}={v:.1%}" for k, v in split_info[split_name]["class_balance"].items())}"""
            )

        return split_info

    def check_group_overlaps(self) -> Dict[str, Set]:
        """Check for overlapping trajectory groups between splits."""
        print("\nChecking for trajectory group overlaps...")

        overlaps = {}
        split_names = list(self.group_ids_by_split.keys())

        for i, split1 in enumerate(split_names):
            for split2 in split_names[i + 1 :]:
                pair_key = f"{split1}_{split2}"
                overlap = (
                    self.group_ids_by_split[split1] & self.group_ids_by_split[split2]
                )
                overlaps[pair_key] = overlap

                if overlap:
                    print(
                        f"   FAIL {split1.capitalize()} <-> {split2.capitalize()}: {len(overlap)} overlapping groups"
                    )
                    if len(overlap) <= 10:
                        print(f"      Overlapping group IDs: {sorted(list(overlap))}")
                    else:
                        sample_overlap = sorted(list(overlap))[:10]
                        print(
                            f"      Sample overlapping group IDs: {sample_overlap} (+{len(overlap) - 10} more)"
                        )
                else:
                    print(
                        f"   PASS {split1.capitalize()} <-> {split2.capitalize()}: No overlap"
                    )

        return overlaps

    def verify_split_coverage(self, original_file: str = "trajectories.pkl.gz") -> Dict:
        """Verify that splits cover all trajectories from original dataset."""
        print("\nVerifying split coverage against original dataset...")

        original_path = self.run_dir / original_file
        if not original_path.exists():
            print(f"   Original file not found: {original_file}")
            return {"status": "skipped", "reason": "Original file not found"}

        try:
            original_trajectories = TrajectoryList.load(str(original_path))
            original_group_ids = set()

            for traj in original_trajectories.trajectories:
                original_group_ids.add(traj.trajectory_group_id)

            all_split_group_ids = set()
            for group_ids in self.group_ids_by_split.values():
                all_split_group_ids.update(group_ids)

            missing_in_splits = original_group_ids - all_split_group_ids
            extra_in_splits = all_split_group_ids - original_group_ids

            coverage_info = {
                "original_groups": len(original_group_ids),
                "split_groups": len(all_split_group_ids),
                "missing_in_splits": len(missing_in_splits),
                "extra_in_splits": len(extra_in_splits),
                "coverage_ratio": (
                    len(all_split_group_ids & original_group_ids)
                    / len(original_group_ids)
                    if original_group_ids
                    else 0
                ),
            }

            print(
                f"""   Coverage analysis:
      Original groups: {coverage_info["original_groups"]}
      Split groups: {coverage_info["split_groups"]}
      Coverage ratio: {coverage_info["coverage_ratio"]:.1%}"""
            )

            if missing_in_splits:
                print(f"      Missing in splits: {len(missing_in_splits)} groups")
            if extra_in_splits:
                print(f"      Extra in splits: {len(extra_in_splits)} groups")

            if missing_in_splits == 0 and extra_in_splits == 0:
                print("   Perfect coverage: All original groups accounted for.")

            return coverage_info

        except Exception as e:
            print(f"   Failed to load original dataset: {e}")
            return {"status": "failed", "reason": str(e)}

    def generate_integrity_report(
        self, split_info: Dict, overlaps: Dict, coverage_info: Dict
    ) -> Dict:
        """Generate comprehensive integrity report."""
        print("\nGenerating integrity report...")

        total_trajectories = sum(info["trajectories"] for info in split_info.values())
        has_overlaps = any(len(overlap) > 0 for overlap in overlaps.values())
        max_overlap = max((len(overlap) for overlap in overlaps.values()), default=0)

        integrity_issues = []
        if has_overlaps:
            integrity_issues.append(
                f"Data leakage detected: {max_overlap} max overlapping groups"
            )
        if coverage_info.get("missing_in_splits", 0) > 0:
            integrity_issues.append(
                f"Missing data: {coverage_info['missing_in_splits']} groups not in splits"
            )
        if coverage_info.get("extra_in_splits", 0) > 0:
            integrity_issues.append(
                f"Extra data: {coverage_info['extra_in_splits']} groups not in original"
            )

        all_states = set()
        for info in split_info.values():
            all_states.update(info["class_balance"].keys())

        balance_consistency = {}
        for state in all_states:
            balances = [
                info["class_balance"].get(state, 0) for info in split_info.values()
            ]
            balance_consistency[state] = {
                "min": min(balances),
                "max": max(balances),
                "range": max(balances) - min(balances),
            }

        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "run_directory": str(self.run_dir),
            "summary": {
                "total_splits": len(self.splits),
                "total_trajectories": total_trajectories,
                "total_unique_groups": len(
                    set().union(*self.group_ids_by_split.values())
                ),
                "has_data_leakage": has_overlaps,
                "max_overlap_groups": max_overlap,
                "integrity_status": "FAILED" if integrity_issues else "PASSED",
                "issues": integrity_issues,
            },
            "split_details": split_info,
            "overlap_analysis": {
                pair: len(overlap_set) for pair, overlap_set in overlaps.items()
            },
            "coverage_analysis": coverage_info,
            "class_balance_analysis": balance_consistency,
        }

        return report

    def save_report(self, report: Dict):
        """Save integrity report to file."""
        report_path = self.run_dir / "split_integrity_report.json"

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"Saved integrity report to: {report_path}")
        return report_path

    def print_summary(self, report: Dict):
        """Print a summary of the integrity check results."""
        print(f"""
{"=" * 80}
SPLIT INTEGRITY VERIFICATION SUMMARY
{"=" * 80}""")

        summary = report["summary"]

        print(
            f"""Dataset Overview:
   Splits analyzed: {summary["total_splits"]}
   Total trajectories: {summary["total_trajectories"]}
   Total unique groups: {summary["total_unique_groups"]}"""
        )

        print("\nData Leakage Check:")
        if summary["has_data_leakage"]:
            print(f"""   FAILED: Data leakage detected
   Max overlapping groups: {summary["max_overlap_groups"]}""")
        else:
            print("   PASSED: No data leakage detected.")

        print(f"\nOverall Status: {summary['integrity_status']}")

        if summary["issues"]:
            print("\nIssues found:")
            for issue in summary["issues"]:
                print(f"   - {issue}")
        else:
            print("\nAll integrity checks passed.")

        print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify trajectory split integrity and detect data leakage"
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        help="Run directory containing trajectory splits (default: use current run)",
    )

    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = get_run_dir()

    print(
        f"""Trajectory Split Integrity Verification
{"=" * 50}
Run directory: {run_dir}"""
    )

    verifier = SplitIntegrityVerifier(run_dir)

    try:
        verifier.load_all_splits()
        split_info = verifier.extract_group_information()
        overlaps = verifier.check_group_overlaps()
        coverage_info = verifier.verify_split_coverage()
        report = verifier.generate_integrity_report(split_info, overlaps, coverage_info)

        verifier.save_report(report)
        verifier.print_summary(report)

        if report["summary"]["integrity_status"] == "FAILED":
            print("\nIntegrity verification failed.")
            return 1
        else:
            print("\nIntegrity verification passed.")
            return 0

    except Exception as e:
        print(f"Integrity verification failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
