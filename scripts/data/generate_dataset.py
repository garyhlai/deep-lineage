"""
Generate synthetic single-cell lineage tracing datasets using LineageSim.

Uses LineageSim for tree + expression generation, then builds trajectories
by sampling ancestor-descendant paths through the lineage tree.
"""

import argparse
import gzip
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from deep_lineage.schema import Cell, Trajectory, TrajectoryList
from lineagesim.simulator import LineageSim
from scripts.utils import (
    generate_run_timestamp,
    get_or_create_run_dir,
    save_run_timestamp,
)

NodeDict = Dict[str, Union[int, float, bool, List[int], None]]
TreeDict = Dict[str, Union[List[NodeDict], int, float]]


class TrajectoryBuilder:
    """Builds trajectories from phylogenetic trees and expression data."""

    def __init__(self, tree: TreeDict, observed_counts: np.ndarray):
        """Initialize with tree structure and expression data.

        Args:
            tree: Phylogenetic tree structure
            observed_counts: Gene expression matrix for ALL nodes
        """
        self.tree = tree
        self.observed_counts = observed_counts
        self.nodes = tree["nodes"]
        assert isinstance(self.nodes, list)
        self.node_id_to_idx = {int(node["id"]): i for i, node in enumerate(self.nodes)}
        self.nodes_by_id = {int(node["id"]): node for node in self.nodes}
        self.leaf_nodes = [node for node in self.nodes if node["is_leaf"]]

    def build_trajectories(
        self,
        max_trajectories: int = 5000,
        fractional_positions: List[float] = [0.25, 0.65, 1.0],
        min_separation: int = 2,
        seed: int = 42,
        balance_t2: bool = False,
        commit_depth: int = 7,
        use_commit_depth_selection: bool = True,
        t0_fateprob_window: Tuple[float, float] | None = None,
        t1_fateprob_window: Tuple[float, float] | None = None,
        prob_deeper_t0: float = 0.0,
    ) -> List[Trajectory]:
        """
        Build true lineage trajectories along root->leaf paths.
        Each trajectory follows a single lineage path with ancestor->descendant->descendant structure.

        Args:
            max_trajectories: Maximum number of trajectories to build
            fractional_positions: Positions along path for t0, t1, t2 (used only if use_commit_depth_selection=False)
            min_separation: Minimum depth separation between consecutive timepoints
            seed: Random seed
            balance_t2: If True, sample accepted trajectories ~1:1 by terminal (t2) fate
            commit_depth: Depth near which cell fate commits (for depth-anchored selection)
            use_commit_depth_selection: If True, choose (t0,t1,t2) by commit depth; else use fractional positions
            t0_fateprob_window: Accept trajectory only if abs(fate_prob(t0)-0.5) in [lo,hi]; None disables gating
            t1_fateprob_window: Accept trajectory only if abs(fate_prob(t1)-0.5) in [lo,hi]; None disables gating
            prob_deeper_t0: Probability of selecting t0 closer to t1 for stronger signal

        Returns:
            List of Trajectory objects with true lineage paths
        """
        np.random.seed(seed)

        print(f"Building path-based trajectories from {len(self.nodes)} tree nodes")
        if use_commit_depth_selection:
            print(
                f"Depth-anchored selection enabled (commit_depth={commit_depth}, "
                f"min_separation={min_separation})"
            )
        else:
            print(
                f"Fractional positions: {fractional_positions} "
                f"(min_separation={min_separation})"
            )

        if balance_t2:
            print("Balancing terminal (t2) fate ~1:1 via trajectory sampling")

        print(f"Found {len(self.leaf_nodes)} leaf nodes")

        trajectories: List[Trajectory] = []
        skipped_short = 0
        skipped_separation = 0

        shuffled_leaves = self.leaf_nodes.copy()
        np.random.shuffle(shuffled_leaves)

        per_class_target = max_trajectories // 2 if balance_t2 else None
        class_counts = {0: 0, 1: 0}

        for leaf in shuffled_leaves:
            if len(trajectories) >= max_trajectories:
                break

            try:
                path = self._get_root_to_leaf_path(leaf)
                if len(path) < 3:
                    skipped_short += 1
                    continue

                if use_commit_depth_selection:
                    selected_nodes = self._select_positions_by_commit_depth(
                        path, commit_depth, min_separation, prob_deeper_t0
                    )
                else:
                    selected_nodes = self._select_path_positions(
                        path, fractional_positions, min_separation
                    )
                if selected_nodes is None:
                    skipped_separation += 1
                    continue

                n0, n1, n2 = selected_nodes

                if t0_fateprob_window is not None:
                    lo, hi = t0_fateprob_window
                    p0 = float(n0.get("fate_prob", 0.5))
                    if not (lo <= abs(p0 - 0.5) <= hi):
                        continue

                if t1_fateprob_window is not None:
                    lo, hi = t1_fateprob_window
                    p1 = float(n1.get("fate_prob", 0.5))
                    if not (lo <= abs(p1 - 0.5) <= hi):
                        continue

                terminal_fate = int(n2["fate"])

                if balance_t2 and per_class_target is not None:
                    if class_counts[terminal_fate] >= per_class_target:
                        continue

                group_id = str(int(n0["id"]))
                trajectory = self._create_trajectory(n0, n1, n2, group_id)

                trajectories.append(trajectory)

                if balance_t2 and per_class_target is not None:
                    class_counts[terminal_fate] += 1
                    if (
                        class_counts[0] >= per_class_target
                        and class_counts[1] >= per_class_target
                    ):
                        break

            except Exception as e:
                print(
                    f"Warning: Failed to build trajectory from leaf {leaf['id']}: {e}"
                )
                continue

        print(f"Built {len(trajectories)} path-based trajectories")
        print(
            f"Skipped: {skipped_short} too short, {skipped_separation} insufficient separation"
        )
        return trajectories

    def _get_root_to_leaf_path(self, leaf: NodeDict) -> List[NodeDict]:
        """Get the complete path from root to leaf node."""
        path = []
        current = leaf

        while current is not None:
            path.append(current)
            parent_id = current["parent"]
            if parent_id is None:
                break
            current = self.nodes_by_id[int(parent_id)]

        path.reverse()
        return path

    def _select_path_positions(
        self,
        path: List[NodeDict],
        fractional_positions: List[float],
        min_separation: int,
    ) -> Union[Tuple[NodeDict, NodeDict, NodeDict], None]:
        """Select three nodes along path based on fractional positions."""
        if len(path) < 3:
            return None

        path_length = len(path)

        idx0 = int(fractional_positions[0] * (path_length - 1))
        idx1 = int(fractional_positions[1] * (path_length - 1))
        idx2 = int(fractional_positions[2] * (path_length - 1))

        if idx1 - idx0 < min_separation or idx2 - idx1 < min_separation:
            return None

        idx0 = max(0, min(idx0, path_length - 3))
        idx1 = max(idx0 + min_separation, min(idx1, path_length - 2))
        idx2 = max(idx1 + min_separation, min(idx2, path_length - 1))

        return path[idx0], path[idx1], path[idx2]

    def _select_positions_by_commit_depth(
        self,
        path: List[NodeDict],
        commit_depth: int,
        min_separation: int,
        prob_deeper_t0: float = 0.0,
    ) -> Union[Tuple[NodeDict, NodeDict, NodeDict], None]:
        """
        Depth-anchored selector:
          - t1: node whose depth is closest to commit_depth (but not too close to leaf)
          - t0: deepest node with depth <= commit_depth-min_separation
          - t2: leaf (>= min_separation after t1)

        Returns None if constraints can't be satisfied for this path.
        """
        if len(path) < 3:
            return None
        depths = [int(n["depth"]) for n in path]
        leaf_idx = len(path) - 1

        idx1 = min(range(len(path)), key=lambda i: abs(depths[i] - commit_depth))
        idx1 = min(idx1, leaf_idx - min_separation)

        if np.random.random() < prob_deeper_t0 and idx1 - min_separation + 1 < len(
            path
        ):
            idx0 = idx1 - min_separation + 1
        else:
            idx0 = max(0, idx1 - min_separation)

        idx2 = leaf_idx
        if idx2 - idx1 < min_separation:
            return None

        return path[idx0], path[idx1], path[idx2]

    def _create_trajectory(
        self,
        n0: NodeDict,
        n1: NodeDict,
        n2: NodeDict,
        group_id: str,
    ) -> Trajectory:
        """Create a Trajectory object from three path nodes."""
        cells = []

        for node, timepoint in [(n0, "t0"), (n1, "t1"), (n2, "t2")]:
            node_idx = self.node_id_to_idx[int(node["id"])]

            cell = Cell(
                idx=node_idx,
                day=timepoint,
                tags=set(),
                expr=self.observed_counts[node_idx],
                state=f"fate_{node['fate']}",
            )
            cells.append(cell)

        return Trajectory(cells=cells, trajectory_group_id=group_id)

    def create_anndata(
        self, trajectories: List[Trajectory], n_genes: int
    ) -> sc.AnnData:
        """Create AnnData object from trajectory data for visualization."""
        cell_data = {}

        for traj in trajectories:
            day0_group_id = traj.trajectory_group_id

            for cell in traj.cells:
                if cell.idx not in cell_data:
                    cell_data[cell.idx] = {
                        "expression": cell.expr,
                        "state": cell.state,
                        "timepoint": cell.day,
                        "day0_group_id": day0_group_id,
                        "cell_id": f"cell_{cell.idx}",
                    }

        cell_indices = sorted(cell_data.keys())
        expressions = np.array([cell_data[idx]["expression"] for idx in cell_indices])

        cell_metadata = []
        for idx in cell_indices:
            data = cell_data[idx]
            cell_metadata.append(
                {
                    "cellID": data["cell_id"],
                    "cell_type": data["state"],
                    "timepoint": data["timepoint"],
                    "day0_group_id": data["day0_group_id"],
                }
            )

        metadata_df = pd.DataFrame(cell_metadata)
        gene_names = [f"gene_{i}" for i in range(n_genes)]
        gene_metadata = pd.DataFrame(index=gene_names)

        adata = sc.AnnData(
            X=sparse.csr_matrix(expressions),
            obs=metadata_df.set_index("cellID"),
            var=gene_metadata,
        )

        print(f"Created AnnData: {adata.shape[0]} cells x {adata.shape[1]} genes")
        print(
            f"Timepoint distribution: {adata.obs['timepoint'].value_counts().to_dict()}"
        )
        print(
            f"Cell type distribution: {adata.obs['cell_type'].value_counts().to_dict()}"
        )
        return adata


def save_intermediate_data(
    tree: TreeDict, observed_counts: np.ndarray, run_dir: Path
) -> None:
    """Save tree and expression data for later trajectory rebuilding."""
    tree_file = run_dir / "tree_data.pkl.gz"
    expression_file = run_dir / "expression_data.pkl.gz"

    print(f"Saving tree data to {tree_file}")
    with gzip.open(tree_file, "wb") as f:
        pickle.dump(tree, f)

    print(f"Saving expression data to {expression_file}")
    with gzip.open(expression_file, "wb") as f:
        pickle.dump(observed_counts, f)


def load_intermediate_data(run_dir: Path) -> Tuple[TreeDict, np.ndarray]:
    """Load tree and expression data for trajectory rebuilding."""
    tree_file = run_dir / "tree_data.pkl.gz"
    expression_file = run_dir / "expression_data.pkl.gz"

    if not tree_file.exists():
        raise FileNotFoundError(
            f"Tree data file not found: {tree_file}. Run without --rebuild_trajectories_only first."
        )

    if not expression_file.exists():
        raise FileNotFoundError(
            f"Expression data file not found: {expression_file}. Run without --rebuild_trajectories_only first."
        )

    print(f"Loading tree data from {tree_file}")
    with gzip.open(tree_file, "rb") as f:
        tree = pickle.load(f)

    print(f"Loading expression data from {expression_file}")
    with gzip.open(expression_file, "rb") as f:
        observed_counts = pickle.load(f)

    print(
        f"Loaded tree with {len(tree['nodes'])} nodes and expression data "
        f"shape {observed_counts.shape}"
    )

    return tree, observed_counts


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic single-cell lineage tracing datasets using LineageSim.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with small dataset
  uv run python scripts/generate_dataset.py --new_run --cells 1000 --genes 500 --max_trajectories 100

  # Production run with balanced terminal classes
  uv run python scripts/generate_dataset.py --new_run --cells 8192 --genes 2000 --beta 0.2 --commit_depth 7 --max_trajectories 8000 --seed 42 --balance_t2

  # Rebuild trajectories with different filtering
  uv run python scripts/generate_dataset.py --rebuild_trajectories_only --balance_t2 --prob_deeper_t0 0.5

  # High-signal dataset for testing (easier classification)
  uv run python scripts/generate_dataset.py --new_run --skip_technical_noise --prob_deeper_t0 1.0
""",
    )

    # Run Management
    run_group = parser.add_argument_group(
        "Run Management", "Control how the script executes"
    )
    run_group.add_argument(
        "--new_run",
        action="store_true",
        help="Start a fresh run with new directory (default: reuse existing)",
    )
    run_group.add_argument(
        "--rebuild_trajectories_only",
        action="store_true",
        help="Skip data generation, only rebuild trajectories from existing tree/expression data",
    )
    run_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # LineageSim Parameters
    sim_group = parser.add_argument_group(
        "LineageSim Parameters", "Parameters for the LineageSim data generator"
    )
    sim_group.add_argument(
        "--cells",
        type=int,
        default=8192,
        help="Total number of cells to simulate (default: 8192)",
    )
    sim_group.add_argument(
        "--genes",
        type=int,
        default=3000,
        help="Number of genes to simulate (default: 3000)",
    )
    sim_group.add_argument(
        "--n_cif",
        type=int,
        default=32,
        help="Number of Cell Identity Factors (default: 32)",
    )
    sim_group.add_argument(
        "--sigma",
        type=float,
        default=0.2,
        help="Brownian motion noise in CIF evolution (default: 0.2)",
    )
    sim_group.add_argument(
        "--beta",
        type=float,
        default=0.2,
        help="Fate signal strength (default: 0.2)",
    )
    sim_group.add_argument(
        "--gene_effect_prob",
        type=float,
        default=0.3,
        help="Probability each gene is affected by CIFs (default: 0.3)",
    )
    sim_group.add_argument(
        "--scale_s",
        type=float,
        default=1.0,
        help="Global scaling for transcription rates (default: 1.0)",
    )
    sim_group.add_argument(
        "--skip_technical_noise",
        action="store_true",
        help="Skip technical noise for maximum signal clarity",
    )
    sim_group.add_argument(
        "--commit_depth",
        type=int,
        default=7,
        help="Tree depth where fate commitment occurs (default: 7)",
    )

    # Trajectory Selection
    traj_group = parser.add_argument_group(
        "Trajectory Selection", "Control sampling and filtering of cell trajectories"
    )
    traj_group.add_argument(
        "--max_trajectories",
        type=int,
        default=5000,
        help="Maximum trajectories to extract from tree (default: 5000)",
    )
    traj_group.add_argument(
        "--balance_t2",
        action="store_true",
        help="Balance terminal cell types to ~50/50 ratio",
    )
    traj_group.add_argument(
        "--prob_deeper_t0",
        type=float,
        default=0.0,
        help="Probability of sampling t0 closer to commitment (default: 0.0)",
    )
    traj_group.add_argument(
        "--min_separation",
        type=int,
        default=2,
        help="Minimum tree depth between consecutive timepoints (default: 2)",
    )
    traj_group.add_argument(
        "--use_fractional_positions",
        action="store_true",
        help="Use fractional position selector instead of depth-anchored",
    )
    traj_group.add_argument(
        "--fractional_positions",
        type=str,
        default="0.25,0.65,1.0",
        help="Fractional positions along paths for t0,t1,t2 (only with --use_fractional_positions)",
    )
    traj_group.add_argument(
        "--t0_fateprob_window",
        type=str,
        default="",
        help='Accept only if |fate_prob(t0)-0.5| in [lo,hi] (e.g., "0.1,0.3")',
    )
    traj_group.add_argument(
        "--t1_fateprob_window",
        type=str,
        default="",
        help='Accept only if |fate_prob(t1)-0.5| in [lo,hi] (e.g., "0.2,0.4")',
    )

    args = parser.parse_args()

    if args.rebuild_trajectories_only and args.new_run:
        raise ValueError(
            "Cannot use --rebuild_trajectories_only with --new_run. "
            "Start a new run first, then rebuild trajectories."
        )

    if args.new_run:
        timestamp = generate_run_timestamp()
        save_run_timestamp(timestamp)
        print(f"Starting new run: {timestamp}")

    run_dir = get_or_create_run_dir()

    if args.rebuild_trajectories_only:
        print(f"Rebuilding trajectories only using data from: {run_dir}")
    else:
        print(f"Using run directory: {run_dir}")

    fractional_positions = [
        float(x.strip()) for x in args.fractional_positions.split(",")
    ]

    t0_window = (
        None
        if args.t0_fateprob_window.strip() == ""
        else tuple(float(x) for x in args.t0_fateprob_window.split(","))
    )

    t1_window = (
        None
        if args.t1_fateprob_window.strip() == ""
        else tuple(float(x) for x in args.t1_fateprob_window.split(","))
    )

    if args.rebuild_trajectories_only:
        tree, observed_counts = load_intermediate_data(run_dir)
    else:
        sim = LineageSim(seed=args.seed)

        tree, observed_counts = sim.generate_dataset(
            n_cells=args.cells,
            n_genes=args.genes,
            n_CIF=args.n_cif,
            sigma=args.sigma,
            beta=args.beta,
            gene_effect_prob=args.gene_effect_prob,
            scale_s=args.scale_s,
            skip_technical_noise=args.skip_technical_noise,
            commit_depth=args.commit_depth,
        )

        save_intermediate_data(tree, observed_counts, run_dir)

    use_commit_depth_selection = not args.use_fractional_positions

    builder = TrajectoryBuilder(tree, observed_counts)
    trajectories = builder.build_trajectories(
        max_trajectories=args.max_trajectories,
        fractional_positions=fractional_positions,
        min_separation=args.min_separation,
        seed=args.seed,
        balance_t2=args.balance_t2,
        commit_depth=args.commit_depth,
        use_commit_depth_selection=use_commit_depth_selection,
        t0_fateprob_window=t0_window,
        t1_fateprob_window=t1_window,
        prob_deeper_t0=args.prob_deeper_t0,
    )

    adata = builder.create_anndata(trajectories, args.genes)

    h5ad_file = run_dir / "dataset.h5ad"
    print(f"Saving dataset to {h5ad_file}")
    adata.write_h5ad(h5ad_file)

    if len(trajectories) == 0:
        raise ValueError("No trajectories were built!")

    traj_file = run_dir / "trajectories.pkl.gz"
    print(f"Saving {len(trajectories)} trajectories to {traj_file}")

    trajectory_list = TrajectoryList(trajectories=trajectories)
    trajectory_list.save(str(traj_file))

    # Summary
    if args.rebuild_trajectories_only:
        print("\n=== Trajectory Rebuild Summary ===")
        print(f"Rebuilt {len(trajectories)} trajectories from existing dataset")
        print(f"Dataset: {adata.shape[0]} cells x {adata.shape[1]} genes (loaded)")
    else:
        print("\n=== Dataset Generation Summary ===")
        print(f"Dataset: {adata.shape[0]} cells x {adata.shape[1]} genes")
        print(f"Trajectories: {len(trajectories)}")

    all_states = []
    for traj in trajectories:
        all_states.extend(traj.to_states())
    state_counts = Counter(all_states)
    print(f"Cell state distribution: {dict(state_counts)}")

    day0_group_ids = [traj.trajectory_group_id for traj in trajectories]
    day0_group_counts = Counter(day0_group_ids)
    print(f"Day0 groups: {len(day0_group_counts)}")
    print(
        f"Trajectories per day0 group (avg): "
        f"{len(trajectories) / len(day0_group_counts):.1f}"
    )

    print(f"""
Files generated in run directory:
  - {h5ad_file}
  - {traj_file}

Next step: uv run python scripts/split_trajectories.py --splits 0.8,0.1,0.1 --seed 42
""")


if __name__ == "__main__":
    main()
