#!/usr/bin/env python3
"""Evaluate regressors on the test set with metrics and visualizations."""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

from deep_lineage.schema import TrajectoryList
from deep_lineage.layers import SelectKthOutput
from deep_lineage.metrics import PearsonCorrelation
from deep_lineage.utils import (
    make_json_serializable,
    evaluate_gene_space,
    collect_predictions_from_dataset,
    normalize_gene_expression,
)
from scripts.utils import get_run_dir
from scripts.utils import load_trajectory_data


class RegressorEvaluator:
    """
    Evaluates regressors with comprehensive metrics and visualizations.
    """

    def __init__(
        self, run_dir: Path, model_name: str, ae_name: str = None, verbose: bool = True
    ):
        self.run_dir = Path(run_dir)
        self.model_name = model_name
        self.ae_name = ae_name
        self.verbose = verbose

        # Load model metadata from results JSON instead of parsing string
        metadata = self._load_model_metadata()
        self.embedding_type = metadata["embedding_type"]
        self.base_model_name = metadata["name"]
        self.input_timepoints = metadata["input_timepoints"]
        self.target_timepoint = metadata["target_timepoint"]

        # Load models
        self.regressor_model = None
        self.encoder_model = None
        self.decoder_model = None
        self._load_models()

    def _load_model_metadata(self) -> Dict[str, Any]:
        """Load model metadata from the results JSON file."""
        # Try to find the results JSON file
        results_path = self.run_dir / f"regressor_{self.model_name}_results.json"

        if not results_path.exists():
            # Fallback: try to find any regressor results file and match by name
            pattern = f"regressor_*_{self.model_name}_results.json"
            matching_files = list(self.run_dir.glob(pattern))
            if not matching_files:
                # Last resort: try old naming convention
                pattern = f"regressor_{self.model_name}_results.json"
                matching_files = list(self.run_dir.glob(pattern))

            if matching_files:
                results_path = matching_files[0]
            else:
                raise FileNotFoundError(
                    f"Could not find results JSON for model '{self.model_name}' in {self.run_dir}. "
                    f"Expected: regressor_{self.model_name}_results.json or regressor_*_{self.model_name}_results.json"
                )

        with open(results_path, "r") as f:
            metadata = json.load(f)

        # Validate required fields
        required_fields = [
            "embedding_type",
            "name",
            "input_timepoints",
            "target_timepoint",
        ]
        missing_fields = [field for field in required_fields if field not in metadata]

        if missing_fields:
            raise ValueError(
                f"Model metadata missing required fields: {missing_fields}. "
                f"This might be an old model trained before metadata was saved."
            )

        if self.verbose:
            print(f"Loaded model metadata from: {results_path}")
            print(f"   Embedding type: {metadata['embedding_type']}")
            print(f"   Base model name: {metadata['name']}")
            print(f"   Input timepoints: {metadata['input_timepoints']}")
            print(f"   Target timepoint: {metadata['target_timepoint']}")

        return metadata

    def _load_models(self):
        """Load regressor, encoder, and decoder models."""
        try:
            regressor_path = self.run_dir / f"regressor_{self.model_name}_final.keras"

            self.regressor_model = load_model(
                str(regressor_path),
                custom_objects={
                    "SelectKthOutput": SelectKthOutput,
                    "PearsonCorrelation": PearsonCorrelation,
                },
            )

            if self.ae_name:
                encoder_path = self.run_dir / f"encoder_{self.ae_name}_final.keras"
                decoder_path = self.run_dir / f"decoder_{self.ae_name}_final.keras"
            else:
                encoder_path = self.run_dir / "encoder_final.keras"
                decoder_path = self.run_dir / "decoder_final.keras"

            self.encoder_model = load_model(str(encoder_path))
            self.decoder_model = load_model(str(decoder_path))

            if self.verbose:
                print(
                    f"""Loaded regressor from: {regressor_path}
Loaded encoder from: {encoder_path}
Loaded decoder from: {decoder_path}"""
                )

        except Exception as e:
            raise RuntimeError(f"Failed to load models: {e}")

    def prepare_streaming_test_data(
        self,
        trajectories: TrajectoryList,
        input_timepoints: List[str],
        target_timepoint: str,
        batch_size: int = 1024,
    ) -> tf.data.Dataset:
        """
        Prepare streaming test data for regression evaluation (consistent with training).

        Args:
            trajectories: Test trajectories
            input_timepoints: Available timepoints for input
            target_timepoint: Timepoint to predict
            batch_size: Batch size for streaming evaluation

        Returns:
            tf.data.Dataset yielding (masked_input, target_encoded, target_raw) batches
        """
        if self.verbose:
            print("Preparing streaming test data")
            print(f"   Input timepoints: {', '.join(input_timepoints)}")
            print(f"   Target timepoint: {target_timepoint}")

        # Map timepoint names to indices
        timepoint_map = {"t0": 0, "t1": 1, "t2": 2}
        input_indices = [timepoint_map[tp] for tp in input_timepoints]
        target_idx = timepoint_map[target_timepoint]

        # Get dimensions for output signature
        sample_traj = next(
            (t for t in trajectories.trajectories if len(t.cells) == 3), None
        )
        if not sample_traj:
            raise ValueError("No valid trajectories with 3 timepoints found")

        n_genes = sample_traj.cells[0].expr.shape[0]

        def gen():
            """Generator for streaming test data - yields RAW data only."""
            for traj in trajectories.trajectories:
                if len(traj.cells) != 3:
                    continue

                # Use built-in method to get expression matrix [3, n_genes]
                expr_matrix = traj.to_expr().astype(np.float32)
                expr_matrix = normalize_gene_expression(expr_matrix, verbose=False)

                # Create masked input: zeros for all timepoints initially
                masked_input = np.zeros_like(expr_matrix)

                # Fill in ONLY input timepoints (target remains zero/masked)
                for idx in input_indices:
                    masked_input[idx] = expr_matrix[idx]

                target_normalized = expr_matrix[target_idx]  # [n_genes]

                yield (
                    masked_input,
                    target_normalized,
                    target_normalized,
                )  # masked_input, target_for_encoding, target_normalized

        # Create dataset with proper output signature
        output_signature = (
            tf.TensorSpec(
                shape=(3, n_genes), dtype=tf.float32
            ),  # masked_input (normalized)
            tf.TensorSpec(
                shape=(n_genes,), dtype=tf.float32
            ),  # target_for_encoding (normalized)
            tf.TensorSpec(shape=(n_genes,), dtype=tf.float32),  # target_normalized
        )

        dataset = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

        # Define batch encoding function (similar to training)
        @tf.function
        def encode_targets(
            masked_inputs, normalized_targets_for_encoding, normalized_targets
        ):
            """Encode the entire batch of normalized targets at once on GPU."""
            # masked_inputs stay as-is [batch, 3, n_genes] (normalized)
            # Encode normalized targets using the encoder model [batch, n_genes] -> [batch, latent_dim]
            encoded_targets = self.encoder_model(
                normalized_targets_for_encoding, training=False
            )
            return masked_inputs, encoded_targets, normalized_targets

        # Apply batching and encoding
        dataset = (
            dataset.batch(batch_size)
            .map(encode_targets, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )

        # Count samples for reporting
        n_samples = sum(1 for t in trajectories.trajectories if len(t.cells) == 3)
        if self.verbose:
            print(f"   Total test samples: {n_samples}")
            print(f"   Input shape per sample: [3, {n_genes}] (raw genes)")
            print(f"   Batch size: {batch_size}")

        return dataset, n_samples

    def evaluate_regressor_streaming(
        self, test_dataset: tf.data.Dataset, n_test_samples: int
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of regressor performance using streaming data.

        Args:
            test_dataset: Streaming test dataset yielding (masked_input, target_encoded, target_raw) batches
            n_test_samples: Total number of test samples

        Returns:
            Dictionary with evaluation metrics
        """
        if self.verbose:
            print(f"Evaluating regressor: {self.model_name}")
            print(f"   Processing {n_test_samples} test samples in streaming mode")

        # Create simple dataset for collecting predictions (just input and encoded target)
        # Note: test_dataset should already be finite (no .repeat() for evaluation)
        simple_dataset = test_dataset.map(lambda x, y_enc, y_raw: (x, y_enc))

        # Collect predictions using shared function (dataset should be finite)
        y_test_encoded, y_pred_encoded = collect_predictions_from_dataset(
            self.regressor_model, simple_dataset, verbose=self.verbose
        )

        # Use shared comprehensive gene space evaluation
        gene_space_results = evaluate_gene_space(
            y_test_encoded,
            y_pred_encoded,
            self.decoder_model,
            verbose=self.verbose,
            title=f"TEST EVALUATION - {self.model_name}",
        )

        # Add model-specific metadata
        results = {
            "model_name": self.model_name,
            "n_test_samples": n_test_samples,
            **gene_space_results,
        }

        return results

    def create_visualizations(self, results: Dict[str, Any], save_dir: Path):
        """
        Create evaluation visualizations using shared results structure.

        Args:
            results: Evaluation results dictionary from shared evaluation function
            save_dir: Directory to save plots
        """
        if self.verbose:
            print("Creating evaluation visualizations...")

        save_dir.mkdir(parents=True, exist_ok=True)

        # Extract data from shared results structure
        latent_space = results["latent_space"]
        gene_space = results["gene_space"]
        per_gene = results["per_gene"]

        # For visualizations that need raw data, we'll create simple plots based on metrics
        # 1. Metrics Comparison Bar Plot
        plt.figure(figsize=(12, 6))

        metrics = ["R²", "Pearson r", "Cosine Sim"]
        latent_values = [
            latent_space["r2"],
            latent_space["pearson"],
            latent_space["cosine"],
        ]
        gene_values = [
            gene_space["r2"],
            gene_space["pearson"],
            gene_space["cosine"],
        ]

        x = np.arange(len(metrics))
        width = 0.35

        plt.bar(x - width / 2, latent_values, width, label="Latent Space", alpha=0.8)
        plt.bar(x + width / 2, gene_values, width, label="Gene Space", alpha=0.8)

        plt.xlabel("Metrics")
        plt.ylabel("Score")
        plt.title(f"Regression Performance Comparison - {self.model_name}")
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([-1, 1])

        # Add value labels on bars
        for i, (lat_val, gene_val) in enumerate(zip(latent_values, gene_values)):
            plt.text(
                i - width / 2,
                lat_val + 0.02,
                f"{lat_val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
            plt.text(
                i + width / 2,
                gene_val + 0.02,
                f"{gene_val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()

        metrics_path = save_dir / f"{self.model_name}_metrics_comparison.png"
        plt.savefig(metrics_path, dpi=300, bbox_inches="tight")
        plt.close()

        # 2. Per-Gene Correlation Histogram
        plt.figure(figsize=(10, 6))

        per_gene_corrs = per_gene["correlations"]
        mean_gene_corr = per_gene["mean_correlation"]

        plt.hist(per_gene_corrs, bins=50, alpha=0.7, edgecolor="black", color="skyblue")
        plt.axvline(
            mean_gene_corr,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_gene_corr:.3f}",
        )

        # Add thresholds
        plt.axvline(0.5, color="green", linestyle=":", alpha=0.7, label="High (>0.5)")
        plt.axvline(
            0.2, color="orange", linestyle=":", alpha=0.7, label="Medium (>0.2)"
        )

        plt.xlabel("Per-Gene Correlation")
        plt.ylabel("Number of Genes")
        plt.title(f"Distribution of Per-Gene Correlations\n{self.model_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add text box with statistics
        n_high = per_gene["well_predicted"]
        n_total = per_gene["total_genes"]
        n_positive = per_gene["positive_correlation"]

        stats_text = (
            f"Well predicted (>0.5): {n_high} ({n_high / n_total * 100:.1f}%)\n"
        )
        stats_text += (
            f"Positive correlation: {n_positive} ({n_positive / n_total * 100:.1f}%)\n"
        )
        stats_text += f"Total genes: {n_total}"

        plt.text(
            0.02,
            0.98,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()

        gene_hist_path = save_dir / f"{self.model_name}_per_gene_correlations.png"
        plt.savefig(gene_hist_path, dpi=300, bbox_inches="tight")
        plt.close()

        if self.verbose:
            print(f"   Saved metrics comparison: {metrics_path}")
            print(f"   Saved per-gene correlations: {gene_hist_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Deep Lineage-style regressor"
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        help="Run directory containing trained models and data (default: use current run)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to evaluate (e.g., 'ae_future', 'ae_intermediate').",
    )
    parser.add_argument(
        "--input_timepoints",
        type=str,
        help="Comma-separated input timepoints (if not specified, inferred from model name)",
    )
    parser.add_argument(
        "--target_timepoint",
        type=str,
        help="Target timepoint to predict (if not specified, inferred from model name)",
    )
    parser.add_argument(
        "--ae_name",
        type=str,
        default=None,
        help="Optional autoencoder name suffix (e.g., 'dropout_30pct_ae'). If not provided, uses default encoder/decoder files.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Get run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = get_run_dir()

    print(f"Evaluating Regressor: {args.model}")
    print(f"Using run directory: {run_dir}")

    # Initialize evaluator to get metadata
    evaluator = RegressorEvaluator(
        run_dir, args.model, ae_name=args.ae_name, verbose=True
    )

    # Get timepoints from model metadata or command line args
    if args.input_timepoints and args.target_timepoint:
        input_timepoints = [tp.strip() for tp in args.input_timepoints.split(",")]
        target_timepoint = args.target_timepoint.strip()

        # Validate that provided timepoints match metadata
        if (
            input_timepoints != evaluator.input_timepoints
            or target_timepoint != evaluator.target_timepoint
        ):
            print("WARNING: Provided timepoints don't match model metadata.")
            print(f"   Provided: {input_timepoints} → {target_timepoint}")
            print(
                f"   Model trained on: {evaluator.input_timepoints} → {evaluator.target_timepoint}"
            )
            print("   Using model metadata.")

    # Use timepoints from model metadata
    input_timepoints = evaluator.input_timepoints
    target_timepoint = evaluator.target_timepoint

    print(f"Input timepoints: {input_timepoints}")
    print(f"Target timepoint: {target_timepoint}")

    # Load test data
    try:
        test_trajectories = load_trajectory_data(run_dir, split="test")
        print(f"Loaded {len(test_trajectories.trajectories)} test trajectories")

    except FileNotFoundError as e:
        print(f"Error loading test data: {e}")
        return

    # Evaluator already initialized above to get metadata

    # Prepare streaming test data (consistent with training approach)
    test_dataset, n_test_samples = evaluator.prepare_streaming_test_data(
        test_trajectories, input_timepoints, target_timepoint, batch_size=1024
    )

    # Evaluate regressor using streaming approach
    results = evaluator.evaluate_regressor_streaming(test_dataset, n_test_samples)

    # Create visualizations
    viz_dir = run_dir / "evaluation_plots" / "regressors"
    evaluator.create_visualizations(results, viz_dir)

    # Save detailed results
    results_path = run_dir / f"regressor_{args.model}_evaluation.json"
    json_results = make_json_serializable(results)
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"Saved evaluation results to: {results_path}")

    print(
        f"""\nRegressor Evaluation complete.
   Model: {args.model}
   Input → Target: {input_timepoints} → {target_timepoint}
   Latent Space R²: {results["latent_space"]["r2"]:.4f}
   Gene Space R²: {results["gene_space"]["r2"]:.4f}
   Latent Space Pearson: {results["latent_space"]["pearson"]:.4f}
   Gene Space Pearson: {results["gene_space"]["pearson"]:.4f}
   Mean per-gene correlation: {results["per_gene"]["mean_correlation"]:.4f}
   Evaluation plots saved to: {viz_dir}"""
    )

    return results


if __name__ == "__main__":
    main()
