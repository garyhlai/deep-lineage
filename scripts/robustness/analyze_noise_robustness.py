"""Analyze noise robustness evaluation results and determine coverage thresholds."""

import argparse
import json
import re
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.utils import get_run_dir

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300


class NoiseRobustnessAnalyzer:
    """Analyzes noise robustness evaluation results."""

    def __init__(self, run_dir: Path, output_dir: Path = None):
        self.run_dir = Path(run_dir)
        self.output_dir = output_dir or (self.run_dir / "noise_robustness_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            "classifier": {},  # noise_type -> rate -> model_name -> results
            "regressor": {},  # noise_type -> rate -> model_name -> results
        }
        self.dropout_stats = {}  # rate -> split -> stats
        self.misid_stats = {}  # rate -> split -> stats

    def load_results(self):
        """Load all evaluation JSON files from the run directory."""
        print("Loading evaluation results...")

        # Load classifier results
        classifier_files = list(self.run_dir.glob("classifier_*_evaluation.json"))
        for file_path in classifier_files:
            self._load_classifier_result(file_path)

        # Load regressor results
        regressor_files = list(self.run_dir.glob("regressor_*_evaluation.json"))
        for file_path in regressor_files:
            self._load_regressor_result(file_path)

        print(f"   Loaded {len(classifier_files)} classifier results")
        print(f"   Loaded {len(regressor_files)} regressor results")

    def load_dropout_statistics(self):
        """Load dropout statistics JSON files."""
        print("Loading dropout statistics...")

        # Look for dropout_stats files
        stats_files = list(self.run_dir.glob("dropout_stats_*_dropout_*pct.json"))

        if not stats_files:
            print("No dropout statistics files found")
            return

        for file_path in stats_files:
            with open(file_path, "r") as f:
                stats = json.load(f)

            dropout_rate = stats["dropout_rate"]
            split = stats["split"]

            # Convert to percentage for consistency
            rate_pct = int(dropout_rate * 100)

            if rate_pct not in self.dropout_stats:
                self.dropout_stats[rate_pct] = {}

            self.dropout_stats[rate_pct][split] = stats

        print(f"   Loaded {len(stats_files)} dropout statistics files")

    def load_misid_statistics(self):
        """Load misidentification statistics JSON files."""
        print("Loading misidentification statistics...")

        # Look for misid_stats files
        stats_files = list(self.run_dir.glob("misid_stats_*_misid_*pct.json"))

        if not stats_files:
            print("No misidentification statistics files found")
            return

        for file_path in stats_files:
            with open(file_path, "r") as f:
                stats = json.load(f)

            misid_rate = stats["misid_rate"]
            split = stats["split"]

            # Convert to percentage for consistency
            rate_pct = int(misid_rate * 100)

            if rate_pct not in self.misid_stats:
                self.misid_stats[rate_pct] = {}

            self.misid_stats[rate_pct][split] = stats

        print(f"   Loaded {len(stats_files)} misidentification statistics files")

    def _parse_model_name(self, model_name: str) -> Dict[str, str]:
        """
        Parse model name to extract noise type, rate, and base model.

        Expected format: {embedding_type}_{base_model}_{noise_type}_{rate}pct_{embedding_type}
        Examples:
            - ae_t0_only_dropout_30pct_ae
            - ae_future_misid_50pct_ae
        """
        # Pattern: noise_type (dropout or misid) followed by rate
        pattern = r"_(dropout|misid)_(\d+)pct"
        match = re.search(pattern, model_name)

        if match:
            noise_type = match.group(1)
            rate = int(match.group(2))
            # Extract base model name (everything before noise_type)
            base_model = model_name[: match.start()]
            return {
                "noise_type": noise_type,
                "rate": rate,
                "base_model": base_model,
            }
        else:
            # This is a clean baseline model (no noise)
            return {
                "noise_type": "clean",
                "rate": 0,
                "base_model": model_name,
            }

    def _load_classifier_result(self, file_path: Path):
        """Load a single classifier evaluation result."""
        with open(file_path, "r") as f:
            data = json.load(f)

        model_name = data["model_name"]
        parsed = self._parse_model_name(model_name)

        noise_type = parsed["noise_type"]
        rate = parsed["rate"]
        base_model = parsed["base_model"]

        # Initialize nested dicts
        if noise_type not in self.results["classifier"]:
            self.results["classifier"][noise_type] = {}
        if rate not in self.results["classifier"][noise_type]:
            self.results["classifier"][noise_type][rate] = {}

        self.results["classifier"][noise_type][rate][base_model] = data

    def _load_regressor_result(self, file_path: Path):
        """Load a single regressor evaluation result."""
        with open(file_path, "r") as f:
            data = json.load(f)

        model_name = data["model_name"]
        parsed = self._parse_model_name(model_name)

        noise_type = parsed["noise_type"]
        rate = parsed["rate"]
        base_model = parsed["base_model"]

        # Initialize nested dicts
        if noise_type not in self.results["regressor"]:
            self.results["regressor"][noise_type] = {}
        if rate not in self.results["regressor"][noise_type]:
            self.results["regressor"][noise_type][rate] = {}

        self.results["regressor"][noise_type][rate][base_model] = data

    def create_classifier_table(
        self, noise_type: str, embedding_type: str = "ae"
    ) -> pd.DataFrame:
        """
        Create comparison table for classifier results.

        Returns DataFrame with columns: Rate, Model, Test Acc, Bal Acc, AUC
        """
        rows = []

        if noise_type not in self.results["classifier"]:
            print(f"No classifier results found for noise type: {noise_type}")
            return pd.DataFrame()

        # Get all rates and sort
        rates = sorted(self.results["classifier"][noise_type].keys())

        # Expected models
        expected_models = [
            f"{embedding_type}_t0_only",
            f"{embedding_type}_t0_t1",
            f"{embedding_type}_t0_t1_t2",
        ]

        for rate in rates:
            rate_results = self.results["classifier"][noise_type][rate]

            for model_name in expected_models:
                if model_name in rate_results:
                    data = rate_results[model_name]
                    rows.append(
                        {
                            "Noise Rate": f"{rate}%",
                            "Model": model_name,
                            "Test Acc": f"{data['accuracy'] * 100:.2f}%",
                            "Bal Acc": f"{data['balanced_accuracy'] * 100:.2f}%",
                            "AUC": f"{data['auc']:.3f}",
                        }
                    )

        df = pd.DataFrame(rows)
        return df

    def create_regressor_table(
        self, noise_type: str, embedding_type: str = "ae"
    ) -> pd.DataFrame:
        """
        Create comparison table for regressor results.

        Returns DataFrame with columns for latent/gene space metrics and per-gene stats
        """
        rows = []

        if noise_type not in self.results["regressor"]:
            print(f"No regressor results found for noise type: {noise_type}")
            return pd.DataFrame()

        rates = sorted(self.results["regressor"][noise_type].keys())
        expected_models = [f"{embedding_type}_forward", f"{embedding_type}_backward"]

        for rate in rates:
            rate_results = self.results["regressor"][noise_type][rate]

            for model_name in expected_models:
                if model_name in rate_results:
                    data = rate_results[model_name]
                    latent = data["latent_space"]
                    gene = data["gene_space"]
                    per_gene = data["per_gene"]

                    rows.append(
                        {
                            "Noise Rate": f"{rate}%",
                            "Model": model_name.replace(f"{embedding_type}_", ""),
                            "Latent Pearson": f"{latent['pearson']:.4f}",
                            "Latent R²": f"{latent['r2']:.4f}",
                            "Gene Pearson": f"{gene['pearson']:.4f}",
                            "Gene R²": f"{gene['r2']:.4f}",
                            "Mean Per-Gene Corr": f"{per_gene['mean_correlation']:.4f}",
                            "Median Per-Gene Corr": f"{per_gene['median_correlation']:.4f}",
                        }
                    )

        df = pd.DataFrame(rows)
        return df

    def plot_classifier_performance(self, embedding_type: str = "ae"):
        """Plot classifier performance vs. noise rate for both noise types."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Classifier Performance vs. Noise Level", fontsize=16, fontweight="bold"
        )

        noise_types = ["dropout", "misid"]
        metrics = ["accuracy", "auc"]
        metric_names = ["Test Accuracy", "AUC"]
        models = ["t0_only", "t0_t1", "t0_t1_t2"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        for noise_idx, noise_type in enumerate(noise_types):
            if noise_type not in self.results["classifier"]:
                continue

            rates_data = self.results["classifier"][noise_type]
            rates = sorted(
                [r for r in rates_data.keys() if r > 0]
            )  # Exclude clean baseline

            for metric_idx, (metric, metric_name) in enumerate(
                zip(metrics, metric_names)
            ):
                ax = axes[metric_idx, noise_idx]

                for model, color in zip(models, colors):
                    model_name = f"{embedding_type}_{model}"
                    values = []

                    for rate in rates:
                        if model_name in rates_data[rate]:
                            value = rates_data[rate][model_name][metric]
                            if metric == "accuracy":
                                value *= 100  # Convert to percentage
                            values.append(value)
                        else:
                            values.append(np.nan)

                    ax.plot(
                        rates, values, marker="o", linewidth=2, label=model, color=color
                    )

                ax.set_xlabel("Noise Rate (%)", fontsize=11)
                ax.set_ylabel(
                    metric_name + (" (%)" if metric == "accuracy" else ""), fontsize=11
                )
                ax.set_title(f"{metric_name} - {noise_type.capitalize()}", fontsize=12)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / "classifier_performance_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"   Saved: {save_path}")

    def plot_regressor_performance(self, embedding_type: str = "ae"):
        """Plot regressor performance vs. noise rate for both noise types."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Regressor Performance vs. Noise Level", fontsize=16, fontweight="bold"
        )

        noise_types = ["dropout", "misid"]
        metrics = ["gene_space.pearson", "gene_space.r2"]
        metric_names = ["Gene Space Pearson", "Gene Space R²"]
        models = ["future", "intermediate"]
        colors = ["#d62728", "#9467bd"]

        for noise_idx, noise_type in enumerate(noise_types):
            if noise_type not in self.results["regressor"]:
                continue

            rates_data = self.results["regressor"][noise_type]
            rates = sorted([r for r in rates_data.keys() if r > 0])

            for metric_idx, (metric, metric_name) in enumerate(
                zip(metrics, metric_names)
            ):
                ax = axes[metric_idx, noise_idx]
                space, metric_key = metric.split(".")

                for model, color in zip(models, colors):
                    model_name = f"{embedding_type}_{model}"
                    values = []

                    for rate in rates:
                        if model_name in rates_data[rate]:
                            value = rates_data[rate][model_name][space][metric_key]
                            values.append(value)
                        else:
                            values.append(np.nan)

                    ax.plot(
                        rates, values, marker="s", linewidth=2, label=model, color=color
                    )

                ax.set_xlabel("Noise Rate (%)", fontsize=11)
                ax.set_ylabel(metric_name, fontsize=11)
                ax.set_title(f"{metric_name} - {noise_type.capitalize()}", fontsize=12)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / "regressor_performance_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"   Saved: {save_path}")

    def determine_minimum_coverage(
        self, threshold: float = 0.9, embedding_type: str = "ae"
    ):
        """
        Determine minimum barcode coverage needed for reliable predictions.

        Args:
            threshold: Performance threshold (e.g., 0.9 means 90% of clean performance)
            embedding_type: ae or pca

        Returns:
            Dict with minimum coverage requirements
        """
        print(
            f"\nDetermining minimum barcode coverage (threshold={threshold * 100:.0f}% of clean)..."
        )

        results = {}

        # Analyze dropout (direct measure of coverage)
        if (
            "dropout" in self.results["classifier"]
            and "clean" in self.results["classifier"]
        ):
            clean_results = self.results["classifier"]["clean"][0]
            dropout_results = self.results["classifier"]["dropout"]

            # For the best model (t0_t1_t2), find minimum coverage
            best_model = f"{embedding_type}_t0_t1_t2"

            if best_model in clean_results:
                clean_acc = clean_results[best_model]["accuracy"]
                target_acc = clean_acc * threshold

                # Find the maximum dropout rate where performance >= target
                valid_rates = []
                for rate in sorted(dropout_results.keys()):
                    if best_model in dropout_results[rate]:
                        noisy_acc = dropout_results[rate][best_model]["accuracy"]
                        if noisy_acc >= target_acc:
                            valid_rates.append(rate)

                if valid_rates:
                    max_dropout = max(valid_rates)
                    min_coverage = 100 - max_dropout
                    results["classifier_min_coverage"] = min_coverage
                    print(f"   Classifier: Requires ≥{min_coverage}% barcode coverage")
                    print(
                        f"   (Can tolerate up to {max_dropout}% dropout while maintaining {threshold * 100:.0f}% of clean performance)"
                    )

        # Similar analysis for regressor
        if (
            "dropout" in self.results["regressor"]
            and "clean" in self.results["regressor"]
        ):
            clean_results = self.results["regressor"]["clean"][0]
            dropout_results = self.results["regressor"]["dropout"]

            forward_model = f"{embedding_type}_forward"

            if forward_model in clean_results:
                clean_r2 = clean_results[forward_model]["gene_space"]["r2"]
                target_r2 = clean_r2 * threshold

                valid_rates = []
                for rate in sorted(dropout_results.keys()):
                    if forward_model in dropout_results[rate]:
                        noisy_r2 = dropout_results[rate][forward_model]["gene_space"][
                            "r2"
                        ]
                        if noisy_r2 >= target_r2:
                            valid_rates.append(rate)

                if valid_rates:
                    max_dropout = max(valid_rates)
                    min_coverage = 100 - max_dropout
                    results["regressor_min_coverage"] = min_coverage
                    print(f"   Regressor: Requires ≥{min_coverage}% barcode coverage")
                    print(
                        f"   (Can tolerate up to {max_dropout}% dropout while maintaining {threshold * 100:.0f}% of clean performance)"
                    )

        return results

    def create_dropout_summary_table(self):
        """Create summary table of trajectory retention across dropout rates."""
        if not self.dropout_stats:
            print("No dropout statistics available")
            return

        print("\nCreating dropout statistics summary...")

        # Prepare data for table
        rows = []
        for rate in sorted(self.dropout_stats.keys()):
            row = {"dropout_rate": rate}

            for split in ["train", "val"]:
                if split in self.dropout_stats[rate]:
                    stats = self.dropout_stats[rate][split]
                    row[f"{split}_original"] = stats["original_trajectories"]
                    row[f"{split}_kept"] = stats["kept_trajectories"]
                    row[f"{split}_retention_pct"] = stats["trajectory_retention_pct"]
                else:
                    row[f"{split}_original"] = None
                    row[f"{split}_kept"] = None
                    row[f"{split}_retention_pct"] = None

            rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Save to CSV
        csv_path = self.output_dir / "dropout_statistics_summary.csv"
        df.to_csv(csv_path, index=False, float_format="%.1f")
        print(f"   Saved: {csv_path}")

        # Print formatted table
        print("\n## Trajectory Retention Statistics")
        print(
            f"{'Dropout Rate':<15} {'Train Original':<15} {'Train Kept':<12} {'Train %':<10} "
            f"{'Val Original':<15} {'Val Kept':<12} {'Val %':<10}"
        )
        print("-" * 100)

        for _, row in df.iterrows():
            rate = f"{row['dropout_rate']}%"
            train_orig = (
                f"{int(row['train_original'])}"
                if pd.notna(row["train_original"])
                else "N/A"
            )
            train_kept = (
                f"{int(row['train_kept'])}" if pd.notna(row["train_kept"]) else "N/A"
            )
            train_pct = (
                f"{row['train_retention_pct']:.1f}%"
                if pd.notna(row["train_retention_pct"])
                else "N/A"
            )
            val_orig = (
                f"{int(row['val_original'])}"
                if pd.notna(row["val_original"])
                else "N/A"
            )
            val_kept = f"{int(row['val_kept'])}" if pd.notna(row["val_kept"]) else "N/A"
            val_pct = (
                f"{row['val_retention_pct']:.1f}%"
                if pd.notna(row["val_retention_pct"])
                else "N/A"
            )

            print(
                f"{rate:<15} {train_orig:<15} {train_kept:<12} {train_pct:<10} "
                f"{val_orig:<15} {val_kept:<12} {val_pct:<10}"
            )

        return df

    def plot_dropout_statistics(self):
        """Plot trajectory retention vs dropout rate."""
        if not self.dropout_stats:
            print("No dropout statistics available for plotting")
            return

        print("\nPlotting trajectory retention curves...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            "Trajectory Retention Under Cell Dropout", fontsize=14, fontweight="bold"
        )

        # Prepare data
        rates = sorted(self.dropout_stats.keys())
        train_retention = []
        val_retention = []

        for rate in rates:
            train_stats = self.dropout_stats[rate].get("train", {})
            val_stats = self.dropout_stats[rate].get("val", {})

            train_retention.append(train_stats.get("trajectory_retention_pct", None))
            val_retention.append(val_stats.get("trajectory_retention_pct", None))

        # Plot 1: Retention percentage
        ax1.plot(
            rates, train_retention, marker="o", linewidth=2, markersize=8, label="Train"
        )
        ax1.plot(
            rates, val_retention, marker="s", linewidth=2, markersize=8, label="Val"
        )
        ax1.axhline(y=90, color="red", linestyle="--", alpha=0.5, label="90% threshold")
        ax1.set_xlabel("Cell Dropout Rate (%)", fontsize=12)
        ax1.set_ylabel("Trajectory Retention (%)", fontsize=12)
        ax1.set_title("Trajectory Retention vs Dropout Rate")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 105])

        # Plot 2: Absolute counts
        train_counts = [
            self.dropout_stats[rate].get("train", {}).get("kept_trajectories", 0)
            for rate in rates
        ]
        val_counts = [
            self.dropout_stats[rate].get("val", {}).get("kept_trajectories", 0)
            for rate in rates
        ]

        x = np.arange(len(rates))
        width = 0.35

        ax2.bar(x - width / 2, train_counts, width, label="Train", alpha=0.8)
        ax2.bar(x + width / 2, val_counts, width, label="Val", alpha=0.8)
        ax2.set_xlabel("Cell Dropout Rate (%)", fontsize=12)
        ax2.set_ylabel("Number of Kept Trajectories", fontsize=12)
        ax2.set_title("Absolute Trajectory Counts")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{r}%" for r in rates])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / "trajectory_retention_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"   Saved: {plot_path}")

    def generate_summary_report(self, embedding_type: str = "ae"):
        """Generate a comprehensive summary report."""
        print("\n" + "=" * 60)
        print("NOISE ROBUSTNESS ANALYSIS SUMMARY")
        print("=" * 60)

        report_lines = []

        # Classifier tables
        for noise_type in ["dropout", "misid"]:
            if noise_type in self.results["classifier"]:
                print(f"\n## Classifier Results - {noise_type.upper()}")
                df = self.create_classifier_table(noise_type, embedding_type)
                print(df.to_string(index=False))
                report_lines.append(f"\n## Classifier Results - {noise_type.upper()}\n")
                report_lines.append(df.to_markdown(index=False))

                # Save CSV
                csv_path = self.output_dir / f"classifier_{noise_type}_results.csv"
                df.to_csv(csv_path, index=False)
                print(f"   Saved: {csv_path}")

        # Regressor tables
        for noise_type in ["dropout", "misid"]:
            if noise_type in self.results["regressor"]:
                print(f"\n## Regressor Results - {noise_type.upper()}")
                df = self.create_regressor_table(noise_type, embedding_type)
                print(df.to_string(index=False))
                report_lines.append(f"\n## Regressor Results - {noise_type.upper()}\n")
                report_lines.append(df.to_markdown(index=False))

                # Save CSV
                csv_path = self.output_dir / f"regressor_{noise_type}_results.csv"
                df.to_csv(csv_path, index=False)
                print(f"   Saved: {csv_path}")

        # Save markdown report
        report_path = self.output_dir / "noise_robustness_summary.md"
        with open(report_path, "w") as f:
            f.write("# Noise Robustness Analysis Report\n")
            f.write("\n".join(report_lines))
        print(f"\nSaved markdown report: {report_path}")

    def run_analysis(self, embedding_type: str = "ae"):
        """Run complete analysis pipeline."""
        print("\n" + "=" * 60)
        print("NOISE ROBUSTNESS ANALYSIS")
        print("=" * 60)

        # Load results
        self.load_results()
        self.load_dropout_statistics()
        self.load_misid_statistics()

        # Dropout statistics (if available)
        if self.dropout_stats:
            self.create_dropout_summary_table()
            self.plot_dropout_statistics()

        # Generate tables and plots
        print("\nGenerating comparison tables and visualizations...")
        self.generate_summary_report(embedding_type)

        print("\nCreating performance curves...")
        self.plot_classifier_performance(embedding_type)
        self.plot_regressor_performance(embedding_type)

        # Determine minimum coverage
        self.determine_minimum_coverage(threshold=0.9, embedding_type=embedding_type)

        print("\n" + "=" * 60)
        print("Analysis complete.")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze noise robustness evaluation results"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory containing evaluation results (default: use get_run_dir())",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for analysis results (default: run_dir/noise_robustness_analysis)",
    )
    args = parser.parse_args()

    # Get run directory
    run_dir = Path(args.run_dir) if args.run_dir else get_run_dir()
    output_dir = Path(args.output_dir) if args.output_dir else None

    # Run analysis
    analyzer = NoiseRobustnessAnalyzer(run_dir, output_dir)
    analyzer.run_analysis(embedding_type="ae")


if __name__ == "__main__":
    main()
