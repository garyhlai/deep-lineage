#!/usr/bin/env python3
"""Collect and compile results from all model training and evaluation jobs."""

import argparse
import json
import csv
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from scripts.utils import get_run_dir


def load_json_safe(filepath: Path) -> Optional[Dict[str, Any]]:
    """Safely load a JSON file, returning None if it doesn't exist or is invalid."""
    if not filepath.exists():
        return None

    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"WARNING: Could not read {filepath.name}: {e}")
        return None


def collect_classifier_results(run_dir: Path) -> Dict[str, Any]:
    """Collect all classifier training and evaluation results."""
    results = {}

    timepoint_configs = ["t0_only", "t0_t1", "t0_t1_t2"]

    for config in timepoint_configs:
        model_key = f"ae_{config}"

        train_file = run_dir / f"classifier_{model_key}_results.json"
        train_data = load_json_safe(train_file)

        eval_file = run_dir / f"classifier_{model_key}_evaluation.json"
        eval_data = load_json_safe(eval_file)

        if train_data or eval_data:
            results[model_key] = {
                "training": train_data,
                "evaluation": eval_data,
                "status": "complete" if (train_data and eval_data) else "partial",
            }

    return results


def collect_regressor_results(run_dir: Path) -> Dict[str, Any]:
    """Collect all regressor training and evaluation results."""
    results = {}

    directions = ["future", "intermediate"]

    for direction in directions:
        model_key = f"ae_{direction}"

        train_file = run_dir / f"regressor_{model_key}_results.json"
        train_data = load_json_safe(train_file)

        eval_file = run_dir / f"regressor_{model_key}_evaluation.json"
        eval_data = load_json_safe(eval_file)

        if train_data or eval_data:
            results[model_key] = {
                "training": train_data,
                "evaluation": eval_data,
                "status": "complete" if (train_data and eval_data) else "partial",
            }

    # Add linear baseline results
    for direction in directions:
        model_key = f"linear_{direction}"

        results_file = run_dir / f"regressor_{model_key}_results.json"
        linear_data = load_json_safe(results_file)

        if linear_data:
            results[model_key] = {
                "training": linear_data.get("training", {}),
                "evaluation": linear_data,
                "status": "complete" if linear_data else "partial",
            }

    return results


def extract_key_metrics(
    model_results: Dict[str, Any], model_type: str
) -> Dict[str, Any]:
    """Extract key metrics from model results for summary."""
    metrics = {
        "status": model_results.get("status", "missing"),
        "train_accuracy": None,
        "val_accuracy": None,
        "test_accuracy": None,
        "train_loss": None,
        "val_loss": None,
        "test_loss": None,
        "epochs": None,
    }

    if model_results.get("training"):
        train = model_results["training"]
        if model_type == "classifier":
            metrics["train_accuracy"] = train.get("final_train_accuracy")
            metrics["val_accuracy"] = train.get("final_val_accuracy")
        metrics["train_loss"] = train.get("final_train_loss")
        metrics["val_loss"] = train.get("final_val_loss")
        metrics["epochs"] = train.get("total_epochs")

    if model_results.get("evaluation"):
        eval_data = model_results["evaluation"]
        if model_type == "classifier":
            metrics["test_accuracy"] = eval_data.get("accuracy")
            metrics["test_balanced_accuracy"] = eval_data.get("balanced_accuracy")
            metrics["test_auc"] = eval_data.get("auc")
        metrics["test_loss"] = eval_data.get("test_loss")

        if model_type == "regressor":
            gene_space = eval_data.get("gene_space", {})
            latent_space = eval_data.get("latent_space", {})
            per_gene = eval_data.get("per_gene", {})

            metrics["test_gene_r2"] = gene_space.get("r2")
            metrics["test_gene_pearson"] = gene_space.get("pearson")
            metrics["test_latent_r2"] = latent_space.get("r2")
            metrics["test_latent_pearson"] = latent_space.get("pearson")
            metrics["test_gene_rmse"] = gene_space.get("rmse")
            metrics["test_mean_gene_correlation"] = per_gene.get("mean_correlation")

            metrics["test_mse"] = eval_data.get("test_mse")
            metrics["test_mae"] = eval_data.get("test_mae")

    return metrics


def format_metric(value: Optional[float], metric_type: str = "accuracy") -> str:
    """Format a metric value for display."""
    if value is None:
        return "N/A"

    if metric_type == "accuracy":
        return f"{value * 100:.2f}%"
    elif metric_type == "auc":
        return f"{value:.3f}"
    elif metric_type == "r2":
        return f"{value:.3f}"
    elif metric_type == "correlation":
        return f"{value:.3f}"
    elif metric_type == "loss":
        return f"{value:.4f}"
    elif metric_type == "epochs":
        return str(int(value))
    else:
        return f"{value:.4f}"


def print_summary_table(classifiers: Dict, regressors: Dict):
    """Print a formatted summary table to console."""
    print("\n" + "=" * 80)
    print("MODEL RESULTS SUMMARY")
    print("=" * 80)

    if classifiers:
        print("\nCLASSIFIERS (Cell Fate Prediction)")
        print("-" * 100)
        print(
            f"{'Model':<20} {'Status':<10} {'Test Acc':<12} {'Bal Acc':<12} {'AUC':<12} {'Epochs':<8}"
        )
        print("-" * 100)

        for model_name in sorted(classifiers.keys()):
            metrics = extract_key_metrics(classifiers[model_name], "classifier")
            status_icon = (
                "PASS"
                if metrics["status"] == "complete"
                else "PARTIAL"
                if metrics["status"] == "partial"
                else "FAIL"
            )

            print(
                f"{model_name:<20} {status_icon:<10} "
                f"{format_metric(metrics['test_accuracy']):<12} "
                f"{format_metric(metrics['test_balanced_accuracy']):<12} "
                f"{format_metric(metrics['test_auc'], 'auc'):<12} "
                f"{format_metric(metrics['epochs'], 'epochs'):<8}"
            )

    if regressors:
        print("\nREGRESSORS (Expression Prediction)")
        print("-" * 100)
        print(
            f"{'Model':<20} {'Status':<10} {'Gene R²':<12} {'Gene Pearson':<12} {'Latent R²':<12} {'Epochs':<8}"
        )
        print("-" * 100)

        for model_name in sorted(regressors.keys()):
            metrics = extract_key_metrics(regressors[model_name], "regressor")
            status_icon = (
                "PASS"
                if metrics["status"] == "complete"
                else "PARTIAL"
                if metrics["status"] == "partial"
                else "FAIL"
            )

            print(
                f"{model_name:<20} {status_icon:<10} "
                f"{format_metric(metrics.get('test_gene_r2'), 'r2'):<12} "
                f"{format_metric(metrics.get('test_gene_pearson'), 'correlation'):<12} "
                f"{format_metric(metrics.get('test_latent_r2'), 'r2'):<12} "
                f"{format_metric(metrics['epochs'], 'epochs'):<8}"
            )

    print("\n" + "=" * 80)


def generate_markdown_report(classifiers: Dict, regressors: Dict, run_dir: Path) -> str:
    """Generate a markdown report of all results."""
    report = []
    report.append("# Model Results Report")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Run Directory:** `{run_dir}`\n")

    total_models = len(classifiers) + len(regressors)
    complete_models = sum(
        1
        for m in {**classifiers, **regressors}.values()
        if m.get("status") == "complete"
    )
    partial_models = sum(
        1
        for m in {**classifiers, **regressors}.values()
        if m.get("status") == "partial"
    )

    report.append("## Summary Statistics")
    report.append(f"- **Total Models:** {total_models}")
    report.append(f"- **Complete (trained & evaluated):** {complete_models}")
    report.append(f"- **Partial (training or evaluation only):** {partial_models}")
    report.append(f"- **Missing:** {total_models - complete_models - partial_models}\n")

    if classifiers:
        report.append("## Classifier Results (Cell Fate Prediction)\n")

        best_val = max(
            (m for m in classifiers.values() if m.get("training")),
            key=lambda x: (
                x["training"].get("final_val_accuracy", 0) if x.get("training") else 0
            ),
            default=None,
        )
        if best_val and best_val.get("training"):
            best_val_acc = best_val["training"].get("final_val_accuracy", 0)
            report.append(f"**Best Validation Accuracy:** {best_val_acc * 100:.2f}%\n")

        report.append("### Detailed Results\n")
        report.append("| Model | Status | Test Acc | Bal Acc | AUC | Epochs |")
        report.append("|-------|--------|----------|---------|-----|--------|")

        for model_name in sorted(classifiers.keys()):
            metrics = extract_key_metrics(classifiers[model_name], "classifier")
            status = (
                "Complete"
                if metrics["status"] == "complete"
                else "Partial"
                if metrics["status"] == "partial"
                else "Missing"
            )

            report.append(
                f"| {model_name} | {status} | "
                f"{format_metric(metrics['test_accuracy'])} | "
                f"{format_metric(metrics['test_balanced_accuracy'])} | "
                f"{format_metric(metrics['test_auc'], 'auc')} | "
                f"{format_metric(metrics['epochs'], 'epochs')} |"
            )

        report.append("")

    if regressors:
        report.append("## Regressor Results (Expression Prediction)\n")

        best_val = min(
            (m for m in regressors.values() if m.get("training")),
            key=lambda x: (
                x["training"].get("final_val_loss", float("inf"))
                if x.get("training")
                else float("inf")
            ),
            default=None,
        )
        if best_val and best_val.get("training"):
            best_val_loss = best_val["training"].get("final_val_loss", 0)
            report.append(f"**Best Validation Loss:** {best_val_loss:.4f}\n")

        report.append("### Detailed Results\n")
        report.append(
            "| Model | Status | Gene R² | Gene Pearson | Latent R² | Latent Pearson | Epochs |"
        )
        report.append(
            "|-------|--------|---------|--------------|-----------|----------------|--------|"
        )

        for model_name in sorted(regressors.keys()):
            metrics = extract_key_metrics(regressors[model_name], "regressor")
            status = (
                "Complete"
                if metrics["status"] == "complete"
                else "Partial"
                if metrics["status"] == "partial"
                else "Missing"
            )

            report.append(
                f"| {model_name} | {status} | "
                f"{format_metric(metrics.get('test_gene_r2'), 'r2')} | "
                f"{format_metric(metrics.get('test_gene_pearson'), 'correlation')} | "
                f"{format_metric(metrics.get('test_latent_r2'), 'r2')} | "
                f"{format_metric(metrics.get('test_latent_pearson'), 'correlation')} | "
                f"{format_metric(metrics['epochs'], 'epochs')} |"
            )

        report.append("")

    return "\n".join(report)


def export_to_csv(classifiers: Dict, regressors: Dict, output_path: Path):
    """Export results to CSV format for further analysis."""
    rows = []

    for model_name, results in classifiers.items():
        metrics = extract_key_metrics(results, "classifier")
        row = {
            "model_type": "classifier",
            "model_name": model_name,
            "embedding": model_name.split("_")[0],
            "config": "_".join(model_name.split("_")[1:]),
            "status": metrics["status"],
            "train_accuracy": metrics["train_accuracy"],
            "val_accuracy": metrics["val_accuracy"],
            "test_accuracy": metrics["test_accuracy"],
            "test_balanced_accuracy": metrics["test_balanced_accuracy"],
            "test_auc": metrics["test_auc"],
            "train_loss": metrics["train_loss"],
            "val_loss": metrics["val_loss"],
            "test_loss": metrics["test_loss"],
            "epochs": metrics["epochs"],
        }
        rows.append(row)

    for model_name, results in regressors.items():
        metrics = extract_key_metrics(results, "regressor")
        row = {
            "model_type": "regressor",
            "model_name": model_name,
            "embedding": model_name.split("_")[0],
            "config": "_".join(model_name.split("_")[1:]),
            "status": metrics["status"],
            "train_accuracy": None,
            "val_accuracy": None,
            "test_accuracy": None,
            "test_balanced_accuracy": None,
            "test_auc": None,
            "train_loss": metrics["train_loss"],
            "val_loss": metrics["val_loss"],
            "test_loss": metrics["test_loss"],
            "test_gene_r2": metrics.get("test_gene_r2"),
            "test_gene_pearson": metrics.get("test_gene_pearson"),
            "test_latent_r2": metrics.get("test_latent_r2"),
            "test_latent_pearson": metrics.get("test_latent_pearson"),
            "test_gene_rmse": metrics.get("test_gene_rmse"),
            "test_mean_gene_correlation": metrics.get("test_mean_gene_correlation"),
            "test_mse": metrics.get("test_mse"),
            "test_mae": metrics.get("test_mae"),
            "epochs": metrics["epochs"],
        }
        rows.append(row)

    if rows:
        fieldnames = [
            "model_type",
            "model_name",
            "embedding",
            "config",
            "status",
            "train_accuracy",
            "val_accuracy",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_auc",
            "train_loss",
            "val_loss",
            "test_loss",
            "test_gene_r2",
            "test_gene_pearson",
            "test_latent_r2",
            "test_latent_pearson",
            "test_gene_rmse",
            "test_mean_gene_correlation",
            "test_mse",
            "test_mae",
            "epochs",
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Exported metrics to: {output_path}")


def main():
    """Main entry point for the results collection script."""
    parser = argparse.ArgumentParser(description="Collect and compile model results")
    parser.add_argument(
        "--run_dir",
        type=str,
        help="Run directory containing result files (default: use current run)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for reports (default: same as run_dir)",
    )
    parser.add_argument(
        "--format",
        nargs="+",
        choices=["console", "json", "markdown", "csv", "all"],
        default=["all"],
        help="Output formats (default: all)",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else get_run_dir()
    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting results from: {run_dir}")

    classifier_results = collect_classifier_results(run_dir)
    regressor_results = collect_regressor_results(run_dir)

    total_found = len(classifier_results) + len(regressor_results)

    if total_found == 0:
        print("No model results found in the run directory.")
        print("   Make sure you've run the training and evaluation scripts first.")
        return

    print(f"Found {total_found} model results")

    formats = set(args.format)
    if "all" in formats:
        formats = {"console", "json", "markdown", "csv"}

    if "console" in formats:
        print_summary_table(classifier_results, regressor_results)

    if "json" in formats:
        all_results = {
            "metadata": {
                "run_directory": str(run_dir),
                "collection_time": datetime.now().isoformat(),
                "total_models": total_found,
            },
            "classifiers": classifier_results,
            "regressors": regressor_results,
        }

        json_path = output_dir / "results_summary.json"
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved JSON summary to: {json_path}")

    if "markdown" in formats:
        report = generate_markdown_report(
            classifier_results, regressor_results, run_dir
        )
        md_path = output_dir / "results_report.md"
        with open(md_path, "w") as f:
            f.write(report)
        print(f"Saved markdown report to: {md_path}")

    if "csv" in formats:
        csv_path = output_dir / "results_metrics.csv"
        export_to_csv(classifier_results, regressor_results, csv_path)

    print("\nResults collection complete.")


if __name__ == "__main__":
    main()
