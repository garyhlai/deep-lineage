"""
Evaluation Script for Reprogramming Dataset (Original Deep Lineage Reproduction)

This script evaluates trained models and produces a summary report comparing results
to the original Deep Lineage paper metrics.

Usage:
    uv run python scripts/evaluate_reprogramming.py \
        --ae_dir runs/reprog_ae \
        --cls_dir runs/reprog_cls \
        --reg_dir runs/reprog_reg
"""

import argparse
import json
from pathlib import Path

from scripts.utils import load_results_json


def print_section(title: str, width: int = 60):
    """Print a section header."""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def evaluate_autoencoder(ae_dir: Path):
    """Evaluate autoencoder results."""
    print_section("AUTOENCODER EVALUATION")

    results = load_results_json(ae_dir / "evaluation_results.json")
    config = load_results_json(ae_dir / "config.json")

    if not results:
        print("  No evaluation results found.")
        return None

    print("\n  Configuration:")
    print(f"    Hidden sizes: {config.get('hidden_sizes', 'N/A')}")
    print(f"    Latent dim: {config.get('latent_dim', 'N/A')}")
    print(f"    Input dropout: {config.get('input_dropout', 'N/A')}")

    print("\n  Test Results:")
    print(f"    MSE: {results.get('mse', 'N/A'):.6f}")
    print(
        f"    Mean Pearson correlation: {results.get('mean_pearson_correlation', 'N/A'):.4f}"
    )
    print(
        f"    Mean Spearman correlation: {results.get('mean_spearman_correlation', 'N/A'):.4f}"
    )

    return results


def evaluate_classifier(cls_dir: Path):
    """Evaluate classifier results."""
    print_section("CLASSIFIER EVALUATION")

    results = load_results_json(cls_dir / "classification_results.json")
    config = load_results_json(cls_dir / "config.json")

    if not results:
        print("  No classification results found.")
        return None

    print("\n  Configuration:")
    print(f"    Model type: {config.get('model_type', 'N/A')}")
    print(f"    Num layers: {config.get('num_layers', 'N/A')}")
    print(f"    Dropout: {config.get('dropout', 'N/A')}")

    print("\n  Test Results:")
    print(
        f"    Accuracy: {results.get('accuracy', 'N/A'):.4f} ({results.get('accuracy', 0) * 100:.2f}%)"
    )
    print(f"    Loss: {results.get('loss', 'N/A'):.6f}")

    print("\n  Confusion Matrix:")
    classes = results.get("classes", ["Class 0", "Class 1"])
    conf_mat = results.get("confusion_matrix", [[0, 0], [0, 0]])
    print(f"    {classes}")
    for i, row in enumerate(conf_mat):
        print(f"    {classes[i]}: {row}")

    return results


def evaluate_regressor(reg_dir: Path):
    """Evaluate regressor results."""
    print_section("REGRESSOR EVALUATION")

    results = load_results_json(reg_dir / "regression_results.json")
    config = load_results_json(reg_dir / "config.json")

    if not results:
        print("  No regression results found.")
        return None

    print("\n  Configuration:")
    print(f"    Model type: {config.get('model_type', 'N/A')}")
    print(f"    Num layers: {config.get('num_layers', 'N/A')}")
    print(f"    Dropout: {config.get('dropout', 'N/A')}")
    print(f"    Target day: Day{config.get('target_day', 'N/A')}")

    print("\n  Test Results:")
    print(f"    MSE: {results.get('mse', 'N/A'):.6f}")
    print(
        f"    Mean Pearson correlation: {results.get('mean_pearson_correlation', 'N/A'):.4f}"
    )
    print(
        f"    Mean Spearman correlation: {results.get('mean_spearman_correlation', 'N/A'):.4f}"
    )
    print(f"    Pearson std: {results.get('pearson_std', 'N/A'):.4f}")
    print(f"    Spearman std: {results.get('spearman_std', 'N/A'):.4f}")

    return results


def print_summary(ae_results: dict, cls_results: dict, reg_results: dict):
    """Print summary comparison with expected paper results."""
    print_section("SUMMARY COMPARISON", width=70)

    print("\n  Expected vs Actual Results:")
    print("  " + "-" * 66)
    print(f"  {'Metric':<35} {'Expected':>12} {'Actual':>12}")
    print("  " + "-" * 66)

    # Autoencoder
    if ae_results:
        ae_pcorr = ae_results.get("mean_pearson_correlation", 0)
        print(
            f"  {'AE Mean Pearson Correlation':<35} {'~0.7-0.9':>12} {ae_pcorr:>12.4f}"
        )

    # Classifier
    if cls_results:
        cls_acc = cls_results.get("accuracy", 0)
        print(f"  {'Classifier Accuracy':<35} {'~85-95%':>12} {cls_acc * 100:>11.2f}%")

    # Regressor
    if reg_results:
        reg_pcorr = reg_results.get("mean_pearson_correlation", 0)
        reg_spcorr = reg_results.get("mean_spearman_correlation", 0)
        print(
            f"  {'Regressor Mean Pearson Corr':<35} {'~0.7-0.9':>12} {reg_pcorr:>12.4f}"
        )
        print(
            f"  {'Regressor Mean Spearman Corr':<35} {'~0.7-0.9':>12} {reg_spcorr:>12.4f}"
        )

    print("  " + "-" * 66)

    # Overall assessment
    print("\n  Assessment:")
    all_good = True

    if ae_results and ae_results.get("mean_pearson_correlation", 0) < 0.5:
        print("    - Autoencoder reconstruction quality is low")
        all_good = False

    if cls_results and cls_results.get("accuracy", 0) < 0.7:
        print("    - Classifier accuracy is below expected")
        all_good = False

    if reg_results and reg_results.get("mean_pearson_correlation", 0) < 0.5:
        print("    - Regressor correlation is below expected")
        all_good = False

    if all_good:
        print("    All metrics are within expected ranges!")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate reprogramming reproduction results"
    )
    parser.add_argument(
        "--ae_dir",
        type=str,
        default="runs/reprog_ae",
        help="Directory with autoencoder results",
    )
    parser.add_argument(
        "--cls_dir",
        type=str,
        default="runs/reprog_cls",
        help="Directory with classifier results",
    )
    parser.add_argument(
        "--reg_dir",
        type=str,
        default="runs/reprog_reg",
        help="Directory with regressor results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/evaluation_summary.json",
        help="Output path for summary JSON",
    )

    args = parser.parse_args()

    ae_dir = Path(args.ae_dir)
    cls_dir = Path(args.cls_dir)
    reg_dir = Path(args.reg_dir)

    print("\n" + "=" * 70)
    print("  DEEP LINEAGE REPRODUCTION - EVALUATION REPORT")
    print("=" * 70)

    # Evaluate each component
    ae_results = evaluate_autoencoder(ae_dir) if ae_dir.exists() else None
    cls_results = evaluate_classifier(cls_dir) if cls_dir.exists() else None
    reg_results = evaluate_regressor(reg_dir) if reg_dir.exists() else None

    # Print summary
    print_summary(ae_results, cls_results, reg_results)

    # Save summary
    summary = {
        "autoencoder": ae_results,
        "classifier": cls_results,
        "regressor": reg_results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary saved to: {output_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
