#!/usr/bin/env python3
"""Evaluate classifiers on the test set with metrics and visualizations."""

import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import tensorflow as tf
from tensorflow.keras.models import load_model

from deep_lineage.schema import TrajectoryList
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

from deep_lineage.utils import make_json_serializable
from scripts.utils import get_run_dir, load_trajectory_data


class ClassifierEvaluator:
    """
    Evaluates classifiers with comprehensive metrics and visualizations.
    """

    def __init__(
        self, run_dir: Path, model_name: str, ae_name: str = None, verbose: bool = True
    ):
        self.run_dir = Path(run_dir)
        self.model_name = model_name
        self.ae_name = ae_name
        self.verbose = verbose

        # Load models
        self.classifier_model = None
        self.encoder_model = None
        self._load_models()

    def _load_models(self):
        """Load classifier and encoder models."""
        try:
            # Load classifier
            classifier_path = self.run_dir / f"classifier_{self.model_name}_final.keras"

            self.classifier_model = load_model(str(classifier_path))

            # Load encoder (no longer needed for evaluation, but kept for parity/logging)
            if self.ae_name:
                encoder_path = self.run_dir / f"encoder_{self.ae_name}_final.keras"
            else:
                encoder_path = self.run_dir / "encoder_final.keras"

            if encoder_path.exists():
                self.encoder_model = load_model(str(encoder_path))
            else:
                self.encoder_model = None

            if self.verbose:
                print(f"Loaded classifier from: {classifier_path}")
                if self.encoder_model is not None:
                    print(f"Loaded encoder from: {encoder_path}")
                else:
                    print("Encoder not found (ok for on-the-fly-encoding models).")

        except Exception as e:
            raise RuntimeError(f"Failed to load models: {e}")

    def prepare_test_data(
        self, trajectories: TrajectoryList, timepoints: List[str]
    ) -> tuple:
        """
        Prepare test data for evaluation.

        Args:
            trajectories: Test trajectories
            timepoints: Timepoints to use for classification

        Returns:
            Tuple of (X_test, y_test, y_test_labels)

        NOTE: For end-to-end models (encoder inside classifier), X_test now contains
        RAW gene expressions per timestep with shape [N, T, G].
        """
        if self.verbose:
            print(f"Preparing test data for timepoints: {', '.join(timepoints)}")

        sequences = []
        labels = []
        label_names = []

        # Map timepoint names to indices
        timepoint_map = {"t0": 0, "t1": 1, "t2": 2}
        timepoint_indices = [timepoint_map[tp] for tp in timepoints]

        for traj in tqdm(
            trajectories.trajectories, desc="Processing test trajectories"
        ):
            if len(traj.cells) != 3:
                continue

            # Collect RAW gene expressions for specified timepoints [T, G]
            raw_seq = []
            for tp_idx in timepoint_indices:
                cell = traj.cells[tp_idx]
                raw_seq.append(np.asarray(cell.expr, dtype=np.float32))
            sequences.append(np.stack(raw_seq, axis=0))

            # Get label from t2 cell (final fate)
            final_cell = traj.cells[2]
            if final_cell.state == "fate_0":
                labels.append([1, 0])  # One-hot: fate_0
                label_names.append("fate_0")
            elif final_cell.state == "fate_1":
                labels.append([0, 1])  # One-hot: fate_1
                label_names.append("fate_1")
            else:
                raise ValueError(f"Unknown cell state: {final_cell.state}")

        # X_test is RAW sequences ([N, T, G]) instead of encoded latents ([N, T, D_enc])
        X_test = np.array(sequences, dtype=np.float32)
        y_test = np.array(labels, dtype=np.float32)

        if self.verbose:
            print(f"   Test samples: {len(sequences)}")
            print(f"   Input shape (raw): {X_test.shape}  # [N, T, G]")
            print(f"   Label shape: {y_test.shape}")
            print(f"   Class distribution: {np.sum(y_test, axis=0)} [fate_0, fate_1]")

        return X_test, y_test, label_names

    def evaluate_classifier(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of classifier performance.

        Args:
            X_test: Test input sequences
            y_test: Test labels (one-hot encoded)

        Returns:
            Dictionary with evaluation metrics
        """
        if self.verbose:
            print(f"Evaluating classifier: {self.model_name}")

        # Get predictions (classifier now expects RAW [N, T, G] if trained end-to-end)
        y_pred_probs = self.classifier_model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        balanced_accuracy = balanced_accuracy_score(y_true_classes, y_pred_classes)

        # Handle binary classification for AUC
        if y_pred_probs.shape[1] == 2:
            auc = roc_auc_score(y_true_classes, y_pred_probs[:, 1])
        else:
            auc = roc_auc_score(
                y_test, y_pred_probs, multi_class="ovr", average="macro"
            )

        f1 = f1_score(y_true_classes, y_pred_classes, average="weighted")
        precision = precision_score(
            y_true_classes, y_pred_classes, average="weighted", zero_division=0
        )
        recall = recall_score(
            y_true_classes, y_pred_classes, average="weighted", zero_division=0
        )

        # Confusion matrix and classification report
        conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
        class_report = classification_report(
            y_true_classes, y_pred_classes, output_dict=True
        )

        # Best threshold via Youden's J statistic
        if y_pred_probs.shape[1] == 2:
            fpr, tpr, thresholds = roc_curve(y_true_classes, y_pred_probs[:, 1])
            youden_j = tpr - fpr
            best_threshold_idx = np.argmax(youden_j)
            best_threshold = thresholds[best_threshold_idx]
            best_youden_j = youden_j[best_threshold_idx]
        else:
            best_threshold = None
            best_youden_j = None

        results = {
            "model_name": self.model_name,
            "n_test_samples": int(X_test.shape[0]),
            "accuracy": float(accuracy),
            "balanced_accuracy": float(balanced_accuracy),
            "auc": float(auc),
            "f1_score": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "confusion_matrix": conf_matrix.tolist(),
            "classification_report": class_report,
            "class_distribution_true": np.bincount(y_true_classes).tolist(),
            "class_distribution_pred": np.bincount(y_pred_classes).tolist(),
            "best_threshold": (
                float(best_threshold) if best_threshold is not None else None
            ),
            "best_youden_j": (
                float(best_youden_j) if best_youden_j is not None else None
            ),
            "predictions": {
                "y_true": y_true_classes.tolist(),
                "y_pred": y_pred_classes.tolist(),
                "y_pred_probs": y_pred_probs.tolist(),
            },
        }

        if self.verbose:
            print(
                f"""Evaluation Results:
   Accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)
   Balanced Accuracy: {balanced_accuracy:.4f} ({balanced_accuracy * 100:.1f}%)
   AUC: {auc:.4f}
   F1 Score: {f1:.4f}
   Precision: {precision:.4f}
   Recall: {recall:.4f}"""
            )
            if best_threshold is not None:
                print(
                    f"   Best threshold (Youden J): {best_threshold:.4f} (J={best_youden_j:.4f})"
                )

        return results

    def create_visualizations(self, results: Dict[str, Any], save_dir: Path):
        """
        Create evaluation visualizations.

        Args:
            results: Evaluation results dictionary
            save_dir: Directory to save plots
        """
        if self.verbose:
            print("Creating evaluation visualizations...")

        save_dir.mkdir(parents=True, exist_ok=True)

        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        conf_matrix = np.array(results["confusion_matrix"])
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["fate_0", "fate_1"],
            yticklabels=["fate_0", "fate_1"],
        )
        plt.title(
            f"Confusion Matrix - {self.model_name}\nAccuracy: {results['accuracy']:.3f}"
        )
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        confusion_path = save_dir / f"{self.model_name}_confusion_matrix.png"
        plt.savefig(confusion_path, dpi=300, bbox_inches="tight")
        plt.close()

        # 2. ROC Curve (for binary classification)
        y_true = np.array(results["predictions"]["y_true"])
        y_pred_probs = np.array(results["predictions"]["y_pred_probs"])

        if y_pred_probs.shape[1] == 2:
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_true, y_pred_probs[:, 1])
            plt.plot(
                fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {results['auc']:.3f})"
            )
            plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Classifier")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {self.model_name}")
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            roc_path = save_dir / f"{self.model_name}_roc_curve.png"
            plt.savefig(roc_path, dpi=300, bbox_inches="tight")
            plt.close()

        # 3. Prediction Distribution
        plt.figure(figsize=(12, 5))

        # Subplot 1: Prediction probabilities by true class
        plt.subplot(1, 2, 1)
        for class_label in [0, 1]:
            class_mask = y_true == class_label
            class_name = "fate_0" if class_label == 0 else "fate_1"
            if y_pred_probs.shape[1] == 2:
                probs_to_plot = y_pred_probs[class_mask, 1]  # Probability of fate_1
            else:
                probs_to_plot = y_pred_probs[class_mask, class_label]

            plt.hist(
                probs_to_plot,
                bins=20,
                alpha=0.7,
                label=f"True {class_name}",
                density=True,
            )

        plt.xlabel("Predicted Probability (fate_1)")
        plt.ylabel("Density")
        plt.title("Prediction Probability Distribution")
        plt.legend()
        if results["best_threshold"] is not None:
            plt.axvline(
                x=results["best_threshold"],
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"Best Threshold ({results['best_threshold']:.3f})",
            )

        # Subplot 2: Class distribution comparison
        plt.subplot(1, 2, 2)
        class_names = ["fate_0", "fate_1"]
        true_counts = results["class_distribution_true"]
        pred_counts = results["class_distribution_pred"]

        x = np.arange(len(class_names))
        width = 0.35

        plt.bar(x - width / 2, true_counts, width, label="True", alpha=0.8)
        plt.bar(x + width / 2, pred_counts, width, label="Predicted", alpha=0.8)

        plt.xlabel("Cell Fate")
        plt.ylabel("Count")
        plt.title("Class Distribution Comparison")
        plt.xticks(x, class_names)
        plt.legend()

        plt.tight_layout()

        dist_path = save_dir / f"{self.model_name}_distributions.png"
        plt.savefig(dist_path, dpi=300, bbox_inches="tight")
        plt.close()

        if self.verbose:
            print(f"   Saved confusion matrix: {confusion_path}")
            if y_pred_probs.shape[1] == 2:
                print(f"   Saved ROC curve: {roc_path}")
            print(f"   Saved distributions: {dist_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Deep Lineage-style classifier"
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
        help="Model name to evaluate (e.g., 'ae_t0_only', 'ae_t0_t1', 'ae_t0_t1_t2').",
    )
    parser.add_argument(
        "--timepoints",
        type=str,
        help="Comma-separated timepoints used by this model (if not specified, inferred from model name)",
    )
    parser.add_argument(
        "--ae_name",
        type=str,
        default=None,
        help="Optional autoencoder name suffix (e.g., 'dropout_30pct_ae'). If not provided, uses default encoder files.",
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

    print(f"Evaluating Classifier: {args.model}")
    print(f"Using run directory: {run_dir}")

    # Infer timepoints from model name if not provided
    if args.timepoints:
        timepoints = [tp.strip() for tp in args.timepoints.split(",")]
    else:
        model_suffix = args.model
        if args.model.startswith("ae_"):
            model_suffix = args.model[3:]

        if "t0_only" in model_suffix or model_suffix == "t0":
            timepoints = ["t0"]
        elif "t0_t1" in model_suffix and "t0_t1_t2" not in model_suffix:
            timepoints = ["t0", "t1"]
        elif "t0_t1_t2" in model_suffix:
            timepoints = ["t0", "t1", "t2"]
        else:
            raise ValueError(
                f"Cannot infer timepoints from model name '{args.model}'. Please specify --timepoints"
            )

    print(f"Timepoints: {timepoints}")

    # Load test data
    try:
        test_trajectories = load_trajectory_data(run_dir, split="test")
        print(f"Loaded {len(test_trajectories.trajectories)} test trajectories")

    except FileNotFoundError as e:
        print(f"Error loading test data: {e}")
        return

    # Initialize evaluator
    evaluator = ClassifierEvaluator(
        run_dir, args.model, ae_name=args.ae_name, verbose=True
    )

    # Prepare test data (RAW [N, T, G] for end-to-end classifier)
    X_test, y_test, label_names = evaluator.prepare_test_data(
        test_trajectories, timepoints
    )

    # Evaluate classifier
    results = evaluator.evaluate_classifier(X_test, y_test)

    # Create visualizations
    viz_dir = run_dir / "evaluation_plots" / "classifiers"
    evaluator.create_visualizations(results, viz_dir)

    # Save detailed results
    results_path = run_dir / f"classifier_{args.model}_evaluation.json"
    json_results = make_json_serializable(results)
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"Saved evaluation results to: {results_path}")

    print(
        f"""\nClassifier Evaluation complete.
   Model: {args.model}
   Timepoints: {timepoints}
   Test Accuracy: {results["accuracy"]:.4f} ({results["accuracy"] * 100:.1f}%)
   Test AUC: {results["auc"]:.4f}
   Balanced Accuracy: {results["balanced_accuracy"]:.4f}
   Evaluation plots saved to: {viz_dir}"""
    )

    return results


if __name__ == "__main__":
    main()
