#!/usr/bin/env python3
"""End-to-end evaluation for the two-stage autoregressive pipeline."""

import numpy as np
import pickle
import gzip
import argparse
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.utils import get_run_dir

DEFAULT_RESULTS_JSON = "autoregressive_generator_results.json"


class AutoregressivePipelineEvaluator:
    """Evaluates the full two-stage autoregressive pipeline."""

    def __init__(
        self,
        run_dir: str,
        eval_coldstart_noise_std: float | None = None,
        match_training_coldstart: bool = True,
        generator_model_path: str = None,
        classifier_model_path: str = None,
        verbose: bool = True,
    ):
        self.run_dir = Path(run_dir)
        self.verbose = verbose

        self.generator_model_path = (
            generator_model_path
            or self.run_dir / "autoregressive_generator_final.keras"
        )
        self.classifier_model_path = (
            classifier_model_path or self.run_dir / "expression_classifier_final.keras"
        )

        self.generator_model = None
        self.classifier_model = None
        self.normalization_params = None
        # Eval cold-start controls
        self.match_training_coldstart = bool(match_training_coldstart)
        self.eval_coldstart_noise_std = (
            None
            if eval_coldstart_noise_std is None
            else float(eval_coldstart_noise_std)
        )

        self._load_models()
        # Numerical stability knobs
        self._eps = 1e-3  # std floor to avoid huge z-scores
        self._clip = 8.0  # clip normalized values to [-_clip, _clip]
        self._load_normalization_params()
        self._maybe_infer_coldstart_from_training()

    def _load_models(self):
        """Load both trained models."""
        try:
            # Load generator model without compilation first, then recompile with simple MSE
            self.generator_model = tf.keras.models.load_model(
                self.generator_model_path, compile=False
            )
            # Recompile with simple MSE loss to match training
            optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4, clipnorm=1.0)
            self.generator_model.compile(
                optimizer=optimizer, loss="mse", metrics=["mae", "mse"]
            )
            if self.verbose:
                print(
                    f"Loaded autoregressive generator from: {self.generator_model_path}"
                )

            self.classifier_model = tf.keras.models.load_model(
                self.classifier_model_path
            )
            if self.verbose:
                print(
                    f"Loaded expression classifier from: {self.classifier_model_path}"
                )

        except Exception as e:
            raise RuntimeError(f"Failed to load models: {e}")

    def _load_normalization_params(self):
        """Load normalization parameters for proper inference."""
        norm_path = self.run_dir / "autoregressive_normalization_params.json"

        if not norm_path.exists():
            if self.verbose:
                print(
                    "No normalization parameters found. Using raw data (may cause issues)."
                )
            return

        try:
            with open(norm_path, "r") as f:
                params_json = json.load(f)

            # Convert back to numpy arrays
            self.normalization_params = {}
            for key, value in params_json.items():
                self.normalization_params[key] = np.array(value)

            if self.verbose:
                print(f"Loaded normalization parameters from: {norm_path}")

        except Exception as e:
            if self.verbose:
                print(f"Failed to load normalization parameters: {e}")
            self.normalization_params = None

    # Note: _expand_stats_for method removed - now using global normalization with scalar parameters

    def _normalize_input(self, X):
        """Normalize input data using training normalization parameters (global)."""
        if self.normalization_params is None:
            return X

        X_mean = self.normalization_params.get("X_mean", None)
        X_std = self.normalization_params.get("X_std", None)
        if X_mean is None or X_std is None:
            return X

        # Convert to scalars if they're arrays (backward compatibility)
        if hasattr(X_mean, "item"):
            X_mean = X_mean.item()
        if hasattr(X_std, "item"):
            X_std = X_std.item()

        # Floor std to avoid inf z-scores
        X_std = max(X_std, self._eps)

        # Simple global normalization
        Xn = (X - X_mean) / X_std

        # Clip and de-nan/inf for stability
        if np.isfinite(self._clip) and self._clip > 0:
            Xn = np.clip(Xn, -self._clip, self._clip)
        Xn = np.nan_to_num(Xn, nan=0.0, posinf=self._clip, neginf=-self._clip)

        if self.verbose:
            try:
                o_min, o_max = float(np.min(X)), float(np.max(X))
                n_min, n_max = float(np.min(Xn)), float(np.max(Xn))
                print("   Input normalization applied (GLOBAL):")
                print(f"      X_mean: {X_mean:.6f}, X_std: {X_std:.6f}")
                print(f"      Original range: [{o_min:.3f}, {o_max:.3f}]")
                print(f"      Normalized range (clipped): [{n_min:.3f}, {n_max:.3f}]")
            except Exception:
                pass

        return Xn

    def _denormalize_output(self, y_pred):
        """Denormalize model output back to original scale (global)."""
        if self.normalization_params is None:
            return y_pred

        y_mean = self.normalization_params.get("y_mean", None)
        y_std = self.normalization_params.get("y_std", None)
        if y_mean is None or y_std is None:
            return y_pred

        # Convert to scalars if they're arrays (backward compatibility)
        if hasattr(y_mean, "item"):
            y_mean = y_mean.item()
        if hasattr(y_std, "item"):
            y_std = y_std.item()

        # Floor std to avoid inf scaling
        y_std = max(y_std, self._eps)

        # Simple global denormalization
        yd = y_pred * y_std + y_mean

        if self.verbose:
            try:
                n_min, n_max = float(np.min(y_pred)), float(np.max(y_pred))
                d_min, d_max = float(np.min(yd)), float(np.max(yd))
                print("   Output denormalization applied (GLOBAL):")
                print(f"      y_mean: {y_mean:.6f}, y_std: {y_std:.6f}")
                print(f"      Normalized range: [{n_min:.3f}, {n_max:.3f}]")
                print(f"      Denormalized range: [{d_min:.3f}, {d_max:.3f}]")
            except Exception:
                pass

        return yd

    def _maybe_infer_coldstart_from_training(self):
        """
        Auto-set eval cold-start noise to match training, unless explicitly provided.
        Looks for <run_dir>/autoregressive_generator_results.json written by the trainer.
        """
        if self.eval_coldstart_noise_std is not None:
            # user explicitly set it; keep as-is
            if self.verbose:
                print(
                    f"   Using eval cold-start noise from CLI: {self.eval_coldstart_noise_std:.3f}"
                )
            return
        if not self.match_training_coldstart:
            self.eval_coldstart_noise_std = 0.0
            if self.verbose:
                print(
                    "   Cold-start eval noise disabled (match_training_coldstart=False)."
                )
            return
        results_path = self.run_dir / DEFAULT_RESULTS_JSON
        self.eval_coldstart_noise_std = 0.0
        if results_path.exists():
            try:
                cfg = json.load(open(results_path, "r"))
                self.eval_coldstart_noise_std = float(
                    cfg.get("coldstart_noise_std", 0.0)
                )
                if self.verbose:
                    print(
                        f"   Eval cold-start noise matched to training: {self.eval_coldstart_noise_std:.3f}"
                    )
            except Exception as e:
                if self.verbose:
                    print(f"   Could not read {results_path.name}: {e}. Using 0.0.")

    def _run_day0_autoregressive(self, X_day0_only, y_true, X_day14_true):
        """Run Day0-only autoregressive prediction and evaluation."""
        if self.verbose:
            print("      Running Day0-only autoregressive generation...")

        # Normalize Day0 input
        X_day0_normalized = self._normalize_input(X_day0_only)
        # Safety: ensure finite
        X_day0_normalized = np.nan_to_num(
            X_day0_normalized, nan=0.0, posinf=self._clip, neginf=-self._clip
        )

        # Autoregressive generation: Day0 → Day7 → Day14
        # We need to generate the sequence step by step
        current_sequence = X_day0_normalized  # Start with [n_samples, 1, n_genes]

        # Generate Day7 (next timestep)
        # Build [Day0, Day0(+ε)] to mirror cold-start augmentation during training
        day0_repeated = np.concatenate([current_sequence, current_sequence], axis=1)
        if (self.eval_coldstart_noise_std or 0.0) > 0.0:
            eps = np.random.normal(
                0.0,
                float(self.eval_coldstart_noise_std),
                size=day0_repeated[:, 1:2, :].shape,
            ).astype(day0_repeated.dtype)
            day0_repeated[:, 1:2, :] = day0_repeated[:, 1:2, :] + eps
        day7_normalized = self.generator_model.predict(day0_repeated, verbose=0)
        day7_pred_normalized = day7_normalized[
            :, 1:2, :
        ]  # Take the second timestep (Day7 prediction)

        # Now we have Day0 and Day7, use them to generate Day14
        day07_sequence = np.concatenate(
            [current_sequence, day7_pred_normalized], axis=1
        )
        day14_normalized = self.generator_model.predict(day07_sequence, verbose=0)
        day14_pred_normalized = day14_normalized[:, 1:2, :]  # Take Day14 prediction

        # Denormalize the final Day14 predictions
        day14_pred = self._denormalize_output(day14_pred_normalized)
        day14_pred_flat = day14_pred.squeeze(axis=1)  # Shape: [n_samples, n_genes]

        if self.verbose:
            print(f"      Generated Day14 from Day0: {day14_pred_flat.shape}")

        # Classify Day14 predictions to get cell types
        y_pred_probs = self.classifier_model.predict(day14_pred_flat, verbose=0)
        y_pred = (y_pred_probs > 0.5).astype(int).ravel()

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_probs.ravel())
        f1 = f1_score(y_true, y_pred, average="weighted")

        # Calculate generation quality metrics
        generation_mse = np.mean((day14_pred_flat - X_day14_true) ** 2)
        generation_correlation = np.corrcoef(
            day14_pred_flat.ravel(), X_day14_true.ravel()
        )[0, 1]

        return {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "auc": auc,
            "f1_score": f1,
            "y_pred": y_pred,
            "y_pred_probs": y_pred_probs.ravel(),
            "X_day14_generated": day14_pred_flat,
            "generation_mse": generation_mse,
            "generation_correlation": generation_correlation,
        }

    def _print_performance_comparison(self, eval_metrics):
        """Print comprehensive performance comparison between different approaches."""
        print(f"\n{'=' * 80}")
        print("COMPREHENSIVE PERFORMANCE COMPARISON")
        print(f"{'=' * 80}")

        # Extract metrics
        pipeline_acc = eval_metrics["accuracy"]
        pipeline_bal_acc = eval_metrics["balanced_accuracy"]
        pipeline_auc = eval_metrics["auc"]
        pipeline_gen_mse = eval_metrics.get("generation_mse", 0)
        pipeline_gen_corr = eval_metrics.get("generation_correlation", 0)

        baselines = eval_metrics.get("baselines", {})

        # Print header
        print(
            f"{'Approach':<35} {'Accuracy':<10} {'Bal. Acc':<10} {'AUC':<8} {'Gen. MSE':<12} {'Gen. Corr':<10}"
        )
        print(f"{'-' * 35} {'-' * 10} {'-' * 10} {'-' * 8} {'-' * 12} {'-' * 10}")

        # Day0+Day7 Autoregressive (main result)
        print(
            f"{'Day0+Day7 → Autoregressive':<35} {pipeline_acc:<10.3f} {pipeline_bal_acc:<10.3f} {pipeline_auc:<8.3f} {pipeline_gen_mse:<12.0f} {pipeline_gen_corr:<10.3f}"
        )

        # Day0 Autoregressive
        if "day0_autoregressive" in baselines:
            day0 = baselines["day0_autoregressive"]
            print(
                f"{'Day0 → Autoregressive':<35} {day0['accuracy']:<10.3f} {day0['balanced_accuracy']:<10.3f} {day0['auc']:<8.3f} {day0['generation_mse']:<12.0f} {day0['generation_correlation']:<10.3f}"
            )

        # True Day14 (upper bound)
        if "true_day14" in baselines:
            true_day14 = baselines["true_day14"]
            print(
                f"{'True Day14 → Classifier':<35} {true_day14['accuracy']:<10.3f} {true_day14['balanced_accuracy']:<10.3f} {true_day14['auc']:<8.3f} {'N/A':<12} {'N/A':<10}"
            )

        # Random (lower bound)
        if "random" in baselines:
            random = baselines["random"]
            print(
                f"{'Random Prediction':<35} {random['accuracy']:<10.3f} {'N/A':<10} {random['auc']:<8.3f} {'N/A':<12} {'N/A':<10}"
            )

        # Analysis section
        print("\nKEY INSIGHTS:")

        if "day0_autoregressive" in baselines:
            day0 = baselines["day0_autoregressive"]
            day7_improvement = pipeline_acc - day0["accuracy"]
            day7_auc_improvement = pipeline_auc - day0["auc"]

            print("   Day7 Information Value:")
            print(
                f"     - Accuracy improvement: {day7_improvement:+.3f} ({day7_improvement / day0['accuracy'] * 100:+.1f}%)"
            )
            print(
                f"     - AUC improvement: {day7_auc_improvement:+.3f} ({day7_auc_improvement / day0['auc'] * 100:+.1f}%)"
            )

            if day7_improvement > 0.05:  # 5% improvement threshold
                print("     - PASS: Day7 information provides SIGNIFICANT benefit")
            elif day7_improvement > 0.02:
                print("     - Day7 information provides MODEST benefit")
            else:
                print("     - FAIL: Day7 information provides MINIMAL benefit")

        if "true_day14" in baselines:
            true_day14 = baselines["true_day14"]
            generation_gap = true_day14["accuracy"] - pipeline_acc

            print("   Generation Quality Impact:")
            print(
                f"     - Performance gap due to generation: {generation_gap:.3f} ({generation_gap / true_day14['accuracy'] * 100:.1f}%)"
            )

            if generation_gap < 0.05:
                print("     - PASS: Generator produces EXCELLENT synthetic expressions")
            elif generation_gap < 0.15:
                print("     - Generator produces GOOD synthetic expressions")
            else:
                print("     - FAIL: Generator quality limits pipeline performance")

        print(f"{'=' * 80}")

    def load_trajectories(self, trajectories_path: str = None, split: str = "test"):
        """Load trajectories for evaluation."""
        if trajectories_path is None:
            # Try to load test split first, fall back to original files
            test_files = [
                f"trajectories_{split}.pkl.gz",
                "trajectories.pkl.gz",
            ]

            for filename in test_files:
                filepath = self.run_dir / filename
                if filepath.exists():
                    trajectories_path = filepath
                    break
            else:
                raise FileNotFoundError(
                    f"No trajectory file found in {self.run_dir}. Looked for: {test_files}"
                )

        if self.verbose:
            print(f"Loading trajectories from: {trajectories_path}")

        with gzip.open(trajectories_path, "rb") as f:
            trajectories = pickle.load(f)

        if self.verbose:
            print(f"Loaded {len(trajectories.trajectories)} trajectories")

            # Log test split information for verification
            group_ids = set()
            for traj in trajectories.trajectories:
                group_ids.add(traj.trajectory_group_id)

            print("   Test split verification:")
            print(f"      Unique Day0 groups: {len(group_ids)}")
            print(
                f"      Group ID range: {min(group_ids) if group_ids else 'N/A'} - {max(group_ids) if group_ids else 'N/A'}"
            )
            sample_groups = sorted(list(group_ids))[:3]
            print(f"      Sample group IDs: {sample_groups}")

        return trajectories

    def prepare_pipeline_data(self, trajectories):
        """
        Prepare data for full pipeline evaluation.

        Pipeline flow:
        1. Input: Day0 + Day7 expressions → Generator → Predicted Day14 expressions
        2. Input: Generated Day14 expressions → Classifier → Predicted cell types
        3. Compare predicted vs true cell types
        """
        X_input = []  # [Day0, Day7] sequences for generator
        X_day14_true = []  # True Day14 expressions
        y_true = []  # True Day14 cell types

        for traj in trajectories.trajectories:
            # Each trajectory has 3 cells: [Day0, Day7, Day14]
            day0_cell = traj.cells[0]  # Day0
            day7_cell = traj.cells[1]  # Day7
            day14_cell = traj.cells[2]  # Day14

            day0_expr = day0_cell.expr
            day7_expr = day7_cell.expr
            day14_expr = day14_cell.expr
            day14_type = day14_cell.state  # 'fate_0' or 'fate_1'

            # Generator input: [Day0, Day7] sequence
            input_seq = [day0_expr, day7_expr]
            X_input.append(input_seq)

            # True Day14 data for comparison
            X_day14_true.append(day14_expr)
            y_true.append(1 if day14_type == "fate_1" else 0)

        X_input = np.array(X_input)  # Shape: [n_samples, 2, n_genes]
        X_day14_true = np.array(X_day14_true)  # Shape: [n_samples, n_genes]
        y_true = np.array(y_true)  # Shape: [n_samples]

        if self.verbose:
            print("Pipeline data prepared:")
            print(f"   Input sequences: {X_input.shape} [samples, timepoints, genes]")
            print(f"   True Day14 expressions: {X_day14_true.shape} [samples, genes]")
            print(f"   True Day14 cell types: {y_true.shape} [samples]")
            print(f"   Class distribution: {np.bincount(y_true)}")

        return X_input, X_day14_true, y_true

    def run_full_pipeline(self, X_input, y_true):
        """
        Run the full two-stage pipeline.

        Args:
            X_input: Input sequences [n_samples, 2, n_genes] for generator
            y_true: True Day14 cell types [n_samples]

        Returns:
            dict with predictions and intermediate results
        """
        if self.verbose:
            print("Running full two-stage pipeline...")

        # Stage 1: Generate Day14 expressions from Day0+Day7
        if self.verbose:
            print("   Stage 1: Autoregressive generation (Day0+Day7 -> Day14)")

        # Normalize input data for generator
        X_input_normalized = self._normalize_input(X_input)

        # Generator expects input shape [n_samples, 2, n_genes]
        # Output shape: [n_samples, 2, n_genes] representing [Day7_pred, Day14_pred]
        generator_output_normalized = self.generator_model.predict(
            X_input_normalized, verbose=0
        )

        # Denormalize generator output
        generator_output = self._denormalize_output(generator_output_normalized)

        # Extract predicted Day14 expressions (second timepoint)
        X_day14_pred = generator_output[:, 1, :]  # Shape: [n_samples, n_genes]

        if self.verbose:
            print(f"      Generated Day14 expressions: {X_day14_pred.shape}")

        # Stage 2: Classify cell types from generated Day14 expressions
        if self.verbose:
            print("   Stage 2: Expression classification (Day14_pred -> cell_types)")

        # Classifier expects input shape [n_samples, n_genes]
        y_pred_probs = self.classifier_model.predict(X_day14_pred, verbose=0)
        y_pred = (y_pred_probs > 0.5).astype(int).ravel()

        if self.verbose:
            print(f"      Predicted cell type probabilities: {y_pred_probs.shape}")
            print(f"      Predicted cell types: {y_pred.shape}")

        return {
            "X_day14_generated": X_day14_pred,
            "y_pred_probs": y_pred_probs.ravel(),
            "y_pred": y_pred,
            "y_true": y_true,
        }

    def evaluate_pipeline(self, results, X_day14_true=None):
        """Evaluate the full pipeline performance."""
        y_true = results["y_true"]
        y_pred = results["y_pred"]
        y_pred_probs = results["y_pred_probs"]

        accuracy = accuracy_score(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_probs)
        f1 = f1_score(y_true, y_pred, average="weighted")
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)

        # Best threshold via Youden's J (optional but useful to report)
        try:
            fpr, tpr, thr = roc_curve(y_true, y_pred_probs)
            youden = tpr - fpr
            j_idx = int(np.argmax(youden))
            best_thr = float(thr[j_idx])
            best_ba = 0.5 + float(youden[j_idx]) / 2.0
        except Exception:
            best_thr = None
            best_ba = None

        # Classification report
        class_report = classification_report(y_true, y_pred, output_dict=True)

        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)

        eval_metrics = {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "auc": auc,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "classification_report": class_report,
            "confusion_matrix": conf_matrix.tolist(),
            "class_distribution_true": np.bincount(y_true).tolist(),
            "class_distribution_pred": np.bincount(y_pred).tolist(),
            "best_threshold": best_thr,
            "best_balanced_accuracy": best_ba,
        }

        if self.verbose:
            print("\nFull Pipeline Evaluation Results:")
            print(f"   End-to-end Accuracy: {accuracy:.3f} ({accuracy * 100:.1f}%)")
            print(
                f"   End-to-end Balanced Accuracy: {balanced_accuracy:.3f} ({balanced_accuracy * 100:.1f}%)"
            )
            print(f"   End-to-end AUC: {auc:.3f}")
            print(f"   End-to-end F1 Score: {f1:.3f}")
            print(f"   End-to-end Precision: {precision:.3f}")
            print(f"   End-to-end Recall: {recall:.3f}")
            print(f"   Class distribution (true): {np.bincount(y_true)}")
            print(f"   Class distribution (pred): {np.bincount(y_pred)}")
            if best_ba is not None:
                print(
                    f"   Best BA (Youden J): {best_ba:.3f} at threshold={best_thr:.3f}"
                )

        # Optional: Compare generated vs true Day14 expressions
        if X_day14_true is not None:
            X_day14_generated = results["X_day14_generated"]
            mse = np.mean((X_day14_generated - X_day14_true) ** 2)
            correlation = np.corrcoef(X_day14_generated.ravel(), X_day14_true.ravel())[
                0, 1
            ]

            eval_metrics["generation_mse"] = mse
            eval_metrics["generation_correlation"] = correlation

            if self.verbose:
                print(f"   Generation MSE: {mse:.6f}")
                print(f"   Generation correlation: {correlation:.3f}")

        return eval_metrics

    def save_evaluation_results(self, eval_metrics, results, suffix=""):
        """Save evaluation results and visualizations."""
        # Save metrics JSON
        results_file = self.run_dir / f"pipeline_evaluation_results{suffix}.json"
        with open(results_file, "w") as f:
            json.dump(eval_metrics, f, indent=2, default=str)

        if self.verbose:
            print(f"Saved evaluation results to: {results_file}")

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        conf_matrix = np.array(eval_metrics["confusion_matrix"])
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Fate_0", "Fate_1"],
            yticklabels=["Fate_0", "Fate_1"],
        )
        plt.title(
            f"Full Pipeline Confusion Matrix\nAccuracy: {eval_metrics['accuracy']:.3f}, AUC: {eval_metrics['auc']:.3f}"
        )
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        confusion_file = self.run_dir / f"pipeline_confusion_matrix{suffix}.png"
        plt.savefig(confusion_file, dpi=300, bbox_inches="tight")
        plt.close()

        if self.verbose:
            print(f"Saved confusion matrix to: {confusion_file}")

        # Plot prediction distribution
        plt.figure(figsize=(12, 5))

        # Subplot 1: Prediction probabilities by true class
        plt.subplot(1, 2, 1)
        y_true = results["y_true"]
        y_pred_probs = results["y_pred_probs"]

        for class_label in [0, 1]:
            class_mask = y_true == class_label
            class_name = "Fate_0" if class_label == 0 else "Fate_1"
            plt.hist(
                y_pred_probs[class_mask],
                bins=20,
                alpha=0.7,
                label=f"True {class_name}",
                density=True,
            )

        plt.xlabel("Predicted Probability (Fate_1)")
        plt.ylabel("Density")
        plt.title("Prediction Probability Distribution")
        plt.legend()
        plt.axvline(x=0.5, color="red", linestyle="--", alpha=0.7, label="Threshold")

        # Subplot 2: ROC-like visualization
        plt.subplot(1, 2, 2)
        sorted_indices = np.argsort(y_pred_probs)
        sorted_true = y_true[sorted_indices]

        # Compute cumulative precision-like metric
        cumulative_positive = np.cumsum(sorted_true)
        total_positive = np.sum(sorted_true)
        precision_curve = cumulative_positive / (np.arange(len(sorted_true)) + 1)
        recall_curve = (
            cumulative_positive / total_positive
            if total_positive > 0
            else np.zeros_like(cumulative_positive)
        )

        plt.plot(recall_curve, precision_curve, "b-", linewidth=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve\n(AUC: {eval_metrics['auc']:.3f})")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        distribution_file = self.run_dir / f"pipeline_prediction_analysis{suffix}.png"
        plt.savefig(distribution_file, dpi=300, bbox_inches="tight")
        plt.close()

        if self.verbose:
            print(f"Saved prediction analysis to: {distribution_file}")

    def compare_with_baselines(self, X_input, X_day14_true, y_true):
        """Compare pipeline with baseline approaches."""
        if self.verbose:
            print("\nComparing with baseline approaches...")

        baselines = {}

        # Baseline 1: Direct classification from Day0+Day7 (skip generation)
        if self.verbose:
            print("   Baseline 1: Direct classification from Day0+Day7")

        try:
            # Use true Day14 expressions to get upper bound
            y_true_day14_probs = self.classifier_model.predict(X_day14_true, verbose=0)
            y_true_day14_pred = (y_true_day14_probs > 0.5).astype(int).ravel()

            baselines["true_day14"] = {
                "accuracy": accuracy_score(y_true, y_true_day14_pred),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_true_day14_pred),
                "auc": roc_auc_score(y_true, y_true_day14_probs.ravel()),
                "f1_score": f1_score(y_true, y_true_day14_pred, average="weighted"),
                "description": "Classifier on true Day14 expressions (upper bound)",
            }

            if self.verbose:
                acc = baselines["true_day14"]["accuracy"]
                balanced_acc = baselines["true_day14"]["balanced_accuracy"]
                auc = baselines["true_day14"]["auc"]
                f1 = baselines["true_day14"]["f1_score"]
                print(
                    f"      True Day14 → Classifier: Accuracy {acc:.3f}, Balanced Acc {balanced_acc:.3f}, AUC {auc:.3f}, F1 {f1:.3f}"
                )

        except Exception as e:
            if self.verbose:
                print(f"      Could not compute true Day14 baseline: {e}")

        # Baseline 2: Day0-only autoregressive prediction
        if self.verbose:
            print("   Baseline 2: Day0-only autoregressive prediction")

        try:
            # Extract Day0 expressions only
            X_day0_only = X_input[:, 0:1, :]  # Shape: [n_samples, 1, n_genes]

            # Run Day0-only autoregressive prediction
            day0_results = self._run_day0_autoregressive(
                X_day0_only, y_true, X_day14_true
            )

            baselines["day0_autoregressive"] = {
                "accuracy": day0_results["accuracy"],
                "balanced_accuracy": day0_results["balanced_accuracy"],
                "auc": day0_results["auc"],
                "f1_score": day0_results["f1_score"],
                "generation_mse": day0_results["generation_mse"],
                "generation_correlation": day0_results["generation_correlation"],
                "description": "Day0-only autoregressive prediction",
            }

            if self.verbose:
                acc = day0_results["accuracy"]
                balanced_acc = day0_results["balanced_accuracy"]
                auc = day0_results["auc"]
                f1 = day0_results["f1_score"]
                gen_mse = day0_results["generation_mse"]
                gen_corr = day0_results["generation_correlation"]
                print(
                    f"      Day0 Autoregressive → Classifier: Accuracy {acc:.3f}, Balanced Acc {balanced_acc:.3f}, AUC {auc:.3f}, F1 {f1:.3f}"
                )
                print(
                    f"      Day0 Generation MSE: {gen_mse:.1f}, Correlation: {gen_corr:.3f}"
                )

        except Exception as e:
            if self.verbose:
                print(f"      Could not compute Day0 autoregressive baseline: {e}")

        # Baseline 3: Random prediction
        np.random.seed(42)
        y_random = np.random.binomial(1, np.mean(y_true), size=len(y_true))
        y_random_probs = np.random.uniform(0, 1, size=len(y_true))

        baselines["random"] = {
            "accuracy": accuracy_score(y_true, y_random),
            "auc": roc_auc_score(y_true, y_random_probs),
            "description": "Random prediction baseline",
        }

        if self.verbose:
            acc = baselines["random"]["accuracy"]
            auc = baselines["random"]["auc"]
            print(f"      Random prediction: Accuracy {acc:.3f}, AUC {auc:.3f}")

        return baselines


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate two-stage autoregressive pipeline"
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Directory containing trained models and data (defaults to current run dir)",
    )
    parser.add_argument(
        "--coldstart_eval_noise_std",
        type=float,
        default=None,
        help="Override: Gaussian noise (normalized units) to add when duplicating Day0 in Day0-only eval. "
        "If not set, will be read from autoregressive_generator_results.json.",
    )
    parser.add_argument(
        "--no_match_training_coldstart",
        action="store_true",
        help="Disable auto-matching of training cold-start noise; use 0.0 unless overridden above.",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    print("Two-Stage Autoregressive Pipeline Evaluation")
    print("=" * 60)

    # Get run directory - use provided or auto-detect
    if args.run_dir is None:
        try:
            run_dir = get_run_dir()
            print(f"Using auto-detected run directory: {run_dir}")
        except ValueError as e:
            print(f"Error: {e}")
            print("   Please specify --run_dir or ensure a pipeline run is active.")
            return None
    else:
        run_dir = args.run_dir
        print(f"Using specified run directory: {run_dir}")

    # Initialize evaluator
    evaluator = AutoregressivePipelineEvaluator(
        run_dir=run_dir,
        eval_coldstart_noise_std=args.coldstart_eval_noise_std,
        match_training_coldstart=not args.no_match_training_coldstart,
        verbose=True,
    )

    # Load data - use test split if available
    trajectories = evaluator.load_trajectories(split="test")
    X_input, X_day14_true, y_true = evaluator.prepare_pipeline_data(trajectories)

    print(f"Using test set: {len(X_input)} samples")

    # Use all test data for evaluation (no further splitting needed)
    X_test = X_input
    X_day14_test_true = X_day14_true
    y_test = y_true

    # Run full pipeline
    results = evaluator.run_full_pipeline(X_test, y_test)

    # Evaluate performance
    eval_metrics = evaluator.evaluate_pipeline(results, X_day14_test_true)

    # Compare with baselines
    baselines = evaluator.compare_with_baselines(X_test, X_day14_test_true, y_test)
    eval_metrics["baselines"] = baselines

    # Add comprehensive performance comparison
    evaluator._print_performance_comparison(eval_metrics)

    # Save results
    evaluator.save_evaluation_results(eval_metrics, results, suffix="_test")

    print("\nTwo-Stage Pipeline Evaluation complete.")
    print(
        f"   Final test accuracy: {eval_metrics['accuracy']:.3f} ({eval_metrics['accuracy'] * 100:.1f}%)"
    )
    print(f"   Final test AUC: {eval_metrics['auc']:.3f}")
    print(f"   Results saved to: {evaluator.run_dir}")

    return eval_metrics


if __name__ == "__main__":
    main()
