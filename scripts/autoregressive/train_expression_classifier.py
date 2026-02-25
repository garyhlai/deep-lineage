#!/usr/bin/env python3
"""Neural network expression classifier (Stage 2 of two-stage pipeline)."""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks, optimizers
from tensorflow.keras.models import save_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from deep_lineage.schema import TrajectoryList
from scripts.utils import get_run_dir, load_trajectory_data, log_trajectory_split_info


def calculate_auc_tf(y_true, y_pred):
    """Calculate AUC using TensorFlow metrics."""
    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        raise ValueError(
            f"AUC calculation requires at least 2 classes, but found {len(unique_labels)} unique labels: {unique_labels}"
        )

    auc_metric = tf.keras.metrics.AUC()
    auc_metric.update_state(y_true, y_pred)
    return float(auc_metric.result().numpy())


class ExpressionClassifier:
    """
    Neural network classifier for cell types from gene expressions.

    Architecture:
    - Multi-layer feedforward network
    - Batch normalization for training stability
    - Dropout for regularization
    - Binary classification output
    """

    def __init__(self, n_genes: int, config: dict = None):
        self.n_genes = n_genes
        self.config = config or self._get_default_config()
        self.model = None
        self.history = None

    def _get_default_config(self) -> dict:
        """Get default configuration for the classifier."""
        return {
            "hidden_layers": [512, 256, 128],  # Hidden layer sizes
            "dropout": 0.3,
            "batch_norm": True,
            "activation": "relu",
            "learning_rate": 1e-3,
            "batch_size": 512,
            "epochs": 100,
            "patience": 15,
            "l2_reg": 1e-5,
        }

    def build_model(self) -> Model:
        """
        Build neural network expression classifier.

        Returns:
            Keras Model for expression classification
        """
        print(f"""Building neural network expression classifier
   Input genes: {self.n_genes}
   Architecture: {self.config["hidden_layers"]} hidden layers
   Dropout: {self.config["dropout"]}, L2 reg: {self.config["l2_reg"]}""")

        inputs = layers.Input(shape=(self.n_genes,), name="gene_expressions")

        l2_reg = (
            tf.keras.regularizers.l2(self.config["l2_reg"])
            if self.config["l2_reg"] > 0
            else None
        )

        x = inputs

        for i, hidden_size in enumerate(self.config["hidden_layers"]):
            x = layers.Dense(
                hidden_size,
                activation=self.config["activation"],
                kernel_regularizer=l2_reg,
                name=f"hidden_{i}",
            )(x)

            if self.config["batch_norm"]:
                x = layers.BatchNormalization(name=f"batch_norm_{i}")(x)

            x = layers.Dropout(self.config["dropout"], name=f"dropout_{i}")(x)

        outputs = layers.Dense(
            1,
            activation="sigmoid",
            name="cell_type_output",
            kernel_regularizer=l2_reg,
        )(x)

        model = Model(inputs=inputs, outputs=outputs, name="expression_classifier")

        optimizer = optimizers.Adam(learning_rate=self.config["learning_rate"])
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

        print(f"""Model summary:
   Total parameters: {model.count_params():,}
   Optimizer: Adam (lr={self.config["learning_rate"]})
   Loss: Binary cross-entropy""")

        self.model = model
        return model

    def prepare_classification_data(
        self, trajectories: TrajectoryList
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for expression classification.

        Extracts all gene expressions and their corresponding cell types
        from trajectories.

        Args:
            trajectories: TrajectoryList with cell trajectory data

        Returns:
            Tuple of (expressions, labels)
        """
        print("Preparing expression classification data...")

        expressions = []
        labels = []

        for traj in trajectories.trajectories:
            for cell in traj.cells:
                expressions.append(cell.expr)

                # Extract cell type from state (e.g., "fate_0" → 0, "fate_1" → 1)
                cell_type = int(cell.state.split("_")[1])
                labels.append(cell_type)

        X = np.array(expressions)
        y = np.array(labels)

        print(f"   Total samples: {len(X)}")
        print(f"   Expression shape: {X.shape} [samples, n_genes]")
        print(f"   Class distribution: {np.bincount(y)}")
        class_balance = np.bincount(y) / len(y) * 100
        print(f"   Class balance: {class_balance[0]:.1f}% / {class_balance[1]:.1f}%")

        class_counts = np.bincount(y)
        total_samples = len(y)
        n_classes = len(class_counts)
        class_weights = {}
        for i in range(n_classes):
            class_weights[i] = total_samples / (n_classes * class_counts[i])

        print(f"   Class weights: {class_weights}")

        return X, y, class_weights

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        class_weights: dict = None,
        test_size: float = 0.2,
        val_size: float = 0.2,
        run_dir: Path = None,
    ) -> Dict[str, Any]:
        """
        Train the expression classifier.

        Args:
            X_train: Training gene expressions [samples, n_genes]
            y_train: Training cell type labels [samples]
            X_val: Validation gene expressions [samples, n_genes] (optional)
            y_val: Validation cell type labels [samples] (optional)
            class_weights: Dictionary with class weights for imbalanced data
            test_size: Fraction of data for testing (used if X_val/y_val not provided)
            val_size: Fraction of training data for validation (used if X_val/y_val not provided)
            run_dir: Directory for saving results

        Returns:
            Dictionary with training metrics and results
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        print("Training expression classifier...")

        if X_val is not None and y_val is not None:
            print("   Using provided train/validation splits")
            print(f"   Train samples: {len(X_train)}")
            print(f"   Validation samples: {len(X_val)}")

            validation_data = (X_val, y_val)
            use_validation_split = False

            X_test, X_val_subset, y_test, y_val_subset = train_test_split(
                X_val, y_val, test_size=0.5, random_state=42, stratify=y_val
            )

        else:
            print("   Splitting training data for validation")
            X_train_split, X_test, y_train_split, y_test = train_test_split(
                X_train, y_train, test_size=test_size, random_state=42, stratify=y_train
            )
            X_train = X_train_split
            y_train = y_train_split

            print(f"   Train samples: {len(X_train)}")
            print(f"   Test samples: {len(X_test)}")

            validation_data = None
            use_validation_split = True

        print(
            f"   Validation method: {'External' if not use_validation_split else f'Split ({val_size:.1%})'}"
        )
        print(
            f"   Epochs: {self.config['epochs']}, Batch size: {self.config['batch_size']}"
        )

        callback_list = []

        if run_dir:
            tensorboard_dir = run_dir / "tensorboard" / "expression_classifier"
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            callback_list.append(
                callbacks.TensorBoard(
                    log_dir=str(tensorboard_dir),
                    histogram_freq=1,
                    write_graph=True,
                    write_images=True,
                    update_freq="epoch",
                )
            )

            checkpoint_path = run_dir / "expression_classifier_best.keras"
            callback_list.append(
                callbacks.ModelCheckpoint(
                    str(checkpoint_path),
                    monitor="val_auc",
                    save_best_only=True,
                    save_weights_only=False,
                    mode="max",
                    verbose=1,
                )
            )

        callback_list.append(
            callbacks.EarlyStopping(
                monitor="val_auc",
                patience=self.config["patience"],
                restore_best_weights=True,
                mode="max",
                verbose=1,
            )
        )

        callback_list.append(
            callbacks.ReduceLROnPlateau(
                monitor="val_auc",
                factor=0.5,
                patience=self.config["patience"] // 2,
                min_lr=1e-6,
                mode="max",
                verbose=1,
            )
        )

        print("Starting training...")
        if class_weights:
            print(f"   Using class weights: {class_weights}")

        fit_args = {
            "x": X_train,
            "y": y_train,
            "batch_size": self.config["batch_size"],
            "epochs": self.config["epochs"],
            "class_weight": class_weights,
            "callbacks": callback_list,
            "verbose": 1,
        }

        if use_validation_split:
            fit_args["validation_split"] = val_size
        else:
            fit_args["validation_data"] = validation_data

        self.history = self.model.fit(**fit_args)

        print("Evaluating on test set...")
        test_metrics = self.model.evaluate(X_test, y_test, verbose=0)
        test_loss, test_accuracy, test_auc, test_precision, test_recall = test_metrics

        y_pred_proba = self.model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)

        from sklearn.metrics import balanced_accuracy_score

        class_report = classification_report(y_test, y_pred, output_dict=True)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)

        results = {
            "test_accuracy": float(test_accuracy),
            "test_balanced_accuracy": float(balanced_acc),
            "test_auc": float(test_auc),
            "test_precision": float(test_precision),
            "test_recall": float(test_recall),
            "test_loss": float(test_loss),
            "classification_report": class_report,
            "total_epochs": len(self.history.history["loss"]),
            "config": self.config,
        }

        print(f"""Training completed.
   Test accuracy: {test_accuracy:.3f} ({test_accuracy:.1%})
   Test balanced accuracy: {balanced_acc:.3f} ({balanced_acc:.1%})
   Test AUC: {test_auc:.3f}
   Test precision: {test_precision:.3f}
   Test recall: {test_recall:.3f}
   Test loss: {test_loss:.4f}
   Total epochs: {results["total_epochs"]}""")

        print("\nClassification Report:")
        for class_name, metrics in class_report.items():
            if isinstance(metrics, dict):
                print(
                    f"   Class {class_name}: Precision={metrics['precision']:.3f}, "
                    f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}"
                )

        if run_dir:
            self._save_confusion_matrix(y_test, y_pred, run_dir)
            self._save_training_plots(run_dir)

        return results, X_test, y_test, y_pred_proba

    def _save_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, run_dir: Path
    ):
        """Save confusion matrix plot."""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["fate_0", "fate_1"],
            yticklabels=["fate_0", "fate_1"],
        )
        plt.title("Expression Classifier - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plot_path = run_dir / "expression_classifier_confusion_matrix.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved confusion matrix to: {plot_path}")

    def _save_training_plots(self, run_dir: Path):
        """Save training history plots."""
        if self.history is None:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Accuracy plot
        axes[0, 0].plot(self.history.history["accuracy"], label="Train")
        axes[0, 0].plot(self.history.history["val_accuracy"], label="Validation")
        axes[0, 0].set_title("Model Accuracy")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Loss plot
        axes[0, 1].plot(self.history.history["loss"], label="Train")
        axes[0, 1].plot(self.history.history["val_loss"], label="Validation")
        axes[0, 1].set_title("Model Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # AUC plot
        axes[1, 0].plot(self.history.history["auc"], label="Train")
        axes[1, 0].plot(self.history.history["val_auc"], label="Validation")
        axes[1, 0].set_title("Model AUC")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("AUC")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Learning rate plot
        if "lr" in self.history.history:
            axes[1, 1].plot(self.history.history["lr"])
            axes[1, 1].set_title("Learning Rate")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Learning Rate")
            axes[1, 1].set_yscale("log")
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "Learning Rate\nNot Available",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )

        plt.tight_layout()

        plot_path = run_dir / "expression_classifier_training_history.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved training plots to: {plot_path}")

    def save(self, filepath: Path):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        save_model(self.model, str(filepath))
        print(f"Saved expression classifier to: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Train neural network expression classifier (Stage 2 of two-stage pipeline)"
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        help="Run directory containing trajectory data (default: use current run)",
    )
    parser.add_argument(
        "--hidden_layers",
        type=str,
        default="512,256,128",
        help="Comma-separated hidden layer sizes (default: 512,256,128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Training batch size (default: 512)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.3, help="Dropout rate (default: 0.3)"
    )
    parser.add_argument(
        "--patience", type=int, default=15, help="Early stopping patience (default: 15)"
    )

    args = parser.parse_args()

    # Get run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = get_run_dir()

    print(f"Using run directory: {run_dir}")

    try:
        train_trajectories = load_trajectory_data(run_dir, split="train")
        print(f"Loaded {len(train_trajectories.trajectories)} training trajectories")
        train_group_ids = log_trajectory_split_info(train_trajectories, "Training")

        val_trajectories = load_trajectory_data(run_dir, split="val")
        print(f"Loaded {len(val_trajectories.trajectories)} validation trajectories")
        val_group_ids = log_trajectory_split_info(val_trajectories, "Validation")

        overlap = train_group_ids & val_group_ids
        if overlap:
            print(
                f"   WARNING: Found {len(overlap)} overlapping group IDs between train/val: {sorted(list(overlap))[:5]}..."
            )
        else:
            print("   No data leakage detected between train/val splits.")

        use_split_data = True
    except FileNotFoundError as e:
        print(f"Split files not found, falling back to single dataset: {e}")
        trajectories = load_trajectory_data(run_dir, split="train")
        print(
            f"Loaded {len(trajectories.trajectories)} trajectories (will split internally)"
        )
        use_split_data = False

    hidden_layers = [int(x.strip()) for x in args.hidden_layers.split(",")]
    config = {
        "hidden_layers": hidden_layers,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
        "patience": args.patience,
        "batch_norm": True,
        "activation": "relu",
        "l2_reg": 1e-5,
    }

    if use_split_data:
        n_genes = len(train_trajectories.trajectories[0].cells[0].expr)
        print(f"Gene expression dimensions: {n_genes}")

        classifier = ExpressionClassifier(n_genes, config)
        classifier.build_model()

        X_train, y_train, class_weights_train = classifier.prepare_classification_data(
            train_trajectories
        )
        X_val, y_val, class_weights_val = classifier.prepare_classification_data(
            val_trajectories
        )

        class_weights = class_weights_train

        results, X_test, y_test, y_pred_proba = classifier.train(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            class_weights=class_weights,
            run_dir=run_dir,
        )
    else:
        n_genes = len(trajectories.trajectories[0].cells[0].expr)
        print(f"Gene expression dimensions: {n_genes}")

        classifier = ExpressionClassifier(n_genes, config)
        classifier.build_model()

        X, y, class_weights = classifier.prepare_classification_data(trajectories)

        results, X_test, y_test, y_pred_proba = classifier.train(
            X, y, class_weights=class_weights, run_dir=run_dir
        )

    model_path = run_dir / "expression_classifier_final.keras"
    classifier.save(model_path)

    results_path = run_dir / "expression_classifier_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved training results to: {results_path}")

    print(f"""\nStage 2 (Expression Classifier) complete.
   Model saved: {model_path}
   Results saved: {results_path}
   Test accuracy: {results["test_accuracy"]:.1%} (vs 72% linear limit)
   Test AUC: {results["test_auc"]:.3f}
   Ready for full pipeline evaluation.""")


if __name__ == "__main__":
    main()
