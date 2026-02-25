#!/usr/bin/env python3
"""Autoregressive LSTM gene expression generator (Stage 1 of two-stage pipeline)."""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks, optimizers
from tensorflow.keras.models import save_model
from tqdm import tqdm

from deep_lineage.schema import TrajectoryList, LSTMConfig
from deep_lineage.metrics import PearsonCorrelation
from scripts.utils import get_run_dir, load_trajectory_data, log_trajectory_split_info


class AutoregressiveExpressionGenerator:
    """
    Autoregressive LSTM model for generating gene expression sequences.

    Architecture:
    - Multi-layer LSTM with layer normalization
    - Linear output layer for gene expression prediction
    - Teacher-forced training with autoregressive inference capability
    """

    def __init__(self, config: LSTMConfig):
        self.config = config
        self.model = None
        self.history = None

    def build_model(self, n_genes: int) -> Model:
        """
        Build autoregressive expression generator model.

        Args:
            n_genes: Number of genes in expression data

        Returns:
            Keras Model for autoregressive expression generation
        """
        print(f"""Building autoregressive expression generator
   Architecture: {self.config.num_layers} layers, {self.config.hidden_dim} hidden units
   Genes: {n_genes}, Dropout: {self.config.dropout}""")

        inputs = layers.Input(shape=(None, n_genes), name="expression_sequence")
        masked = layers.Masking(mask_value=0.0)(inputs)

        l2_reg = (
            tf.keras.regularizers.l2(self.config.l2) if self.config.l2 > 0 else None
        )

        x = masked
        RNN = layers.LSTM if self.config.cell_type == "LSTM" else layers.GRU

        for layer_idx in range(self.config.num_layers - 1):
            x = RNN(
                self.config.hidden_dim,
                return_sequences=True,
                dropout=self.config.dropout,
                kernel_regularizer=l2_reg,
                recurrent_regularizer=l2_reg,
                name=f"lstm_layer_{layer_idx}",
            )(x)
            x = layers.LayerNormalization(name=f"layer_norm_{layer_idx}")(x)

        lstm_out = RNN(
            self.config.hidden_dim,
            return_sequences=True,
            dropout=self.config.dropout,
            kernel_regularizer=l2_reg,
            recurrent_regularizer=l2_reg,
            name="final_lstm",
        )(x)

        lstm_out = layers.LayerNormalization(name="final_layer_norm")(lstm_out)

        outputs = layers.Dense(
            n_genes,
            activation="linear",
            name="expression_output",
            kernel_regularizer=l2_reg,
        )(lstm_out)

        model = Model(
            inputs=inputs, outputs=outputs, name="autoregressive_expression_generator"
        )

        optimizer = optimizers.Adam(
            learning_rate=self.config.learning_rate,
            clipnorm=1.0,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
        )

        model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=[
                "mae",
                "mse",
                PearsonCorrelation(),
                tf.keras.metrics.R2Score(),
                tf.keras.metrics.CosineSimilarity(axis=-1),
            ],
        )
        print("   Using MSE loss with correlation metrics for evaluation")

        print(f"""Model summary:
   Total parameters: {model.count_params():,}
   Optimizer: Adam (lr={self.config.learning_rate})
   Loss: MSE + correlation metrics (Pearson, R², cosine similarity)""")

        self.model = model
        return model

    def prepare_autoregressive_data(
        self, trajectories: TrajectoryList, *, fit_stats: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for autoregressive training (like language modeling).

        Creates sequence-to-sequence pairs:
        - Input: [Day0, Day7] → Target: [Day7, Day14] (shifted sequence)

        This is exactly like language modeling:
        - Input tokens: [Day0_expr, Day7_expr]
        - Target tokens: [Day7_expr, Day14_expr] (same sequence shifted by 1)

        Args:
            trajectories: TrajectoryList with 3-timepoint trajectories

        Returns:
            Tuple of (input_sequences, target_sequences)
        """
        print("Preparing autoregressive training data (language model style)...")

        input_sequences = []
        target_sequences = []

        for traj in tqdm(trajectories.trajectories, desc="Processing trajectories"):
            if len(traj.cells) != 3:
                continue  # Skip non-3-timepoint trajectories

            day0_expr = traj.cells[0].expr  # Day0 expression
            day7_expr = traj.cells[1].expr  # Day7 expression
            day14_expr = traj.cells[2].expr  # Day14 expression

            # Language model style: input sequence → target sequence (shifted)
            input_seq = [day0_expr, day7_expr]  # Input: [Day0, Day7]
            target_seq = [day7_expr, day14_expr]  # Target: [Day7, Day14] (shifted by 1)

            input_sequences.append(input_seq)
            target_sequences.append(target_seq)

        print(f"""   Generated {len(input_sequences)} sequence pairs
   Each sequence length: 2 timesteps
   Training pattern: [Day0, Day7] -> [Day7, Day14] (autoregressive)""")

        n_genes = len(trajectories.trajectories[0].cells[0].expr)
        seq_len = 2  # Always 2 timesteps for our trajectory data

        # Create arrays
        X = np.zeros((len(input_sequences), seq_len, n_genes), dtype=np.float32)
        y = np.zeros((len(target_sequences), seq_len, n_genes), dtype=np.float32)

        for i, (input_seq, target_seq) in enumerate(
            zip(input_sequences, target_sequences)
        ):
            for t, (inp_expr, tgt_expr) in enumerate(zip(input_seq, target_seq)):
                X[i, t] = inp_expr
                y[i, t] = tgt_expr

        print(f"""   Input shape: {X.shape} [batch, seq_len, n_genes]
   Target shape: {y.shape} [batch, seq_len, n_genes]
   Model will learn: X[t] -> y[t] for each timestep t""")

        # Fit stats on TRAIN only; reuse for VAL/TEST
        if fit_stats or not hasattr(self, "normalization_params"):
            print("   CALCULATING normalization parameters (global z-score)")

            X_mean = np.mean(X).astype(np.float32)
            X_std = np.std(X).astype(np.float32)
            y_mean = np.mean(y).astype(np.float32)
            y_std = np.std(y).astype(np.float32)

            X_std = np.maximum(X_std, 1e-3)
            y_std = np.maximum(y_std, 1e-3)

            self.normalization_params = {
                "X_mean": X_mean,
                "X_std": X_std,
                "y_mean": y_mean,
                "y_std": y_std,
            }

            print(f"""      X_mean: {X_mean:.6f}, X_std: {X_std:.6f}
      y_mean: {y_mean:.6f}, y_std: {y_std:.6f}""")

        else:
            print("   REUSING stored normalization parameters (global)")

            X_mean = self.normalization_params["X_mean"]
            X_std = self.normalization_params["X_std"]
            y_mean = self.normalization_params["y_mean"]
            y_std = self.normalization_params["y_std"]

            print(f"""      X_mean: {X_mean:.6f}, X_std: {X_std:.6f}
      y_mean: {y_mean:.6f}, y_std: {y_std:.6f}""")

        X_normalized = (X - X_mean) / X_std
        y_normalized = (y - y_mean) / y_std

        if not np.all(np.isfinite(X_normalized)) or not np.all(
            np.isfinite(y_normalized)
        ):
            raise ValueError("Non-finite values after normalization (check std floor).")

        print(f"""      Global z-score normalization completed
      X_normalized: mean={np.mean(X_normalized):.6f}, std={np.std(X_normalized):.6f}
      y_normalized: mean={np.mean(y_normalized):.6f}, std={np.std(y_normalized):.6f}""")

        if abs(np.std(X_normalized) - 1.0) > 0.1:
            print(
                f"      WARNING: X_normalized std ({np.std(X_normalized):.3f}) far from 1.0"
            )
        if abs(np.std(y_normalized) - 1.0) > 0.1:
            print(
                f"      WARNING: y_normalized std ({np.std(y_normalized):.3f}) far from 1.0"
            )

        return X_normalized, y_normalized

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        validation_split: float = 0.2,
        run_dir: Path = None,
    ) -> Dict[str, Any]:
        """
        Train the autoregressive expression generator.

        Args:
            X_train: Training input sequences [batch, max_seq_len, n_genes]
            y_train: Training target sequences [batch, max_seq_len, n_genes]
            X_val: Validation input sequences [batch, max_seq_len, n_genes] (optional)
            y_val: Validation target sequences [batch, max_seq_len, n_genes] (optional)
            validation_split: Fraction of data for validation (used if X_val/y_val not provided)
            run_dir: Directory for saving results

        Returns:
            Dictionary with training metrics and results
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        print("Training autoregressive expression generator...")

        if X_val is not None and y_val is not None:
            print("   Using provided train/validation splits")
            print(f"   Train samples: {len(X_train)}")
            print(f"   Validation samples: {len(X_val)}")
            validation_data = (X_val, y_val)
            use_validation_split = False
        else:
            print("   Using internal validation split")
            print(f"   Training samples: {len(X_train)}")
            print(f"   Validation split: {validation_split:.1%}")
            validation_data = None
            use_validation_split = True

        print(f"   Epochs: {self.config.epochs}, Batch size: {self.config.batch_size}")

        callback_list = []

        if run_dir:
            tensorboard_dir = run_dir / "tensorboard" / "autoregressive_generator"
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

            checkpoint_path = run_dir / "autoregressive_generator_best.keras"
            callback_list.append(
                callbacks.ModelCheckpoint(
                    str(checkpoint_path),
                    monitor="val_loss",
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1,
                )
            )

        callback_list.append(
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.patience,
                restore_best_weights=True,
                verbose=1,
            )
        )

        callback_list.append(
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.9,
                patience=self.config.patience // 2,
                min_lr=1e-7,
                verbose=1,
            )
        )

        def cosine_decay_schedule(epoch, lr):
            if epoch < 50:  # Warmup phase
                return lr
            else:
                # Cosine decay after warmup
                decay_steps = self.config.epochs - 50
                cosine_decay = 0.5 * (1 + np.cos(np.pi * (epoch - 50) / decay_steps))
                return (
                    self.config.learning_rate * 0.1
                    + (self.config.learning_rate - self.config.learning_rate * 0.1)
                    * cosine_decay
                )

        callback_list.append(
            callbacks.LearningRateScheduler(cosine_decay_schedule, verbose=0)
        )

        print("Starting training...")

        fit_args = {
            "x": X_train,
            "y": y_train,
            "batch_size": self.config.batch_size,
            "epochs": self.config.epochs,
            "callbacks": callback_list,
            "verbose": 1,
        }

        if use_validation_split:
            fit_args["validation_split"] = validation_split
        else:
            fit_args["validation_data"] = validation_data

        self.history = self.model.fit(**fit_args)

        final_train_loss = self.history.history["loss"][-1]
        final_val_loss = self.history.history["val_loss"][-1]
        final_train_mae = self.history.history["mae"][-1]
        final_val_mae = self.history.history["val_mae"][-1]

        final_train_pearson = self.history.history["pearson_correlation"][-1]
        final_val_pearson = self.history.history["val_pearson_correlation"][-1]
        final_train_r2 = self.history.history["r2_score"][-1]
        final_val_r2 = self.history.history["val_r2_score"][-1]
        final_train_cosine = self.history.history["cosine_similarity"][-1]
        final_val_cosine = self.history.history["val_cosine_similarity"][-1]

        results = {
            "final_train_loss": float(final_train_loss),
            "final_val_loss": float(final_val_loss),
            "final_train_mae": float(final_train_mae),
            "final_val_mae": float(final_val_mae),
            "final_train_pearson": float(final_train_pearson),
            "final_val_pearson": float(final_val_pearson),
            "final_train_r2": float(final_train_r2),
            "final_val_r2": float(final_val_r2),
            "final_train_cosine": float(final_train_cosine),
            "final_val_cosine": float(final_val_cosine),
            "total_epochs": len(self.history.history["loss"]),
            "config": self.config.model_dump(),
        }

        print(f"""Training completed.
   Final train loss: {final_train_loss:.4f}
   Final val loss (MSE): {final_val_loss:.4f}
   Final train MAE: {final_train_mae:.4f}
   Final val MAE: {final_val_mae:.4f}
   Correlation Metrics:
      Train Pearson: {final_train_pearson:.4f} | Val Pearson: {final_val_pearson:.4f}
      Train R²: {final_train_r2:.4f} | Val R²: {final_val_r2:.4f}
      Train Cosine: {final_train_cosine:.4f} | Val Cosine: {final_val_cosine:.4f}
   Total epochs: {results["total_epochs"]}""")

        return results

    def predict_autoregressive(
        self, initial_expressions: np.ndarray, n_steps: int = 2
    ) -> np.ndarray:
        """
        Generate expressions autoregressively (like language model inference).

        Args:
            initial_expressions: Starting expressions [batch, n_genes]
            n_steps: Number of autoregressive steps to generate

        Returns:
            Full sequence including initial + generated [batch, n_steps+1, n_genes]
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        sequence = initial_expressions[:, np.newaxis, :]  # [batch, 1, n_genes]

        for step in range(n_steps):
            if sequence.shape[1] == 1:
                model_input = sequence
            else:
                model_input = sequence[:, -2:, :]

            predictions = self.model(model_input)
            next_expr = predictions[:, -1:, :]
            sequence = tf.concat([sequence, next_expr], axis=1)

        return sequence.numpy()

    def save(self, filepath: Path):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        save_model(self.model, str(filepath))
        print(f"Saved autoregressive generator to: {filepath}")

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive evaluation with detailed correlation analysis.

        Args:
            X: Input sequences [batch, seq_len, n_genes]
            y: Target sequences [batch, seq_len, n_genes]

        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        print("Running comprehensive evaluation...")

        y_pred = self.model.predict(X, verbose=0)

        from deep_lineage.metrics import (
            compute_correlation_metrics,
            per_gene_correlation_analysis,
        )

        global_metrics = compute_correlation_metrics(y, y_pred)
        gene_analysis = per_gene_correlation_analysis(y, y_pred)

        mse = np.mean((y - y_pred) ** 2)
        mae = np.mean(np.abs(y - y_pred))
        rmse = np.sqrt(mse)

        evaluation_results = {
            # Basic metrics
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            # Global correlation metrics
            "global_pearson_correlation": global_metrics["pearson_correlation"],
            "global_spearman_correlation": global_metrics["spearman_correlation"],
            "global_r2_score": global_metrics["r2_score"],
            "global_cosine_similarity": global_metrics["cosine_similarity"],
            # Per-gene analysis
            "mean_gene_correlation": gene_analysis["mean_gene_correlation"],
            "median_gene_correlation": gene_analysis["median_gene_correlation"],
            "std_gene_correlation": gene_analysis["std_gene_correlation"],
            "genes_with_positive_correlation": gene_analysis[
                "genes_with_positive_correlation"
            ],
            "genes_with_strong_correlation": gene_analysis[
                "genes_with_strong_correlation"
            ],
            "total_genes": gene_analysis["total_genes"],
            "fraction_positive_genes": gene_analysis["genes_with_positive_correlation"]
            / gene_analysis["total_genes"],
            "fraction_strong_genes": gene_analysis["genes_with_strong_correlation"]
            / gene_analysis["total_genes"],
        }

        print(f"""Evaluation Results:
   MSE: {evaluation_results["mse"]:.4f}
   RMSE: {evaluation_results["rmse"]:.4f}
   Global Pearson: {evaluation_results["global_pearson_correlation"]:.4f}
   Global R²: {evaluation_results["global_r2_score"]:.4f}
   Mean Gene Correlation: {evaluation_results["mean_gene_correlation"]:.4f}
   Genes with Positive Correlation: {evaluation_results["genes_with_positive_correlation"]}/{evaluation_results["total_genes"]} ({evaluation_results["fraction_positive_genes"]:.1%})
   Genes with Strong Correlation (>0.5): {evaluation_results["genes_with_strong_correlation"]}/{evaluation_results["total_genes"]} ({evaluation_results["fraction_strong_genes"]:.1%})""")

        return evaluation_results

    def save_normalization_params(self, run_dir: Path):
        """Save normalization parameters for inference."""
        if not hasattr(self, "normalization_params"):
            print("No normalization parameters found. Skipping save.")
            return

        import json

        norm_path = run_dir / "autoregressive_normalization_params.json"

        params_json = {}
        for key, value in self.normalization_params.items():
            if hasattr(value, "tolist"):
                params_json[key] = value.tolist()
            else:
                params_json[key] = value

        with open(norm_path, "w") as f:
            json.dump(params_json, f, indent=2)

        print(f"Saved normalization parameters to: {norm_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train autoregressive gene expression generator (Stage 1 of two-stage pipeline)"
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        help="Run directory containing trajectory data (default: use current run)",
    )
    parser.add_argument(
        "--num_layers", type=int, default=3, help="Number of LSTM layers (default: 3)"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension of LSTM layers (default: 256)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs (default: 200)",
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
        "--patience", type=int, default=25, help="Early stopping patience (default: 25)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for augmentation"
    )

    args = parser.parse_args()

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

    config = LSTMConfig(
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        patience=args.patience,
    )

    generator = AutoregressiveExpressionGenerator(config)

    if use_split_data:
        n_genes = len(train_trajectories.trajectories[0].cells[0].expr)
        print(f"Gene expression dimensions: {n_genes}")

        generator.build_model(n_genes)

        X_train, y_train = generator.prepare_autoregressive_data(
            train_trajectories, fit_stats=True
        )
        X_val, y_val = generator.prepare_autoregressive_data(
            val_trajectories, fit_stats=False
        )

        results = generator.train(
            X_train, y_train, X_val=X_val, y_val=y_val, run_dir=run_dir
        )

        print("\nRunning post-training evaluation on validation set...")
        val_evaluation = generator.evaluate(X_val, y_val)
        results.update({"validation_evaluation": val_evaluation})
    else:
        n_genes = len(trajectories.trajectories[0].cells[0].expr)
        print(f"Gene expression dimensions: {n_genes}")

        generator.build_model(n_genes)

        X, y = generator.prepare_autoregressive_data(trajectories, fit_stats=True)

        results = generator.train(X, y, run_dir=run_dir)

        print("\nRunning post-training evaluation on sample data...")
        sample_size = min(1000, len(X))  # Evaluate on up to 1000 samples
        eval_indices = np.random.choice(len(X), sample_size, replace=False)
        sample_evaluation = generator.evaluate(X[eval_indices], y[eval_indices])
        results.update({"sample_evaluation": sample_evaluation})

    model_path = run_dir / "autoregressive_generator_final.keras"
    generator.save(model_path)

    generator.save_normalization_params(run_dir)

    results_path = run_dir / "autoregressive_generator_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved training results to: {results_path}")

    print(f"""\nStage 1 (Autoregressive Generator) complete.
   Model saved: {model_path}
   Results saved: {results_path}
   Ready for Stage 2: Neural network expression classifier""")


if __name__ == "__main__":
    main()
