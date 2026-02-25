"""
Regressor Training for Reprogramming Dataset (Original Deep Lineage Reproduction)

This script trains a bidirectional LSTM/GRU regressor to predict gene expression at Day28
from earlier timepoints.

IMPORTANT: The regressor works in GENE SPACE, not latent space!
- Input: First N timepoints gene expression (default: Days 6,9,12,15 = 4 timepoints)
- Output: Day28 gene expression (n_genes)

The --input_days argument controls how many input timepoints to use (default: 4).

Reference: original_deep_lineage/Regression_Model_Training_On_Reprogramming/
           Train_Regression_On_Reprogramming_Time_Series_Two_Classes_Day28.py
"""

import argparse
import numpy as np
import h5py
import json
from pathlib import Path
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
import pandas as pd

from scripts.utils import compute_correlations


def load_reprogramming_data_for_regression(
    dataset_path: str,
    target_day_idx: int = 5,
    input_days: int = None,
) -> Tuple[np.ndarray, ...]:
    """
    Load reprogramming dataset and prepare for regression.

    Splits the time series:
    - Input: first `input_days` timepoints (e.g., Days 6,9,12,15 if input_days=4)
    - Target: timepoint at target_day_idx (e.g., Day28)

    Args:
        dataset_path: Path to H5 dataset
        target_day_idx: Index of target day (default 5 = Day28)
        input_days: Number of input timepoints (default None = use all before target)

    Returns:
        X_train, X_val, X_test: Input time series (N, input_days, n_genes)
        y_train, y_val, y_test: Target expression (N, n_genes)
    """
    print(f"Loading dataset from {dataset_path}...")

    with h5py.File(dataset_path, "r") as f:
        X_train_full = np.array(f["X_train"])
        X_val_full = np.array(f["X_val"])
        X_test_full = np.array(f["X_test"])

        config = json.loads(f.attrs["config"])
        timepoints = config.get(
            "timepoints", ["Day6", "Day9", "Day12", "Day15", "Day21", "Day28"]
        )

    print(
        f"Full shapes: train={X_train_full.shape}, val={X_val_full.shape}, test={X_test_full.shape}"
    )
    print(f"Timepoints: {timepoints}")
    print(f"Target day index: {target_day_idx} ({timepoints[target_day_idx]})")

    # Determine number of input days
    if input_days is None:
        input_days = target_day_idx  # Use all days before target
    else:
        input_days = min(
            input_days, target_day_idx
        )  # Can't use more days than available

    input_timepoint_names = timepoints[:input_days]
    print(f"Input timepoints: {input_timepoint_names} ({input_days} days)")

    # Split: input = first input_days, target = day at target_day_idx
    def split_input_target(X_full, n_input, target_idx):
        X_input = X_full[:, :n_input, :].copy()
        y_target = X_full[:, target_idx, :].copy()
        return X_input, y_target

    X_train, y_train = split_input_target(X_train_full, input_days, target_day_idx)
    X_val, y_val = split_input_target(X_val_full, input_days, target_day_idx)
    X_test, y_test = split_input_target(X_test_full, input_days, target_day_idx)

    print("\nRegression data shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_regressor(
    timesteps: int,
    features: int,
    output_features: int,
    num_layers: int = 4,
    dropout: float = 0.25,
    model_type: str = "LSTM",
) -> Model:
    """
    Build bidirectional RNN regressor matching original architecture.

    Architecture (from original):
        Masking(mask_value=0.0)
        (N-1) × Bidirectional(LSTM/GRU, return_sequences=True, merge_mode='ave')
        Bidirectional(LSTM/GRU, return_sequences=False, merge_mode='ave')
        Dense(output_features)  # linear activation for gene expression

    Args:
        timesteps: Number of input timepoints (5 for Days 6-21)
        features: Input feature dimension (n_genes)
        output_features: Output dimension (n_genes)
        num_layers: Number of RNN layers (default 4)
        dropout: Dropout rate (default 0.25)
        model_type: "LSTM" or "GRU"

    Returns:
        Compiled regressor model
    """
    print("\nBuilding regressor:")
    print(f"  Timesteps: {timesteps}")
    print(f"  Features: {features}")
    print(f"  Output features: {output_features}")
    print(f"  Num layers: {num_layers}")
    print(f"  Dropout: {dropout}")
    print(f"  Model type: {model_type}")

    RNNCell = layers.LSTM if model_type == "LSTM" else layers.GRU

    inputs = layers.Input(shape=(timesteps, features), name="gene_expression_input")

    # Masking for zero-padded timepoints
    x = layers.Masking(mask_value=0.0, name="masking")(inputs)

    # Bidirectional RNN layers (N-1 with return_sequences=True)
    for i in range(num_layers - 1):
        x = layers.Bidirectional(
            RNNCell(features, return_sequences=True, dropout=dropout),
            merge_mode="ave",
            name=f"birnn_{i + 1}",
        )(x)

    # Final RNN layer (return_sequences=False)
    x = layers.Bidirectional(
        RNNCell(features, return_sequences=False, dropout=dropout),
        merge_mode="ave",
        name=f"birnn_{num_layers}",
    )(x)

    # Output layer (linear activation for gene expression)
    outputs = layers.Dense(output_features, name="predicted_expression")(x)

    model = Model(inputs, outputs, name="regressor")

    model.compile(loss="mse", optimizer="adam", metrics=["mse"])

    print(f"  Total params: {model.count_params():,}")

    return model


def train_regressor(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100000,
    batch_size: int = 1024,
    patience: int = 100,
    output_dir: Path = None,
):
    """Train regressor with original parameters."""
    print("\nTraining regressor:")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Validation samples: {len(X_val):,}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Patience: {patience}")

    callback_list = []

    # Early stopping (note: min_delta=5e-5 for regression, different from classifier)
    es = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        min_delta=5e-5,
        restore_best_weights=True,
        verbose=1,
    )
    callback_list.append(es)

    # Model checkpoint
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        cp = callbacks.ModelCheckpoint(
            filepath=str(output_dir / "regressor_best.keras"),
            save_best_only=True,
            monitor="val_loss",
            verbose=0,
        )
        callback_list.append(cp)

    # Train
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callback_list,
        shuffle=True,
        verbose=1,
    )

    return history


def evaluate_regressor(
    model: Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path = None,
):
    """Evaluate regressor and compute metrics."""
    print("\nEvaluating regressor...")

    # Predictions
    y_pred = model.predict(X_test, verbose=0)

    # MSE
    mse = np.mean((y_test - y_pred) ** 2)

    # Per-sample correlations (matching original)
    pcorr, spcorr = compute_correlations(y_test, y_pred)

    mean_pcorr = np.mean(pcorr)
    mean_spcorr = np.mean(spcorr)

    print("\nTest Results:")
    print(f"  MSE: {mse:.6f}")
    print(f"  Mean Pearson correlation: {mean_pcorr:.4f}")
    print(f"  Mean Spearman correlation: {mean_spcorr:.4f}")
    print(f"  Pearson std: {np.std(pcorr):.4f}")
    print(f"  Spearman std: {np.std(spcorr):.4f}")

    results = {
        "mse": float(mse),
        "mean_pearson_correlation": float(mean_pcorr),
        "mean_spearman_correlation": float(mean_spcorr),
        "pearson_std": float(np.std(pcorr)),
        "spearman_std": float(np.std(spcorr)),
    }

    if output_dir:
        with open(output_dir / "regression_results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Save correlation distributions
        pcorr.to_csv(output_dir / "pearson_correlations_test.csv")
        spcorr.to_csv(output_dir / "spearman_correlations_test.csv")

        # Save predictions
        pd.DataFrame(y_pred).to_csv(output_dir / "predictions_test.csv", index=False)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train regressor for reprogramming dataset (original Deep Lineage reproduction)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="reprogramming_dataset.h5",
        help="Path to reprogramming dataset H5 file",
    )
    parser.add_argument(
        "--target_day",
        type=int,
        default=28,
        help="Target day to predict (default: 28)",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=4,
        help="Number of bidirectional RNN layers (default: 4)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.25,
        help="Dropout rate (default: 0.25)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="LSTM",
        choices=["LSTM", "GRU"],
        help="RNN cell type (default: LSTM)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100000,
        help="Maximum epochs (default: 100000)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size (default: 1024)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Early stopping patience (default: 100)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/reprog_reg",
        help="Output directory for models and results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--input_days",
        type=int,
        default=4,
        help="Number of input timepoints to use (default: 4 = Days 6,9,12,15)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with single batch",
    )

    args = parser.parse_args()

    if args.debug:
        args.epochs = 2
        args.patience = 1
        args.batch_size = 32
        print("DEBUG MODE: epochs=2, patience=1, batch_size=32, single batch of data")

    # Map target day to index
    day_to_idx = {6: 0, 9: 1, 12: 2, 15: 3, 21: 4, 28: 5}
    if args.target_day not in day_to_idx:
        raise ValueError(
            f"Invalid target_day {args.target_day}. Must be one of {list(day_to_idx.keys())}"
        )
    target_day_idx = day_to_idx[args.target_day]

    # Set seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = {
        "target_day": args.target_day,
        "target_day_idx": target_day_idx,
        "input_days": args.input_days,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "model_type": args.model_type,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "seed": args.seed,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = (
        load_reprogramming_data_for_regression(
            args.dataset, target_day_idx, input_days=args.input_days
        )
    )

    if args.debug:
        X_train, y_train = X_train[: args.batch_size], y_train[: args.batch_size]
        X_val, y_val = X_train, y_train  # Same as train for overfit check
        X_test, y_test = X_train, y_train

    # Build regressor
    timesteps = X_train.shape[1]
    features = X_train.shape[2]
    output_features = y_train.shape[1]

    model = build_regressor(
        timesteps=timesteps,
        features=features,
        output_features=output_features,
        num_layers=args.num_layers,
        dropout=args.dropout,
        model_type=args.model_type,
    )

    # Train
    history = train_regressor(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        output_dir=output_dir,
    )

    # Save training history
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(output_dir / "training_history.csv", index=False)

    # Evaluate
    results = evaluate_regressor(model, X_test, y_test, output_dir)

    # Save final model
    model.save(output_dir / "regressor.keras")

    print(
        f"""
Training complete!
  Output directory: {output_dir}
  Mean Pearson correlation: {results["mean_pearson_correlation"]:.4f}
  Mean Spearman correlation: {results["mean_spearman_correlation"]:.4f}
  MSE: {results["mse"]:.6f}
  Model saved: regressor.keras

Next step: Run evaluation script:
  uv run python scripts/evaluate_reprogramming.py
"""
    )


if __name__ == "__main__":
    main()
