"""
Autoencoder Training for Reprogramming Dataset (Original Deep Lineage Reproduction)

Uses StandardAutoencoder with input_dropout=0.2 and l2=0 to match
the original Deep Lineage paper architecture.

Reference: original_deep_lineage/Autoencoders/Train_Autoencoders_Reprogramming.py
"""

import argparse
import numpy as np
import h5py
import json
from pathlib import Path
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import callbacks
import pandas as pd

from deep_lineage.schema import AEConfig
from deep_lineage.models.autoencoder import StandardAutoencoder
from scripts.utils import compute_correlations


def load_reprogramming_data(
    dataset_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load reprogramming dataset and flatten time series to get all individual cells.

    The dataset has shape (N, 6, n_genes) for 6 timepoints.
    We flatten to get all non-zero cells for autoencoder training.
    """
    print(f"Loading dataset from {dataset_path}...")

    with h5py.File(dataset_path, "r") as f:
        X_train_ts = np.array(f["X_train"])
        X_val_ts = np.array(f["X_val"])
        X_test_ts = np.array(f["X_test"])

        config = json.loads(f.attrs["config"])
        print(f"Dataset config: {config}")

    print(
        f"Time series shapes: train={X_train_ts.shape}, val={X_val_ts.shape}, test={X_test_ts.shape}"
    )

    def flatten_nonzero_cells(X_ts):
        n_samples, n_timepoints, n_genes = X_ts.shape
        X_flat = X_ts.reshape(-1, n_genes)
        row_sums = np.abs(X_flat).sum(axis=1)
        nonzero_mask = row_sums > 0
        X_nonzero = X_flat[nonzero_mask]
        print(
            f"  Flattened: {X_flat.shape} -> {X_nonzero.shape} (removed {X_flat.shape[0] - X_nonzero.shape[0]} zero cells)"
        )
        return X_nonzero

    X_train = flatten_nonzero_cells(X_train_ts)
    X_val = flatten_nonzero_cells(X_val_ts)
    X_test = flatten_nonzero_cells(X_test_ts)

    return X_train, X_val, X_test


def evaluate_autoencoder(
    autoencoder: StandardAutoencoder, X_test: np.ndarray, output_dir: Path = None
):
    """Evaluate autoencoder reconstruction quality."""
    print("\nEvaluating autoencoder...")
    X_pred = autoencoder.model.predict(X_test, verbose=0)

    mse = np.mean((X_test - X_pred) ** 2)
    pcorr, spcorr = compute_correlations(X_test, X_pred)

    mean_pcorr = np.mean(pcorr)
    mean_spcorr = np.mean(spcorr)

    print("\nTest Results:")
    print(f"  MSE: {mse:.6f}")
    print(f"  Mean Pearson correlation: {mean_pcorr:.4f}")
    print(f"  Mean Spearman correlation: {mean_spcorr:.4f}")

    results = {
        "mse": float(mse),
        "mean_pearson_correlation": float(mean_pcorr),
        "mean_spearman_correlation": float(mean_spcorr),
        "pearson_std": float(np.std(pcorr)),
        "spearman_std": float(np.std(spcorr)),
    }

    if output_dir:
        pcorr.to_csv(output_dir / "pearson_correlations_test.csv")
        spcorr.to_csv(output_dir / "spearman_correlations_test.csv")
        with open(output_dir / "evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train autoencoder for reprogramming dataset (original Deep Lineage reproduction)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="reprogramming_dataset.h5",
        help="Path to reprogramming dataset H5 file",
    )
    parser.add_argument(
        "--hidden_sizes",
        type=str,
        default="500,100",
        help="Comma-separated hidden layer sizes (default: 500,100)",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=75,
        help="Latent dimension (default: 75)",
    )
    parser.add_argument(
        "--input_dropout",
        type=float,
        default=0.2,
        help="Input dropout rate (default: 0.2)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10000,
        help="Maximum epochs (default: 10000)",
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
        default="runs/reprog_ae",
        help="Output directory for models and results",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")

    args = parser.parse_args()

    if args.debug:
        args.epochs = 2
        args.patience = 1
        args.batch_size = 32
        print("DEBUG MODE: epochs=2, patience=1, batch_size=32, single batch of data")

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    hidden_sizes = [int(x.strip()) for x in args.hidden_sizes.split(",")]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "hidden_sizes": hidden_sizes,
        "latent_dim": args.latent_dim,
        "input_dropout": args.input_dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "seed": args.seed,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    X_train, X_val, X_test = load_reprogramming_data(args.dataset)

    if args.debug:
        X_train = X_train[: args.batch_size]
        X_val = X_train
        X_test = X_train

    n_genes = X_train.shape[1]

    # Use StandardAutoencoder with input_dropout and l2=0 (matching original)
    ae_config = AEConfig(
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=1e-3,  # Adam default
        patience=args.patience,
        l2=0,
        input_dropout=args.input_dropout,
    )

    autoencoder = StandardAutoencoder(ae_config)
    autoencoder.build_model(n_genes, hidden_sizes)

    # Setup callbacks matching original training parameters
    callback_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            min_delta=5e-6,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(output_dir / "autoencoder_best.keras"),
            save_best_only=True,
            monitor="val_loss",
            verbose=0,
        ),
    ]

    autoencoder.train(X_train, X_val, callbacks_list=callback_list)

    # Save training history
    hist_df = pd.DataFrame(autoencoder.history.history)
    hist_df.to_csv(output_dir / "training_history.csv", index=False)

    # Evaluate
    evaluate_autoencoder(autoencoder, X_test, output_dir)

    # Save final models
    autoencoder.encoder.save(output_dir / "encoder.keras")
    autoencoder.decoder.save(output_dir / "decoder.keras")
    autoencoder.model.save(output_dir / "autoencoder.keras")

    print(
        f"""
Training complete!
  Output directory: {output_dir}
  Models saved: encoder.keras, decoder.keras, autoencoder.keras

Next step: Train classifier with:
  uv run python scripts/real_training/train_classifier_reprogramming.py --encoder {output_dir}/encoder.keras
"""
    )


if __name__ == "__main__":
    main()
