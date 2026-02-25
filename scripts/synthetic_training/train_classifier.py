"""Train LSTM classifiers for cell fate prediction."""

import argparse
import json
import numpy as np
from pathlib import Path
import tensorflow as tf

from deep_lineage.schema import LSTMConfig
from deep_lineage.models.classifier import Classifier, _infer_dims
from deep_lineage.utils import make_json_serializable
from scripts.utils import get_run_dir, load_trajectory_data


def main():
    parser = argparse.ArgumentParser(description="Train Deep Lineage classifier")
    parser.add_argument(
        "--run_dir",
        type=str,
        help="Run directory containing trajectory data and encoder (default: use current run)",
    )
    parser.add_argument(
        "--timepoints",
        type=str,
        required=True,
        help="Comma-separated timepoints to use (e.g., 't0', 't0,t1', 't0,t1,t2')",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name for this classifier (e.g., 't0_only', 't0_t1', 't0,t1,t2')",
    )
    parser.add_argument(
        "--ae_name",
        type=str,
        default=None,
        help="Optional autoencoder name suffix (e.g., 'dropout_30pct_ae').",
    )
    parser.add_argument(
        "--train_trajectories",
        type=str,
        default=None,
        help="Optional custom path to training trajectory file (overrides default lookup)",
    )
    parser.add_argument(
        "--val_trajectories",
        type=str,
        default=None,
        help="Optional custom path to validation trajectory file (overrides default lookup)",
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of LSTM layers (default: 2)"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Hidden dimension (default: 256)"
    )
    parser.add_argument(
        "--epochs", type=int, default=300, help="Number of epochs (default: 300)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="Batch size (default: 1024)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.25, help="Dropout rate (default: 0.25)"
    )
    parser.add_argument(
        "--patience", type=int, default=25, help="Early stopping patience (default: 25)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    timepoints = [tp.strip() for tp in args.timepoints.split(",")]
    valid_timepoints = {"t0", "t1", "t2"}
    if not all(tp in valid_timepoints for tp in timepoints):
        raise ValueError(
            f"Invalid timepoints: {timepoints}. Valid options: {valid_timepoints}"
        )

    run_dir = Path(args.run_dir) if args.run_dir else get_run_dir()
    print(f"Training classifier: {args.name}")
    print(f"   Timepoints: {timepoints}")
    print(f"Using run directory: {run_dir}")

    try:
        train_trajectories = load_trajectory_data(
            run_dir, split="train", custom_path=args.train_trajectories
        )
        print(f"Loaded {len(train_trajectories.trajectories)} training trajectories")

        val_trajectories = load_trajectory_data(
            run_dir, split="val", custom_path=args.val_trajectories
        )
        print(f"Loaded {len(val_trajectories.trajectories)} validation trajectories")

    except FileNotFoundError as e:
        print(f"Error loading trajectory data: {e}")
        return

    config = LSTMConfig(
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        patience=args.patience,
    )

    classifier = Classifier(config, timepoints, args.name, ae_name=args.ae_name)

    classifier.load_encoder(run_dir)

    print("\nPreparing classification data...")
    train_ds = classifier.prepare_data(
        train_trajectories, timepoints, config.batch_size
    )
    val_ds = classifier.prepare_data(val_trajectories, timepoints, config.batch_size)

    n_timesteps, n_genes, n_classes = _infer_dims(train_trajectories, timepoints)
    print(f"Model input dimensions: {n_timesteps} timesteps x {n_genes} genes")

    classifier.build_model(n_timesteps, n_genes, n_classes)

    results = classifier.train(train_ds, val_ds, run_dir=run_dir)

    model_path = run_dir / f"classifier_ae_{args.name}_final.keras"
    classifier.save(model_path)

    results_path = run_dir / f"classifier_ae_{args.name}_results.json"
    json_results = make_json_serializable(results)
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"Saved training results to: {results_path}")

    print(
        f"""\nClassifier Training Complete.
   Name: {args.name}
   Timepoints: {timepoints}
   Final validation accuracy: {results["final_val_accuracy"]:.4f} ({results["final_val_accuracy"] * 100:.1f}%)
   Model saved: {model_path}
   Results saved: {results_path}"""
    )


if __name__ == "__main__":
    main()
