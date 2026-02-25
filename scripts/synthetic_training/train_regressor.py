"""Train LSTM regressors to predict missing timepoints."""

import argparse
import json
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import callbacks

from deep_lineage.schema import LSTMConfig
from deep_lineage.models.regressor import Regressor
from deep_lineage.utils import (
    make_json_serializable,
    evaluate_gene_space,
    collect_predictions_from_dataset,
)
from scripts.utils import get_run_dir, load_trajectory_data


class GeneSpaceEvaluator(callbacks.Callback):
    """Periodically evaluate predictions in gene expression space."""

    def __init__(
        self,
        decoder_model,
        encoder_model,
        val_dataset,
        val_samples,
        input_timepoints,
        target_timepoint,
        eval_frequency=10,
    ):
        super().__init__()
        self.decoder_model = decoder_model
        self.encoder_model = encoder_model
        self.val_dataset = val_dataset
        self.val_samples = val_samples
        self.input_timepoints = input_timepoints
        self.target_timepoint = target_timepoint
        self.eval_frequency = eval_frequency

        self.timepoint_map = {"t0": 0, "t1": 1, "t2": 2}
        self.input_indices = [self.timepoint_map[tp] for tp in input_timepoints]
        self.target_idx = self.timepoint_map[target_timepoint]

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.eval_frequency != 0:
            return

        y_true_latent, y_pred_latent = collect_predictions_from_dataset(
            self.model, self.val_dataset, verbose=False
        )

        results = evaluate_gene_space(
            y_true_latent,
            y_pred_latent,
            self.decoder_model,
            verbose=False,
            title=f"GENE SPACE Evaluation (Epoch {epoch + 1})",
        )

        gene_space = results["gene_space"]
        per_gene = results["per_gene"]
        r2_corr = gene_space["pearson"] ** 2

        print(
            f"""
GENE SPACE Evaluation (Epoch {epoch + 1}):
   Overall: Pearson={gene_space["pearson"]:.4f} | r²={r2_corr:.4f} | MSE={gene_space["mse"]:.4f} | Cosine={gene_space["cosine"]:.4f}
   Per-gene: Mean corr={per_gene["mean_correlation"]:.4f} +/- {per_gene["std_correlation"]:.4f}
   Well-predicted genes (r>0.5): {per_gene["well_predicted"]}/{per_gene["total_genes"]} ({100 * per_gene["well_predicted"] / per_gene["total_genes"]:.1f}%)"""
        )


class ClearMetricsLogger(callbacks.Callback):
    """Custom callback for clear metric logging during training."""

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return

        print(f"\n{'=' * 80}")

        lr = float(self.model.optimizer.learning_rate)
        lr_str = f"{lr:.2e}"

        print(f"Epoch {epoch + 1}/{self.params.get('epochs', '?')} - LR: {lr_str}")
        print(f"{'-' * 80}")

        print("LATENT SPACE Metrics:")

        train_mse = logs.get("loss", 0)
        train_mae = logs.get("mae", 0)
        train_pearson = logs.get("pearson_correlation", 0)
        train_r2 = train_pearson**2
        train_cosine = logs.get("cosine_similarity", 0)

        print(
            f"   Train: MSE={train_mse:.4f} | MAE={train_mae:.4f} | "
            f"Pearson={train_pearson:.4f} | R²={train_r2:.4f} | "
            f"Cosine={train_cosine:.4f}"
        )

        val_mse = logs.get("val_loss", 0)
        val_mae = logs.get("val_mae", 0)
        val_pearson = logs.get("val_pearson_correlation", 0)
        val_r2 = val_pearson**2
        val_cosine = logs.get("val_cosine_similarity", 0)

        print(
            f"   Valid: MSE={val_mse:.4f} | MAE={val_mae:.4f} | "
            f"Pearson={val_pearson:.4f} | R²={val_r2:.4f} | "
            f"Cosine={val_cosine:.4f}"
        )

        if epoch > 0:
            if val_mse < getattr(self, "best_val_loss", float("inf")):
                print("   Validation loss improved.")
                self.best_val_loss = val_mse
        else:
            self.best_val_loss = val_mse

        print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(description="Train Deep Lineage regressor")
    parser.add_argument(
        "--run_dir",
        type=str,
        help="Run directory containing trajectory data and encoder/decoder (default: use current run)",
    )
    parser.add_argument(
        "--input_timepoints",
        type=str,
        required=True,
        help="Comma-separated input timepoints (e.g., 't0,t2' or 't0,t1')",
    )
    parser.add_argument(
        "--target_timepoint",
        type=str,
        required=True,
        help="Target timepoint to predict (e.g., 't1' or 't2')",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name for this regressor (e.g., 'future', 'intermediate')",
    )
    parser.add_argument(
        "--ae_name",
        type=str,
        default=None,
        help="Optional autoencoder name suffix.",
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

    input_timepoints = [tp.strip() for tp in args.input_timepoints.split(",")]
    target_timepoint = args.target_timepoint.strip()

    valid_timepoints = {"t0", "t1", "t2"}
    if not all(tp in valid_timepoints for tp in input_timepoints):
        raise ValueError(
            f"Invalid input timepoints: {input_timepoints}. Valid options: {valid_timepoints}"
        )
    if target_timepoint not in valid_timepoints:
        raise ValueError(
            f"Invalid target timepoint: {target_timepoint}. Valid options: {valid_timepoints}"
        )
    if target_timepoint in input_timepoints:
        raise ValueError(
            f"Target timepoint {target_timepoint} cannot be in input timepoints {input_timepoints}"
        )

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = get_run_dir()

    print(f"Training regressor: {args.name}")
    print(f"   Input timepoints: {input_timepoints}")
    print(f"   Target timepoint: {target_timepoint}")
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

    regressor = Regressor(
        config,
        input_timepoints,
        target_timepoint,
        args.name,
        ae_name=args.ae_name,
    )

    regressor.load_encoder_decoder(run_dir)

    n_timesteps = 3
    n_genes = train_trajectories.trajectories[0].cells[0].expr.shape[0]

    print(f"Model dimensions: {n_timesteps} timesteps x {n_genes} genes (raw input)")

    regressor.build_model(n_timesteps, n_genes)

    print("\nCreating streaming datasets...")
    train_dataset, train_samples = regressor.prepare_regression_data(
        train_trajectories, input_timepoints, target_timepoint, config.batch_size
    )
    val_dataset, val_samples = regressor.prepare_regression_data(
        val_trajectories, input_timepoints, target_timepoint, config.batch_size
    )

    extra_callbacks = [ClearMetricsLogger()]
    if regressor.decoder_model is not None:
        extra_callbacks.append(
            GeneSpaceEvaluator(
                decoder_model=regressor.decoder_model,
                encoder_model=regressor.encoder_model,
                val_dataset=val_dataset,
                val_samples=val_samples,
                input_timepoints=input_timepoints,
                target_timepoint=target_timepoint,
                eval_frequency=10,
            )
        )

    results = regressor.train(
        train_dataset,
        val_dataset,
        val_samples,
        run_dir=run_dir,
        extra_callbacks=extra_callbacks,
    )

    gene_space_results = regressor.evaluate_in_gene_space(val_dataset)
    results["gene_space_evaluation"] = gene_space_results

    model_path = run_dir / f"regressor_ae_{args.name}_final.keras"
    regressor.save(model_path)

    results_path = run_dir / f"regressor_ae_{args.name}_results.json"
    json_results = make_json_serializable(results)
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"Saved training results to: {results_path}")

    print(
        f"""\nRegressor Training Complete.
   Name: {args.name}
   Input timepoints: {input_timepoints}
   Target timepoint: {target_timepoint}

   LATENT SPACE (Final validation):
      Pearson: {results["final_val_pearson"]:.4f} | r2: {results["final_val_r2"]:.4f}"""
    )
    if "gene_space_evaluation" in results:
        gene_eval = results["gene_space_evaluation"]["gene_space"]
        per_gene = results["gene_space_evaluation"]["per_gene"]
        r2_corr = gene_eval["pearson"] ** 2
        print(
            f"""\n   GENE SPACE (Final validation):
      Pearson: {gene_eval["pearson"]:.4f} | r²: {r2_corr:.4f}
      Mean per-gene correlation: {per_gene["mean_correlation"]:.4f}"""
        )
    print(
        f"""\n   Model saved: {model_path}
   Results saved: {results_path}"""
    )


if __name__ == "__main__":
    main()
