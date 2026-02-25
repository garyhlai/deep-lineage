"""Train autoencoder for gene expression compression."""

import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import tensorflow as tf
from tensorflow.keras import callbacks
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

from deep_lineage.schema import AEConfig
from deep_lineage.models.autoencoder import (
    BaseAutoencoder,
    create_autoencoder,
)
from scripts.utils import (
    get_run_dir,
    load_trajectory_data,
    log_trajectory_split_info,
    check_data_leakage,
    add_common_training_args,
    add_model_training_args,
    set_random_seeds,
    save_results_json,
    get_keras_callbacks,
)


class AutoencoderQualityCallback(callbacks.Callback):
    """Monitor autoencoder reconstruction quality during training."""

    def __init__(
        self,
        X_val: np.ndarray,
        check_every: int = 5,
        quality_thresholds: Dict[str, float] = None,
    ):
        super().__init__()
        self.X_val = X_val
        self.check_every = check_every

        self.thresholds = quality_thresholds or {
            "r2_threshold": 0.5,
            "mean_gene_correlation": 0.7,
            "mean_sample_correlation": 0.8,
            "fraction_good_genes": 0.8,
        }

        self.quality_history = []

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.check_every != 0:
            return

        X_reconstructed = self.model.predict(self.X_val, verbose=0)

        quality_metrics = self._calculate_quality_metrics(self.X_val, X_reconstructed)
        quality_metrics["epoch"] = epoch + 1
        self.quality_history.append(quality_metrics)

        self._print_quality_assessment(quality_metrics)

    def _calculate_quality_metrics(
        self, X_true: np.ndarray, X_pred: np.ndarray
    ) -> Dict[str, float]:
        r2 = r2_score(X_true, X_pred)
        overall_corr, _ = pearsonr(X_true.ravel(), X_pred.ravel())

        per_sample_correlations = []
        for i in range(len(X_true)):
            if np.std(X_true[i]) > 1e-6 and np.std(X_pred[i]) > 1e-6:
                corr, _ = pearsonr(X_true[i], X_pred[i])
                per_sample_correlations.append(corr if not np.isnan(corr) else 0.0)
            else:
                per_sample_correlations.append(0.0)
        mean_sample_corr = np.mean(per_sample_correlations)

        per_gene_correlations = []
        for j in range(X_true.shape[1]):
            if np.std(X_true[:, j]) > 1e-6 and np.std(X_pred[:, j]) > 1e-6:
                corr, _ = pearsonr(X_true[:, j], X_pred[:, j])
                per_gene_correlations.append(corr if not np.isnan(corr) else 0.0)
            else:
                per_gene_correlations.append(0.0)
        mean_gene_corr = np.mean(per_gene_correlations)
        fraction_good_genes = np.mean(np.array(per_gene_correlations) > 0.5)

        return {
            "r2": r2,
            "overall_correlation": overall_corr,
            "mean_sample_correlation": mean_sample_corr,
            "mean_gene_correlation": mean_gene_corr,
            "fraction_good_genes": fraction_good_genes,
            "per_sample_correlations": per_sample_correlations,
            "per_gene_correlations": per_gene_correlations,
        }

    def _print_quality_assessment(self, metrics: Dict[str, Any]):
        epoch = metrics["epoch"]

        print(f"\nQUALITY CHECK - Epoch {epoch}")
        print("=" * 50)

        r2_good = metrics["r2"] >= self.thresholds["r2_threshold"]
        gene_corr_good = (
            metrics["mean_gene_correlation"] >= self.thresholds["mean_gene_correlation"]
        )
        sample_corr_good = (
            metrics["mean_sample_correlation"]
            >= self.thresholds["mean_sample_correlation"]
        )
        gene_fraction_good = (
            metrics["fraction_good_genes"] >= self.thresholds["fraction_good_genes"]
        )

        status_r2 = "PASS" if r2_good else "FAIL"
        status_gene = "PASS" if gene_corr_good else "FAIL"
        status_sample = "PASS" if sample_corr_good else "FAIL"
        status_fraction = "PASS" if gene_fraction_good else "FAIL"

        print(
            f"  [{status_r2}] R² Score: {metrics['r2']:.4f} (target: ≥{self.thresholds['r2_threshold']:.1f})"
        )
        print(
            f"  [{status_gene}] Mean Gene Correlation: {metrics['mean_gene_correlation']:.4f} (target: ≥{self.thresholds['mean_gene_correlation']:.1f})"
        )
        print(
            f"  [{status_sample}] Mean Sample Correlation: {metrics['mean_sample_correlation']:.4f} (target: ≥{self.thresholds['mean_sample_correlation']:.1f})"
        )
        print(
            f"  [{status_fraction}] Good Genes Fraction: {metrics['fraction_good_genes']:.1%} (target: ≥{self.thresholds['fraction_good_genes']:.0%})"
        )

        all_good = (
            r2_good and gene_corr_good and sample_corr_good and gene_fraction_good
        )
        mostly_good = (
            sum([r2_good, gene_corr_good, sample_corr_good, gene_fraction_good]) >= 3
        )

        if all_good:
            print("EXCELLENT. Autoencoder is good enough for downstream tasks.")
        elif mostly_good:
            print("DECENT. Autoencoder should work, but could be improved.")
        else:
            print("POOR. Autoencoder needs significant improvement.")
            print("   Consider: more epochs, higher capacity, better architecture")

        print("=" * 50)

    def get_final_assessment(self) -> Dict[str, Any]:
        if not self.quality_history:
            return {"assessment": "No quality checks performed"}

        final_metrics = self.quality_history[-1]

        r2_good = final_metrics["r2"] >= self.thresholds["r2_threshold"]
        gene_corr_good = (
            final_metrics["mean_gene_correlation"]
            >= self.thresholds["mean_gene_correlation"]
        )
        sample_corr_good = (
            final_metrics["mean_sample_correlation"]
            >= self.thresholds["mean_sample_correlation"]
        )
        gene_fraction_good = (
            final_metrics["fraction_good_genes"]
            >= self.thresholds["fraction_good_genes"]
        )

        passed_checks = sum(
            [r2_good, gene_corr_good, sample_corr_good, gene_fraction_good]
        )

        if passed_checks == 4:
            quality_level = "excellent"
            recommendation = "Autoencoder is ready for downstream tasks"
        elif passed_checks >= 3:
            quality_level = "good"
            recommendation = "Autoencoder should work adequately"
        elif passed_checks >= 2:
            quality_level = "fair"
            recommendation = (
                "Autoencoder may work but consider retraining with better parameters"
            )
        else:
            quality_level = "poor"
            recommendation = "Autoencoder will likely fail downstream. Retrain with higher capacity/better architecture"

        return {
            "quality_level": quality_level,
            "recommendation": recommendation,
            "passed_checks": passed_checks,
            "total_checks": 4,
            "final_metrics": final_metrics,
            "quality_history": self.quality_history,
        }


def run_diagnostics(
    autoencoder: BaseAutoencoder,
    X_train: np.ndarray,
    X_val: np.ndarray,
    hidden_sizes: List[int],
    config: AEConfig,
):
    """Run comprehensive diagnostics on data and model without training."""
    import hashlib

    print(f"""
{"=" * 80}
DRY RUN DIAGNOSTICS - STANDARD AUTOENCODER
{"=" * 80}

DATA VERIFICATION
{"-" * 40}

Training Data:
   Shape: {X_train.shape}
   Dtype: {X_train.dtype}
   Memory: {X_train.nbytes / 1024**2:.2f} MB
   Mean: {np.mean(X_train):.6f}
   Std: {np.std(X_train):.6f}
   Min: {np.min(X_train):.6f}
   Max: {np.max(X_train):.6f}
   Percentiles [25, 50, 75]: {np.percentile(X_train, [25, 50, 75])}""")

    n_clipped_train = np.sum((X_train <= -10) | (X_train >= 10))
    pct_clipped_train = (n_clipped_train / X_train.size) * 100
    train_hash = hashlib.md5(X_train.tobytes()).hexdigest()[:8]
    val_hash = hashlib.md5(X_val.tobytes()).hexdigest()[:8]
    batch_size = min(5, len(X_train))

    print(f"""   Clipped values: {n_clipped_train:,} ({pct_clipped_train:.2f}%)
   Data hash: {train_hash}

Validation Data:
   Shape: {X_val.shape}
   Dtype: {X_val.dtype}
   Memory: {X_val.nbytes / 1024**2:.2f} MB
   Mean: {np.mean(X_val):.6f}
   Std: {np.std(X_val):.6f}
   Min: {np.min(X_val):.6f}
   Max: {np.max(X_val):.6f}
   Data hash: {val_hash}

First Batch Sample (first 5 samples, first 10 genes):""")
    print(X_train[:batch_size, :10])

    n_genes = X_train.shape[1]
    model, encoder, decoder = autoencoder.build_model(n_genes, hidden_sizes)

    print(f"""

MODEL ARCHITECTURE VERIFICATION
{"-" * 40}

StandardAutoencoder Architecture:
   Hidden sizes: {hidden_sizes}
   Latent dim: {config.latent_dim}""")

    print("\n   Encoder layers:")
    for layer in encoder.layers:
        if hasattr(layer, "units"):
            print(f"      {layer.name}: {layer.units} units")

    print("\n   Decoder layers:")
    for layer in decoder.layers:
        if hasattr(layer, "units"):
            print(f"      {layer.name}: {layer.units} units")

    print(f"""
   Total parameters: {model.count_params():,}
   Encoder parameters: {encoder.count_params():,}
   Decoder parameters: {decoder.count_params():,}""")

    test_batch = X_train[:32]
    encoded = encoder(test_batch, training=False)
    reconstructed = decoder(encoded, training=False)
    mse = np.mean((test_batch - reconstructed) ** 2)

    print(f"""

FORWARD PASS VERIFICATION
{"-" * 40}

Test batch shape: {test_batch.shape}
   Encoded shape: {encoded.shape}
   Encoded stats: mean={np.mean(encoded):.4f}, std={np.std(encoded):.4f}
   Encoded range: [{np.min(encoded):.4f}, {np.max(encoded):.4f}]

   Reconstructed shape: {reconstructed.shape}
   Reconstructed stats: mean={np.mean(reconstructed):.4f}, std={np.std(reconstructed):.4f}
   Reconstructed range: [{np.min(reconstructed):.4f}, {np.max(reconstructed):.4f}]

   Initial MSE: {mse:.6f}""")

    print(f"""

GRADIENT FLOW VERIFICATION
{"-" * 40}""")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="mse",
    )

    with tf.GradientTape() as tape:
        predictions = model(test_batch, training=True)
        loss = tf.reduce_mean(tf.square(test_batch - predictions))

    gradients = tape.gradient(loss, model.trainable_weights)

    print("\nGradient statistics by layer:")
    for i, (grad, weight) in enumerate(zip(gradients, model.trainable_weights)):
        if grad is not None:
            grad_norm = tf.norm(grad).numpy()
            grad_mean = tf.reduce_mean(grad).numpy()
            grad_std = tf.math.reduce_std(grad).numpy()
            print(
                f"   Layer {weight.name[:30]:30s}: norm={grad_norm:.6f}, mean={grad_mean:.6e}, std={grad_std:.6e}"
            )

    input_range = np.max(X_train) - np.min(X_train)
    output_range = np.max(reconstructed) - np.min(reconstructed)
    range_ratio = output_range / input_range
    range_status = (
        f"WARNING: Output range is {range_ratio:.1%} of input range - model may have activation issues"
        if range_ratio < 0.5
        else f"Output range is {range_ratio:.1%} of input range"
    )

    print(f"""

DIAGNOSTIC SUMMARY
{"-" * 40}
   Model type: STANDARD AUTOENCODER
   Data preprocessed identically: yes
   Input range: [{np.min(X_train):.2f}, {np.max(X_train):.2f}]
   Output range: [{np.min(reconstructed):.2f}, {np.max(reconstructed):.2f}]
   {range_status}""")

    print("\n" + "=" * 80)
    print("DRY RUN COMPLETE - No training performed")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train Deep Lineage-style autoencoder for gene expression compression"
    )
    add_common_training_args(parser)
    add_model_training_args(parser)
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=75,
        help="Dimension of autoencoder latent space (default: 75)",
    )
    parser.add_argument(
        "--hidden_sizes",
        type=str,
        required=True,
        help="Comma-separated hidden layer sizes (e.g., '800,400,200,100'). Required.",
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
        "--name",
        type=str,
        default=None,
        help="Optional experiment name suffix for encoder/decoder files (e.g., 'dropout_30pct')",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run diagnostics only without training.",
    )
    parser.set_defaults(batch_size=2048, epochs=300, patience=50, learning_rate=5e-4)
    args = parser.parse_args()
    print(f"Parsed arguments: {args}")

    set_random_seeds(args.seed)

    run_dir = Path(args.run_dir) if args.run_dir else get_run_dir()
    print(f"Using run directory: {run_dir}")

    try:
        train_trajectories = load_trajectory_data(
            run_dir, split="train", custom_path=args.train_trajectories
        )
        print(f"Loaded {len(train_trajectories.trajectories)} training trajectories")
        train_group_ids = log_trajectory_split_info(train_trajectories, "Training")

        val_trajectories = load_trajectory_data(
            run_dir, split="val", custom_path=args.val_trajectories
        )
        print(f"Loaded {len(val_trajectories.trajectories)} validation trajectories")
        val_group_ids = log_trajectory_split_info(val_trajectories, "Validation")

        check_data_leakage(train_group_ids, val_group_ids)

    except FileNotFoundError as e:
        print(f"Error loading trajectory data: {e}")
        return

    config = AEConfig(
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
    )

    autoencoder = create_autoencoder(config)

    print("Using Standard Autoencoder architecture")

    print("\nPreparing training data...")
    X_train = autoencoder.prepare_data(train_trajectories)
    X_val = autoencoder.prepare_data(val_trajectories)

    n_genes = X_train.shape[1]
    print(f"Gene expression dimensions: {n_genes}")

    try:
        hidden_sizes = [int(x.strip()) for x in args.hidden_sizes.split(",")]
        print(f"Using hidden layer sizes: {hidden_sizes}")
    except ValueError:
        print("Invalid hidden_sizes format. Use comma-separated integers.")
        return

    if args.dry_run:
        run_diagnostics(
            autoencoder=autoencoder,
            X_train=X_train,
            X_val=X_val,
            hidden_sizes=hidden_sizes,
            config=config,
        )
        return

    autoencoder.build_model(n_genes, hidden_sizes=hidden_sizes)

    quality_callback = AutoencoderQualityCallback(X_val=X_val, check_every=5)
    extra_callbacks = get_keras_callbacks(
        run_dir=run_dir,
        model_name="autoencoder",
        monitor="val_loss",
        patience=config.patience,
    )
    extra_callbacks.append(quality_callback)

    results = autoencoder.train(X_train, X_val, callbacks_list=extra_callbacks)

    quality_assessment = quality_callback.get_final_assessment()
    results["quality_assessment"] = quality_assessment

    autoencoder.save_models(run_dir, name=args.name)

    results_path = run_dir / "autoencoder_results.json"
    save_results_json(results, results_path)

    print(
        f"""\nAutoencoder Training Complete.
   Latent dimension: {config.latent_dim}
   Final validation correlation: {results["val_reconstruction_correlation"]:.4f}
   Models saved to: {run_dir}
   Ready for temporal classification and regression training."""
    )


if __name__ == "__main__":
    main()
