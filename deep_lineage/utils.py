"""Shared utilities for the Deep Lineage pipeline."""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
import tensorflow as tf
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from deep_lineage.schema import TrajectoryList


def normalize_gene_expression(
    expressions: np.ndarray, verbose: bool = True
) -> np.ndarray:
    """
    Normalize gene expression data following the Deep Lineage protocol.

    Pipeline: log1p -> per-gene standardization (mean=0, std=1) -> clip to [-3, 3].

    Args:
        expressions: Gene expression matrix [n_cells, n_genes]
        verbose: Whether to print normalization statistics

    Returns:
        Normalized gene expression matrix with same shape
    """
    if verbose:
        print(
            f"""Normalizing gene expression data...
   Original range: [{np.min(expressions):.3f}, {np.max(expressions):.3f}]
   Original statistics: mean={np.mean(expressions):.3f}, std={np.std(expressions):.3f}"""
        )

    expressions = np.log1p(expressions)

    if verbose:
        print(
            f"""   Log1p normalized range: [{np.min(expressions):.3f}, {np.max(expressions):.3f}]
   Log1p normalized statistics: mean={np.mean(expressions):.3f}, std={np.std(expressions):.3f}"""
        )

    gene_means = np.mean(expressions, axis=0, keepdims=True)
    gene_stds = np.std(expressions, axis=0, keepdims=True)
    gene_stds = np.maximum(gene_stds, 1e-7)

    expressions = (expressions - gene_means) / gene_stds

    if verbose:
        print(
            f"""   Before clipping range: [{np.min(expressions):.3f}, {np.max(expressions):.3f}]
   Before clipping percentiles [1, 50, 99, 99.9]: {np.percentile(expressions, [1, 50, 99, 99.9])}"""
        )

    expressions = np.clip(expressions, -3, 3)

    if verbose:
        n_clipped = np.sum((expressions <= -3) | (expressions >= 3))
        pct_clipped = (n_clipped / expressions.size) * 100
        print(
            f"""   After clipping range: [{np.min(expressions):.3f}, {np.max(expressions):.3f}]
   Final statistics: mean={np.mean(expressions):.6f}, std={np.std(expressions):.3f}
   Values clipped: {n_clipped:,} ({pct_clipped:.2f}% of total)"""
        )

    return expressions


def make_json_serializable(obj):
    """Convert Path objects and other non-JSON-serializable objects to JSON-safe types."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, (set, tuple)):
        return [make_json_serializable(v) for v in obj]
    else:
        return obj


def prepare_autoencoder_data(trajectories: TrajectoryList) -> np.ndarray:
    """
    Prepare gene expression data for autoencoder training.

    Flattens all cells across all trajectories and applies normalization.

    Args:
        trajectories: TrajectoryList with trajectory data

    Returns:
        Gene expression matrix [n_cells, n_genes]
    """
    print("Preparing autoencoder training data...")

    expressions = []
    for traj in tqdm(trajectories.trajectories, desc="Processing trajectories"):
        for cell in traj.cells:
            expressions.append(cell.expr)

    expressions = np.array(expressions, dtype=np.float32)

    print(
        f"""   Total cells: {expressions.shape[0]:,}
   Genes per cell: {expressions.shape[1]:,}
   Data shape for training: {expressions.shape}
   Memory usage: {expressions.nbytes / 1024**2:.1f} MB"""
    )

    expressions = normalize_gene_expression(expressions, verbose=True)
    return expressions


def compute_per_sample_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute per-sample Pearson correlations.

    Args:
        y_true: True values [n_samples, n_dims]
        y_pred: Predicted values [n_samples, n_dims]

    Returns:
        Array of per-sample Pearson correlations [n_samples]
    """
    n_samples = y_true.shape[0]
    pearson_corrs = []

    for i in range(n_samples):
        true_sample = y_true[i, :]
        pred_sample = y_pred[i, :]

        if np.std(true_sample) > 1e-6 and np.std(pred_sample) > 1e-6:
            corr, _ = pearsonr(true_sample, pred_sample)
            pearson_corrs.append(corr if not np.isnan(corr) else 0.0)
        else:
            pearson_corrs.append(0.0)

    return np.array(pearson_corrs)


def evaluate_gene_space(
    y_true_latent: np.ndarray,
    y_pred_latent: np.ndarray,
    decoder_model,
    verbose: bool = True,
    title: str = "GENE SPACE EVALUATION",
) -> Dict[str, Any]:
    """
    Gene space evaluation utility.

    Args:
        y_true_latent: True latent vectors [samples, latent_dim]
        y_pred_latent: Predicted latent vectors [samples, latent_dim]
        decoder_model: Decoder to convert latent -> gene space
        verbose: Whether to print results
        title: Title for the evaluation output

    Returns:
        Dictionary with comprehensive metrics
    """
    y_true_gene = decoder_model.predict(y_true_latent, verbose=0)
    y_pred_gene = decoder_model.predict(y_pred_latent, verbose=0)

    latent_pearson_arr = compute_per_sample_pearson(y_true_latent, y_pred_latent)
    latent_pearson = float(np.mean(latent_pearson_arr))
    latent_pearson_std = float(np.std(latent_pearson_arr))
    latent_r2 = latent_pearson**2
    latent_mse = mean_squared_error(y_true_latent, y_pred_latent)
    latent_mae = mean_absolute_error(y_true_latent, y_pred_latent)
    latent_cosine = np.mean(
        [
            cosine_similarity([y_true_latent[i]], [y_pred_latent[i]])[0, 0]
            for i in range(len(y_true_latent))
        ]
    )

    gene_pearson_arr = compute_per_sample_pearson(y_true_gene, y_pred_gene)
    gene_pearson = float(np.mean(gene_pearson_arr))
    gene_pearson_std = float(np.std(gene_pearson_arr))
    gene_spearman, _ = spearmanr(y_true_gene.flatten(), y_pred_gene.flatten())
    gene_r2 = gene_pearson**2
    gene_mse = mean_squared_error(y_true_gene, y_pred_gene)
    gene_mae = mean_absolute_error(y_true_gene, y_pred_gene)

    gene_cosine = np.mean(
        [
            cosine_similarity([y_true_gene[i]], [y_pred_gene[i]])[0, 0]
            for i in range(len(y_true_gene))
        ]
    )

    n_genes = y_true_gene.shape[1]
    gene_correlations = []
    for gene_idx in range(n_genes):
        true_gene = y_true_gene[:, gene_idx]
        pred_gene = y_pred_gene[:, gene_idx]
        if np.std(true_gene) > 1e-6 and np.std(pred_gene) > 1e-6:
            corr, _ = pearsonr(true_gene, pred_gene)
            gene_correlations.append(corr if not np.isnan(corr) else 0.0)
        else:
            gene_correlations.append(0.0)

    gene_correlations = np.array(gene_correlations)

    mean_gene_corr = np.mean(gene_correlations)
    std_gene_corr = np.std(gene_correlations)
    median_gene_corr = np.median(gene_correlations)

    excellent_genes = np.sum(gene_correlations > 0.7)
    good_genes = np.sum((gene_correlations > 0.5) & (gene_correlations <= 0.7))
    moderate_genes = np.sum((gene_correlations > 0.3) & (gene_correlations <= 0.5))
    poor_genes = np.sum(gene_correlations <= 0.3)
    well_predicted = np.sum(gene_correlations > 0.5)
    positive_correlation = np.sum(gene_correlations > 0.0)

    if verbose:
        print(
            f"""
{title}
{"=" * 80}

LATENT SPACE (Training objective):
   Pearson: {latent_pearson:.4f} ± {latent_pearson_std:.4f} | R²: {latent_r2:.4f} | Cosine: {latent_cosine:.4f}
   MSE: {latent_mse:.4f} | MAE: {latent_mae:.4f}

GENE SPACE (Biological relevance):
   Pearson: {gene_pearson:.4f} ± {gene_pearson_std:.4f} | R²: {gene_r2:.4f} | Cosine: {gene_cosine:.4f}
   Spearman: {gene_spearman:.4f} | MSE: {gene_mse:.4f} | MAE: {gene_mae:.4f}

PER-GENE ANALYSIS:
   Mean correlation: {mean_gene_corr:.4f} ± {std_gene_corr:.4f}
   Median correlation: {median_gene_corr:.4f}
   Excellent (r>0.7): {excellent_genes}/{n_genes} ({100 * excellent_genes / n_genes:.1f}%)
   Good (0.5<r≤0.7): {good_genes}/{n_genes} ({100 * good_genes / n_genes:.1f}%)
   Moderate (0.3<r≤0.5): {moderate_genes}/{n_genes} ({100 * moderate_genes / n_genes:.1f}%)
   Poor (r≤0.3): {poor_genes}/{n_genes} ({100 * poor_genes / n_genes:.1f}%)
{"=" * 80}"""
        )

    return {
        "latent_space": {
            "pearson": float(latent_pearson),
            "pearson_std": float(latent_pearson_std),
            "r2": float(latent_r2),
            "mse": float(latent_mse),
            "mae": float(latent_mae),
            "cosine": float(latent_cosine),
        },
        "gene_space": {
            "pearson": float(gene_pearson),
            "pearson_std": float(gene_pearson_std),
            "spearman": float(gene_spearman),
            "r2": float(gene_r2),
            "mse": float(gene_mse),
            "mae": float(gene_mae),
            "cosine": float(gene_cosine),
        },
        "per_gene": {
            "mean_correlation": float(mean_gene_corr),
            "std_correlation": float(std_gene_corr),
            "median_correlation": float(median_gene_corr),
            "excellent_genes": int(excellent_genes),
            "good_genes": int(good_genes),
            "moderate_genes": int(moderate_genes),
            "poor_genes": int(poor_genes),
            "well_predicted": int(well_predicted),
            "positive_correlation": int(positive_correlation),
            "total_genes": int(n_genes),
            "correlations": gene_correlations.tolist(),
        },
        "n_samples_evaluated": len(y_true_latent),
    }


def collect_predictions_from_dataset(
    model: tf.keras.Model, dataset: tf.data.Dataset, verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect predictions from a streaming dataset.

    Args:
        model: Model to use for predictions
        dataset: Dataset yielding (X_batch, y_true_latent) pairs
        verbose: Whether to show progress

    Returns:
        Tuple of (y_true_latent, y_pred_latent) arrays
    """
    all_y_true_latent = []
    all_y_pred_latent = []

    for batch_idx, (X_batch, y_true_latent) in enumerate(dataset):
        if verbose and batch_idx % 10 == 0:
            print(f"   Processing batch {batch_idx + 1}...")

        y_pred_latent = model.predict(X_batch, verbose=0)

        all_y_true_latent.append(y_true_latent.numpy())
        all_y_pred_latent.append(y_pred_latent)

    y_true_latent = np.concatenate(all_y_true_latent, axis=0)
    y_pred_latent = np.concatenate(all_y_pred_latent, axis=0)

    return y_true_latent, y_pred_latent
