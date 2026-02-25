"""Metrics for gene expression prediction evaluation."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Metric
from tensorflow.keras.saving import register_keras_serializable
from typing import Dict, Any
from scipy.stats import spearmanr


@register_keras_serializable()
class PearsonCorrelation(Metric):
    """
    Pearson correlation coefficient metric for Keras.

    Computes correlation between predicted and true values across the batch.
    Useful for gene expression where relative patterns matter more than
    absolute values.
    """

    def __init__(self, name="pearson_correlation", **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum_x = self.add_weight(name="sum_x", initializer="zeros")
        self.sum_y = self.add_weight(name="sum_y", initializer="zeros")
        self.sum_xx = self.add_weight(name="sum_xx", initializer="zeros")
        self.sum_yy = self.add_weight(name="sum_yy", initializer="zeros")
        self.sum_xy = self.add_weight(name="sum_xy", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])

        mask = tf.logical_and(
            tf.math.is_finite(y_true_flat), tf.math.is_finite(y_pred_flat)
        )
        y_true_clean = tf.boolean_mask(y_true_flat, mask)
        y_pred_clean = tf.boolean_mask(y_pred_flat, mask)

        n = tf.cast(tf.shape(y_true_clean)[0], tf.float32)

        self.sum_x.assign_add(tf.reduce_sum(y_pred_clean))
        self.sum_y.assign_add(tf.reduce_sum(y_true_clean))
        self.sum_xx.assign_add(tf.reduce_sum(y_pred_clean * y_pred_clean))
        self.sum_yy.assign_add(tf.reduce_sum(y_true_clean * y_true_clean))
        self.sum_xy.assign_add(tf.reduce_sum(y_pred_clean * y_true_clean))
        self.count.assign_add(n)

    def result(self):
        n = self.count
        numerator = n * self.sum_xy - self.sum_x * self.sum_y
        denominator = tf.sqrt(
            (n * self.sum_xx - self.sum_x * self.sum_x)
            * (n * self.sum_yy - self.sum_y * self.sum_y)
        )
        correlation = tf.where(denominator > 1e-10, numerator / denominator, 0.0)
        return correlation

    def reset_state(self):
        self.sum_x.assign(0.0)
        self.sum_y.assign(0.0)
        self.sum_xx.assign(0.0)
        self.sum_yy.assign(0.0)
        self.sum_xy.assign(0.0)
        self.count.assign(0.0)


def compute_correlation_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive correlation metrics for evaluation.

    Args:
        y_true: True values [samples, ..., features]
        y_pred: Predicted values [samples, ..., features]

    Returns:
        Dictionary with correlation metrics
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
    y_true_clean = y_true_flat[mask]
    y_pred_clean = y_pred_flat[mask]

    if len(y_true_clean) == 0:
        return {
            "pearson_correlation": 0.0,
            "spearman_correlation": 0.0,
            "r2_score": 0.0,
            "cosine_similarity": 0.0,
        }

    # Pearson correlation
    try:
        pearson_r = np.corrcoef(y_true_clean, y_pred_clean)[0, 1]
        if np.isnan(pearson_r):
            pearson_r = 0.0
    except (ValueError, FloatingPointError):
        pearson_r = 0.0

    # Spearman correlation
    try:
        spearman_r, _ = spearmanr(y_true_clean, y_pred_clean)
        if np.isnan(spearman_r):
            spearman_r = 0.0
    except (ValueError, FloatingPointError):
        spearman_r = 0.0

    # R² score
    try:
        ss_res = np.sum((y_true_clean - y_pred_clean) ** 2)
        ss_tot = np.sum((y_true_clean - np.mean(y_true_clean)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        r2 = np.clip(r2, -1.0, 1.0)
    except (ValueError, FloatingPointError):
        r2 = 0.0

    # Cosine similarity
    try:
        if y_true.ndim > 1:
            y_true_samples = y_true.reshape(y_true.shape[0], -1)
            y_pred_samples = y_pred.reshape(y_pred.shape[0], -1)

            cos_similarities = []
            for i in range(len(y_true_samples)):
                true_vec = y_true_samples[i]
                pred_vec = y_pred_samples[i]

                norm_true = np.linalg.norm(true_vec)
                norm_pred = np.linalg.norm(pred_vec)

                if norm_true > 1e-10 and norm_pred > 1e-10:
                    cos_sim = np.dot(true_vec, pred_vec) / (norm_true * norm_pred)
                    cos_similarities.append(cos_sim)

            cosine_sim = np.mean(cos_similarities) if cos_similarities else 0.0
        else:
            norm_true = np.linalg.norm(y_true_clean)
            norm_pred = np.linalg.norm(y_pred_clean)
            cosine_sim = (
                np.dot(y_true_clean, y_pred_clean) / (norm_true * norm_pred)
                if norm_true > 1e-10 and norm_pred > 1e-10
                else 0.0
            )
    except (ValueError, FloatingPointError):
        cosine_sim = 0.0

    return {
        "pearson_correlation": float(pearson_r),
        "spearman_correlation": float(spearman_r),
        "r2_score": float(r2),
        "cosine_similarity": float(cosine_sim),
    }


def per_gene_correlation_analysis(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Any]:
    """
    Compute per-gene correlation analysis for gene expression data.

    Args:
        y_true: True expressions [samples, timesteps, genes]
        y_pred: Predicted expressions [samples, timesteps, genes]

    Returns:
        Dictionary with per-gene analysis results
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    n_samples, n_timesteps, n_genes = y_true.shape
    gene_correlations = []

    for gene_idx in range(n_genes):
        true_vals = y_true[:, :, gene_idx].flatten()
        pred_vals = y_pred[:, :, gene_idx].flatten()

        mask = np.isfinite(true_vals) & np.isfinite(pred_vals)
        true_clean = true_vals[mask]
        pred_clean = pred_vals[mask]

        if (
            len(true_clean) > 1
            and np.std(true_clean) > 1e-10
            and np.std(pred_clean) > 1e-10
        ):
            try:
                correlation = np.corrcoef(true_clean, pred_clean)[0, 1]
                if np.isfinite(correlation):
                    gene_correlations.append(correlation)
                else:
                    gene_correlations.append(0.0)
            except (ValueError, FloatingPointError):
                gene_correlations.append(0.0)
        else:
            gene_correlations.append(0.0)

    gene_correlations = np.array(gene_correlations)

    return {
        "per_gene_correlations": gene_correlations.tolist(),
        "mean_gene_correlation": float(np.mean(gene_correlations)),
        "median_gene_correlation": float(np.median(gene_correlations)),
        "std_gene_correlation": float(np.std(gene_correlations)),
        "min_gene_correlation": float(np.min(gene_correlations)),
        "max_gene_correlation": float(np.max(gene_correlations)),
        "genes_with_positive_correlation": int(np.sum(gene_correlations > 0)),
        "genes_with_strong_correlation": int(np.sum(gene_correlations > 0.5)),
        "total_genes": int(len(gene_correlations)),
    }
