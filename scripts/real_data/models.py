"""Models, training helpers and metrics for the real-data analyses."""

from __future__ import annotations

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    r2_score,
    roc_auc_score,
)

from deep_lineage.models.autoencoder import StandardAutoencoder
from deep_lineage.schema import AEConfig

MISSING = -1


# --------------------------------------------------------------------------- encoding


def train_autoencoder(
    X_train, X_val, latent_dim=75, hidden=(500, 100), epochs=500, patience=20
):
    """Deep Lineage autoencoder (hidden layers 500 and 100, 75 latent dimensions)."""
    config = AEConfig(
        latent_dim=latent_dim,
        batch_size=512,
        epochs=epochs,
        learning_rate=1e-3,
        patience=patience,
        l2=0,
        input_dropout=0.2,
    )
    ae = StandardAutoencoder(config)
    ae.build_model(X_train.shape[1], list(hidden))
    stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True, verbose=0
    )
    ae.model.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        batch_size=config.batch_size,
        epochs=epochs,
        callbacks=[stop],
        verbose=0,
        shuffle=True,
    )
    return ae


def series_latent(
    idx: np.ndarray, Z: np.ndarray, n_days: int | None = None
) -> np.ndarray:
    """(series, days, latent) array; absent days are all-zero so the Masking layer skips them."""
    idx = idx[:, :n_days] if n_days else idx
    out = np.zeros(idx.shape + (Z.shape[1],), dtype=np.float32)
    present = idx != MISSING
    out[present] = Z[idx[present]]
    return out


def series_genes(idx: np.ndarray, X: np.ndarray) -> np.ndarray:
    """(series, days, genes) array with zeros for absent days."""
    out = np.zeros(idx.shape + (X.shape[1],), dtype=np.float32)
    present = idx != MISSING
    out[present] = X[idx[present]]
    return out


def encode_genes(encoder, X: np.ndarray) -> np.ndarray:
    return encoder.predict(X, batch_size=4096, verbose=0).astype(np.float32)


def encode_series_genes(encoder, G: np.ndarray) -> np.ndarray:
    """Encode a (series, days, genes) array; absent (all-zero) days stay zero."""
    n, t, g = G.shape
    flat = G.reshape(-1, g)
    present = np.abs(flat).sum(axis=1) > 0
    Z = np.zeros((n * t, encoder.output_shape[-1]), dtype=np.float32)
    if present.any():
        Z[present] = encode_genes(encoder, flat[present])
    return Z.reshape(n, t, -1)


# --------------------------------------------------------------------------- networks


def build_classifier(
    n_days, n_features, n_classes=2, n_layers=2, dropout=0.25, cell="LSTM"
):
    """Bidirectional RNN classifier (Masking, Bi-LSTM layers with averaged directions, softmax)."""
    rnn = keras.layers.LSTM if cell == "LSTM" else keras.layers.GRU
    inputs = keras.Input((n_days, n_features))
    x = keras.layers.Masking(mask_value=0.0)(inputs)
    for i in range(n_layers):
        last = i == n_layers - 1
        x = keras.layers.Bidirectional(
            rnn(n_features, return_sequences=not last, dropout=dropout),
            merge_mode="ave",
        )(x)
    outputs = keras.layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def build_regressor(
    n_days, n_features, n_genes, n_layers=2, units=256, dropout=0.25, cell="LSTM"
):
    """Bidirectional RNN that maps the latent embeddings of observed days to the gene
    expression of a held-out day."""
    rnn = keras.layers.LSTM if cell == "LSTM" else keras.layers.GRU
    inputs = keras.Input((n_days, n_features))
    x = keras.layers.Masking(mask_value=0.0)(inputs)
    for i in range(n_layers):
        last = i == n_layers - 1
        x = keras.layers.Bidirectional(
            rnn(units, return_sequences=not last, dropout=dropout), merge_mode="ave"
        )(x)
    outputs = keras.layers.Dense(n_genes)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def fit(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=300,
    patience=20,
    batch_size=256,
    class_weight=None,
):
    """Train with early stopping on the validation loss; returns the epochs run."""
    stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True, verbose=0
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[stop],
        shuffle=True,
        verbose=0,
    )
    return len(history.history["loss"])


def balanced_class_weight(labels: np.ndarray, n_classes: int = 2) -> dict:
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    return {c: float(len(labels) / (n_classes * counts[c])) for c in range(n_classes)}


def one_hot(labels: np.ndarray, n_classes: int = 2) -> np.ndarray:
    return np.eye(n_classes, dtype=np.float32)[labels]


# --------------------------------------------------------------------------- metrics


def classification_metrics(labels: np.ndarray, prob: np.ndarray) -> dict:
    """Metrics for a binary task; ``prob`` is the probability of class 1."""
    labels = np.asarray(labels)
    prob = np.clip(np.asarray(prob, dtype=float), 1e-7, 1 - 1e-7)
    pred = (prob >= 0.5).astype(int)
    out = {
        "n": len(labels),
        "class_counts": np.bincount(labels, minlength=2).tolist(),
        "accuracy": float(accuracy_score(labels, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "cross_entropy": float(log_loss(labels, np.c_[1 - prob, prob], labels=[0, 1])),
        "confusion_matrix": confusion_matrix(labels, pred, labels=[0, 1]).tolist(),
    }
    out["auc"] = float(roc_auc_score(labels, prob)) if len(set(labels)) == 2 else None
    return out


def lineage_level(labels, prob, lineages) -> dict:
    """Average the predicted probability over the series of each lineage and score it
    against the lineage's majority outcome."""
    import pandas as pd

    df = pd.DataFrame({"label": labels, "prob": prob, "lineage": lineages})
    agg = df.groupby("lineage").agg(
        label=("label", lambda s: int(s.mean() >= 0.5)), prob=("prob", "mean")
    )
    return classification_metrics(agg["label"].to_numpy(), agg["prob"].to_numpy())


def rowwise_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (a * b).sum(axis=1) / denom, np.nan)


def regression_metrics(
    y_true, y_pred, labels, classes, random_reference, rng, lineages=None
) -> dict:
    """Per-series Pearson r between predicted and held-out expression, a random-cell
    control (the same prediction compared with a random test cell of the same day), and
    the squared Pearson correlation of gene means (R2) for each class."""
    r = rowwise_pearson(y_pred, y_true)
    random_cells = random_reference[
        rng.integers(len(random_reference), size=len(y_pred))
    ]
    r_random = rowwise_pearson(y_pred, random_cells)
    out = {
        "n": len(r),
        "pearson_r_mean": float(np.nanmean(r)),
        "pearson_r_random_mean": float(np.nanmean(r_random)),
        "per_class": {},
        "series_r": r.tolist(),
        "series_r_random": r_random.tolist(),
        "series_label": np.asarray(labels).tolist(),
        "series_lineage": []
        if lineages is None
        else np.asarray(lineages).astype(str).tolist(),
    }
    for c, name in enumerate(classes):
        m = np.asarray(labels) == c
        if m.sum() < 2:
            continue
        mean_pred, mean_true = y_pred[m].mean(axis=0), y_true[m].mean(axis=0)
        out["per_class"][name] = {
            "n": int(m.sum()),
            "pearson_r_mean": float(np.nanmean(r[m])),
            "gene_mean_R2": float(pearsonr(mean_pred, mean_true)[0] ** 2),
        }
    return out


def profile_similarity(a: np.ndarray, b: np.ndarray) -> dict:
    """Similarity of two mean expression profiles: Pearson r and coefficient of determination."""
    return {"pearson_r": float(pearsonr(a, b)[0]), "r2_score": float(r2_score(b, a))}


# --------------------------------------------------------------------------- attribution


def gene_space_classifier(encoder, classifier, n_days, n_genes):
    """Classifier that takes gene expression per day: encode each day, zero absent days."""
    inputs = keras.Input((n_days, n_genes))
    z = keras.layers.TimeDistributed(encoder)(inputs)
    present = keras.ops.cast(
        keras.ops.any(keras.ops.not_equal(inputs, 0.0), axis=-1, keepdims=True),
        "float32",
    )
    outputs = classifier(z * present)
    return keras.Model(inputs, outputs)


def day_shap_values(model, G, day, background, target, n_samples=32):
    """SHAP values (shap.GradientExplainer) of the genes of one day toward class
    ``target``, with the other days of each series held at their observed values.

    G: (series, days, genes) gene-space inputs with ``day`` observed.
    background: (cells, genes) reference cells observed on that day (training cells).
    Returns (series, genes) SHAP values.
    """
    import shap

    # The explained model takes the genes of one day, inserts them into the series held
    # in ``context`` and returns the probability of ``target``.
    context = tf.Variable(G[:1], trainable=False)
    genes = keras.Input(G.shape[2:])

    def insert_day(x):
        full = tf.tile(context, [tf.shape(x)[0], 1, 1])
        return tf.concat([full[:, :day], x[:, None], full[:, day + 1 :]], axis=1)

    prob = model(keras.layers.Lambda(insert_day)(genes))
    # shap reads the output size from the last layer, so the slice must be a layer.
    prob = keras.layers.Lambda(lambda p: p[:, target : target + 1])(prob)
    explainer = shap.GradientExplainer(keras.Model([genes], prob), background)
    out = np.zeros((len(G), G.shape[2]))
    for i in range(len(G)):
        context.assign(G[i : i + 1])
        values = explainer.shap_values(G[i : i + 1, day], nsamples=n_samples)
        out[i] = np.asarray(values, dtype=float).reshape(-1)
    return out


def perturb(G_day: np.ndarray, genes_up, genes_down) -> np.ndarray:
    """Overexpress (five times baseline expression) or knock out (zero) genes in
    log1p-expression space."""
    x = G_day.copy()
    if len(genes_up):
        x[:, genes_up] = np.log1p(5.0 * np.expm1(x[:, genes_up]))
    if len(genes_down):
        x[:, genes_down] = 0.0
    return x
