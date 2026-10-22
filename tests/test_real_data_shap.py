"""Checks that day_shap_values gives the SHAP values of one day's genes with the other
days held fixed, by comparing it with the expected-gradients formula written out.

Run with:  uv run --with pytest python -m pytest tests/test_real_data_shap.py
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import numpy as np
import tensorflow as tf

from scripts.real_data.models import (
    build_classifier,
    day_shap_values,
    gene_space_classifier,
)

N_GENES, N_DAYS, DAY, TARGET, N_SAMPLES = 20, 3, 1, 1, 4000


def expected_gradients(model, x, day, background, target, n_samples):
    """Average of (x - b) * gradient at b + alpha (x - b) on ``day`` over random
    background cells b and alpha in [0, 1], with the other days of ``x`` unchanged."""
    rng = np.random.default_rng()
    b = background[rng.integers(len(background), size=n_samples)]
    alpha = rng.random((n_samples, 1)).astype(np.float32)
    mixed = np.repeat(x[None], n_samples, axis=0)
    mixed[:, day] = b + alpha * (x[day] - b)
    mixed = tf.convert_to_tensor(mixed)
    with tf.GradientTape() as tape:
        tape.watch(mixed)
        prob = model(mixed, training=False)[:, target]
    grad = tape.gradient(prob, mixed).numpy()[:, day]
    return (grad * (x[day] - b)).mean(axis=0)


def test_day_shap_values_match_expected_gradients():
    rng = np.random.default_rng()
    genes = keras.Input((N_GENES,))
    hidden = keras.layers.Dense(16, activation="tanh")(genes)
    encoder = keras.Model(genes, keras.layers.Dense(4)(hidden))
    model = gene_space_classifier(encoder, build_classifier(N_DAYS, 4), N_DAYS, N_GENES)

    def expression(*shape):
        return np.log1p(rng.poisson(1.0, size=shape)).astype(np.float32)

    background, G = expression(200, N_GENES), expression(2, N_DAYS, N_GENES)

    values = day_shap_values(model, G, DAY, background, TARGET, n_samples=N_SAMPLES)
    for i in range(len(G)):
        direct = expected_gradients(model, G[i], DAY, background, TARGET, N_SAMPLES)
        assert np.corrcoef(values[i], direct)[0, 1] > 0.9
