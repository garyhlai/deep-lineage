"""Reusable Keras layers for the Deep Lineage pipeline."""

from tensorflow.keras import layers


class SelectKthOutput(layers.Layer):
    """Select the k-th timestep from a sequence output."""

    def __init__(self, k: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def call(self, inputs):
        return inputs[:, self.k, :]

    def get_config(self):
        config = super().get_config()
        config.update({"k": self.k})
        return config
