"""Autoencoder models for gene expression compression."""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, List
from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, regularizers
from tensorflow.keras.models import save_model

from deep_lineage.schema import TrajectoryList, AEConfig
from deep_lineage.utils import prepare_autoencoder_data


class BaseAutoencoder(ABC):
    """Abstract base class for autoencoder models."""

    def __init__(self, config: AEConfig):
        self.config = config
        self.model = None
        self.encoder = None
        self.decoder = None
        self.history = None

    @abstractmethod
    def build_model(
        self, n_genes: int, hidden_sizes: List[int]
    ) -> Tuple[Model, Model, Model]:
        pass

    @abstractmethod
    def prepare_data(self, trajectories: TrajectoryList) -> np.ndarray:
        pass

    @abstractmethod
    def train(
        self, X_train: np.ndarray, X_val: np.ndarray, run_dir: Path = None
    ) -> Dict[str, Any]:
        pass


class StandardAutoencoder(BaseAutoencoder):
    """
    Standard autoencoder for gene expression compression.

    Architecture:
    - Multi-layer encoder with batch normalization and LeakyReLU
    - Optional input dropout (for reprogramming experiments)
    - Multi-layer decoder with batch normalization and LeakyReLU
    - Linear output layer for gene expression reconstruction
    """

    def build_model(
        self, n_genes: int, hidden_sizes: List[int]
    ) -> Tuple[Model, Model, Model]:
        print("Building autoencoder architecture")

        num_layers = len(hidden_sizes)
        print(f"""   Input dimensions: {n_genes} genes
   Hidden layers: {num_layers}
   Latent dimensions: {self.config.latent_dim}""")
        if self.config.input_dropout > 0:
            print(f"   Input dropout: {self.config.input_dropout}")

        arch_str = f"{n_genes}"
        for size in hidden_sizes:
            arch_str += f" -> {size}"
        arch_str += f" -> {self.config.latent_dim}"
        print(f"   Architecture: {arch_str}")

        l2_reg = regularizers.l2(self.config.l2) if self.config.l2 > 0 else None

        encoder_input = layers.Input(shape=(n_genes,), name="gene_expression")
        x = encoder_input

        if self.config.input_dropout > 0:
            x = layers.Dropout(self.config.input_dropout, name="input_dropout")(x)

        for i, hidden_size in enumerate(hidden_sizes, 1):
            x = layers.Dense(
                hidden_size, name=f"encoder_layer{i}", kernel_regularizer=l2_reg
            )(x)
            x = layers.BatchNormalization(name=f"encoder_bn{i}")(x)
            x = layers.LeakyReLU(alpha=0.3, name=f"encoder_leaky{i}")(x)

        latent = layers.Dense(self.config.latent_dim, name="latent_representation")(x)
        encoder = Model(encoder_input, latent, name="encoder")

        decoder_input = layers.Input(
            shape=(self.config.latent_dim,), name="latent_input"
        )
        y = decoder_input
        reversed_sizes = list(reversed(hidden_sizes))
        for i, hidden_size in enumerate(reversed_sizes, 1):
            y = layers.Dense(
                hidden_size, name=f"decoder_layer{i}", kernel_regularizer=l2_reg
            )(y)
            if i < len(reversed_sizes):
                y = layers.BatchNormalization(name=f"decoder_bn{i}")(y)
            y = layers.LeakyReLU(alpha=0.3, name=f"decoder_leaky{i}")(y)

        decoder_output = layers.Dense(
            n_genes,
            activation="linear",
            name="reconstructed_expression",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer=tf.keras.initializers.Zeros(),
        )(y)
        decoder = Model(decoder_input, decoder_output, name="decoder")

        autoencoder_output = decoder(encoder(encoder_input))
        autoencoder = Model(encoder_input, autoencoder_output, name="autoencoder")

        optimizer = optimizers.Adam(
            learning_rate=self.config.learning_rate,
            clipnorm=1.0,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
        )

        autoencoder.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=["mae", "mse", tf.keras.metrics.CosineSimilarity(axis=-1)],
        )

        print(f"""Model summary:
   Encoder parameters: {encoder.count_params():,}
   Decoder parameters: {decoder.count_params():,}
   Total parameters: {autoencoder.count_params():,}
   Optimizer: Adam (lr={self.config.learning_rate})
   Loss: MSE reconstruction{f" with L2={self.config.l2}" if self.config.l2 > 0 else ""}""")

        self.model = autoencoder
        self.encoder = encoder
        self.decoder = decoder

        return autoencoder, encoder, decoder

    def prepare_data(self, trajectories: TrajectoryList) -> np.ndarray:
        return prepare_autoencoder_data(trajectories)

    def train(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        run_dir: Path = None,
        callbacks_list: list = None,
    ) -> Dict[str, Any]:
        """
        Train the autoencoder.

        Args:
            X_train: Training gene expressions [n_train_cells, n_genes]
            X_val: Validation gene expressions [n_val_cells, n_genes]
            run_dir: Directory for saving results
            callbacks_list: Optional list of Keras callbacks (overrides default)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        print(f"""Training Deep Lineage-style autoencoder...
   Training samples: {len(X_train):,}
   Validation samples: {len(X_val):,}
   Epochs: {self.config.epochs}, Batch size: {self.config.batch_size}""")

        callback_list = callbacks_list or []

        if run_dir and not callbacks_list:
            from scripts.utils import get_keras_callbacks

            callback_list.extend(
                get_keras_callbacks(
                    run_dir=run_dir,
                    model_name="autoencoder",
                    monitor="val_loss",
                    patience=self.config.patience,
                )
            )

        print("Starting training...")

        self.history = self.model.fit(
            X_train,
            X_train,
            validation_data=(X_val, X_val),
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            callbacks=callback_list,
            verbose=1,
            shuffle=True,
        )

        final_train_loss = self.history.history["loss"][-1]
        final_val_loss = self.history.history["val_loss"][-1]
        final_train_mae = self.history.history["mae"][-1]
        final_val_mae = self.history.history["val_mae"][-1]

        X_val_reconstructed = self.model.predict(X_val, verbose=0)
        val_correlation = np.corrcoef(X_val.ravel(), X_val_reconstructed.ravel())[0, 1]

        results = {
            "final_train_loss": float(final_train_loss),
            "final_val_loss": float(final_val_loss),
            "final_train_mae": float(final_train_mae),
            "final_val_mae": float(final_val_mae),
            "val_reconstruction_correlation": float(val_correlation),
            "total_epochs": len(self.history.history["loss"]),
            "config": self.config.model_dump(),
        }

        print(f"""Training completed.
   Final train loss (MSE): {final_train_loss:.6f}
   Final val loss (MSE): {final_val_loss:.6f}
   Final train MAE: {final_train_mae:.6f}
   Final val MAE: {final_val_mae:.6f}
   Validation reconstruction correlation: {val_correlation:.4f}
   Total epochs: {results["total_epochs"]}""")

        return results

    def save_models(self, run_dir: Path, name: str = None):
        """Save the trained models separately."""
        if self.model is None or self.encoder is None or self.decoder is None:
            raise ValueError("Models not trained. Call train() first.")

        if name:
            autoencoder_filename = f"autoencoder_{name}_final.keras"
            encoder_filename = f"encoder_{name}_final.keras"
            decoder_filename = f"decoder_{name}_final.keras"
        else:
            autoencoder_filename = "autoencoder_final.keras"
            encoder_filename = "encoder_final.keras"
            decoder_filename = "decoder_final.keras"

        autoencoder_path = run_dir / autoencoder_filename
        encoder_path = run_dir / encoder_filename
        decoder_path = run_dir / decoder_filename

        save_model(self.model, str(autoencoder_path))
        save_model(self.encoder, str(encoder_path))
        save_model(self.decoder, str(decoder_path))

        print(f"""Saved models:
   Autoencoder: {autoencoder_path}
   Encoder: {encoder_path}
   Decoder: {decoder_path}""")


def create_autoencoder(config: AEConfig) -> StandardAutoencoder:
    """Factory function for creating a StandardAutoencoder."""
    return StandardAutoencoder(config)
