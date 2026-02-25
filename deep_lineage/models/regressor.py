"""Regressor for gene expression prediction."""

import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks
from tensorflow.keras.models import save_model, load_model

from deep_lineage.schema import TrajectoryList, LSTMConfig
from deep_lineage.layers import SelectKthOutput
from deep_lineage.metrics import PearsonCorrelation
from deep_lineage.utils import (
    normalize_gene_expression,
    evaluate_gene_space,
    collect_predictions_from_dataset,
)


class Regressor:
    """
    Regressor following Deep Lineage methodology.

    Uses Bidirectional LSTM with masking to predict missing timepoints
    in encoded gene expression space.
    """

    def __init__(
        self,
        config: LSTMConfig,
        input_timepoints: List[str],
        target_timepoint: str,
        name: str,
        ae_name: str = None,
    ):
        self.config = config
        self.input_timepoints = input_timepoints
        self.target_timepoint = target_timepoint
        self.name = name
        self.ae_name = ae_name
        self.model = None
        self.encoder_model = None
        self.decoder_model = None
        self.history = None
        self.normalization_params = None

    def load_encoder_decoder(self, run_dir: Path):
        """Load the trained autoencoder encoder and decoder models."""
        if self.ae_name:
            encoder_path = run_dir / f"encoder_{self.ae_name}_final.keras"
            decoder_path = run_dir / f"decoder_{self.ae_name}_final.keras"
        else:
            encoder_path = run_dir / "encoder_final.keras"
            decoder_path = run_dir / "decoder_final.keras"

        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder model not found at {encoder_path}")

        self.encoder_model = load_model(str(encoder_path))
        self.encoder_model.trainable = False
        print(f"Loaded encoder from: {encoder_path} (frozen)")

        if not decoder_path.exists():
            raise FileNotFoundError(f"Decoder model not found at {decoder_path}")
        self.decoder_model = load_model(str(decoder_path))
        print(f"Loaded decoder from: {decoder_path}")

    def build_model(self, n_timesteps: int, n_genes: int) -> Model:
        """
        Build regressor with on-the-fly encoding.

        Args:
            n_timesteps: Number of timesteps in input sequence (always 3 for t0,t1,t2)
            n_genes: Number of genes per timestep (raw input dimension)
        """
        if self.encoder_model is None:
            raise ValueError(
                "Encoder model not loaded. Call load_encoder_decoder() first."
            )

        print(f"""Building regressor: {self.name}
   Input: {n_timesteps} timesteps x {n_genes} genes (raw)
   Target: {self.target_timepoint}
   Architecture: {self.config.num_layers} Bidirectional {self.config.cell_type} layers""")

        inputs = layers.Input(
            shape=(n_timesteps, n_genes), name="raw_masked_timeseries"
        )

        encoded = layers.TimeDistributed(self.encoder_model, name="td_encoder")(inputs)
        masked = layers.Masking(mask_value=0.0, name="masking")(encoded)

        l2_reg = (
            tf.keras.regularizers.l2(self.config.l2) if self.config.l2 > 0 else None
        )

        latent_dim = self.encoder_model.output_shape[-1]

        x = masked
        RNN = layers.LSTM if self.config.cell_type == "LSTM" else layers.GRU

        for layer_idx in range(self.config.num_layers):
            x = layers.Bidirectional(
                RNN(
                    latent_dim,
                    return_sequences=True,
                    dropout=self.config.dropout,
                    kernel_regularizer=l2_reg,
                    recurrent_regularizer=l2_reg,
                    name=f"{self.config.cell_type.lower()}_layer_{layer_idx}",
                ),
                merge_mode="ave",
                name=f"bidirectional_{layer_idx}",
            )(x)

        timepoint_map = {"t0": 0, "t1": 1, "t2": 2}
        target_idx = timepoint_map[self.target_timepoint]

        selected_output = SelectKthOutput(
            k=target_idx, name=f"select_{self.target_timepoint}"
        )(x)

        outputs = layers.Dense(
            latent_dim,
            activation="linear",
            name=f"{self.target_timepoint}_prediction",
            kernel_regularizer=l2_reg,
        )(selected_output)

        model = Model(inputs=inputs, outputs=outputs, name=f"regressor_{self.name}")

        optimizer = optimizers.Adam(
            learning_rate=self.config.learning_rate,
            clipnorm=1.0,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
        )

        model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=[
                "mae",
                "mse",
                PearsonCorrelation(),
                tf.keras.metrics.CosineSimilarity(axis=-1),
            ],
        )

        print(f"""Model summary:
   Total parameters: {model.count_params():,}
   Optimizer: Adam (lr={self.config.learning_rate})
   Loss: MSE + correlation metrics""")

        self.model = model
        return model

    def prepare_regression_data(
        self,
        trajectories: TrajectoryList,
        input_timepoints: List[str],
        target_timepoint: str,
        batch_size: int,
    ) -> tuple:
        """
        Prepare streaming regression data with masking.

        Returns:
            Tuple of (tf.data.Dataset, n_samples)
        """
        print(f"""Preparing streaming regression data
   Input timepoints: {", ".join(input_timepoints)}
   Target timepoint: {target_timepoint}
   Batch size: {batch_size}""")

        if self.encoder_model is None or self.decoder_model is None:
            raise ValueError(
                "Encoder/decoder models not loaded. Call load_encoder_decoder() first."
            )

        timepoint_map = {"t0": 0, "t1": 1, "t2": 2}
        input_indices = [timepoint_map[tp] for tp in input_timepoints]
        target_idx = timepoint_map[target_timepoint]

        sample_traj = next(
            (t for t in trajectories.trajectories if len(t.cells) == 3), None
        )
        if not sample_traj:
            raise ValueError("No valid trajectories with 3 timepoints found")

        n_genes = sample_traj.cells[0].expr.shape[0]
        latent_dim = self.encoder_model.output_shape[-1]

        def gen():
            for traj in trajectories.trajectories:
                if len(traj.cells) != 3:
                    continue

                expr_matrix = traj.to_expr().astype(np.float32)
                expr_matrix = normalize_gene_expression(expr_matrix, verbose=False)

                masked_input = np.zeros_like(expr_matrix)
                for idx in input_indices:
                    masked_input[idx] = expr_matrix[idx]

                target_normalized = expr_matrix[target_idx]
                yield masked_input, target_normalized

        output_signature = (
            tf.TensorSpec(shape=(3, n_genes), dtype=tf.float32),
            tf.TensorSpec(shape=(n_genes,), dtype=tf.float32),
        )

        dataset = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

        @tf.function
        def encode_targets(masked_inputs, normalized_targets):
            encoded_targets = self.encoder_model(normalized_targets, training=False)
            return masked_inputs, encoded_targets

        dataset = (
            dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=True)
            .batch(batch_size)
            .map(encode_targets, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )

        n_samples = sum(1 for t in trajectories.trajectories if len(t.cells) == 3)
        print(f"""   Total samples: {n_samples}
   Input shape per sample: [3, {n_genes}] (normalized genes)
   Target shape: [{n_genes}] -> [{latent_dim}] (normalized -> encoded on GPU)""")

        return dataset, n_samples

    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        val_samples: int,
        run_dir: Path = None,
        extra_callbacks: list = None,
    ) -> Dict[str, Any]:
        """Train the regressor."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        print(f"""Training regressor: {self.name}
   Using streaming dataset with batch size: {self.config.batch_size}
   Epochs: {self.config.epochs}""")

        callback_list = list(extra_callbacks or [])

        if run_dir:
            tensorboard_dir = run_dir / "tensorboard" / f"regressor_{self.name}"
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            callback_list.append(
                callbacks.TensorBoard(
                    log_dir=str(tensorboard_dir),
                    histogram_freq=1,
                    write_graph=True,
                    write_images=True,
                    update_freq="epoch",
                )
            )

            checkpoint_path = run_dir / f"regressor_{self.name}_best.keras"
            callback_list.append(
                callbacks.ModelCheckpoint(
                    str(checkpoint_path),
                    monitor="val_loss",
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1,
                )
            )

        callback_list.append(
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.patience,
                restore_best_weights=True,
                verbose=1,
            )
        )

        callback_list.append(
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=self.config.patience // 2,
                min_lr=1e-7,
                verbose=1,
            )
        )

        print("Starting training...")

        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config.epochs,
            callbacks=callback_list,
            verbose=1,
        )

        final_train_loss = self.history.history["loss"][-1]
        final_val_loss = self.history.history["val_loss"][-1]
        final_train_mae = self.history.history["mae"][-1]
        final_val_mae = self.history.history["val_mae"][-1]

        final_train_pearson = self.history.history["pearson_correlation"][-1]
        final_val_pearson = self.history.history["val_pearson_correlation"][-1]
        final_train_r2 = float(final_train_pearson**2)
        final_val_r2 = float(final_val_pearson**2)

        results = {
            "input_timepoints": self.input_timepoints,
            "target_timepoint": self.target_timepoint,
            "name": self.name,
            "embedding_type": "ae",
            "final_train_loss": float(final_train_loss),
            "final_val_loss": float(final_val_loss),
            "final_train_mae": float(final_train_mae),
            "final_val_mae": float(final_val_mae),
            "final_train_pearson": float(final_train_pearson),
            "final_val_pearson": float(final_val_pearson),
            "final_train_r2": final_train_r2,
            "final_val_r2": final_val_r2,
            "total_epochs": len(self.history.history["loss"]),
            "config": self.config.model_dump(),
        }

        print(f"""Training completed.
   Final train loss (MSE): {final_train_loss:.6f}
   Final val loss (MSE): {final_val_loss:.6f}
   Final train MAE: {final_train_mae:.6f}
   Final val MAE: {final_val_mae:.6f}
   Correlation Metrics:
      Train Pearson: {final_train_pearson:.4f} (r²={final_train_r2:.4f}) | Val Pearson: {final_val_pearson:.4f} (r²={final_val_r2:.4f})
   Total epochs: {results["total_epochs"]}""")

        return results

    def evaluate_in_gene_space(self, val_dataset: tf.data.Dataset) -> Dict[str, Any]:
        """Gene space evaluation after training."""
        y_true_latent, y_pred_latent = collect_predictions_from_dataset(
            self.model, val_dataset, verbose=False
        )

        return evaluate_gene_space(
            y_true_latent,
            y_pred_latent,
            self.decoder_model,
            verbose=True,
            title="FINAL GENE SPACE EVALUATION",
        )

    def save(self, filepath: Path):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        save_model(self.model, str(filepath))
        print(f"Saved regressor to: {filepath}")
