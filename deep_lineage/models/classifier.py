"""Classifier for cell fate prediction."""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, List
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks
from tensorflow.keras.models import save_model, load_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from deep_lineage.schema import TrajectoryList, LSTMConfig


def _infer_dims(
    trajectories: TrajectoryList, timepoints: List[str]
) -> Tuple[int, int, int]:
    """Infer (n_timesteps, n_genes, n_classes)."""
    n_timesteps = len(timepoints)
    for traj in trajectories.trajectories:
        if len(traj.cells) == 3:
            n_genes = int(np.asarray(traj.cells[0].expr).shape[0])
            return n_timesteps, n_genes, 2
    raise RuntimeError(
        "Could not infer input dims; need a trajectory with 3 timepoints."
    )


class Classifier:
    """
    Classifier following Deep Lineage methodology.

    Uses Bidirectional LSTM to classify cell fate from encoded time series data.
    Supports different temporal contexts (t0 only, t0+t1, t0+t1+t2).
    """

    def __init__(
        self,
        config: LSTMConfig,
        timepoints: List[str],
        name: str,
        ae_name: str = None,
    ):
        self.config = config
        self.timepoints = timepoints
        self.name = name
        self.ae_name = ae_name
        self.model = None
        self.encoder_model = None
        self.history = None

    def load_encoder(self, run_dir: Path):
        """Load the trained autoencoder encoder model."""
        if self.ae_name:
            encoder_path = run_dir / f"encoder_{self.ae_name}_final.keras"
        else:
            encoder_path = run_dir / "encoder_final.keras"

        if not encoder_path.exists():
            raise FileNotFoundError(f"Autoencoder encoder not found at {encoder_path}")

        self.encoder_model = load_model(str(encoder_path))
        self.encoder_model.trainable = False
        print(f"Loaded encoder from: {encoder_path} (frozen)")

    def build_model(
        self, n_timesteps: int, latent_dim: int, n_classes: int = 2
    ) -> Model:
        """
        Build classifier.

        Args:
            n_timesteps: Number of timesteps in input sequence
            latent_dim: Number of genes per timestep (encoder applied on-the-fly)
            n_classes: Number of output classes
        """
        print(f"""Building classifier: {self.name}
   Input: {n_timesteps} timesteps x {latent_dim} genes
   Architecture: {self.config.num_layers} Bidirectional {self.config.cell_type} layers
   Output: {n_classes} classes, Dropout: {self.config.dropout}""")

        if self.encoder_model is None:
            raise ValueError("Encoder model not loaded. Call load_encoder() first.")

        inputs = layers.Input(shape=(n_timesteps, latent_dim), name="raw_timeseries")

        encoded = layers.TimeDistributed(self.encoder_model, name="td_encoder")(inputs)

        l2_reg = (
            tf.keras.regularizers.l2(self.config.l2) if self.config.l2 > 0 else None
        )

        x = encoded
        RNN = layers.LSTM if self.config.cell_type == "LSTM" else layers.GRU

        for layer_idx in range(self.config.num_layers - 1):
            x = layers.Bidirectional(
                RNN(
                    self.config.hidden_dim,
                    return_sequences=True,
                    dropout=self.config.dropout,
                    kernel_regularizer=l2_reg,
                    recurrent_regularizer=l2_reg,
                    name=f"{self.config.cell_type.lower()}_layer_{layer_idx}",
                ),
                merge_mode="ave",
                name=f"bidirectional_{layer_idx}",
            )(x)

        lstm_out = layers.Bidirectional(
            RNN(
                self.config.hidden_dim,
                return_sequences=False,
                dropout=self.config.dropout,
                kernel_regularizer=l2_reg,
                recurrent_regularizer=l2_reg,
                name=f"final_{self.config.cell_type.lower()}",
            ),
            merge_mode="ave",
            name="final_bidirectional",
        )(x)

        outputs = layers.Dense(
            n_classes,
            activation="softmax",
            name="cell_fate_prediction",
            kernel_regularizer=l2_reg,
        )(lstm_out)

        model = Model(
            inputs=inputs,
            outputs=outputs,
            name=f"classifier_{self.name}",
        )

        optimizer = optimizers.Adam(
            learning_rate=self.config.learning_rate,
            clipnorm=1.0,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
        )

        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy", "categorical_crossentropy"],
        )

        print(f"""Model summary:
   Total parameters: {model.count_params():,}
   Optimizer: Adam (lr={self.config.learning_rate})
   Loss: categorical_crossentropy + accuracy""")

        self.model = model
        return model

    def prepare_data(
        self,
        trajectories: TrajectoryList,
        timepoints: List[str],
        batch_size: int,
    ) -> tf.data.Dataset:
        """
        Prepare classification data as a streaming Dataset.

        Returns:
            tf.data.Dataset yielding (raw_sequence [T,G], one_hot_label [2])
        """
        print(f"Preparing classification data for: {', '.join(timepoints)}")

        if self.encoder_model is None:
            raise ValueError("Encoder model not loaded. Call load_encoder() first.")

        timepoint_map = {"t0": 0, "t1": 1, "t2": 2}
        timepoint_indices = [timepoint_map[tp] for tp in timepoints]

        def _one_hot(state: str) -> np.ndarray:
            if state == "fate_0":
                return np.array([1.0, 0.0], dtype=np.float32)
            if state == "fate_1":
                return np.array([0.0, 1.0], dtype=np.float32)
            raise ValueError(f"Unknown cell state: {state}")

        def gen():
            for traj in trajectories.trajectories:
                if len(traj.cells) != 3:
                    continue
                seq = np.stack(
                    [
                        np.asarray(traj.cells[tp_idx].expr, dtype=np.float32)
                        for tp_idx in timepoint_indices
                    ],
                    axis=0,
                )
                label = _one_hot(traj.cells[2].state)
                yield seq, label

        n_timesteps, n_genes, _ = _infer_dims(trajectories, timepoints)

        output_signature = (
            tf.TensorSpec(shape=(n_timesteps, n_genes), dtype=tf.float32),
            tf.TensorSpec(shape=(2,), dtype=tf.float32),
        )

        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
        ds = (
            ds.shuffle(8192, reshuffle_each_iteration=True)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        return ds

    def train(
        self,
        X_train: tf.data.Dataset,
        X_val: tf.data.Dataset,
        run_dir: Path = None,
    ) -> Dict[str, Any]:
        """Train the classifier using streaming datasets."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        print(f"""Training classifier: {self.name}
   Epochs: {self.config.epochs}, Batch size: {self.config.batch_size}""")

        callback_list = []

        if run_dir:
            tensorboard_dir = run_dir / "tensorboard" / f"classifier_{self.name}"
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

            checkpoint_path = run_dir / f"classifier_{self.name}_best.keras"
            callback_list.append(
                callbacks.ModelCheckpoint(
                    str(checkpoint_path),
                    monitor="val_accuracy",
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1,
                    mode="max",
                )
            )

        callback_list.append(
            callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=self.config.patience,
                restore_best_weights=True,
                verbose=1,
                mode="max",
            )
        )

        callback_list.append(
            callbacks.ReduceLROnPlateau(
                monitor="val_accuracy",
                factor=0.5,
                patience=self.config.patience // 2,
                min_lr=1e-7,
                verbose=1,
                mode="max",
            )
        )

        print("Starting training...")

        self.history = self.model.fit(
            X_train,
            validation_data=X_val,
            epochs=self.config.epochs,
            callbacks=callback_list,
            verbose=1,
        )

        final_train_loss = self.history.history["loss"][-1]
        final_val_loss = self.history.history["val_loss"][-1]
        final_train_acc = self.history.history["accuracy"][-1]
        final_val_acc = self.history.history["val_accuracy"][-1]

        y_true, y_pred = [], []
        for Xb, yb in X_val:
            probs = self.model.predict(Xb, verbose=0)
            y_pred.append(np.argmax(probs, axis=1))
            y_true.append(np.argmax(yb.numpy(), axis=1))
        y_pred = np.concatenate(y_pred, axis=0) if y_pred else np.array([])
        y_true = np.concatenate(y_true, axis=0) if y_true else np.array([])

        if y_true.size > 0:
            val_accuracy = accuracy_score(y_true, y_pred)
            val_confusion = confusion_matrix(y_true, y_pred)
            val_report = classification_report(y_true, y_pred, output_dict=True)
        else:
            val_accuracy, val_confusion, val_report = 0.0, np.zeros((2, 2), int), {}

        results = {
            "timepoints": self.timepoints,
            "name": self.name,
            "final_train_loss": float(final_train_loss),
            "final_val_loss": float(final_val_loss),
            "final_train_accuracy": float(final_train_acc),
            "final_val_accuracy": float(final_val_acc),
            "detailed_val_accuracy": float(val_accuracy),
            "confusion_matrix": val_confusion.tolist(),
            "classification_report": val_report,
            "total_epochs": len(self.history.history["loss"]),
            "config": self.config.model_dump(),
        }

        print(f"""Training completed.
   Final train accuracy: {final_train_acc:.4f} ({final_train_acc * 100:.1f}%)
   Final val accuracy: {final_val_acc:.4f} ({final_val_acc * 100:.1f}%)
   Detailed val accuracy: {val_accuracy:.4f}
   Total epochs: {results["total_epochs"]}""")

        return results

    def save(self, filepath: Path):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        save_model(self.model, str(filepath))
        print(f"Saved classifier to: {filepath}")
