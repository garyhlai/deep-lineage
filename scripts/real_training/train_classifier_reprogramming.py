"""
Classifier Training for Reprogramming Dataset (Original Deep Lineage Reproduction)

Trains a bidirectional LSTM/GRU classifier matching the original architecture:
- Masking layer for zero-padded timepoints
- N bidirectional RNN layers with merge_mode='ave'
- Softmax output for 2-class classification (Reprogrammed vs Failed)

Reference: original_deep_lineage/Classification_Model_Training_On_Reprogramming/
           Train_Classification_On_Reprogramming_Encoded_Time_Series_Two_Classes.py
"""

import argparse
import numpy as np
import h5py
import json
from pathlib import Path
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd


def load_reprogramming_data(dataset_path: str) -> Tuple[np.ndarray, ...]:
    """Load reprogramming dataset for classification."""
    print(f"Loading dataset from {dataset_path}...")

    with h5py.File(dataset_path, "r") as f:
        X_train = np.array(f["X_train"])
        X_val = np.array(f["X_val"])
        X_test = np.array(f["X_test"])
        y_train = np.array(f["y_train"])
        y_val = np.array(f["y_val"])
        y_test = np.array(f["y_test"])

        json.loads(f.attrs["config"])
        classes = json.loads(f.attrs["classes"])

    print("Dataset shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"  Classes: {classes}")

    return X_train, X_val, X_test, y_train, y_val, y_test, classes


def encode_time_series(X: np.ndarray, encoder: Model) -> np.ndarray:
    """Encode time series data using the autoencoder."""
    N, T, n_genes = X.shape
    latent_dim = encoder.output_shape[-1]

    X_flat = X.reshape(-1, n_genes)
    X_encoded_flat = encoder.predict(X_flat, verbose=0)
    X_encoded = X_encoded_flat.reshape(N, T, latent_dim)

    return X_encoded


def build_classifier(
    timesteps: int,
    features: int,
    num_classes: int,
    num_layers: int = 4,
    dropout: float = 0.25,
    model_type: str = "LSTM",
) -> Model:
    """Build bidirectional RNN classifier matching original architecture."""
    print("\nBuilding classifier:")
    print(f"  Timesteps: {timesteps}")
    print(f"  Features: {features}")
    print(f"  Num classes: {num_classes}")
    print(f"  Num layers: {num_layers}")
    print(f"  Dropout: {dropout}")
    print(f"  Model type: {model_type}")

    RNNCell = layers.LSTM if model_type == "LSTM" else layers.GRU

    inputs = layers.Input(shape=(timesteps, features), name="encoded_input")

    x = layers.Masking(mask_value=0.0, name="masking")(inputs)

    for i in range(num_layers - 1):
        x = layers.Bidirectional(
            RNNCell(features, return_sequences=True, dropout=dropout),
            merge_mode="ave",
            name=f"birnn_{i + 1}",
        )(x)

    x = layers.Bidirectional(
        RNNCell(features, return_sequences=False, dropout=dropout),
        merge_mode="ave",
        name=f"birnn_{num_layers}",
    )(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = Model(inputs, outputs, name="classifier")

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["categorical_crossentropy", "accuracy"],
    )

    print(f"  Total params: {model.count_params():,}")

    return model


def train_classifier(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100000,
    batch_size: int = 1024,
    patience: int = 100,
    output_dir: Path = None,
):
    """Train classifier with original parameters."""
    print("\nTraining classifier:")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Validation samples: {len(X_val):,}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Patience: {patience}")

    callback_list = []

    es = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        min_delta=5e-6,
        restore_best_weights=True,
        verbose=1,
    )
    callback_list.append(es)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        cp = callbacks.ModelCheckpoint(
            filepath=str(output_dir / "classifier_best.keras"),
            save_best_only=True,
            monitor="val_loss",
            verbose=0,
        )
        callback_list.append(cp)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callback_list,
        shuffle=True,
        verbose=1,
    )

    return history


def evaluate_classifier(
    model: Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: list,
    output_dir: Path = None,
):
    """Evaluate classifier and compute metrics."""
    print("\nEvaluating classifier...")

    y_pred = model.predict(X_test, verbose=0)

    y_test_classes = np.argmax(y_test, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)

    test_loss = model.evaluate(X_test, y_test, verbose=0)[0]

    print("\nTest Results:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"  Loss (categorical crossentropy): {test_loss:.6f}")
    print("\n  Confusion Matrix:")
    print(f"    {classes}")
    print(conf_matrix)

    print("\n  Classification Report:")
    print(
        classification_report(
            y_test_classes,
            y_pred_classes,
            target_names=classes,
            labels=list(range(len(classes))),
        )
    )

    results = {
        "accuracy": float(accuracy),
        "loss": float(test_loss),
        "confusion_matrix": conf_matrix.tolist(),
        "classes": classes,
    }

    if output_dir:
        with open(output_dir / "classification_results.json", "w") as f:
            json.dump(results, f, indent=2)

        pd.DataFrame(
            {
                "y_true": y_test_classes,
                "y_pred": y_pred_classes,
                "prob_class0": y_pred[:, 0],
                "prob_class1": y_pred[:, 1],
            }
        ).to_csv(output_dir / "predictions_test.csv", index=False)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train classifier for reprogramming dataset (original Deep Lineage reproduction)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="reprogramming_dataset.h5",
        help="Path to reprogramming dataset H5 file",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        required=True,
        help="Path to trained encoder model (.keras)",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=4,
        help="Number of bidirectional RNN layers (default: 4)",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.25, help="Dropout rate (default: 0.25)"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="LSTM",
        choices=["LSTM", "GRU"],
        help="RNN cell type (default: LSTM)",
    )
    parser.add_argument(
        "--epochs", type=int, default=100000, help="Maximum epochs (default: 100000)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="Batch size (default: 1024)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Early stopping patience (default: 100)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/reprog_cls",
        help="Output directory for models and results",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--input_days",
        type=int,
        default=4,
        help="Number of input timepoints to use for prediction (default: 4 = Days 6,9,12,15)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode with single batch"
    )

    args = parser.parse_args()

    if args.debug:
        args.epochs = 2
        args.patience = 1
        args.batch_size = 32
        print("DEBUG MODE: epochs=2, patience=1, batch_size=32, single batch of data")

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "encoder_path": args.encoder,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "model_type": args.model_type,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "seed": args.seed,
        "input_days": args.input_days,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    X_train, X_val, X_test, y_train, y_val, y_test, classes = load_reprogramming_data(
        args.dataset
    )

    if args.debug:
        X_train, y_train = X_train[: args.batch_size], y_train[: args.batch_size]
        X_val, y_val = X_train, y_train
        X_test, y_test = X_train, y_train

    print(f"\nLoading encoder from {args.encoder}...")
    encoder = load_model(args.encoder)
    latent_dim = encoder.output_shape[-1]
    print(f"  Encoder latent dimension: {latent_dim}")

    print("\nEncoding time series data...")
    X_train_enc = encode_time_series(X_train, encoder)
    X_val_enc = encode_time_series(X_val, encoder)
    X_test_enc = encode_time_series(X_test, encoder)

    print(
        f"  Full encoded shapes: train={X_train_enc.shape}, val={X_val_enc.shape}, test={X_test_enc.shape}"
    )

    if args.input_days < X_train_enc.shape[1]:
        print(f"\n  Using first {args.input_days} timepoints for early fate prediction")
        X_train_enc = X_train_enc[:, : args.input_days, :]
        X_val_enc = X_val_enc[:, : args.input_days, :]
        X_test_enc = X_test_enc[:, : args.input_days, :]
        print(
            f"  Sliced shapes: train={X_train_enc.shape}, val={X_val_enc.shape}, test={X_test_enc.shape}"
        )

    timesteps = X_train_enc.shape[1]
    features = X_train_enc.shape[2]
    num_classes = y_train.shape[1]

    model = build_classifier(
        timesteps=timesteps,
        features=features,
        num_classes=num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        model_type=args.model_type,
    )

    history = train_classifier(
        model,
        X_train_enc,
        y_train,
        X_val_enc,
        y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        output_dir=output_dir,
    )

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(output_dir / "training_history.csv", index=False)

    results = evaluate_classifier(model, X_test_enc, y_test, classes, output_dir)

    model.save(output_dir / "classifier.keras")

    print(
        f"""
Training complete!
  Output directory: {output_dir}
  Accuracy: {results["accuracy"]:.4f}
  Model saved: classifier.keras

Next step: Train regressor with:
  uv run python scripts/real_training/train_regressor_reprogramming.py --encoder {args.encoder}
"""
    )


if __name__ == "__main__":
    main()
