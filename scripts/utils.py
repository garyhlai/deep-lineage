"""
Utilities for the Deep Lineage pipeline.

Combines run directory management, data loading, argument parsing,
training callbacks, and shared helper functions.
"""

import argparse
import json
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import tensorflow as tf
from tensorflow.keras.models import load_model

from deep_lineage.schema import TrajectoryList
from deep_lineage.utils import make_json_serializable


# Run directory management


def get_timestamp_file() -> Path:
    """Path to the file storing the current run timestamp."""
    return Path("local_bucket/.current_run_timestamp")


def generate_run_timestamp() -> str:
    """Generate a human-readable timestamp string."""
    return time.strftime("%Y%m%d-%H%M%S")


def save_run_timestamp(timestamp: str) -> None:
    """Save a timestamp to the timestamp file."""
    os.makedirs("local_bucket", exist_ok=True)
    with open(get_timestamp_file(), "w") as f:
        f.write(timestamp)


def get_run_timestamp() -> Optional[str]:
    """Get the saved run timestamp if available."""
    timestamp_file = get_timestamp_file()
    if timestamp_file.exists():
        with open(timestamp_file, "r") as f:
            return f.read().strip()
    return None


def get_or_create_run_dir() -> Path:
    """Get the current run directory, or create a new one if none exists."""
    timestamp = get_run_timestamp()

    if not timestamp:
        timestamp = generate_run_timestamp()
        save_run_timestamp(timestamp)

    run_dir = Path(f"local_bucket/run_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def get_run_dir() -> Path:
    """Get the current run directory without creating a new one."""
    timestamp = get_run_timestamp()
    if not timestamp:
        raise ValueError(
            "No active run found. Start a new pipeline run with process_data.py first."
        )

    run_dir = Path(f"local_bucket/run_{timestamp}")
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def clean_previous_run() -> None:
    """Remove the current run timestamp if it exists."""
    timestamp_file = get_timestamp_file()
    if timestamp_file.exists():
        timestamp_file.unlink()


# Data loading


def load_trajectory_data(
    run_dir: Path, split: str = "train", custom_path: str = None
) -> TrajectoryList:
    """Load trajectory data from run directory for specified split."""
    if custom_path:
        custom_path = Path(custom_path)
        if not custom_path.exists():
            raise FileNotFoundError(f"Custom trajectory file not found: {custom_path}")
        print(f"Loading {split} trajectories from custom path: {custom_path}")
        return TrajectoryList.load(str(custom_path))

    split_files = [
        f"trajectories_{split}.pkl.gz",
        "trajectories.pkl.gz",
    ]

    for filename in split_files:
        filepath = run_dir / filename
        if filepath.exists():
            print(f"Loading {split} trajectories from: {filepath}")
            return TrajectoryList.load(str(filepath))

    raise FileNotFoundError(
        f"No trajectory file found in {run_dir} for split '{split}'. Looked for: {split_files}"
    )


def log_trajectory_split_info(trajectories: TrajectoryList, split_name: str):
    """Log trajectory group information for verification."""
    group_ids = set()
    trajectory_count = len(trajectories.trajectories)

    for traj in trajectories.trajectories:
        group_ids.add(traj.trajectory_group_id)

    info = f"""   {split_name} split verification:
      Trajectories: {trajectory_count}
      Unique trajectory groups: {len(group_ids)}"""
    if group_ids:
        sample_groups = sorted(list(group_ids))[:3]
        info += f"""
      Group ID range: {min(group_ids)} - {max(group_ids)}
      Sample group IDs: {sample_groups}"""
    print(info)

    return group_ids


def check_data_leakage(train_group_ids: set, val_group_ids: set):
    """Check for data leakage between train and validation splits."""
    overlap = train_group_ids & val_group_ids
    if overlap:
        print(
            f"   WARNING: Found {len(overlap)} overlapping group IDs between train/val"
        )
    else:
        print("   No data leakage detected between train/val splits.")
    return len(overlap) == 0


# Model loading


def load_trained_models(run_dir: Path) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """Load trained autoencoder and encoder models."""
    autoencoder_path = run_dir / "autoencoder_final.keras"
    encoder_path = run_dir / "encoder_final.keras"

    if not autoencoder_path.exists():
        autoencoder_path = run_dir / "autoencoder_best.keras"
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder model not found at {encoder_path}")

    print(f"""Loading models:
   Autoencoder: {autoencoder_path}
   Encoder: {encoder_path}""")

    autoencoder = load_model(str(autoencoder_path))
    encoder = load_model(str(encoder_path))

    return autoencoder, encoder


# Argparse helpers


def add_common_training_args(parser: argparse.ArgumentParser):
    """Add common training arguments to argument parser."""
    parser.add_argument(
        "--run_dir",
        type=str,
        help="Run directory containing trajectory data (default: use current run)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")


def add_model_training_args(parser: argparse.ArgumentParser):
    """Add model-specific training arguments to argument parser."""
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--patience", type=int, default=50, help="Early stopping patience"
    )


# Training helpers


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_keras_callbacks(
    run_dir: Path, model_name: str, monitor: str = "val_loss", patience: int = 10
) -> List[tf.keras.callbacks.Callback]:
    """Get standard Keras callbacks for training."""
    callbacks_list = []

    tensorboard_dir = run_dir / "tensorboard" / model_name
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    callbacks_list.append(
        tf.keras.callbacks.TensorBoard(
            log_dir=str(tensorboard_dir),
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq="epoch",
        )
    )

    checkpoint_path = run_dir / f"{model_name}_best.keras"
    callbacks_list.append(
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        )
    )

    callbacks_list.append(
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        )
    )

    callbacks_list.append(
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.95,
            patience=patience // 2,
            min_lr=1e-7,
            verbose=1,
        )
    )

    return callbacks_list


# Results I/O


def save_results_json(results: Dict[str, Any], filepath: Path):
    """Save results dictionary to JSON file."""
    json_results = make_json_serializable(results)
    with open(filepath, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved results to: {filepath}")


def load_results_json(filepath: Path) -> Dict[str, Any]:
    """Load results dictionary from JSON file, returning {} on missing file."""
    if not Path(filepath).exists():
        return {}
    with open(filepath, "r") as f:
        return json.load(f)


# Shared computation helpers


def compute_correlations(X_true: np.ndarray, X_pred: np.ndarray):
    """Compute per-sample Pearson and Spearman correlations."""
    df_true = pd.DataFrame(X_true)
    df_pred = pd.DataFrame(X_pred)
    pcorr = df_pred.corrwith(df_true, axis=1, method="pearson")
    spcorr = df_pred.corrwith(df_true, axis=1, method="spearman")
    return pcorr, spcorr
