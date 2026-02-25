"""
Validate reprogramming datasets for correctness.

Usage:
    uv run python scripts/validate_reprogramming_dataset.py
"""

import h5py
import json
import os
import sys

import numpy as np

EXPECTED_TIMEPOINTS = 6
EXPECTED_GENES = 1000
EXPECTED_CLASSES = 2

CUMULATIVE_PATH = "reprogramming_dataset_cumulative.h5"
SINGLE_PATH = "reprogramming_dataset_single.h5"


def check_shapes(f):
    """Verify array dimensions are consistent."""
    passed = True

    for split in ["train", "val", "test"]:
        X = f[f"X_{split}"]
        y = f[f"y_{split}"]

        if X.shape[0] != y.shape[0]:
            print(f"  FAIL: X_{split} and y_{split} sample count mismatch")
            passed = False

        if (
            len(X.shape) != 3
            or X.shape[1] != EXPECTED_TIMEPOINTS
            or X.shape[2] != EXPECTED_GENES
        ):
            print(
                f"  FAIL: X_{split} shape {X.shape}, expected (n, {EXPECTED_TIMEPOINTS}, {EXPECTED_GENES})"
            )
            passed = False

        if len(y.shape) != 2 or y.shape[1] != EXPECTED_CLASSES:
            print(
                f"  FAIL: y_{split} shape {y.shape}, expected (n, {EXPECTED_CLASSES})"
            )
            passed = False

    if passed:
        print("  PASS: All shapes correct")
    return passed


def check_dtypes(f):
    """Verify arrays are float32."""
    passed = True

    for split in ["train", "val", "test"]:
        for prefix in ["X", "y"]:
            key = f"{prefix}_{split}"
            if f[key].dtype != np.float32:
                print(f"  FAIL: {key} dtype is {f[key].dtype}, expected float32")
                passed = False

    if passed:
        print("  PASS: All dtypes are float32")
    return passed


def check_class_balance(f):
    """Check class distribution in each split."""
    passed = True

    for split in ["train", "val", "test"]:
        y = f[f"y_{split}"][:]
        failed_count = np.sum(y[:, 0] == 1)
        reprog_count = np.sum(y[:, 1] == 1)

        print(f"  {split}: Failed={failed_count}, Reprogrammed={reprog_count}")

        if failed_count != reprog_count:
            print("    WARN: Classes not balanced")

    return passed


def check_nan_inf(f):
    """Check for NaN or Inf values."""
    passed = True

    for split in ["train", "val", "test"]:
        X = f[f"X_{split}"][:]
        y = f[f"y_{split}"][:]

        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print(f"  FAIL: X_{split} contains NaN or Inf")
            passed = False

        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            print(f"  FAIL: y_{split} contains NaN or Inf")
            passed = False

    if passed:
        print("  PASS: No NaN or Inf values")
    return passed


def check_metadata(f):
    """Check metadata exists."""
    passed = True

    if "gene_names" not in f:
        print("  FAIL: gene_names missing")
        passed = False
    else:
        n_genes = len(f["gene_names"])
        if n_genes != EXPECTED_GENES:
            print(
                f"  FAIL: gene_names has {n_genes} entries, expected {EXPECTED_GENES}"
            )
            passed = False

    if "config" not in f.attrs:
        print("  FAIL: config attribute missing")
        passed = False
    else:
        config = json.loads(f.attrs["config"])
        print(f"  Config: {config}")

    if "classes" not in f.attrs:
        print("  FAIL: classes attribute missing")
        passed = False
    else:
        classes = json.loads(f.attrs["classes"])
        if classes != ["Failed", "Reprogrammed"]:
            print(f"  FAIL: classes is {classes}, expected ['Failed', 'Reprogrammed']")
            passed = False

    if passed:
        print("  PASS: Metadata complete")
    return passed


def validate_dataset(path):
    """Run all validation checks on a dataset."""
    print(f"Validating {path}...\n")

    with h5py.File(path, "r") as f:
        results = []

        print("1. Shape checks:")
        results.append(check_shapes(f))

        print("\n2. Dtype checks:")
        results.append(check_dtypes(f))

        print("\n3. Class balance:")
        results.append(check_class_balance(f))

        print("\n4. NaN/Inf checks:")
        results.append(check_nan_inf(f))

        print("\n5. Metadata:")
        results.append(check_metadata(f))

    return all(results)


def check_test_sets_match(cumulative_path, single_path):
    """Verify test sets are identical between cumulative and single datasets."""
    print("Comparing test sets...")

    with h5py.File(cumulative_path, "r") as f_cum:
        X_test_cum = f_cum["X_test"][:]
        y_test_cum = f_cum["y_test"][:]

    with h5py.File(single_path, "r") as f_single:
        X_test_single = f_single["X_test"][:]
        y_test_single = f_single["y_test"][:]

    passed = True

    if X_test_cum.shape != X_test_single.shape:
        print(
            f"  FAIL: X_test shapes differ: {X_test_cum.shape} vs {X_test_single.shape}"
        )
        passed = False
    elif not np.allclose(X_test_cum, X_test_single):
        print("  FAIL: X_test values differ")
        passed = False

    if y_test_cum.shape != y_test_single.shape:
        print(
            f"  FAIL: y_test shapes differ: {y_test_cum.shape} vs {y_test_single.shape}"
        )
        passed = False
    elif not np.array_equal(y_test_cum, y_test_single):
        print("  FAIL: y_test values differ")
        passed = False

    if passed:
        print("  PASS: Test sets are identical")

    return passed


def main():
    results = []

    # Check which datasets exist
    cumulative_exists = os.path.exists(CUMULATIVE_PATH)
    single_exists = os.path.exists(SINGLE_PATH)

    if not cumulative_exists and not single_exists:
        print("ERROR: No datasets found")
        print(f"  Expected: {CUMULATIVE_PATH} or {SINGLE_PATH}")
        sys.exit(1)

    # Validate cumulative dataset
    if cumulative_exists:
        results.append(validate_dataset(CUMULATIVE_PATH))
        print()

    # Validate single dataset
    if single_exists:
        results.append(validate_dataset(SINGLE_PATH))
        print()

    # Compare test sets if both exist
    if cumulative_exists and single_exists:
        print("=" * 40)
        print("Cross-dataset validation:\n")
        results.append(check_test_sets_match(CUMULATIVE_PATH, SINGLE_PATH))
        print()

    # Final result
    print("=" * 40)
    if all(results):
        print("ALL VALIDATIONS PASSED")
    else:
        print("VALIDATION FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
