"""Basic regression check for the voltage fingerprinting pipeline."""

import sys
from pathlib import Path

# Allow running as a standalone script from project root
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from dataset_loader import CANDatasetLoader
from voltage_fingerprinting import VoltageFingerprinter


def _train_test_split(X, y, df, train_ratio: float = 0.7, seed: int = 42):
    """Simple shuffled train/test split that keeps ECU IDs aligned."""
    rng = np.random.default_rng(seed)
    n = len(X)
    indices = np.arange(n)
    rng.shuffle(indices)

    split = int(train_ratio * n)
    train_idx = indices[:split]
    test_idx = indices[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    ecu_train = df.iloc[train_idx]["ecu_id"].to_numpy()
    ecu_test = df.iloc[test_idx]["ecu_id"].to_numpy()

    return (X_train, y_train, ecu_train), (X_test, y_test, ecu_test)


def _print_summary(y_true, y_pred) -> None:
    """Print a compact summary of classification performance."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    print("\n=== Voltage Fingerprinting Summary ===")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Precision    : {prec:.4f}")
    print(f"Recall (TPR) : {rec:.4f}")
    print(f"F1-score     : {f1:.4f}")
    print()
    print("Confusion matrix (normal=0, attack=1):")
    print(f"  TP: {tp:4d}   FP: {fp:4d}")
    print(f"  TN: {tn:4d}   FN: {fn:4d}")

    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0

    print()
    print(f"TPR (recall attacks)     : {tpr:.4f}")
    print(f"FPR (false alarms)       : {fpr:.4f}")
    print(f"TNR (specificity normal) : {tnr:.4f}")
    print("======================================\n")


def _print_sample_predictions(
    fingerprinter, X_test, y_test, ecu_test, y_pred, max_samples: int = 10
) -> None:
    """Show a few concrete examples to sanity–check behavior."""
    print("Example predictions (first few test samples):")
    print("-" * 72)
    n = min(max_samples, len(X_test))
    for i in range(n):
        claimed_ecu = int(ecu_test[i])
        pred_ecu, confidence = fingerprinter.predict(X_test[i])
        is_anomaly, _ = fingerprinter.detect_anomaly(X_test[i], claimed_ecu)

        actual = "ATTACK" if y_test[i] == 1 else "normal"
        detected = "ATTACK" if is_anomaly else "normal"
        ok = "✓" if y_test[i] == y_pred[i] else "✗"

        print(
            f"{ok} sample {i+1:2d}: "
            f"actual={actual:6s}, detected={detected:6s}, "
            f"claimed_ecu=0x{claimed_ecu:03X}, "
            f"pred_ecu=0x{pred_ecu:03X}, conf={confidence:.3f}"
        )
    print()


def test_voltage_fingerprinting():
    """End-to-end test of the voltage fingerprinting pipeline on synthetic data."""

    print("\n=== Voltage fingerprinting test ===")

    # 1) Generate synthetic voltage traces with ECU IDs and labels.
    loader = CANDatasetLoader(data_path=".")
    voltage_df = loader._create_sample_voltage_data(n_samples=500)

    print(
        f"Generated {len(voltage_df)} samples "
        f"(normal={np.sum(voltage_df['label'] == 0)}, "
        f"attack={np.sum(voltage_df['label'] == 1)})"
    )

    # 2) Preprocess into model inputs.
    X, y = loader.preprocess_voltage_data(voltage_df)

    # 3) Train/test split.
    (X_train, y_train, ecu_train), (X_test, y_test, ecu_test) = _train_test_split(
        X, y, voltage_df
    )
    print(
        f"Train: {len(X_train)}  (normal={np.sum(y_train == 0)}, attack={np.sum(y_train == 1)})"
    )
    print(
        f"Test : {len(X_test)}  (normal={np.sum(y_test == 0)}, attack={np.sum(y_test == 1)})"
    )

    # 4) Train the fingerprinter on normal traffic only.
    fingerprinter = VoltageFingerprinter(threshold=0.4)
    normal_mask = y_train == 0
    fingerprinter.train(X_train[normal_mask], ecu_train[normal_mask])

    # 5) Run detection on test set.
    y_pred = []
    for signal, ecu_id in zip(X_test, ecu_test):
        is_anomaly, _ = fingerprinter.detect_anomaly(signal, ecu_id)
        y_pred.append(1 if is_anomaly else 0)
    y_pred = np.asarray(y_pred, dtype=int)

    # 6) Report metrics and a few sample predictions.
    _print_summary(y_test, y_pred)
    _print_sample_predictions(fingerprinter, X_test, y_test, ecu_test, y_pred)

    # Light sanity check for automated runs: require non-trivial accuracy.
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.6, f"Voltage fingerprinting accuracy too low: {accuracy:.3f}"

    return accuracy


if __name__ == "__main__":
    test_voltage_fingerprinting()
