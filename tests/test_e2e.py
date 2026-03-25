#!/usr/bin/env python
"""End-to-end test script for tclf.

This script demonstrates the complete pipeline:
1. Create synthetic data
2. Train a classifier
3. Save the model
4. Load the model
5. Make predictions
6. Evaluate results

Run this script with:
    python tests/test_e2e.py
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from tclf.classical_classifier import ClassicalClassifier


def create_synthetic_data(n_samples: int = 100, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create synthetic training and test data.

    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, test_data)
    """
    np.random.seed(seed)

    def make_df(n: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "trade_price": np.random.uniform(99, 101, n),
                "trade_size": np.random.randint(100, 1000, n),
                "bid_ex": np.random.uniform(98.5, 99.5, n),
                "ask_ex": np.random.uniform(100.5, 101.5, n),
                "bid_best": np.random.uniform(98.8, 99.8, n),
                "ask_best": np.random.uniform(100.2, 101.2, n),
                "bid_size_ex": np.random.randint(500, 2000, n),
                "ask_size_ex": np.random.randint(500, 2000, n),
                "price_ex_lag": np.random.uniform(99, 101, n),
                "price_ex_lead": np.random.uniform(99, 101, n),
                "price_all_lag": np.random.uniform(99, 101, n),
                "price_all_lead": np.random.uniform(99, 101, n),
            }
        )

    train_data = make_df(n_samples)
    test_data = make_df(n_samples // 4)

    return train_data, test_data


def create_true_labels(test_data: pd.DataFrame, seed: int = 42) -> np.ndarray:
    """Create synthetic true labels for evaluation.

    Args:
        test_data: Test data
        seed: Random seed

    Returns:
        Array of labels (-1 or 1)
    """
    np.random.seed(seed)
    return np.random.choice([-1, 1], size=len(test_data))


def test_basic_pipeline() -> None:
    """Test basic classification pipeline."""
    print("\n" + "=" * 60)
    print("TEST: Basic Pipeline")
    print("=" * 60)

    train_data, test_data = create_synthetic_data()

    clf = ClassicalClassifier(
        layers=[("quote", "ex")],
        random_state=42,
    )

    columns = ["trade_price", "bid_ex", "ask_ex"]

    print(f"Training on {len(train_data)} samples...")
    clf.fit(train_data[columns])

    print(f"Predicting on {len(test_data)} samples...")
    predictions = clf.predict(test_data[columns])
    probabilities = clf.predict_proba(test_data[columns])

    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Unique predictions: {np.unique(predictions)}")
    print("✓ Basic pipeline test passed")


def test_save_load_pipeline() -> None:
    """Test train -> save -> load -> predict pipeline."""
    print("\n" + "=" * 60)
    print("TEST: Save/Load Pipeline")
    print("=" * 60)

    train_data, test_data = create_synthetic_data()
    columns = ["trade_price", "bid_ex", "ask_ex"]

    clf = ClassicalClassifier(
        layers=[("quote", "ex")],
        random_state=42,
    )

    print("Training classifier...")
    clf.fit(train_data[columns])

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        model_path = f.name

    try:
        print(f"Saving model to {model_path}...")
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)

        print("Loading model...")
        with open(model_path, "rb") as f:
            loaded_clf = pickle.load(f)

        print("Making predictions with loaded model...")
        original_pred = clf.predict(test_data[columns])
        loaded_pred = loaded_clf.predict(test_data[columns])

        assert np.array_equal(original_pred, loaded_pred), "Predictions don't match!"
        print("✓ Save/Load pipeline test passed")

    finally:
        Path(model_path).unlink(missing_ok=True)


def test_multi_layer_pipeline() -> None:
    """Test with multiple classification layers."""
    print("\n" + "=" * 60)
    print("TEST: Multi-Layer Pipeline")
    print("=" * 60)

    train_data, test_data = create_synthetic_data()

    clf = ClassicalClassifier(
        layers=[
            ("quote", "ex"),
            ("quote", "best"),
            ("tick", "all"),
        ],
        random_state=42,
    )

    columns = [
        "trade_price", "bid_ex", "ask_ex",
        "bid_best", "ask_best",
        "price_all_lag", "price_all_lead",
    ]

    print("Training multi-layer classifier...")
    clf.fit(train_data[columns])

    print("Making predictions...")
    predictions = clf.predict(test_data[columns])

    print(f"Predictions: {predictions[:10]}")
    print("✓ Multi-layer pipeline test passed")


def test_evaluation_pipeline() -> None:
    """Test model evaluation with metrics."""
    print("\n" + "=" * 60)
    print("TEST: Evaluation Pipeline")
    print("=" * 60)

    train_data, test_data = create_synthetic_data()
    true_labels = create_true_labels(test_data)

    clf = ClassicalClassifier(
        layers=[("quote", "ex")],
        random_state=42,
    )

    columns = ["trade_price", "bid_ex", "ask_ex"]

    print("Training classifier...")
    clf.fit(train_data[columns])

    print("Making predictions...")
    predictions = clf.predict(test_data[columns])

    print("Calculating metrics...")
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=["Sell", "Buy"]))
    print("✓ Evaluation pipeline test passed")


def test_numpy_input_pipeline() -> None:
    """Test with NumPy array input."""
    print("\n" + "=" * 60)
    print("TEST: NumPy Input Pipeline")
    print("=" * 60)

    train_data, test_data = create_synthetic_data()

    feature_names = ["trade_price", "bid_ex", "ask_ex"]
    train_array = train_data[feature_names].to_numpy()
    test_array = test_data[feature_names].to_numpy()

    clf = ClassicalClassifier(
        layers=[("quote", "ex")],
        features=feature_names,
        random_state=42,
    )

    print(f"Training on NumPy array with shape {train_array.shape}...")
    clf.fit(train_array)

    print("Making predictions...")
    predictions = clf.predict(test_array)

    print(f"Predictions shape: {predictions.shape}")
    print("✓ NumPy input pipeline test passed")


def test_all_rules() -> None:
    """Test all classification rules."""
    print("\n" + "=" * 60)
    print("TEST: All Classification Rules")
    print("=" * 60)

    train_data, _ = create_synthetic_data(n_samples=50)

    rules = [
        ("tick", "ex", ["trade_price", "price_ex_lag"]),
        ("rev_tick", "ex", ["trade_price", "price_ex_lead"]),
        ("quote", "ex", ["trade_price", "bid_ex", "ask_ex"]),
        ("lr", "ex", ["trade_price", "bid_ex", "ask_ex", "price_ex_lag"]),
        ("rev_lr", "ex", ["trade_price", "bid_ex", "ask_ex", "price_ex_lead"]),
        ("emo", "ex", ["trade_price", "bid_ex", "ask_ex", "price_ex_lag"]),
        ("rev_emo", "ex", ["trade_price", "bid_ex", "ask_ex", "price_ex_lead"]),
        ("clnv", "ex", ["trade_price", "bid_ex", "ask_ex", "price_ex_lag"]),
        ("rev_clnv", "ex", ["trade_price", "bid_ex", "ask_ex", "price_ex_lead"]),
        ("trade_size", "ex", ["trade_size", "bid_size_ex", "ask_size_ex"]),
        ("depth", "ex", ["trade_price", "bid_ex", "ask_ex", "bid_size_ex", "ask_size_ex"]),
        ("nan", "ex", ["trade_price"]),
    ]

    for rule_name, subset, columns in rules:
        clf = ClassicalClassifier(layers=[(rule_name, subset)], random_state=42)
        clf.fit(train_data[columns])
        predictions = clf.predict(train_data[columns])
        print(f"  {rule_name:12} -> predictions: {np.unique(predictions, return_counts=True)}")

    print("✓ All rules test passed")


def test_strategies() -> None:
    """Test different strategies for unclassified trades."""
    print("\n" + "=" * 60)
    print("TEST: Classification Strategies")
    print("=" * 60)

    train_data, test_data = create_synthetic_data(n_samples=50)
    columns = ["trade_price"]

    print("Testing 'random' strategy...")
    clf_random = ClassicalClassifier(
        layers=[("nan", "ex")],
        strategy="random",
        random_state=42,
    )
    clf_random.fit(train_data[columns])
    pred_random = clf_random.predict(test_data[columns])
    print(f"  Random predictions: {np.unique(pred_random, return_counts=True)}")

    print("Testing 'const' strategy...")
    clf_const = ClassicalClassifier(
        layers=[("nan", "ex")],
        strategy="const",
    )
    clf_const.fit(train_data[columns])
    pred_const = clf_const.predict(test_data[columns])
    print(f"  Const predictions: {np.unique(pred_const, return_counts=True)}")

    print("✓ Strategies test passed")


def run_all_tests() -> None:
    """Run all end-to-end tests."""
    print("\n" + "=" * 60)
    print("TCLF End-to-End Tests")
    print("=" * 60)

    test_basic_pipeline()
    test_save_load_pipeline()
    test_multi_layer_pipeline()
    test_evaluation_pipeline()
    test_numpy_input_pipeline()
    test_all_rules()
    test_strategies()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
