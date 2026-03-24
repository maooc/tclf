"""Demo script for tclf trade classification.

This script demonstrates how to use the ClassicalClassifier
with sample trade data.
"""

import numpy as np
import pandas as pd

from tclf.classical_classifier import ClassicalClassifier


def demo_basic():
    """Basic demo with quote rule."""
    print("=" * 60)
    print("Demo 1: Basic Quote Rule Classification")
    print("=" * 60)

    X = pd.DataFrame(
        [
            [1.5, 1, 3],
            [2.5, 1, 3],
            [1.5, 3, 1],
            [2.5, 3, 1],
            [1, np.nan, 1],
            [3, np.nan, np.nan],
        ],
        columns=["trade_price", "bid_ex", "ask_ex"],
    )

    print("Input Data:")
    print(X)
    print()

    clf = ClassicalClassifier(layers=[("quote", "ex")], strategy="random")
    clf.fit(X)
    predictions = clf.predict(X)
    probabilities = clf.predict_proba(X)

    print("Predictions (1=buy, -1=sell):")
    print(predictions)
    print()
    print("Probabilities:")
    print(probabilities)
    print()


def demo_advanced():
    """Advanced demo with multiple layers."""
    print("=" * 60)
    print("Demo 2: Advanced Multi-Layer Classification")
    print("=" * 60)

    X = np.array(
        [
            [1.5, 1, 3, 2, 2.5],
            [2.5, 1, 3, 1, 3],
            [1.5, 3, 1, 1, 3],
            [2.5, 3, 1, 1, 3],
            [1, np.nan, 1, 1, 3],
            [3, np.nan, np.nan, 1, 3],
        ]
    )
    y_true = np.array([-1, 1, 1, -1, -1, 1])
    features = ["trade_price", "bid_ex", "ask_ex", "bid_best", "ask_best"]

    print("Features:", features)
    print("Input Data Shape:", X.shape)
    print()

    clf = ClassicalClassifier(
        layers=[("quote", "ex"), ("quote", "best")],
        strategy="random",
        features=features,
    )
    clf.fit(X)

    predictions = clf.predict(X)
    print("Predictions:", predictions)
    print("True Labels:", y_true)

    accuracy = np.mean(predictions == y_true)
    print(f"Accuracy: {accuracy:.2%}")
    print()


def demo_tick_rule():
    """Demo with tick rule."""
    print("=" * 60)
    print("Demo 3: Tick Rule Classification")
    print("=" * 60)

    X = pd.DataFrame(
        {
            "trade_price": [10.0, 10.5, 10.3, 10.1, 10.4],
            "price_ex_lag": [np.nan, 10.0, 10.5, 10.3, 10.1],
        }
    )

    print("Input Data:")
    print(X)
    print()

    clf = ClassicalClassifier(layers=[("tick", "ex")], strategy="random")
    clf.fit(X)
    predictions = clf.predict(X)

    print("Predictions (1=buy, -1=sell):")
    print(predictions)
    print()


if __name__ == "__main__":
    demo_basic()
    demo_advanced()
    demo_tick_rule()
    print("=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)
