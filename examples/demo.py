#!/usr/bin/env python3
"""Demo script for tclf trade classification."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from tclf.classical_classifier import ClassicalClassifier

def demo_basic_usage():
    """Basic usage demonstration."""
    print("=" * 60)
    print("Basic tclf Demo")
    print("=" * 60)
    
    # Create sample data
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
    
    print("\nInput Data:")
    print(X)
    
    # Create and fit classifier
    clf = ClassicalClassifier(layers=[("quote", "ex")], strategy="random")
    clf.fit(X)
    
    # Make predictions
    predictions = clf.predict(X)
    probabilities = clf.predict_proba(X)
    
    print("\nPredictions (-1 = sell, 1 = buy):")
    print(predictions)
    
    print("\nProbabilities:")
    print(probabilities)

def demo_advanced_usage():
    """Advanced usage demonstration."""
    print("\n" + "=" * 60)
    print("Advanced tclf Demo")
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
    
    print("\nFeatures:", features)
    print("\nTrue labels:", y_true)
    
    clf = ClassicalClassifier(
        layers=[("quote", "ex"), ("quote", "best")], 
        strategy="random", 
        features=features
    )
    clf.fit(X)
    
    predictions = clf.predict(X)
    acc = accuracy_score(y_true, predictions)
    
    print("\nPredictions:", predictions)
    print(f"Accuracy: {acc:.4f}")

def demo_multiple_algorithms():
    """Demonstrate multiple algorithms."""
    print("\n" + "=" * 60)
    print("Multiple Algorithms Demo")
    print("=" * 60)
    
    # Algorithms that don't need lag/lead columns
    algorithms = [
        ("quote", "Quote Rule"),
        ("depth", "Depth Rule"),
        ("trade_size", "Trade Size Rule"),
    ]
    
    # Try multiple paths for the sample data file
    possible_paths = [
        "/app/examples/sample_data.csv",
        "examples/sample_data.csv",
        "./examples/sample_data.csv",
    ]
    
    X = None
    for path in possible_paths:
        try:
            X = pd.read_csv(path)
            break
        except FileNotFoundError:
            continue
    
    if X is None:
        # Create sample data if file not found
        print("Sample data file not found, creating sample data...")
        X = pd.DataFrame(
            [
                [1.5, 1, 3, 2, 2.5],
                [2.5, 1, 3, 1, 3],
                [1.5, 3, 1, 1, 3],
                [2.5, 3, 1, 1, 3],
                [1.0, np.nan, 1, 1, 3],
                [3.0, np.nan, np.nan, 1, 3],
                [2.0, 1.5, 2.5, 1.5, 2.5],
                [2.0, 2.0, 2.0, 2.0, 2.0],
                [1.8, 1.5, 2.5, 1.5, 2.5],
                [2.2, 1.5, 2.5, 1.5, 2.5],
            ],
            columns=["trade_price", "bid_ex", "ask_ex", "bid_best", "ask_best"],
        )
    
    print("\nComparing different algorithms:")
    print("-" * 40)
    
    for algo, name in algorithms:
        clf = ClassicalClassifier(layers=[(algo, "ex")], strategy="const", random_state=42)
        clf.fit(X)
        pred = clf.predict(X)
        print(f"{name}: {pred}")

if __name__ == "__main__":
    demo_basic_usage()
    demo_advanced_usage()
    demo_multiple_algorithms()
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
