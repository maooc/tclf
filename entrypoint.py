#!/usr/bin/env python
"""Entrypoint script for Docker container.

Supports multiple modes:
- api: Start REST API server (default)
- demo: Run demo classification script
- train: Run training script with provided data file
"""

import argparse
import sys


def run_api():
    """Start the FastAPI REST API server."""
    import uvicorn
    from app import app

    uvicorn.run(app, host="0.0.0.0", port=8000)


def run_demo():
    """Run demo classification with sample data."""
    import numpy as np
    import pandas as pd
    from tclf.classical_classifier import ClassicalClassifier

    print("=" * 60)
    print("tclf Demo - Trade Classification")
    print("=" * 60)

    X = pd.DataFrame(
        [
            [1.5, 1, 3, 2, 2.5],
            [2.5, 1, 3, 1, 3],
            [1.5, 3, 1, 1, 3],
            [2.5, 3, 1, 1, 3],
            [1, np.nan, 1, 1, 3],
            [3, np.nan, np.nan, 1, 3],
        ],
        columns=["trade_price", "bid_ex", "ask_ex", "bid_best", "ask_best"],
    )

    print("\nInput Data:")
    print(X.to_string())

    clf = ClassicalClassifier(
        layers=[("quote", "ex"), ("quote", "best")], strategy="random"
    )
    clf.fit(X)
    predictions = clf.predict(X)
    probabilities = clf.predict_proba(X)

    print("\n" + "-" * 60)
    print("Results:")
    print("-" * 60)

    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        direction = "BUY " if pred == 1 else "SELL"
        print(
            f"Trade {i + 1}: {direction} (prob: sell={prob[0]:.2f}, buy={prob[1]:.2f})"
        )

    print("\n" + "=" * 60)
    print(f"Total trades classified: {len(predictions)}")
    print(f"Buy-initiated:  {sum(predictions == 1)}")
    print(f"Sell-initiated: {sum(predictions == -1)}")
    print("=" * 60)


def run_train(data_file: str, output_file: str | None = None):
    """Run classification on provided data file.

    Args:
        data_file: Path to input CSV file.
        output_file: Optional path to save results.
    """
    import pandas as pd
    from tclf.classical_classifier import ClassicalClassifier

    print("=" * 60)
    print("tclf Training/Prediction Mode")
    print("=" * 60)

    try:
        df = pd.read_csv(data_file)
        print(f"\nLoaded {len(df)} trades from: {data_file}")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error loading data file: {e}")
        sys.exit(1)

    clf = ClassicalClassifier(
        layers=[("quote", "ex"), ("quote", "best")], strategy="random"
    )
    clf.fit(df)
    predictions = clf.predict(df)
    probabilities = clf.predict_proba(df)

    df_result = df.copy()
    df_result["prediction"] = predictions
    df_result["prob_sell"] = probabilities[:, 0]
    df_result["prob_buy"] = probabilities[:, 1]

    print("\n" + "-" * 60)
    print("Classification Results:")
    print("-" * 60)
    print(f"Total trades classified: {len(predictions)}")
    print(f"Buy-initiated:  {sum(predictions == 1)}")
    print(f"Sell-initiated: {sum(predictions == -1)}")

    if output_file:
        df_result.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    else:
        print("\nResults preview:")
        print(df_result.head(10).to_string())

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="tclf - Trade Classification with Python",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python entrypoint.py api              # Start REST API server
  python entrypoint.py demo             # Run demo classification
  python entrypoint.py train data.csv   # Classify trades from CSV
  python entrypoint.py train data.csv -o results.csv  # Save results
        """,
    )

    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")

    subparsers.add_parser("api", help="Start REST API server")

    subparsers.add_parser("demo", help="Run demo classification")

    train_parser = subparsers.add_parser("train", help="Classify trades from CSV file")
    train_parser.add_argument("data_file", help="Path to input CSV file")
    train_parser.add_argument(
        "-o", "--output", help="Path to output CSV file (optional)"
    )

    args = parser.parse_args()

    if args.mode == "api" or args.mode is None:
        run_api()
    elif args.mode == "demo":
        run_demo()
    elif args.mode == "train":
        run_train(args.data_file, args.output)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
