"""Pytest fixtures for tclf tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def minimal_columns() -> list[str]:
    """Return minimal set of column names for basic tests."""
    return ["trade_price", "bid_ex", "ask_ex"]


@pytest.fixture
def full_columns() -> list[str]:
    """Return full set of column names for comprehensive tests."""
    return [
        "ask_size_ex",
        "bid_size_ex",
        "ask_best",
        "bid_best",
        "ask_ex",
        "bid_ex",
        "trade_price",
        "trade_size",
        "price_ex_lag",
        "price_ex_lead",
        "price_best_lag",
        "price_best_lead",
        "price_all_lag",
        "price_all_lead",
    ]


@pytest.fixture
def synthetic_train_data(full_columns: list[str]) -> pd.DataFrame:
    """Generate synthetic training data with all required columns.

    Returns:
        pd.DataFrame: Training data with all required columns.
    """
    np.random.seed(42)
    n_samples = 100

    data = {
        "trade_price": np.random.uniform(10, 100, n_samples),
        "bid_ex": np.random.uniform(9, 99, n_samples),
        "ask_ex": np.random.uniform(11, 101, n_samples),
        "bid_best": np.random.uniform(9, 99, n_samples),
        "ask_best": np.random.uniform(11, 101, n_samples),
        "trade_size": np.random.randint(1, 1000, n_samples),
        "bid_size_ex": np.random.randint(1, 500, n_samples),
        "ask_size_ex": np.random.randint(1, 500, n_samples),
        "price_ex_lag": np.random.uniform(10, 100, n_samples),
        "price_ex_lead": np.random.uniform(10, 100, n_samples),
        "price_best_lag": np.random.uniform(10, 100, n_samples),
        "price_best_lead": np.random.uniform(10, 100, n_samples),
        "price_all_lag": np.random.uniform(10, 100, n_samples),
        "price_all_lead": np.random.uniform(10, 100, n_samples),
    }

    df = pd.DataFrame(data)
    # Ensure ask > bid for valid spread
    df["ask_ex"] = np.maximum(df["ask_ex"], df["bid_ex"] + 0.01)
    df["ask_best"] = np.maximum(df["ask_best"], df["bid_best"] + 0.01)
    return df


@pytest.fixture
def synthetic_test_data() -> pd.DataFrame:
    """Generate small synthetic test data for predictable results.

    Returns:
        pd.DataFrame: Small test dataset with all required columns.
    """
    return pd.DataFrame(
        {
            "trade_price": [15.0, 25.0, 20.0, 20.0, 20.0],
            "bid_ex": [10.0, 20.0, 19.0, 19.0, np.nan],
            "ask_ex": [20.0, 30.0, 21.0, 21.0, np.nan],
            "bid_best": [10.0, 20.0, 19.0, 19.0, 19.0],
            "ask_best": [20.0, 30.0, 21.0, 21.0, 21.0],
            "price_ex_lag": [10.0, 30.0, 20.0, 20.0, 20.0],
            "price_ex_lead": [20.0, 10.0, 20.0, 20.0, 20.0],
            "price_best_lag": [10.0, 30.0, 20.0, 20.0, 20.0],
            "price_best_lead": [20.0, 10.0, 20.0, 20.0, 20.0],
            "price_all_lag": [10.0, 30.0, 20.0, 20.0, 20.0],
            "price_all_lead": [20.0, 10.0, 20.0, 20.0, 20.0],
            "trade_size": [100, 200, 150, 150, 150],
            "bid_size_ex": [100, 200, 150, 150, 150],
            "ask_size_ex": [200, 100, 150, 150, 150],
        }
    )


@pytest.fixture
def empty_dataframe() -> pd.DataFrame:
    """Return an empty DataFrame."""
    return pd.DataFrame()


@pytest.fixture
def dataframe_with_missing_columns() -> pd.DataFrame:
    """Return a DataFrame with missing required columns."""
    return pd.DataFrame({
        "trade_price": [1.0, 2.0, 3.0],
        "some_other_column": [4.0, 5.0, 6.0],
    })


@pytest.fixture
def dataframe_with_nans() -> pd.DataFrame:
    """Return a DataFrame with NaN values."""
    return pd.DataFrame({
        "trade_price": [1.0, np.nan, 3.0],
        "bid_ex": [0.5, 1.5, np.nan],
        "ask_ex": [1.5, 2.5, 3.5],
    })


@pytest.fixture
def dataframe_with_infs() -> pd.DataFrame:
    """Return a DataFrame with infinity values."""
    return pd.DataFrame({
        "trade_price": [1.0, np.inf, 3.0],
        "bid_ex": [0.5, 1.5, -np.inf],
        "ask_ex": [1.5, 2.5, 3.5],
    })


@pytest.fixture
def single_row_data() -> pd.DataFrame:
    """Return a single row DataFrame."""
    return pd.DataFrame({
        "trade_price": [15.0],
        "bid_ex": [10.0],
        "ask_ex": [20.0],
    })


@pytest.fixture
def large_synthetic_data() -> pd.DataFrame:
    """Generate larger synthetic dataset for performance tests.

    Returns:
        pd.DataFrame: Larger dataset with 1000 rows.
    """
    np.random.seed(123)
    n_samples = 1000

    data = {
        "trade_price": np.random.uniform(10, 100, n_samples),
        "bid_ex": np.random.uniform(9, 99, n_samples),
        "ask_ex": np.random.uniform(11, 101, n_samples),
        "bid_best": np.random.uniform(9, 99, n_samples),
        "ask_best": np.random.uniform(11, 101, n_samples),
        "trade_size": np.random.randint(1, 1000, n_samples),
        "bid_size_ex": np.random.randint(1, 500, n_samples),
        "ask_size_ex": np.random.randint(1, 500, n_samples),
        "price_ex_lag": np.random.uniform(10, 100, n_samples),
        "price_ex_lead": np.random.uniform(10, 100, n_samples),
    }

    df = pd.DataFrame(data)
    df["ask_ex"] = np.maximum(df["ask_ex"], df["bid_ex"] + 0.01)
    df["ask_best"] = np.maximum(df["ask_best"], df["bid_best"] + 0.01)
    return df


@pytest.fixture
def numpy_array_data() -> np.ndarray:
    """Return data as numpy array for testing non-DataFrame input."""
    return np.array([
        [15.0, 10.0, 20.0],
        [25.0, 20.0, 30.0],
        [20.0, 19.0, 21.0],
    ])


@pytest.fixture
def feature_names() -> list[str]:
    """Return feature names corresponding to numpy array columns."""
    return ["trade_price", "bid_ex", "ask_ex"]
