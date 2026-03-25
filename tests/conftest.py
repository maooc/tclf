"""Shared pytest fixtures for tclf tests."""

from __future__ import annotations

from typing import Generator

import numpy as np
import pandas as pd
import pytest

from tclf.classical_classifier import ClassicalClassifier


@pytest.fixture
def minimal_df() -> pd.DataFrame:
    """Minimal DataFrame with basic columns for simple tests.

    Returns:
        pd.DataFrame: minimal test dataframe
    """
    return pd.DataFrame(
        {
            "trade_price": [100.0, 101.0, 99.0, 100.5],
            "bid_ex": [99.5, 100.5, 98.5, 100.0],
            "ask_ex": [100.5, 101.5, 99.5, 101.0],
        }
    )


@pytest.fixture
def synthetic_train_data() -> pd.DataFrame:
    """Synthetic training data with all required columns.

    Returns:
        pd.DataFrame: training data
    """
    np.random.seed(42)
    n_samples = 20

    return pd.DataFrame(
        {
            "trade_price": np.random.uniform(99, 101, n_samples),
            "trade_size": np.random.randint(100, 1000, n_samples),
            "bid_ex": np.random.uniform(98.5, 99.5, n_samples),
            "ask_ex": np.random.uniform(100.5, 101.5, n_samples),
            "bid_best": np.random.uniform(98.8, 99.8, n_samples),
            "ask_best": np.random.uniform(100.2, 101.2, n_samples),
            "bid_size_ex": np.random.randint(500, 2000, n_samples),
            "ask_size_ex": np.random.randint(500, 2000, n_samples),
            "price_ex_lag": np.random.uniform(99, 101, n_samples),
            "price_ex_lead": np.random.uniform(99, 101, n_samples),
            "price_best_lag": np.random.uniform(99, 101, n_samples),
            "price_best_lead": np.random.uniform(99, 101, n_samples),
            "price_all_lag": np.random.uniform(99, 101, n_samples),
            "price_all_lead": np.random.uniform(99, 101, n_samples),
        }
    )


@pytest.fixture
def synthetic_test_data() -> pd.DataFrame:
    """Synthetic test data for prediction.

    Returns:
        pd.DataFrame: test data
    """
    return pd.DataFrame(
        {
            "trade_price": [100.0, 101.0, 99.0, 100.5, 100.25],
            "bid_ex": [99.5, 100.5, 98.5, 100.0, 99.75],
            "ask_ex": [100.5, 101.5, 99.5, 101.0, 100.75],
            "bid_best": [99.4, 100.4, 98.4, 99.9, 99.65],
            "ask_best": [100.6, 101.6, 99.6, 101.1, 100.85],
            "price_ex_lag": [99.8, 100.8, 99.2, 100.3, 100.1],
        }
    )


@pytest.fixture
def synthetic_labels() -> np.ndarray:
    """Synthetic labels for test data.

    Returns:
        np.ndarray: array of labels (-1 or 1)
    """
    return np.array([-1, 1, -1, 1, -1])


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """Empty DataFrame for exception testing.

    Returns:
        pd.DataFrame: empty dataframe
    """
    return pd.DataFrame()


@pytest.fixture
def df_with_nans() -> pd.DataFrame:
    """DataFrame with NaN values for NaN handling tests.

    Returns:
        pd.DataFrame: dataframe with NaN values
    """
    return pd.DataFrame(
        {
            "trade_price": [100.0, np.nan, 99.0, 100.5],
            "bid_ex": [99.5, 100.5, np.nan, 100.0],
            "ask_ex": [100.5, np.nan, 99.5, 101.0],
        }
    )


@pytest.fixture
def df_with_infs() -> pd.DataFrame:
    """DataFrame with Inf values for edge case testing.

    Returns:
        pd.DataFrame: dataframe with Inf values
    """
    return pd.DataFrame(
        {
            "trade_price": [100.0, np.inf, 99.0, -np.inf],
            "bid_ex": [99.5, 100.5, 98.5, 100.0],
            "ask_ex": [100.5, 101.5, 99.5, 101.0],
        }
    )


@pytest.fixture
def df_missing_columns() -> pd.DataFrame:
    """DataFrame missing required columns.

    Returns:
        pd.DataFrame: dataframe with missing columns
    """
    return pd.DataFrame(
        {
            "trade_price": [100.0, 101.0],
            "trade_size": [500, 600],
        }
    )


@pytest.fixture
def np_array_data() -> np.ndarray:
    """NumPy array data for testing non-DataFrame input.

    Returns:
        np.ndarray: numpy array
    """
    return np.array(
        [
            [100.0, 99.5, 100.5, 99.8],
            [101.0, 100.5, 101.5, 100.8],
            [99.0, 98.5, 99.5, 99.2],
        ]
    )


@pytest.fixture
def feature_names() -> list[str]:
    """Feature names for numpy array input.

    Returns:
        list[str]: list of feature names
    """
    return ["trade_price", "bid_ex", "ask_ex", "price_ex_lag"]


@pytest.fixture
def fitted_classifier(synthetic_train_data: pd.DataFrame) -> ClassicalClassifier:
    """Pre-fitted classifier for testing.

    Args:
        synthetic_train_data: training data fixture

    Returns:
        ClassicalClassifier: fitted classifier
    """
    columns = ["trade_price", "bid_ex", "ask_ex"]
    clf = ClassicalClassifier(
        layers=[("quote", "ex")],
        random_state=42,
    )
    return clf.fit(synthetic_train_data[columns])


@pytest.fixture
def temp_model_path(tmp_path: Generator[str, None, None]) -> str:
    """Temporary path for saving/loading models.

    Args:
        tmp_path: pytest tmp_path fixture

    Returns:
        str: path to temporary file
    """
    return str(tmp_path / "test_model.pkl")


@pytest.fixture
def multi_layer_classifier(synthetic_train_data: pd.DataFrame) -> ClassicalClassifier:
    """Classifier with multiple layers for complex tests.

    Args:
        synthetic_train_data: training data fixture

    Returns:
        ClassicalClassifier: fitted multi-layer classifier
    """
    columns = ["trade_price", "bid_ex", "ask_ex", "bid_best", "ask_best", "price_all_lag"]
    clf = ClassicalClassifier(
        layers=[
            ("quote", "ex"),
            ("quote", "best"),
        ],
        random_state=42,
    )
    return clf.fit(synthetic_train_data[columns])
