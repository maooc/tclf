"""Pytest fixtures for tclf test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tclf.classical_classifier import ClassicalClassifier


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Generate mini synthetic trade data for testing.

    Returns:
        pd.DataFrame: DataFrame with all required columns for various classification rules.
    """
    np.random.seed(42)
    n_samples = 10

    return pd.DataFrame(
        {
            "trade_price": np.random.uniform(1.0, 100.0, n_samples),
            "trade_size": np.random.randint(1, 1000, n_samples),
            "ask_ex": np.random.uniform(1.0, 100.0, n_samples) + 0.5,
            "bid_ex": np.random.uniform(1.0, 100.0, n_samples) - 0.5,
            "ask_best": np.random.uniform(1.0, 100.0, n_samples) + 0.3,
            "bid_best": np.random.uniform(1.0, 100.0, n_samples) - 0.3,
            "ask_size_ex": np.random.randint(100, 10000, n_samples),
            "bid_size_ex": np.random.randint(100, 10000, n_samples),
            "ask_size_best": np.random.randint(100, 10000, n_samples),
            "bid_size_best": np.random.randint(100, 10000, n_samples),
            "price_ex_lag": np.random.uniform(1.0, 100.0, n_samples),
            "price_ex_lead": np.random.uniform(1.0, 100.0, n_samples),
            "price_best_lag": np.random.uniform(1.0, 100.0, n_samples),
            "price_best_lead": np.random.uniform(1.0, 100.0, n_samples),
            "price_all_lag": np.random.uniform(1.0, 100.0, n_samples),
            "price_all_lead": np.random.uniform(1.0, 100.0, n_samples),
        }
    )


@pytest.fixture
def sample_data_numpy(sample_data: pd.DataFrame) -> np.ndarray:
    """Convert sample DataFrame to numpy array.

    Args:
        sample_data: Input DataFrame fixture.

    Returns:
        np.ndarray: Numpy array representation of the data.
    """
    return sample_data.to_numpy()


@pytest.fixture
def feature_names(sample_data: pd.DataFrame) -> list[str]:
    """Get feature names from sample data.

    Args:
        sample_data: Input DataFrame fixture.

    Returns:
        list[str]: List of feature names.
    """
    return sample_data.columns.tolist()


@pytest.fixture(params=["random", "const"])
def strategy(request: pytest.FixtureRequest) -> str:
    """Parameterized fixture for different filling strategies.

    Returns:
        str: Either 'random' or 'const' strategy.
    """
    return request.param


@pytest.fixture
def fitted_classifier(sample_data: pd.DataFrame) -> ClassicalClassifier:
    """Create and fit a basic classifier for testing.

    Args:
        sample_data: Input DataFrame fixture.

    Returns:
        ClassicalClassifier: Fitted classifier instance.
    """
    clf = ClassicalClassifier(
        layers=[("quote", "ex"), ("tick", "all")],
        random_state=42,
        strategy="random",
    )
    clf.fit(sample_data)
    return clf


@pytest.fixture
def minimal_columns() -> list[str]:
    """Minimal required columns for basic classification.

    Returns:
        list[str]: List of minimal column names.
    """
    return ["trade_price", "ask_ex", "bid_ex"]


@pytest.fixture
def minimal_data() -> pd.DataFrame:
    """Minimal dataset with just enough columns for basic classification.

    Returns:
        pd.DataFrame: Minimal dataset.
    """
    return pd.DataFrame(
        {
            "trade_price": [10.0, 11.0, 10.5, 11.5],
            "ask_ex": [10.5, 11.5, 11.0, 12.0],
            "bid_ex": [9.5, 10.5, 10.0, 11.0],
        }
    )


@pytest.fixture
def edge_case_data() -> pd.DataFrame:
    """Data with edge cases like NaN values, equal prices.

    Returns:
        pd.DataFrame: Edge case dataset.
    """
    return pd.DataFrame(
        {
            "trade_price": [10.0, 10.0, np.nan, 11.0, 10.5],
            "ask_ex": [10.0, 10.0, 10.5, 11.0, 10.5],
            "bid_ex": [10.0, 10.0, 9.5, 10.0, 10.5],
            "price_ex_lag": [9.0, np.nan, 10.0, 11.0, 10.0],
            "price_all_lag": [9.0, 10.0, np.nan, 11.0, 10.0],
        }
    )
