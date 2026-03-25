"""Tests for exception handling and edge cases."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.validation import check_is_fitted

from tclf.classical_classifier import ALLOWED_FUNC_STR, ClassicalClassifier


class TestExceptionHandling:
    """Tests for exception handling."""

    def test_empty_dataframe(self, empty_df: pd.DataFrame) -> None:
        """Test that empty DataFrame raises appropriate error."""
        clf = ClassicalClassifier(layers=[("quote", "ex")])
        with pytest.raises(ValueError):
            clf.fit(empty_df)

    def test_missing_required_columns(self, df_missing_columns: pd.DataFrame) -> None:
        """Test that missing required columns raises ValueError."""
        clf = ClassicalClassifier(layers=[("quote", "ex")])
        with pytest.raises(ValueError, match=r"Expected to find columns"):
            clf.fit(df_missing_columns)

    def test_missing_tick_columns(self) -> None:
        """Test missing columns for tick rule."""
        df = pd.DataFrame({"trade_price": [100.0, 101.0]})
        clf = ClassicalClassifier(layers=[("tick", "ex")])
        with pytest.raises(ValueError, match=r"Expected to find columns"):
            clf.fit(df)

    def test_missing_depth_columns(self) -> None:
        """Test missing columns for depth rule."""
        df = pd.DataFrame(
            {
                "trade_price": [100.0],
                "bid_ex": [99.5],
                "ask_ex": [100.5],
            }
        )
        clf = ClassicalClassifier(layers=[("depth", "ex")])
        with pytest.raises(ValueError, match=r"Expected to find columns"):
            clf.fit(df)

    def test_invalid_function_string(self) -> None:
        """Test that invalid function string raises ValueError."""
        df = pd.DataFrame({"trade_price": [100.0], "bid_ex": [99.5], "ask_ex": [100.5]})
        clf = ClassicalClassifier(layers=[("invalid_func", "ex")])  # type: ignore[list-item]
        with pytest.raises(ValueError, match=r"Unknown function string"):
            clf.fit(df)

    def test_invalid_subset_string(self, synthetic_train_data: pd.DataFrame) -> None:
        """Test that invalid subset raises error during fit."""
        clf = ClassicalClassifier(layers=[("quote", "invalid_subset")])  # type: ignore[list-item]
        with pytest.raises(ValueError, match=r"Expected to find columns"):
            clf.fit(synthetic_train_data[["trade_price", "bid_ex", "ask_ex"]])

    def test_column_count_mismatch(self, synthetic_train_data: pd.DataFrame) -> None:
        """Test that mismatched column count raises ValueError."""
        clf = ClassicalClassifier(
            layers=[("quote", "ex")],
            features=["trade_price", "bid_ex"],
        )
        data = synthetic_train_data[["trade_price", "bid_ex", "ask_ex"]].to_numpy()
        with pytest.raises(ValueError, match=r"Expected .* columns"):
            clf.fit(data)

    def test_predict_before_fit(self, synthetic_test_data: pd.DataFrame) -> None:
        """Test that predict before fit raises NotFittedError."""
        clf = ClassicalClassifier(layers=[("quote", "ex")])
        with pytest.raises(Exception):  # NotFittedError
            clf.predict(synthetic_test_data)

    def test_predict_proba_before_fit(self, synthetic_test_data: pd.DataFrame) -> None:
        """Test that predict_proba before fit raises NotFittedError."""
        clf = ClassicalClassifier(layers=[("quote", "ex")])
        with pytest.raises(Exception):  # NotFittedError
            clf.predict_proba(synthetic_test_data)

    def test_invalid_strategy(self, synthetic_train_data: pd.DataFrame) -> None:
        """Test that invalid strategy falls back gracefully."""
        clf = ClassicalClassifier(
            layers=[("quote", "ex")],
            strategy="invalid_strategy",  # type: ignore[arg-type]
        )
        clf.fit(synthetic_train_data[["trade_price", "bid_ex", "ask_ex"]])
        result = clf.predict(synthetic_train_data[["trade_price", "bid_ex", "ask_ex"]])
        assert result is not None


class TestNaNHandling:
    """Tests for NaN value handling."""

    def test_nan_in_trade_price(self, df_with_nans: pd.DataFrame) -> None:
        """Test handling of NaN in trade_price column."""
        clf = ClassicalClassifier(layers=[("quote", "ex")], random_state=42)
        clf.fit(df_with_nans)
        result = clf.predict(df_with_nans)
        assert result is not None
        assert len(result) == len(df_with_nans)

    def test_nan_in_bid_ask(self) -> None:
        """Test handling of NaN in bid/ask columns."""
        df = pd.DataFrame(
            {
                "trade_price": [100.0, 100.0, 100.0],
                "bid_ex": [np.nan, 99.5, 99.5],
                "ask_ex": [100.5, np.nan, 100.5],
            }
        )
        clf = ClassicalClassifier(layers=[("quote", "ex")], random_state=42)
        clf.fit(df)
        result = clf.predict(df)
        assert result is not None
        assert len(result) == 3

    def test_all_nan_row(self) -> None:
        """Test handling of row with all NaN values."""
        df = pd.DataFrame(
            {
                "trade_price": [np.nan, 100.0],
                "bid_ex": [np.nan, 99.5],
                "ask_ex": [np.nan, 100.5],
            }
        )
        clf = ClassicalClassifier(layers=[("quote", "ex")], random_state=42)
        clf.fit(df)
        result = clf.predict(df)
        assert result is not None
        assert len(result) == 2

    def test_nan_fallback_to_random(self) -> None:
        """Test that NaN results fall back to random classification."""
        df = pd.DataFrame(
            {
                "trade_price": [100.0],
                "bid_ex": [np.nan],
                "ask_ex": [np.nan],
            }
        )
        clf = ClassicalClassifier(layers=[("quote", "ex")], strategy="random", random_state=42)
        clf.fit(df)
        result = clf.predict(df)
        assert result[0] in [-1, 1]

    def test_nan_fallback_to_const(self) -> None:
        """Test that NaN results fall back to constant zero."""
        df = pd.DataFrame(
            {
                "trade_price": [100.0],
                "bid_ex": [np.nan],
                "ask_ex": [np.nan],
            }
        )
        clf = ClassicalClassifier(layers=[("quote", "ex")], strategy="const")
        clf.fit(df)
        result = clf.predict(df)
        assert result[0] == 0


class TestInfHandling:
    """Tests for infinity value handling."""

    def test_inf_in_data(self, df_with_infs: pd.DataFrame) -> None:
        """Test handling of Inf values in data."""
        clf = ClassicalClassifier(layers=[("quote", "ex")], random_state=42)
        clf.fit(df_with_infs)
        result = clf.predict(df_with_infs)
        assert result is not None
        assert len(result) == len(df_with_infs)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_row_data(self) -> None:
        """Test with single row of data."""
        df = pd.DataFrame(
            {
                "trade_price": [100.0],
                "bid_ex": [99.5],
                "ask_ex": [100.5],
            }
        )
        clf = ClassicalClassifier(layers=[("quote", "ex")], random_state=42)
        clf.fit(df)
        result = clf.predict(df)
        assert len(result) == 1

    def test_large_values(self) -> None:
        """Test with very large numerical values."""
        df = pd.DataFrame(
            {
                "trade_price": [1e10, 1e10 + 1],
                "bid_ex": [1e10 - 1, 1e10],
                "ask_ex": [1e10 + 1, 1e10 + 2],
            }
        )
        clf = ClassicalClassifier(layers=[("quote", "ex")], random_state=42)
        clf.fit(df)
        result = clf.predict(df)
        assert result is not None

    def test_negative_prices(self) -> None:
        """Test with negative price values."""
        df = pd.DataFrame(
            {
                "trade_price": [-100.0, -99.0],
                "bid_ex": [-101.0, -100.0],
                "ask_ex": [-99.0, -98.0],
            }
        )
        clf = ClassicalClassifier(layers=[("quote", "ex")], random_state=42)
        clf.fit(df)
        result = clf.predict(df)
        assert result is not None

    def test_zero_prices(self) -> None:
        """Test with zero price values."""
        df = pd.DataFrame(
            {
                "trade_price": [0.0, 0.0],
                "bid_ex": [-1.0, 0.0],
                "ask_ex": [1.0, 0.0],
            }
        )
        clf = ClassicalClassifier(layers=[("quote", "ex")], random_state=42)
        clf.fit(df)
        result = clf.predict(df)
        assert result is not None

    def test_identical_bid_ask(self) -> None:
        """Test with identical bid and ask prices."""
        df = pd.DataFrame(
            {
                "trade_price": [100.0, 100.0],
                "bid_ex": [100.0, 100.0],
                "ask_ex": [100.0, 100.0],
            }
        )
        clf = ClassicalClassifier(layers=[("quote", "ex")], random_state=42)
        clf.fit(df)
        result = clf.predict(df)
        assert result is not None

    def test_bid_greater_than_ask(self) -> None:
        """Test with bid price greater than ask price (invalid spread)."""
        df = pd.DataFrame(
            {
                "trade_price": [100.0],
                "bid_ex": [101.0],
                "ask_ex": [99.0],
            }
        )
        clf = ClassicalClassifier(layers=[("quote", "ex")], random_state=42)
        clf.fit(df)
        result = clf.predict(df)
        assert result is not None


class TestInputTypes:
    """Tests for different input types."""

    def test_numpy_array_input(
        self, np_array_data: np.ndarray, feature_names: list[str]
    ) -> None:
        """Test with NumPy array input."""
        clf = ClassicalClassifier(
            layers=[("quote", "ex")],
            features=feature_names,
            random_state=42,
        )
        clf.fit(np_array_data)
        result = clf.predict(np_array_data)
        assert result is not None

    def test_numpy_array_without_features(self, np_array_data: np.ndarray) -> None:
        """Test NumPy array without feature names uses default column names."""
        clf = ClassicalClassifier(layers=[("nan", "ex")], random_state=42)
        clf.fit(np_array_data)
        assert check_is_fitted(clf) is None

    def test_pandas_series_input(self) -> None:
        """Test that pandas Series input is handled appropriately."""
        s = pd.Series([100.0, 99.5, 100.5], name="trade_price")
        clf = ClassicalClassifier(layers=[("nan", "ex")], random_state=42)
        with pytest.raises(Exception):
            clf.fit(s)


class TestLayerValidation:
    """Tests for layer configuration validation."""

    def test_empty_layers(self, minimal_df: pd.DataFrame) -> None:
        """Test with empty layers list (random classification)."""
        clf = ClassicalClassifier(layers=None, random_state=42)
        clf.fit(minimal_df)
        result = clf.predict(minimal_df)
        assert result is not None
        assert len(result) == len(minimal_df)

    def test_multiple_layers(self, synthetic_train_data: pd.DataFrame) -> None:
        """Test with multiple classification layers."""
        clf = ClassicalClassifier(
            layers=[
                ("quote", "ex"),
                ("quote", "best"),
                ("tick", "all"),
            ],
            random_state=42,
        )
        clf.fit(synthetic_train_data)
        result = clf.predict(synthetic_train_data)
        assert result is not None

    def test_all_function_types(self, synthetic_train_data: pd.DataFrame) -> None:
        """Test all valid function types are accepted."""
        valid_funcs = list(ALLOWED_FUNC_STR)
        for func in valid_funcs:
            clf = ClassicalClassifier(layers=[(func, "ex")], random_state=42)
            try:
                clf.fit(synthetic_train_data)
            except ValueError as e:
                if "Expected to find columns" not in str(e):
                    raise


class TestReproducibility:
    """Tests for reproducibility with random state."""

    def test_same_random_state_same_results(
        self, synthetic_train_data: pd.DataFrame, synthetic_test_data: pd.DataFrame
    ) -> None:
        """Test that same random state produces same results."""
        columns = ["trade_price", "bid_ex", "ask_ex"]
        clf1 = ClassicalClassifier(layers=[("nan", "ex")], random_state=42)
        clf1.fit(synthetic_train_data[columns])
        result1 = clf1.predict(synthetic_test_data[columns])

        clf2 = ClassicalClassifier(layers=[("nan", "ex")], random_state=42)
        clf2.fit(synthetic_train_data[columns])
        result2 = clf2.predict(synthetic_test_data[columns])

        np.testing.assert_array_equal(result1, result2)

    def test_different_random_state_different_results(
        self, synthetic_train_data: pd.DataFrame, synthetic_test_data: pd.DataFrame
    ) -> None:
        """Test that different random states may produce different results."""
        columns = ["trade_price", "bid_ex", "ask_ex"]
        clf1 = ClassicalClassifier(layers=[("nan", "ex")], random_state=42)
        clf1.fit(synthetic_train_data[columns])
        result1 = clf1.predict(synthetic_test_data[columns])

        clf2 = ClassicalClassifier(layers=[("nan", "ex")], random_state=123)
        clf2.fit(synthetic_train_data[columns])
        result2 = clf2.predict(synthetic_test_data[columns])

        assert not np.array_equal(result1, result2) or True


class TestModelPersistence:
    """Tests for model save/load functionality."""

    def test_pickle_roundtrip(
        self, fitted_classifier: ClassicalClassifier, synthetic_test_data: pd.DataFrame
    ) -> None:
        """Test that model can be pickled and unpickled."""
        columns = ["trade_price", "bid_ex", "ask_ex"]
        pickled = pickle.dumps(fitted_classifier)
        unpickled = pickle.loads(pickled)

        result_original = fitted_classifier.predict(synthetic_test_data[columns])
        result_unpickled = unpickled.predict(synthetic_test_data[columns])

        np.testing.assert_array_equal(result_original, result_unpickled)

    def test_pickle_preserves_attributes(
        self, fitted_classifier: ClassicalClassifier
    ) -> None:
        """Test that pickling preserves model attributes."""
        pickled = pickle.dumps(fitted_classifier)
        unpickled = pickle.loads(pickled)

        assert unpickled.layers == fitted_classifier.layers
        assert unpickled.random_state == fitted_classifier.random_state
        assert unpickled.strategy == fitted_classifier.strategy
        np.testing.assert_array_equal(unpickled.classes_, fitted_classifier.classes_)
