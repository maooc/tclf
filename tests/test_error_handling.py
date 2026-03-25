"""Error handling and edge case tests for ClassicalClassifier."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tclf.classical_classifier import ClassicalClassifier


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_empty_dataframe(self) -> None:
        """Test behavior with empty DataFrame."""
        empty_df = pd.DataFrame()
        clf = ClassicalClassifier(layers=[("quote", "ex")])

        with pytest.raises(ValueError):
            clf.fit(empty_df)

    def test_empty_numpy_array(self) -> None:
        """Test behavior with empty numpy array."""
        empty_arr = np.array([])
        clf = ClassicalClassifier(layers=[("quote", "ex")], features=["trade_price"])

        with pytest.raises(ValueError):
            clf.fit(empty_arr)

    def test_missing_required_columns(self, sample_data: pd.DataFrame) -> None:
        """Test error when required columns are missing."""
        # Remove required columns
        incomplete_data = sample_data.drop(["ask_ex", "bid_ex"], axis=1, errors="ignore")
        clf = ClassicalClassifier(layers=[("quote", "ex")])

        with pytest.raises(ValueError, match="Expected to find columns"):
            clf.fit(incomplete_data)

    def test_invalid_function_name(self, sample_data: pd.DataFrame) -> None:
        """Test error with invalid classification function name."""
        # Using invalid function name
        clf = ClassicalClassifier(layers=[("invalid_func", "ex")])  # type: ignore

        with pytest.raises(ValueError, match="Unknown function string"):
            clf.fit(sample_data)

    def test_invalid_strategy(self) -> None:
        """Test error with invalid strategy parameter."""
        # The strategy type checking is done by Literal type checker at runtime
        # This should work even with invalid string, but may behave unexpectedly
        # Let's test that it doesn't crash completely
        df = pd.DataFrame(
            {"trade_price": [1.0], "ask_ex": [1.1], "bid_ex": [0.9]}
        )
        clf = ClassicalClassifier(
            layers=[("quote", "ex")],
            strategy="invalid_strategy",  # type: ignore
        )
        clf.fit(df)
        # With invalid strategy, it should default to const-like behavior
        predictions = clf.predict(df)
        assert len(predictions) == 1

    def test_predict_before_fit(self, sample_data: pd.DataFrame) -> None:
        """Test error when predicting before fitting."""
        clf = ClassicalClassifier(layers=[("quote", "ex")])

        with pytest.raises(ValueError, match="This ClassicalClassifier instance is not fitted yet"):
            clf.predict(sample_data)

    def test_predict_proba_before_fit(self, sample_data: pd.DataFrame) -> None:
        """Test error when predict_proba before fitting."""
        clf = ClassicalClassifier(layers=[("quote", "ex")])

        with pytest.raises(ValueError, match="This ClassicalClassifier instance is not fitted yet"):
            clf.predict_proba(sample_data)

    def test_invalid_sample_weight(self, sample_data: pd.DataFrame) -> None:
        """Test error with invalid sample_weight."""
        clf = ClassicalClassifier(layers=[("quote", "ex")])

        # Wrong length
        with pytest.raises(ValueError):
            clf.fit(sample_data, sample_weight=np.ones(len(sample_data) - 1))

        # Contains NaN
        invalid_weight = np.ones(len(sample_data))
        invalid_weight[0] = np.nan
        with pytest.raises(ValueError):
            clf.fit(sample_data, sample_weight=invalid_weight)

    def test_all_nan_columns(self) -> None:
        """Test behavior with all-NaN columns."""
        df = pd.DataFrame(
            {
                "trade_price": [np.nan, np.nan, np.nan],
                "ask_ex": [1.1, 1.2, 1.3],
                "bid_ex": [0.9, 0.8, 0.7],
            }
        )
        clf = ClassicalClassifier(layers=[("quote", "ex")], random_state=42)
        clf.fit(df)
        predictions = clf.predict(df)
        # Should handle NaN values gracefully
        assert len(predictions) == 3
        assert not np.isnan(predictions).any()

    def test_infinite_values(self) -> None:
        """Test behavior with infinite values."""
        df = pd.DataFrame(
            {
                "trade_price": [1.0, np.inf, -np.inf],
                "ask_ex": [1.1, 1.2, 1.3],
                "bid_ex": [0.9, 0.8, 0.7],
            }
        )
        clf = ClassicalClassifier(layers=[("quote", "ex")], random_state=42)
        clf.fit(df)
        predictions = clf.predict(df)
        # Should handle infinite values gracefully
        assert len(predictions) == 3

    def test_single_row_data(self) -> None:
        """Test behavior with single row of data."""
        df = pd.DataFrame(
            {"trade_price": [1.0], "ask_ex": [1.1], "bid_ex": [0.9]}
        )
        clf = ClassicalClassifier(layers=[("quote", "ex")], random_state=42)
        clf.fit(df)
        predictions = clf.predict(df)
        assert len(predictions) == 1

    def test_zero_variance_data(self) -> None:
        """Test behavior with zero variance data."""
        df = pd.DataFrame(
            {
                "trade_price": [1.0, 1.0, 1.0],
                "ask_ex": [1.0, 1.0, 1.0],
                "bid_ex": [1.0, 1.0, 1.0],
            }
        )
        clf = ClassicalClassifier(layers=[("quote", "ex")], random_state=42)
        clf.fit(df)
        predictions = clf.predict(df)
        assert len(predictions) == 3
        # With equal prices at bid/ask/mid, it should fall back to strategy
        assert not np.isnan(predictions).any()

    def test_no_layers_provided(self, sample_data: pd.DataFrame) -> None:
        """Test behavior when no layers are provided."""
        clf = ClassicalClassifier(layers=None, random_state=42)
        clf.fit(sample_data)
        predictions = clf.predict(sample_data)
        # Should classify all using strategy when no layers are provided
        assert len(predictions) == len(sample_data)

    def test_empty_layers_list(self, sample_data: pd.DataFrame) -> None:
        """Test behavior with empty layers list."""
        clf = ClassicalClassifier(layers=[], random_state=42)
        clf.fit(sample_data)
        predictions = clf.predict(sample_data)
        assert len(predictions) == len(sample_data)

    def test_invalid_subset(self) -> None:
        """Test behavior with invalid subset name."""
        df = pd.DataFrame(
            {"trade_price": [1.0], "ask_ex": [1.1], "bid_ex": [0.9]}
        )
        # Using subset that doesn't match column patterns
        clf = ClassicalClassifier(layers=[("quote", "invalid_subset")])
        with pytest.raises(ValueError, match="Expected to find columns"):
            clf.fit(df)

    def test_predict_different_column_count(self, fitted_classifier: ClassicalClassifier) -> None:
        """Test error when predict data has different column count."""
        # Data with fewer columns
        df = pd.DataFrame({"trade_price": [1.0], "ask_ex": [1.1]})  # Missing columns
        with pytest.raises(ValueError):
            fitted_classifier.predict(df)

    def test_negative_prices(self) -> None:
        """Test behavior with negative prices (invalid but should handle gracefully)."""
        df = pd.DataFrame(
            {
                "trade_price": [-1.0, 2.0, 3.0],
                "ask_ex": [1.1, 2.2, 3.3],
                "bid_ex": [0.9, 1.8, 2.7],
            }
        )
        clf = ClassicalClassifier(layers=[("quote", "ex")], random_state=42)
        clf.fit(df)
        predictions = clf.predict(df)
        assert len(predictions) == 3
        assert not np.isnan(predictions).any()

    def test_fit_predict_different_dtypes(self, sample_data: pd.DataFrame) -> None:
        """Test behavior when fit and predict have different dtypes."""
        clf = ClassicalClassifier(layers=[("quote", "ex")], random_state=42)
        clf.fit(sample_data)

        # Convert to float32 for prediction
        data_float32 = sample_data.astype(np.float32)
        predictions = clf.predict(data_float32)
        assert len(predictions) == len(sample_data)

    def test_large_values(self) -> None:
        """Test behavior with very large numeric values."""
        df = pd.DataFrame(
            {
                "trade_price": [1e10, 2e10, 3e10],
                "ask_ex": [1.1e10, 2.2e10, 3.3e10],
                "bid_ex": [0.9e10, 1.8e10, 2.7e10],
            }
        )
        clf = ClassicalClassifier(layers=[("quote", "ex")], random_state=42)
        clf.fit(df)
        predictions = clf.predict(df)
        assert len(predictions) == 3
        assert not np.isnan(predictions).any()
