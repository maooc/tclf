"""Unit tests for tclf classical classifier.

Tests cover:
- Core training functions
- Prediction functions
- Data preprocessing/validation
- Exception handling for invalid inputs
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.utils.validation import check_is_fitted

from tclf.classical_classifier import (
    ALLOWED_FUNC_STR,
    ClassicalClassifier,
)


class TestInitialization:
    """Test classifier initialization."""

    def test_default_initialization(self) -> None:
        """Test classifier initializes with default parameters."""
        clf = ClassicalClassifier()
        assert clf.layers is None
        assert clf.random_state == 42
        assert clf.strategy == "random"
        assert clf.features is None

    def test_custom_initialization(self) -> None:
        """Test classifier initializes with custom parameters."""
        clf = ClassicalClassifier(
            layers=[("quote", "ex")],
            random_state=123,
            strategy="const",
            features=["a", "b"],
        )
        assert clf.layers == [("quote", "ex")]
        assert clf.random_state == 123
        assert clf.strategy == "const"
        assert clf.features == ["a", "b"]

    def test_allowed_functions(self) -> None:
        """Test that all allowed functions are valid."""
        expected_functions = [
            "tick",
            "rev_tick",
            "quote",
            "lr",
            "rev_lr",
            "emo",
            "rev_emo",
            "clnv",
            "rev_clnv",
            "trade_size",
            "depth",
            "nan",
        ]
        for func in expected_functions:
            assert func in ALLOWED_FUNC_STR


class TestFitFunction:
    """Test the fit function."""

    def test_fit_returns_self(self, synthetic_train_data: pd.DataFrame) -> None:
        """Test fit returns the classifier instance."""
        clf = ClassicalClassifier()
        result = clf.fit(synthetic_train_data)
        assert result is clf

    def test_fit_creates_attributes(self, synthetic_train_data: pd.DataFrame) -> None:
        """Test fit creates required attributes."""
        clf = ClassicalClassifier(layers=[("quote", "ex")])
        clf.fit(synthetic_train_data)

        assert hasattr(clf, "classes_")
        assert hasattr(clf, "columns_")
        assert hasattr(clf, "func_mapping_")
        assert hasattr(clf, "_layers")
        assert check_is_fitted(clf) is None

    def test_fit_with_numpy_array(
        self, numpy_array_data: np.ndarray, feature_names: list[str]
    ) -> None:
        """Test fit works with numpy array and feature names."""
        clf = ClassicalClassifier(layers=[("quote", "ex")], features=feature_names)
        clf.fit(numpy_array_data)
        assert clf.columns_ == feature_names

    def test_fit_without_layers(self, synthetic_train_data: pd.DataFrame) -> None:
        """Test fit works without layers (uses strategy only)."""
        clf = ClassicalClassifier()
        clf.fit(synthetic_train_data)
        assert clf._layers == []

    def test_fit_invalid_function_string(self, synthetic_train_data: pd.DataFrame) -> None:
        """Test fit raises error for invalid function string."""
        clf = ClassicalClassifier(layers=[("invalid_func", "ex")])
        with pytest.raises(ValueError, match="Unknown function string"):
            clf.fit(synthetic_train_data)

    def test_fit_missing_columns(self, dataframe_with_missing_columns: pd.DataFrame) -> None:
        """Test fit raises error for missing required columns."""
        clf = ClassicalClassifier(layers=[("quote", "ex")])
        with pytest.raises(ValueError, match="Expected to find columns"):
            clf.fit(dataframe_with_missing_columns)

    def test_fit_invalid_column_length(
        self, numpy_array_data: np.ndarray
    ) -> None:
        """Test fit raises error when features length doesn't match data."""
        clf = ClassicalClassifier(
            layers=[("quote", "ex")], features=["only_one_column"]
        )
        with pytest.raises(ValueError, match="Expected"):
            clf.fit(numpy_array_data)

    def test_fit_empty_dataframe(self, empty_dataframe: pd.DataFrame) -> None:
        """Test fit raises error for empty DataFrame."""
        clf = ClassicalClassifier()
        # Empty DataFrame should raise ValueError
        with pytest.raises(ValueError):
            clf.fit(empty_dataframe)

    def test_fit_with_sample_weights(self, synthetic_train_data: pd.DataFrame) -> None:
        """Test fit accepts sample weights."""
        clf = ClassicalClassifier()
        sample_weights = np.ones(len(synthetic_train_data))
        clf.fit(synthetic_train_data, sample_weight=sample_weights)
        assert check_is_fitted(clf) is None


class TestPredictFunction:
    """Test the predict function."""

    def test_predict_output_shape(
        self, synthetic_train_data: pd.DataFrame, synthetic_test_data: pd.DataFrame
    ) -> None:
        """Test predict returns correct output shape."""
        clf = ClassicalClassifier(layers=[("quote", "ex")])
        clf.fit(synthetic_train_data)
        predictions = clf.predict(synthetic_test_data)

        assert predictions.shape == (len(synthetic_test_data),)

    def test_predict_output_values(
        self, synthetic_train_data: pd.DataFrame, synthetic_test_data: pd.DataFrame
    ) -> None:
        """Test predict returns valid classification values."""
        clf = ClassicalClassifier(layers=[("quote", "ex")])
        clf.fit(synthetic_train_data)
        predictions = clf.predict(synthetic_test_data)

        # Values should be -1, 0, or 1
        assert all(np.isin(predictions, [-1, 0, 1]))

    def test_predict_without_fit(self, synthetic_test_data: pd.DataFrame) -> None:
        """Test predict raises error if not fitted."""
        clf = ClassicalClassifier()
        with pytest.raises(Exception):  # sklearn raises NotFittedError
            clf.predict(synthetic_test_data)

    def test_predict_with_numpy_array(
        self, numpy_array_data: np.ndarray, feature_names: list[str]
    ) -> None:
        """Test predict works with numpy array."""
        clf = ClassicalClassifier(layers=[("quote", "ex")], features=feature_names)
        clf.fit(numpy_array_data)
        predictions = clf.predict(numpy_array_data)
        assert len(predictions) == len(numpy_array_data)

    def test_predict_consistency(
        self, synthetic_train_data: pd.DataFrame, synthetic_test_data: pd.DataFrame
    ) -> None:
        """Test predict returns consistent results with same random state."""
        clf1 = ClassicalClassifier(layers=[("quote", "ex")], random_state=42)
        clf2 = ClassicalClassifier(layers=[("quote", "ex")], random_state=42)

        clf1.fit(synthetic_train_data)
        clf2.fit(synthetic_train_data)

        pred1 = clf1.predict(synthetic_test_data)
        pred2 = clf2.predict(synthetic_test_data)

        assert_array_equal(pred1, pred2)


class TestPredictProbaFunction:
    """Test the predict_proba function."""

    def test_predict_proba_output_shape(
        self, synthetic_train_data: pd.DataFrame, synthetic_test_data: pd.DataFrame
    ) -> None:
        """Test predict_proba returns correct output shape."""
        clf = ClassicalClassifier(layers=[("quote", "ex")])
        clf.fit(synthetic_train_data)
        probabilities = clf.predict_proba(synthetic_test_data)

        assert probabilities.shape == (len(synthetic_test_data), 2)

    def test_predict_proba_probability_range(
        self, synthetic_train_data: pd.DataFrame, synthetic_test_data: pd.DataFrame
    ) -> None:
        """Test predict_proba returns probabilities in [0, 1]."""
        clf = ClassicalClassifier(layers=[("quote", "ex")])
        clf.fit(synthetic_train_data)
        probabilities = clf.predict_proba(synthetic_test_data)

        assert (probabilities >= 0).all()
        assert (probabilities <= 1).all()

    def test_predict_proba_sums_to_one(
        self, synthetic_train_data: pd.DataFrame, synthetic_test_data: pd.DataFrame
    ) -> None:
        """Test predict_proba probabilities sum to 1."""
        clf = ClassicalClassifier(layers=[("quote", "ex")])
        clf.fit(synthetic_train_data)
        probabilities = clf.predict_proba(synthetic_test_data)

        assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_predict_proba_const_strategy(
        self, synthetic_train_data: pd.DataFrame, synthetic_test_data: pd.DataFrame
    ) -> None:
        """Test predict_proba with const strategy."""
        clf = ClassicalClassifier(layers=[("nan", "ex")], strategy="const")
        clf.fit(synthetic_train_data)
        probabilities = clf.predict_proba(synthetic_test_data)

        # With nan rule and const strategy, all should be (0.5, 0.5)
        assert_allclose(probabilities, 0.5)


class TestScoreFunction:
    """Test the score function."""

    def test_score_returns_float(
        self, synthetic_train_data: pd.DataFrame, synthetic_test_data: pd.DataFrame
    ) -> None:
        """Test score returns a float value."""
        clf = ClassicalClassifier(layers=[("quote", "ex")])
        clf.fit(synthetic_train_data)

        y_true = np.random.choice([-1, 1], size=len(synthetic_test_data))
        score = clf.score(synthetic_test_data, y_true)

        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_score_perfect_prediction(
        self, synthetic_train_data: pd.DataFrame
    ) -> None:
        """Test score with perfect predictions."""
        clf = ClassicalClassifier(layers=[("nan", "ex")], strategy="const")
        clf.fit(synthetic_train_data)

        # With const strategy, all predictions are 0
        y_true = np.zeros(len(synthetic_train_data))
        score = clf.score(synthetic_train_data, y_true)

        assert score == 1.0


class TestClassificationRules:
    """Test individual classification rules."""

    def test_tick_rule(self) -> None:
        """Test tick rule classification."""
        # Create minimal train data
        train_data = pd.DataFrame({
            "trade_price": [15.0, 25.0, 20.0],
            "price_ex_lag": [10.0, 30.0, 20.0],
        })
        clf = ClassicalClassifier(layers=[("tick", "ex")], features=list(train_data.columns))
        clf.fit(train_data)

        # Create test data with known outcomes
        test_data = pd.DataFrame({
            "trade_price": [15.0, 5.0, 10.0],
            "price_ex_lag": [10.0, 15.0, 10.0],
        })
        predictions = clf.predict(test_data)

        # Price > lag -> buy (1), Price < lag -> sell (-1)
        assert predictions[0] == 1  # 15 > 10
        assert predictions[1] == -1  # 5 < 15

    def test_rev_tick_rule(self) -> None:
        """Test reverse tick rule classification."""
        train_data = pd.DataFrame({
            "trade_price": [10.0, 15.0, 12.0],
            "price_ex_lead": [15.0, 10.0, 12.0],
        })
        clf = ClassicalClassifier(layers=[("rev_tick", "ex")], features=list(train_data.columns))
        clf.fit(train_data)

        test_data = pd.DataFrame({
            "trade_price": [10.0, 15.0, 10.0],
            "price_ex_lead": [15.0, 10.0, 10.0],
        })
        predictions = clf.predict(test_data)

        # Lead > price -> sell (-1), Lead < price -> buy (1)
        assert predictions[0] == -1  # 15 > 10
        assert predictions[1] == 1  # 10 < 15

    def test_quote_rule(self) -> None:
        """Test quote rule classification."""
        train_data = pd.DataFrame({
            "trade_price": [15.0, 25.0, 20.0],
            "bid_ex": [10.0, 20.0, 19.0],
            "ask_ex": [20.0, 30.0, 21.0],
        })
        clf = ClassicalClassifier(layers=[("quote", "ex")], strategy="const", features=list(train_data.columns))
        clf.fit(train_data)

        test_data = pd.DataFrame({
            "trade_price": [15.0, 5.0, 10.0],
            "bid_ex": [10.0, 10.0, 10.0],
            "ask_ex": [20.0, 20.0, 20.0],
        })
        predictions = clf.predict(test_data)

        # Midpoint is 15. Price > mid -> buy (1), Price < mid -> sell (-1), Price == mid -> 0
        assert predictions[0] == 0  # 15 == mid, unclassified with const
        assert predictions[1] == -1  # 5 < 15

    def test_quote_rule_with_random_strategy(self) -> None:
        """Test quote rule with random strategy for midspread trades."""
        train_data = pd.DataFrame({
            "trade_price": [15.0, 25.0, 20.0],
            "bid_ex": [10.0, 20.0, 19.0],
            "ask_ex": [20.0, 30.0, 21.0],
        })
        clf = ClassicalClassifier(
            layers=[("quote", "ex")], strategy="random", random_state=42,
            features=list(train_data.columns)
        )
        clf.fit(train_data)

        test_data = pd.DataFrame({
            "trade_price": [15.0],  # Exactly at midpoint
            "bid_ex": [10.0],
            "ask_ex": [20.0],
        })
        predictions = clf.predict(test_data)

        # Should be randomly assigned -1 or 1
        assert predictions[0] in [-1, 1]

    def test_lr_rule(self) -> None:
        """Test LR (Lee-Ready) rule classification."""
        train_data = pd.DataFrame({
            "trade_price": [15.0, 25.0, 20.0],
            "bid_ex": [10.0, 20.0, 19.0],
            "ask_ex": [20.0, 30.0, 21.0],
            "price_ex_lag": [10.0, 30.0, 20.0],
        })
        clf = ClassicalClassifier(layers=[("lr", "ex")], features=list(train_data.columns))
        clf.fit(train_data)

        # LR uses quote rule first, then tick rule for midspread
        test_data = pd.DataFrame({
            "trade_price": [15.0, 5.0, 15.0],
            "bid_ex": [10.0, 10.0, 10.0],
            "ask_ex": [20.0, 20.0, 20.0],
            "price_ex_lag": [10.0, 10.0, 20.0],
        })
        predictions = clf.predict(test_data)

        # First at mid, classified by tick rule (15 > 10 -> buy)
        # Second below mid -> sell
        # Third at mid, classified by tick rule (15 < 20 -> sell)
        assert predictions[0] == 1
        assert predictions[1] == -1
        assert predictions[2] == -1

    def test_emo_rule(self) -> None:
        """Test EMO rule classification."""
        train_data = pd.DataFrame({
            "trade_price": [20.0, 10.0, 15.0],
            "bid_ex": [10.0, 10.0, 10.0],
            "ask_ex": [20.0, 20.0, 20.0],
            "price_ex_lag": [15.0, 15.0, 10.0],
        })
        clf = ClassicalClassifier(layers=[("emo", "ex")], features=list(train_data.columns))
        clf.fit(train_data)

        test_data = pd.DataFrame({
            "trade_price": [20.0, 10.0, 15.0],
            "bid_ex": [10.0, 10.0, 10.0],
            "ask_ex": [20.0, 20.0, 20.0],
            "price_ex_lag": [15.0, 15.0, 10.0],
        })
        predictions = clf.predict(test_data)

        # At ask -> buy, at bid -> sell, otherwise tick rule
        assert predictions[0] == 1  # At ask
        assert predictions[1] == -1  # At bid

    def test_clnv_rule(self) -> None:
        """Test CLNV rule classification."""
        train_data = pd.DataFrame({
            "trade_price": [17.0, 13.0, 15.0],
            "bid_ex": [10.0, 10.0, 10.0],
            "ask_ex": [20.0, 20.0, 20.0],
            "price_ex_lag": [15.0, 15.0, 16.0],
        })
        clf = ClassicalClassifier(layers=[("clnv", "ex")], features=list(train_data.columns))
        clf.fit(train_data)

        # CLNV uses quote rule for upper/lower 30%, tick rule for middle
        test_data = pd.DataFrame({
            "trade_price": [17.0, 13.0, 15.0],
            "bid_ex": [10.0, 10.0, 10.0],
            "ask_ex": [20.0, 20.0, 20.0],
            "price_ex_lag": [15.0, 15.0, 16.0],
        })
        predictions = clf.predict(test_data)

        # 17 in upper 30% (17-20), quote rule -> buy
        # 13 in lower 30% (10-13), quote rule -> sell
        # 15 in middle, tick rule
        assert predictions[0] == 1
        assert predictions[1] == -1

    def test_trade_size_rule(self) -> None:
        """Test trade size rule classification."""
        train_data = pd.DataFrame({
            "trade_size": [100, 200, 150],
            "bid_size_ex": [100, 150, 100],
            "ask_size_ex": [200, 200, 150],
        })
        clf = ClassicalClassifier(layers=[("trade_size", "ex")], features=list(train_data.columns))
        clf.fit(train_data)

        test_data = pd.DataFrame({
            "trade_size": [100, 200, 150],
            "bid_size_ex": [100, 150, 100],
            "ask_size_ex": [200, 200, 150],
        })
        predictions = clf.predict(test_data)

        # Trade size matches bid size -> buy (1)
        # Trade size matches ask size -> sell (-1)
        assert predictions[0] == 1
        assert predictions[1] == -1

    def test_depth_rule(self) -> None:
        """Test depth rule classification."""
        train_data = pd.DataFrame({
            "trade_price": [15.0, 15.0, 20.0],
            "bid_ex": [10.0, 10.0, 10.0],
            "ask_ex": [20.0, 20.0, 20.0],
            "bid_size_ex": [100, 200, 150],
            "ask_size_ex": [200, 100, 150],
        })
        clf = ClassicalClassifier(layers=[("depth", "ex")], features=list(train_data.columns))
        clf.fit(train_data)

        test_data = pd.DataFrame({
            "trade_price": [15.0, 15.0],
            "bid_ex": [10.0, 10.0],
            "ask_ex": [20.0, 20.0],
            "bid_size_ex": [100, 200],
            "ask_size_ex": [200, 100],
        })
        predictions = clf.predict(test_data)

        # At mid, ask_size > bid_size -> buy (1)
        # At mid, ask_size < bid_size -> sell (-1)
        assert predictions[0] == 1
        assert predictions[1] == -1

    def test_nan_rule(self, synthetic_train_data: pd.DataFrame) -> None:
        """Test nan rule (no classification)."""
        clf = ClassicalClassifier(layers=[("nan", "ex")], strategy="const")
        clf.fit(synthetic_train_data)

        predictions = clf.predict(synthetic_train_data)

        # All should be unclassified (0 with const strategy)
        assert (predictions == 0).all()


class TestLayerStacking:
    """Test stacking multiple classification layers."""

    def test_two_layer_stack(
        self, synthetic_train_data: pd.DataFrame, synthetic_test_data: pd.DataFrame
    ) -> None:
        """Test stacking two classification rules."""
        clf = ClassicalClassifier(
            layers=[("quote", "ex"), ("tick", "ex")], random_state=42
        )
        clf.fit(synthetic_train_data)
        predictions = clf.predict(synthetic_test_data)

        assert len(predictions) == len(synthetic_test_data)

    def test_multiple_layer_stack(
        self, synthetic_train_data: pd.DataFrame
    ) -> None:
        """Test stacking multiple classification rules."""
        clf = ClassicalClassifier(
            layers=[
                ("quote", "ex"),
                ("emo", "ex"),
                ("tick", "ex"),
            ],
            random_state=42,
        )
        clf.fit(synthetic_train_data)
        predictions = clf.predict(synthetic_train_data)

        assert len(predictions) == len(synthetic_train_data)

    def test_layer_override(self) -> None:
        """Test that first layer results are not overridden."""
        # Create train data with columns needed for both layers
        train_data = pd.DataFrame({
            "trade_price": [5.0, 25.0, 15.0],
            "bid_ex": [10.0, 10.0, 10.0],
            "ask_ex": [20.0, 20.0, 20.0],
            "price_ex_lag": [10.0, 30.0, 20.0],
        })
        # If first layer classifies all, second layer shouldn't change anything
        clf = ClassicalClassifier(
            layers=[("quote", "ex"), ("tick", "ex")], strategy="const",
            features=list(train_data.columns)
        )
        clf.fit(train_data)

        # Create data where quote rule classifies everything
        test_data = pd.DataFrame({
            "trade_price": [5.0, 25.0],
            "bid_ex": [10.0, 10.0],
            "ask_ex": [20.0, 20.0],
            "price_ex_lag": [10.0, 30.0],
        })
        predictions = clf.predict(test_data)

        # Both should be classified by quote rule
        assert predictions[0] == -1  # Below mid
        assert predictions[1] == 1  # Above mid


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_row_prediction(self) -> None:
        """Test prediction on single row."""
        train_data = pd.DataFrame({
            "trade_price": [15.0, 25.0],
            "bid_ex": [10.0, 20.0],
            "ask_ex": [20.0, 30.0],
        })
        clf = ClassicalClassifier(layers=[("quote", "ex")], features=list(train_data.columns))
        clf.fit(train_data)

        single_row = pd.DataFrame({
            "trade_price": [15.0],
            "bid_ex": [10.0],
            "ask_ex": [20.0],
        })
        predictions = clf.predict(single_row)

        assert len(predictions) == 1

    def test_data_with_nans(self) -> None:
        """Test handling of NaN values."""
        train_data = pd.DataFrame({
            "trade_price": [15.0, 25.0],
            "bid_ex": [10.0, 20.0],
            "ask_ex": [20.0, 30.0],
        })
        clf = ClassicalClassifier(layers=[("quote", "ex")], strategy="const", features=list(train_data.columns))
        clf.fit(train_data)

        test_data = pd.DataFrame({
            "trade_price": [1.0, np.nan, 3.0],
            "bid_ex": [0.5, 1.5, np.nan],
            "ask_ex": [1.5, 2.5, 3.5],
        })
        # Should handle NaN without raising error
        predictions = clf.predict(test_data)
        assert len(predictions) == len(test_data)

    def test_data_with_infs(self) -> None:
        """Test handling of infinity values."""
        train_data = pd.DataFrame({
            "trade_price": [15.0, 25.0],
            "bid_ex": [10.0, 20.0],
            "ask_ex": [20.0, 30.0],
        })
        clf = ClassicalClassifier(layers=[("quote", "ex")], strategy="const", features=list(train_data.columns))
        clf.fit(train_data)

        test_data = pd.DataFrame({
            "trade_price": [1.0, np.inf, 3.0],
            "bid_ex": [0.5, 1.5, -np.inf],
            "ask_ex": [1.5, 2.5, 3.5],
        })
        # Should handle infinity without raising error
        predictions = clf.predict(test_data)
        assert len(predictions) == len(test_data)

    def test_bid_greater_than_ask(self) -> None:
        """Test handling when bid > ask (invalid spread)."""
        train_data = pd.DataFrame({
            "trade_price": [15.0, 25.0],
            "bid_ex": [10.0, 20.0],
            "ask_ex": [20.0, 30.0],
        })
        clf = ClassicalClassifier(layers=[("quote", "ex")], strategy="const", features=list(train_data.columns))
        clf.fit(train_data)

        test_data = pd.DataFrame({
            "trade_price": [15.0],
            "bid_ex": [20.0],
            "ask_ex": [10.0],
        })
        predictions = clf.predict(test_data)

        # Should handle invalid spread
        assert len(predictions) == 1

    def test_all_nan_input(self) -> None:
        """Test handling when all inputs are NaN."""
        train_data = pd.DataFrame({
            "trade_price": [15.0, 25.0],
            "bid_ex": [10.0, 20.0],
            "ask_ex": [20.0, 30.0],
        })
        clf = ClassicalClassifier(layers=[("quote", "ex")], strategy="const", features=list(train_data.columns))
        clf.fit(train_data)

        test_data = pd.DataFrame({
            "trade_price": [np.nan],
            "bid_ex": [np.nan],
            "ask_ex": [np.nan],
        })
        predictions = clf.predict(test_data)

        assert predictions[0] == 0


class TestStrategyParameter:
    """Test different strategy parameters."""

    def test_random_strategy(
        self, synthetic_train_data: pd.DataFrame, synthetic_test_data: pd.DataFrame
    ) -> None:
        """Test random strategy for unclassified trades."""
        clf = ClassicalClassifier(
            layers=[("nan", "ex")], strategy="random", random_state=42
        )
        clf.fit(synthetic_train_data)
        predictions = clf.predict(synthetic_test_data)

        # All should be randomly assigned -1 or 1
        assert all(np.isin(predictions, [-1, 1]))

    def test_const_strategy(
        self, synthetic_train_data: pd.DataFrame, synthetic_test_data: pd.DataFrame
    ) -> None:
        """Test const strategy for unclassified trades."""
        clf = ClassicalClassifier(layers=[("nan", "ex")], strategy="const")
        clf.fit(synthetic_train_data)
        predictions = clf.predict(synthetic_test_data)

        # All should be 0
        assert (predictions == 0).all()

    def test_random_state_reproducibility(
        self, synthetic_train_data: pd.DataFrame, synthetic_test_data: pd.DataFrame
    ) -> None:
        """Test that random state ensures reproducibility."""
        clf1 = ClassicalClassifier(
            layers=[("nan", "ex")], strategy="random", random_state=42
        )
        clf2 = ClassicalClassifier(
            layers=[("nan", "ex")], strategy="random", random_state=42
        )

        clf1.fit(synthetic_train_data)
        clf2.fit(synthetic_train_data)

        pred1 = clf1.predict(synthetic_test_data)
        pred2 = clf2.predict(synthetic_test_data)

        assert_array_equal(pred1, pred2)


class TestSklearnCompatibility:
    """Test scikit-learn compatibility."""

    def test_is_classifier(self) -> None:
        """Test that classifier is recognized as sklearn classifier."""
        from sklearn.base import is_classifier

        clf = ClassicalClassifier()
        assert is_classifier(clf)

    def test_get_params(self) -> None:
        """Test get_params method."""
        clf = ClassicalClassifier(layers=[("quote", "ex")], random_state=42)
        params = clf.get_params()

        assert "layers" in params
        assert "random_state" in params
        assert "strategy" in params
        assert params["random_state"] == 42

    def test_set_params(self) -> None:
        """Test set_params method."""
        clf = ClassicalClassifier()
        clf.set_params(random_state=123, strategy="const")

        assert clf.random_state == 123
        assert clf.strategy == "const"

    def test_clone(self) -> None:
        """Test that classifier can be cloned."""
        from sklearn.base import clone

        clf = ClassicalClassifier(layers=[("quote", "ex")], random_state=42)
        clf_clone = clone(clf)

        assert clf_clone.random_state == clf.random_state
        assert clf_clone.strategy == clf.strategy
