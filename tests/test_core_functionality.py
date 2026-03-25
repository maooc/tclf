"""Core functionality tests for ClassicalClassifier."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.validation import check_is_fitted

from tclf.classical_classifier import ALLOWED_FUNC_STR, ClassicalClassifier


class TestCoreFunctionality:
    """Tests for core classifier functionality."""

    def test_fit_basic(self, sample_data: pd.DataFrame) -> None:
        """Test basic fitting functionality."""
        clf = ClassicalClassifier(layers=[("quote", "ex")], random_state=42)
        result = clf.fit(sample_data)
        assert isinstance(result, ClassicalClassifier)
        assert check_is_fitted(clf) is None
        assert hasattr(clf, "classes_")
        assert hasattr(clf, "func_mapping_")

    def test_fit_without_y(self, sample_data: pd.DataFrame) -> None:
        """Test that fit works without y (unsupervised)."""
        clf = ClassicalClassifier(layers=[("quote", "ex")])
        result = clf.fit(sample_data, y=None)
        assert isinstance(result, ClassicalClassifier)

    def test_fit_with_sample_weight(self, sample_data: pd.DataFrame) -> None:
        """Test fit with sample weights."""
        clf = ClassicalClassifier(layers=[("quote", "ex")])
        sample_weight = np.ones(len(sample_data))
        result = clf.fit(sample_data, sample_weight=sample_weight)
        assert isinstance(result, ClassicalClassifier)

    def test_predict_basic(self, fitted_classifier: ClassicalClassifier, sample_data: pd.DataFrame) -> None:
        """Test basic prediction functionality."""
        predictions = fitted_classifier.predict(sample_data)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(sample_data)
        assert all(pred in [-1, 1] for pred in predictions)

    def test_predict_proba_basic(self, fitted_classifier: ClassicalClassifier, sample_data: pd.DataFrame) -> None:
        """Test basic predict_proba functionality."""
        probas = fitted_classifier.predict_proba(sample_data)
        assert isinstance(probas, np.ndarray)
        assert probas.shape == (len(sample_data), 2)
        assert np.all(probas >= 0) and np.all(probas <= 1)
        assert np.allclose(probas.sum(axis=1), 1.0)

    def test_strategy_const_behavior(self, sample_data: pd.DataFrame) -> None:
        """Test const strategy returns 0.5 probabilities for unclassified."""
        clf = ClassicalClassifier(layers=[("nan", "ex")], strategy="const")
        clf.fit(sample_data)
        probas = clf.predict_proba(sample_data)
        assert np.allclose(probas, 0.5)

    def test_strategy_random_behavior(self, sample_data: pd.DataFrame) -> None:
        """Test random strategy produces random predictions."""
        clf1 = ClassicalClassifier(layers=[("nan", "ex")], strategy="random", random_state=42)
        clf2 = ClassicalClassifier(layers=[("nan", "ex")], strategy="random", random_state=42)
        clf3 = ClassicalClassifier(layers=[("nan", "ex")], strategy="random", random_state=123)

        clf1.fit(sample_data)
        clf2.fit(sample_data)
        clf3.fit(sample_data)

        pred1 = clf1.predict(sample_data)
        pred2 = clf2.predict(sample_data)
        pred3 = clf3.predict(sample_data)

        # Same random state should produce same predictions
        np.testing.assert_array_equal(pred1, pred2)
        # Different random states should produce different predictions
        assert not np.array_equal(pred1, pred3)

    def test_numpy_input(self, sample_data: pd.DataFrame, sample_data_numpy: np.ndarray, feature_names: list[str]) -> None:
        """Test classifier works with numpy input."""
        clf = ClassicalClassifier(
            layers=[("quote", "ex")],
            features=feature_names,
            random_state=42,
        )
        clf.fit(sample_data_numpy)
        predictions = clf.predict(sample_data_numpy)
        assert len(predictions) == len(sample_data_numpy)

    def test_multiple_layers(self, sample_data: pd.DataFrame) -> None:
        """Test classification with multiple layers."""
        clf = ClassicalClassifier(
            layers=[("quote", "ex"), ("tick", "all"), ("emo", "best")],
            random_state=42,
        )
        clf.fit(sample_data)
        predictions = clf.predict(sample_data)
        assert len(predictions) == len(sample_data)

    @pytest.mark.parametrize("func_name", ALLOWED_FUNC_STR)
    def test_all_classification_functions(self, sample_data: pd.DataFrame, func_name: str) -> None:
        """Test all individual classification functions."""
        # Determine appropriate subset based on function
        if func_name in ["tick", "rev_tick", "trade_size", "depth", "nan"]:
            subset = "ex"
        else:
            subset = "ex" if "ex" in sample_data.columns else "best"

        clf = ClassicalClassifier(layers=[(func_name, subset)], random_state=42)

        # Some functions need specific columns, skip if not available
        try:
            clf.fit(sample_data)
            predictions = clf.predict(sample_data)
            assert len(predictions) == len(sample_data)
        except ValueError as e:
            if "Expected to find columns" in str(e):
                pytest.skip(f"Missing required columns for {func_name}: {e}")
            raise

    def test_score_function(self, fitted_classifier: ClassicalClassifier, sample_data: pd.DataFrame) -> None:
        """Test score calculation."""
        y_true = np.random.choice([-1, 1], size=len(sample_data))
        score = fitted_classifier.score(sample_data, y_true)
        assert 0.0 <= score <= 1.0

    def test_edge_case_predictions(self, edge_case_data: pd.DataFrame) -> None:
        """Test predictions with edge case data (NaN values, equal prices)."""
        clf = ClassicalClassifier(
            layers=[("quote", "ex"), ("tick", "ex")],
            random_state=42,
        )
        clf.fit(edge_case_data)
        predictions = clf.predict(edge_case_data)
        assert len(predictions) == len(edge_case_data)
        # All predictions should be valid (no NaN in output)
        assert not np.isnan(predictions).any()


class TestDataPreprocessing:
    """Tests for data preprocessing and validation."""

    def test_column_validation(self, sample_data: pd.DataFrame) -> None:
        """Test that column validation correctly identifies required columns."""
        # This should work as all columns are present
        clf = ClassicalClassifier(
            layers=[("quote", "ex"), ("tick", "all"), ("emo", "best")],
        )
        clf.fit(sample_data)  # Should not raise

    def test_features_parameter(self, sample_data: pd.DataFrame) -> None:
        """Test features parameter functionality."""
        features = sample_data.columns.tolist()
        clf = ClassicalClassifier(
            layers=[("quote", "ex")],
            features=features,
        )
        # Should work with numpy array when features are provided
        clf.fit(sample_data.to_numpy())
        assert clf.features == features

    def test_feature_column_mismatch(self, sample_data: pd.DataFrame) -> None:
        """Test error when feature count doesn't match data columns."""
        clf = ClassicalClassifier(
            layers=[("quote", "ex")],
            features=["trade_price", "ask_ex"],  # Only 2 features
        )
        # Data has more columns than features
        with pytest.raises(ValueError, match="Expected"):
            clf.fit(sample_data.to_numpy())

    def test_force_all_finite(self, sample_data: pd.DataFrame) -> None:
        """Test that NaN values are allowed in input."""
        # Create data with NaN values
        nan_data = sample_data.copy()
        nan_data.iloc[0, 0] = np.nan

        clf = ClassicalClassifier(layers=[("quote", "ex")])
        # Should not raise due to allow_nan tag
        clf.fit(nan_data)
        predictions = clf.predict(nan_data)
        assert not np.isnan(predictions).any()
