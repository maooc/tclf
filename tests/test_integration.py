"""Integration and end-to-end tests for tclf.

These tests verify the complete pipeline:
- Training → Saving → Loading → Prediction
- Full workflow with different configurations
- Performance tests
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from tclf.classical_classifier import ClassicalClassifier


class TestFullPipeline:
    """End-to-end tests for the complete classification pipeline."""

    def test_basic_pipeline_with_dataframe(self) -> None:
        """Test complete pipeline with DataFrame input."""
        # Create synthetic data
        np.random.seed(42)
        n_samples = 50

        train_data = pd.DataFrame({
            "trade_price": np.random.uniform(10, 100, n_samples),
            "bid_ex": np.random.uniform(9, 99, n_samples),
            "ask_ex": np.random.uniform(11, 101, n_samples),
            "price_ex_lag": np.random.uniform(10, 100, n_samples),
        })
        train_data["ask_ex"] = np.maximum(train_data["ask_ex"], train_data["bid_ex"] + 0.01)

        test_data = pd.DataFrame({
            "trade_price": [15.0, 25.0, 20.0],
            "bid_ex": [10.0, 20.0, 19.0],
            "ask_ex": [20.0, 30.0, 21.0],
            "price_ex_lag": [10.0, 30.0, 20.0],
        })

        # Full pipeline
        clf = ClassicalClassifier(
            layers=[("quote", "ex"), ("tick", "ex")],
            strategy="random",
            random_state=42,
        )

        # Train
        clf.fit(train_data)
        assert hasattr(clf, "classes_")

        # Predict
        predictions = clf.predict(test_data)
        assert len(predictions) == len(test_data)

        # Predict probabilities
        probabilities = clf.predict_proba(test_data)
        assert probabilities.shape == (len(test_data), 2)

        # Score
        y_true = np.array([-1, 1, 1])
        score = clf.score(test_data, y_true)
        assert 0 <= score <= 1

    def test_basic_pipeline_with_numpy(self) -> None:
        """Test complete pipeline with numpy array input."""
        features = ["trade_price", "bid_ex", "ask_ex", "price_ex_lag"]

        train_data = np.array([
            [15.0, 10.0, 20.0, 10.0],
            [25.0, 20.0, 30.0, 30.0],
            [20.0, 19.0, 21.0, 20.0],
            [18.0, 15.0, 25.0, 15.0],
        ])

        test_data = np.array([
            [15.0, 10.0, 20.0, 10.0],
            [25.0, 20.0, 30.0, 30.0],
        ])

        # Full pipeline with numpy
        clf = ClassicalClassifier(
            layers=[("quote", "ex"), ("tick", "ex")],
            strategy="random",
            random_state=42,
            features=features,
        )

        clf.fit(train_data)
        predictions = clf.predict(test_data)
        probabilities = clf.predict_proba(test_data)

        assert len(predictions) == len(test_data)
        assert probabilities.shape == (len(test_data), 2)

    def test_pipeline_with_all_rules(self) -> None:
        """Test pipeline using all available classification rules."""
        np.random.seed(42)
        n_samples = 100

        train_data = pd.DataFrame({
            "trade_price": np.random.uniform(10, 100, n_samples),
            "bid_ex": np.random.uniform(9, 99, n_samples),
            "ask_ex": np.random.uniform(11, 101, n_samples),
            "bid_best": np.random.uniform(9, 99, n_samples),
            "ask_best": np.random.uniform(11, 101, n_samples),
            "price_ex_lag": np.random.uniform(10, 100, n_samples),
            "price_ex_lead": np.random.uniform(10, 100, n_samples),
            "price_best_lag": np.random.uniform(10, 100, n_samples),
            "price_best_lead": np.random.uniform(10, 100, n_samples),
            "price_all_lag": np.random.uniform(10, 100, n_samples),
            "price_all_lead": np.random.uniform(10, 100, n_samples),
            "trade_size": np.random.randint(1, 1000, n_samples),
            "bid_size_ex": np.random.randint(1, 500, n_samples),
            "ask_size_ex": np.random.randint(1, 500, n_samples),
        })
        train_data["ask_ex"] = np.maximum(train_data["ask_ex"], train_data["bid_ex"] + 0.01)
        train_data["ask_best"] = np.maximum(train_data["ask_best"], train_data["bid_best"] + 0.01)

        # Test all rules in a comprehensive stack
        clf = ClassicalClassifier(
            layers=[
                ("quote", "best"),
                ("emo", "best"),
                ("lr", "ex"),
                ("tick", "ex"),
            ],
            strategy="random",
            random_state=42,
        )

        clf.fit(train_data)
        predictions = clf.predict(train_data)
        probabilities = clf.predict_proba(train_data)

        assert len(predictions) == len(train_data)
        assert probabilities.shape == (len(train_data), 2)


class TestModelPersistence:
    """Tests for model saving and loading."""

    def test_pickle_save_load(self, tmp_path: Path) -> None:
        """Test saving and loading model with pickle."""
        # Create data
        train_data = pd.DataFrame({
            "trade_price": [15.0, 25.0, 20.0],
            "bid_ex": [10.0, 20.0, 19.0],
            "ask_ex": [20.0, 30.0, 21.0],
        })

        test_data = pd.DataFrame({
            "trade_price": [18.0, 22.0],
            "bid_ex": [15.0, 20.0],
            "ask_ex": [25.0, 28.0],
        })

        # Train and predict with original model
        clf = ClassicalClassifier(
            layers=[("quote", "ex")],
            strategy="random",
            random_state=42,
        )
        clf.fit(train_data)
        original_predictions = clf.predict(test_data)

        # Save model
        model_path = tmp_path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)

        # Load model
        with open(model_path, "rb") as f:
            loaded_clf = pickle.load(f)

        # Predict with loaded model
        loaded_predictions = loaded_clf.predict(test_data)

        # Predictions should be identical
        assert_array_equal(original_predictions, loaded_predictions)

    def test_pickle_with_all_rules(self, tmp_path: Path) -> None:
        """Test pickle with complex model configuration."""
        train_data = pd.DataFrame({
            "trade_price": [15.0, 25.0, 20.0, 18.0],
            "bid_ex": [10.0, 20.0, 19.0, 15.0],
            "ask_ex": [20.0, 30.0, 21.0, 25.0],
            "bid_best": [10.0, 20.0, 19.0, 15.0],
            "ask_best": [20.0, 30.0, 21.0, 25.0],
            "price_ex_lag": [10.0, 30.0, 20.0, 15.0],
            "price_ex_lead": [20.0, 10.0, 20.0, 20.0],
        })

        test_data = pd.DataFrame({
            "trade_price": [16.0, 24.0],
            "bid_ex": [12.0, 18.0],
            "ask_ex": [22.0, 28.0],
            "bid_best": [12.0, 18.0],
            "ask_best": [22.0, 28.0],
            "price_ex_lag": [12.0, 28.0],
            "price_ex_lead": [18.0, 22.0],
        })

        clf = ClassicalClassifier(
            layers=[
                ("quote", "ex"),
                ("quote", "best"),
                ("tick", "ex"),
                ("rev_tick", "ex"),
            ],
            strategy="random",
            random_state=42,
        )
        clf.fit(train_data)
        original_predictions = clf.predict(test_data)
        original_proba = clf.predict_proba(test_data)

        # Save and load
        model_path = tmp_path / "complex_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)

        with open(model_path, "rb") as f:
            loaded_clf = pickle.load(f)

        loaded_predictions = loaded_clf.predict(test_data)
        loaded_proba = loaded_clf.predict_proba(test_data)

        assert_array_equal(original_predictions, loaded_predictions)
        assert_allclose(original_proba, loaded_proba)

    def test_pickle_numpy_array_model(self, tmp_path: Path) -> None:
        """Test pickle with model trained on numpy array."""
        features = ["trade_price", "bid_ex", "ask_ex"]
        train_data = np.array([
            [15.0, 10.0, 20.0],
            [25.0, 20.0, 30.0],
            [20.0, 19.0, 21.0],
        ])

        test_data = np.array([
            [16.0, 12.0, 22.0],
            [24.0, 18.0, 28.0],
        ])

        clf = ClassicalClassifier(
            layers=[("quote", "ex")],
            strategy="random",
            random_state=42,
            features=features,
        )
        clf.fit(train_data)
        original_predictions = clf.predict(test_data)

        # Save and load
        model_path = tmp_path / "numpy_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)

        with open(model_path, "rb") as f:
            loaded_clf = pickle.load(f)

        loaded_predictions = loaded_clf.predict(test_data)
        assert_array_equal(original_predictions, loaded_predictions)


class TestWorkflowScenarios:
    """Test realistic workflow scenarios."""

    def test_exchange_then_best_classification(self) -> None:
        """Test workflow: classify on exchange level, then best level."""
        np.random.seed(42)
        n_samples = 100

        data = pd.DataFrame({
            "trade_price": np.random.uniform(10, 100, n_samples),
            "bid_ex": np.random.uniform(9, 99, n_samples),
            "ask_ex": np.random.uniform(11, 101, n_samples),
            "bid_best": np.random.uniform(9.5, 99.5, n_samples),
            "ask_best": np.random.uniform(10.5, 100.5, n_samples),
        })
        data["ask_ex"] = np.maximum(data["ask_ex"], data["bid_ex"] + 0.01)
        data["ask_best"] = np.maximum(data["ask_best"], data["bid_best"] + 0.01)

        # Common workflow: try exchange first, then best
        clf = ClassicalClassifier(
            layers=[("quote", "ex"), ("quote", "best")],
            strategy="random",
            random_state=42,
        )

        clf.fit(data)
        predictions = clf.predict(data)
        probabilities = clf.predict_proba(data)

        assert len(predictions) == len(data)
        assert probabilities.shape == (len(data), 2)

    def test_hybrid_classification_stack(self) -> None:
        """Test complex hybrid classification stack."""
        np.random.seed(42)
        n_samples = 100

        data = pd.DataFrame({
            "trade_price": np.random.uniform(10, 100, n_samples),
            "bid_ex": np.random.uniform(9, 99, n_samples),
            "ask_ex": np.random.uniform(11, 101, n_samples),
            "bid_best": np.random.uniform(9, 99, n_samples),
            "ask_best": np.random.uniform(11, 101, n_samples),
            "price_ex_lag": np.random.uniform(10, 100, n_samples),
            "price_all_lag": np.random.uniform(10, 100, n_samples),
            "trade_size": np.random.randint(1, 1000, n_samples),
            "bid_size_ex": np.random.randint(1, 500, n_samples),
            "ask_size_ex": np.random.randint(1, 500, n_samples),
        })
        data["ask_ex"] = np.maximum(data["ask_ex"], data["bid_ex"] + 0.01)
        data["ask_best"] = np.maximum(data["ask_best"], data["bid_best"] + 0.01)

        # Complex stack: best quote -> ex quote -> emo -> lr -> tick
        clf = ClassicalClassifier(
            layers=[
                ("quote", "best"),
                ("quote", "ex"),
                ("emo", "ex"),
                ("lr", "ex"),
                ("tick", "all"),
            ],
            strategy="random",
            random_state=42,
        )

        clf.fit(data)
        predictions = clf.predict(data)

        assert len(predictions) == len(data)

    def test_option_trade_classification_workflow(self) -> None:
        """Test workflow inspired by option trade classification."""
        np.random.seed(42)
        n_samples = 200

        # Simulate option trade data
        data = pd.DataFrame({
            "trade_price": np.random.uniform(1, 50, n_samples),
            "bid_ex": np.random.uniform(0.5, 49.5, n_samples),
            "ask_ex": np.random.uniform(1.5, 50.5, n_samples),
            "bid_best": np.random.uniform(0.5, 49.5, n_samples),
            "ask_best": np.random.uniform(1.5, 50.5, n_samples),
            "price_ex_lag": np.random.uniform(1, 50, n_samples),
            "price_ex_lead": np.random.uniform(1, 50, n_samples),
            "price_best_lag": np.random.uniform(1, 50, n_samples),
            "price_best_lead": np.random.uniform(1, 50, n_samples),
            "trade_size": np.random.randint(1, 100, n_samples),
            "bid_size_ex": np.random.randint(1, 50, n_samples),
            "ask_size_ex": np.random.randint(1, 50, n_samples),
        })
        data["ask_ex"] = np.maximum(data["ask_ex"], data["bid_ex"] + 0.01)
        data["ask_best"] = np.maximum(data["ask_best"], data["bid_best"] + 0.01)

        # Option trade classification stack
        clf = ClassicalClassifier(
            layers=[
                ("trade_size", "ex"),
                ("depth", "ex"),
                ("quote", "best"),
                ("clnv", "best"),
                ("tick", "ex"),
            ],
            strategy="random",
            random_state=42,
        )

        clf.fit(data)
        predictions = clf.predict(data)
        probabilities = clf.predict_proba(data)

        assert len(predictions) == len(data)
        assert probabilities.shape == (len(data), 2)

        # Verify all predictions are valid
        assert all(np.isin(predictions, [-1, 0, 1]))


class TestPerformance:
    """Performance-related integration tests."""

    def test_large_dataset_handling(self) -> None:
        """Test handling of larger datasets."""
        np.random.seed(42)
        n_samples = 10000

        data = pd.DataFrame({
            "trade_price": np.random.uniform(10, 100, n_samples),
            "bid_ex": np.random.uniform(9, 99, n_samples),
            "ask_ex": np.random.uniform(11, 101, n_samples),
            "price_ex_lag": np.random.uniform(10, 100, n_samples),
        })
        data["ask_ex"] = np.maximum(data["ask_ex"], data["bid_ex"] + 0.01)

        clf = ClassicalClassifier(
            layers=[("quote", "ex"), ("tick", "ex")],
            strategy="random",
            random_state=42,
        )

        clf.fit(data)
        predictions = clf.predict(data)
        probabilities = clf.predict_proba(data)

        assert len(predictions) == n_samples
        assert probabilities.shape == (n_samples, 2)

    def test_multiple_fit_predict_cycles(self) -> None:
        """Test multiple fit/predict cycles on same classifier instance."""
        train_data1 = pd.DataFrame({
            "trade_price": [15.0, 25.0],
            "bid_ex": [10.0, 20.0],
            "ask_ex": [20.0, 30.0],
        })

        train_data2 = pd.DataFrame({
            "trade_price": [30.0, 40.0],
            "bid_ex": [25.0, 35.0],
            "ask_ex": [35.0, 45.0],
        })

        test_data = pd.DataFrame({
            "trade_price": [20.0, 35.0],
            "bid_ex": [15.0, 30.0],
            "ask_ex": [25.0, 40.0],
        })

        clf = ClassicalClassifier(layers=[("quote", "ex")], strategy="const")

        # First cycle
        clf.fit(train_data1)
        predictions1 = clf.predict(test_data)

        # Second cycle (refit)
        clf.fit(train_data2)
        predictions2 = clf.predict(test_data)

        # Both should work without error
        assert len(predictions1) == len(test_data)
        assert len(predictions2) == len(test_data)


class TestErrorHandling:
    """Test error handling in integration scenarios."""

    def test_prediction_with_wrong_features(self) -> None:
        """Test prediction fails when test data has wrong features."""
        train_data = pd.DataFrame({
            "trade_price": [15.0, 25.0],
            "bid_ex": [10.0, 20.0],
            "ask_ex": [20.0, 30.0],
        })

        # Missing required columns
        test_data = pd.DataFrame({
            "trade_price": [20.0],
            "wrong_column": [15.0],
        })

        clf = ClassicalClassifier(layers=[("quote", "ex")])
        clf.fit(train_data)

        with pytest.raises(Exception):
            clf.predict(test_data)

    def test_prediction_with_mismatched_numpy_shape(self) -> None:
        """Test prediction fails with wrong numpy array shape."""
        features = ["trade_price", "bid_ex", "ask_ex"]
        train_data = np.array([
            [15.0, 10.0, 20.0],
            [25.0, 20.0, 30.0],
        ])

        # Wrong number of columns
        test_data = np.array([
            [20.0, 15.0],
        ])

        clf = ClassicalClassifier(
            layers=[("quote", "ex")], features=features
        )
        clf.fit(train_data)

        with pytest.raises(Exception):
            clf.predict(test_data)


class TestReproducibility:
    """Test result reproducibility."""

    def test_full_pipeline_reproducibility(self) -> None:
        """Test that full pipeline produces reproducible results."""
        np.random.seed(42)
        n_samples = 100

        data = pd.DataFrame({
            "trade_price": np.random.uniform(10, 100, n_samples),
            "bid_ex": np.random.uniform(9, 99, n_samples),
            "ask_ex": np.random.uniform(11, 101, n_samples),
            "bid_best": np.random.uniform(9, 99, n_samples),
            "ask_best": np.random.uniform(11, 101, n_samples),
            "price_ex_lag": np.random.uniform(10, 100, n_samples),
            "price_ex_lead": np.random.uniform(10, 100, n_samples),
        })
        data["ask_ex"] = np.maximum(data["ask_ex"], data["bid_ex"] + 0.01)
        data["ask_best"] = np.maximum(data["ask_best"], data["bid_best"] + 0.01)

        # Run pipeline twice with same random state
        results = []
        for _ in range(2):
            clf = ClassicalClassifier(
                layers=[
                    ("quote", "ex"),
                    ("quote", "best"),
                    ("tick", "ex"),
                ],
                strategy="random",
                random_state=42,
            )
            clf.fit(data)
            predictions = clf.predict(data)
            probabilities = clf.predict_proba(data)
            results.append((predictions, probabilities))

        # Results should be identical
        assert_array_equal(results[0][0], results[1][0])
        assert_allclose(results[0][1], results[1][1])
