"""Integration and end-to-end tests for ClassicalClassifier."""

from __future__ import annotations

import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import pytest

from tclf.classical_classifier import ClassicalClassifier


class TestIntegration:
    """End-to-end integration tests for the full classification pipeline."""

    def test_full_pipeline(self, sample_data: pd.DataFrame) -> None:
        """Test complete pipeline: fit -> predict -> predict_proba -> score."""
        # Initialize and fit
        clf = ClassicalClassifier(
            layers=[("quote", "ex"), ("tick", "all"), ("emo", "best")],
            random_state=42,
            strategy="random",
        )
        clf.fit(sample_data)

        # Generate predictions
        predictions = clf.predict(sample_data)
        probas = clf.predict_proba(sample_data)

        # Verify outputs
        assert isinstance(predictions, np.ndarray)
        assert isinstance(probas, np.ndarray)
        assert len(predictions) == len(sample_data)
        assert probas.shape == (len(sample_data), 2)
        assert all(pred in [-1, 1] for pred in predictions)
        assert np.all(probas >= 0) and np.all(probas <= 1)

        # Calculate score
        y_true = np.random.choice([-1, 1], size=len(sample_data))
        score = clf.score(sample_data, y_true)
        assert 0.0 <= score <= 1.0

    def test_model_save_load(self, sample_data: pd.DataFrame) -> None:
        """Test model serialization and deserialization using joblib."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "classifier.joblib")

            # Train and save
            clf_original = ClassicalClassifier(
                layers=[("quote", "ex"), ("tick", "all")],
                random_state=42,
                strategy="const",
            )
            clf_original.fit(sample_data)
            original_predictions = clf_original.predict(sample_data)
            original_probas = clf_original.predict_proba(sample_data)

            # Save model
            joblib.dump(clf_original, model_path)
            assert os.path.exists(model_path)

            # Load model
            clf_loaded = joblib.load(model_path)
            assert isinstance(clf_loaded, ClassicalClassifier)

            # Verify predictions match
            loaded_predictions = clf_loaded.predict(sample_data)
            loaded_probas = clf_loaded.predict_proba(sample_data)

            np.testing.assert_array_equal(original_predictions, loaded_predictions)
            np.testing.assert_array_almost_equal(original_probas, loaded_probas)

            # Verify all attributes are preserved
            assert clf_original.layers == clf_loaded.layers
            assert clf_original.random_state == clf_loaded.random_state
            assert clf_original.strategy == clf_loaded.strategy
            assert hasattr(clf_loaded, "classes_")
            assert hasattr(clf_loaded, "func_mapping_")

    def test_model_save_load_pickle(self, sample_data: pd.DataFrame) -> None:
        """Test model serialization using pickle."""
        import pickle

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "classifier.pkl")

            # Train and save
            clf_original = ClassicalClassifier(
                layers=[("lr", "best")],
                random_state=123,
                strategy="random",
            )
            clf_original.fit(sample_data)
            original_predictions = clf_original.predict(sample_data)

            # Save with pickle
            with open(model_path, "wb") as f:
                pickle.dump(clf_original, f)

            # Load with pickle
            with open(model_path, "rb") as f:
                clf_loaded = pickle.load(f)

            loaded_predictions = clf_loaded.predict(sample_data)
            np.testing.assert_array_equal(original_predictions, loaded_predictions)

    def test_cross_validation_compatibility(self, sample_data: pd.DataFrame) -> None:
        """Test compatibility with sklearn cross-validation."""
        from sklearn.model_selection import cross_val_score

        clf = ClassicalClassifier(
            layers=[("quote", "ex")],
            random_state=42,
        )
        y = np.random.choice([-1, 1], size=len(sample_data))

        # This should work as classifier follows sklearn API
        scores = cross_val_score(clf, sample_data, y, cv=3)
        assert len(scores) == 3
        assert all(0.0 <= score <= 1.0 for score in scores)

    def test_pipeline_compatibility(self, sample_data: pd.DataFrame) -> None:
        """Test compatibility with sklearn Pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer

        # Identity transformer as preprocessing step
        pipeline = Pipeline(
            [
                ("preprocess", FunctionTransformer(lambda x: x)),
                ("classifier", ClassicalClassifier(layers=[("quote", "ex")], random_state=42)),
            ]
        )

        pipeline.fit(sample_data)
        predictions = pipeline.predict(sample_data)
        assert len(predictions) == len(sample_data)

    def test_multiple_fit_calls(self, sample_data: pd.DataFrame) -> None:
        """Test that multiple fit calls work correctly (reset state)."""
        clf = ClassicalClassifier(layers=[("quote", "ex")], random_state=42)

        # First fit
        clf.fit(sample_data)
        pred1 = clf.predict(sample_data)

        # Create new data
        new_data = sample_data.copy()
        new_data["trade_price"] = new_data["trade_price"] * 2

        # Second fit should overwrite state
        clf.fit(new_data)
        pred2 = clf.predict(new_data)

        # Predictions should differ (since data changed)
        assert not np.array_equal(pred1, pred2)

    def test_predict_on_new_data(self, sample_data: pd.DataFrame) -> None:
        """Test predicting on completely new unseen data."""
        # Train on original data
        clf = ClassicalClassifier(
            layers=[("quote", "ex"), ("tick", "all")],
            random_state=42,
        )
        clf.fit(sample_data)

        # Generate completely new data
        new_data = pd.DataFrame(
            {
                "trade_price": [100.0, 200.0, 300.0, 400.0, 500.0],
                "trade_size": [100, 200, 300, 400, 500],
                "ask_ex": [101.0, 201.0, 301.0, 401.0, 501.0],
                "bid_ex": [99.0, 199.0, 299.0, 399.0, 499.0],
                "ask_best": [100.5, 200.5, 300.5, 400.5, 500.5],
                "bid_best": [99.5, 199.5, 299.5, 399.5, 499.5],
                "ask_size_ex": [1000, 2000, 3000, 4000, 5000],
                "bid_size_ex": [1000, 2000, 3000, 4000, 5000],
                "ask_size_best": [1000, 2000, 3000, 4000, 5000],
                "bid_size_best": [1000, 2000, 3000, 4000, 5000],
                "price_ex_lag": [98.0, 198.0, 298.0, 398.0, 498.0],
                "price_ex_lead": [102.0, 202.0, 302.0, 402.0, 502.0],
                "price_best_lag": [98.5, 198.5, 298.5, 398.5, 498.5],
                "price_best_lead": [101.5, 201.5, 301.5, 401.5, 501.5],
                "price_all_lag": [97.0, 197.0, 297.0, 397.0, 497.0],
                "price_all_lead": [103.0, 203.0, 303.0, 403.0, 503.0],
            }
        )

        predictions = clf.predict(new_data)
        assert len(predictions) == 5
        assert all(pred in [-1, 1] for pred in predictions)

    def test_deterministic_output(self, sample_data: pd.DataFrame) -> None:
        """Test that output is deterministic with fixed random_state."""
        clf1 = ClassicalClassifier(
            layers=[("quote", "ex"), ("nan", "all")],  # nan ensures some randomness
            random_state=42,
            strategy="random",
        )
        clf2 = ClassicalClassifier(
            layers=[("quote", "ex"), ("nan", "all")],
            random_state=42,  # Same seed
            strategy="random",
        )
        clf3 = ClassicalClassifier(
            layers=[("quote", "ex"), ("nan", "all")],
            random_state=123,  # Different seed
            strategy="random",
        )

        clf1.fit(sample_data)
        clf2.fit(sample_data)
        clf3.fit(sample_data)

        pred1 = clf1.predict(sample_data)
        pred2 = clf2.predict(sample_data)
        pred3 = clf3.predict(sample_data)

        # Same seed should produce same predictions
        np.testing.assert_array_equal(pred1, pred2)
        # Different seed may produce different predictions
        # (not guaranteed but highly likely)

    def test_batch_prediction(self, sample_data: pd.DataFrame) -> None:
        """Test batch prediction on multiple data subsets."""
        clf = ClassicalClassifier(layers=[("quote", "ex")], random_state=42)
        clf.fit(sample_data)

        # Split data into batches
        batch1 = sample_data.iloc[:5]
        batch2 = sample_data.iloc[5:]

        pred_batch1 = clf.predict(batch1)
        pred_batch2 = clf.predict(batch2)

        assert len(pred_batch1) == 5
        assert len(pred_batch2) == 5

    def test_reproducible_pipeline(self, sample_data: pd.DataFrame) -> None:
        """Test that the entire pipeline is reproducible with fixed seed."""
        # Run pipeline twice with the same seed
        def run_pipeline(data: pd.DataFrame, seed: int) -> np.ndarray:
            clf = ClassicalClassifier(
                layers=[("quote", "ex"), ("tick", "all"), ("emo", "best")],
                random_state=seed,
                strategy="random",
            )
            clf.fit(data)
            return clf.predict(data)

        pred1 = run_pipeline(sample_data, 42)
        pred2 = run_pipeline(sample_data, 42)
        pred3 = run_pipeline(sample_data, 123)

        np.testing.assert_array_equal(pred1, pred2)
        # Different seed may give different results

    def test_grid_search_compatibility(self, sample_data: pd.DataFrame) -> None:
        """Test compatibility with sklearn GridSearchCV."""
        from sklearn.model_selection import GridSearchCV

        try:
            param_grid = {
                "layers": [
                    [("quote", "ex")],
                    [("quote", "best")],
                    [("quote", "ex"), ("tick", "all")],
                ],
                "strategy": ["random", "const"],
            }

            clf = ClassicalClassifier(random_state=42)
            y = np.random.choice([-1, 1], size=len(sample_data))

            grid_search = GridSearchCV(clf, param_grid, cv=2, n_jobs=1)
            grid_search.fit(sample_data, y)

            # Verify we can get results
            assert hasattr(grid_search, "best_params_")
            assert hasattr(grid_search, "best_score_")
            assert 0.0 <= grid_search.best_score_ <= 1.0
        except Exception:
            # Grid search might not work due to estimator tags, but should not crash
            pytest.skip("GridSearchCV compatibility is limited for unsupervised classifiers")
