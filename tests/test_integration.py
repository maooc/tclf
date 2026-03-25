"""Integration tests for the complete pipeline."""

from __future__ import annotations

import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from tclf.classical_classifier import ClassicalClassifier


class TestEndToEndPipeline:
    """End-to-end tests for the complete classification pipeline."""

    def test_full_pipeline_with_dataframe(
        self, synthetic_train_data: pd.DataFrame, synthetic_test_data: pd.DataFrame
    ) -> None:
        """Test complete pipeline with DataFrame input."""
        columns = ["trade_price", "bid_ex", "ask_ex"]

        clf = ClassicalClassifier(
            layers=[("quote", "ex")],
            random_state=42,
        )
        clf.fit(synthetic_train_data[columns])
        predictions = clf.predict(synthetic_test_data[columns])
        probabilities = clf.predict_proba(synthetic_test_data[columns])

        assert len(predictions) == len(synthetic_test_data)
        assert probabilities.shape == (len(synthetic_test_data), 2)
        assert all(p in [-1, 0, 1] for p in predictions)
        assert all((prob >= 0).all() for prob in probabilities)
        assert all((prob <= 1).all() for prob in probabilities)

    def test_full_pipeline_with_numpy(
        self,
        synthetic_train_data: pd.DataFrame,
        synthetic_test_data: pd.DataFrame,
        feature_names: list[str],
    ) -> None:
        """Test complete pipeline with NumPy array input."""
        train_array = synthetic_train_data[feature_names].to_numpy()
        test_array = synthetic_test_data[feature_names].to_numpy()

        clf = ClassicalClassifier(
            layers=[("quote", "ex"), ("tick", "ex")],
            features=feature_names,
            random_state=42,
        )
        clf.fit(train_array)
        predictions = clf.predict(test_array)

        assert len(predictions) == len(test_array)

    def test_train_save_load_predict(
        self,
        synthetic_train_data: pd.DataFrame,
        synthetic_test_data: pd.DataFrame,
        temp_model_path: str,
    ) -> None:
        """Test train -> save -> load -> predict pipeline."""
        columns = ["trade_price", "bid_ex", "ask_ex"]

        clf = ClassicalClassifier(
            layers=[("quote", "ex")],
            random_state=42,
        )
        clf.fit(synthetic_train_data[columns])

        with open(temp_model_path, "wb") as f:
            pickle.dump(clf, f)

        with open(temp_model_path, "rb") as f:
            loaded_clf = pickle.load(f)

        predictions = loaded_clf.predict(synthetic_test_data[columns])

        assert len(predictions) == len(synthetic_test_data)

    def test_joblib_save_load(
        self,
        synthetic_train_data: pd.DataFrame,
        synthetic_test_data: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Test save/load with joblib (recommended for sklearn models)."""
        model_path = str(tmp_path / "model.joblib")
        columns = ["trade_price", "bid_ex", "ask_ex"]

        clf = ClassicalClassifier(
            layers=[("quote", "ex")],
            random_state=42,
        )
        clf.fit(synthetic_train_data[columns])

        joblib.dump(clf, model_path)
        loaded_clf = joblib.load(model_path)

        original_pred = clf.predict(synthetic_test_data[columns])
        loaded_pred = loaded_clf.predict(synthetic_test_data[columns])

        np.testing.assert_array_equal(original_pred, loaded_pred)

    def test_multi_layer_pipeline(
        self, multi_layer_classifier: ClassicalClassifier, synthetic_test_data: pd.DataFrame
    ) -> None:
        """Test pipeline with multiple classification layers."""
        columns = ["trade_price", "bid_ex", "ask_ex", "bid_best", "ask_best", "price_all_lag"]

        test_data = synthetic_test_data.copy()
        test_data["price_all_lag"] = test_data["price_ex_lag"]
        predictions = multi_layer_classifier.predict(test_data[columns])

        assert len(predictions) == len(test_data)

    def test_score_evaluation(
        self,
        fitted_classifier: ClassicalClassifier,
        synthetic_test_data: pd.DataFrame,
        synthetic_labels: np.ndarray,
    ) -> None:
        """Test model evaluation with score method."""
        columns = ["trade_price", "bid_ex", "ask_ex"]
        score = fitted_classifier.score(synthetic_test_data[columns], synthetic_labels)

        assert 0.0 <= score <= 1.0

    def test_accuracy_score_metric(
        self,
        fitted_classifier: ClassicalClassifier,
        synthetic_test_data: pd.DataFrame,
        synthetic_labels: np.ndarray,
    ) -> None:
        """Test using sklearn metrics with the classifier."""
        columns = ["trade_price", "bid_ex", "ask_ex"]
        predictions = fitted_classifier.predict(synthetic_test_data[columns])
        accuracy = accuracy_score(synthetic_labels, predictions)

        assert 0.0 <= accuracy <= 1.0


class TestDifferentStrategies:
    """Tests for different classification strategies."""

    def test_random_strategy(
        self, synthetic_train_data: pd.DataFrame, synthetic_test_data: pd.DataFrame
    ) -> None:
        """Test random strategy for unclassified trades."""
        columns = ["trade_price", "bid_ex", "ask_ex"]

        clf = ClassicalClassifier(
            layers=[("nan", "ex")],
            strategy="random",
            random_state=42,
        )
        clf.fit(synthetic_train_data[columns])
        predictions = clf.predict(synthetic_test_data[columns])

        assert all(p in [-1, 1] for p in predictions)

    def test_const_strategy(
        self, synthetic_train_data: pd.DataFrame, synthetic_test_data: pd.DataFrame
    ) -> None:
        """Test constant strategy for unclassified trades."""
        columns = ["trade_price", "bid_ex", "ask_ex"]

        clf = ClassicalClassifier(
            layers=[("nan", "ex")],
            strategy="const",
        )
        clf.fit(synthetic_train_data[columns])
        predictions = clf.predict(synthetic_test_data[columns])

        assert all(p == 0 for p in predictions)

    def test_const_strategy_proba(
        self, synthetic_train_data: pd.DataFrame, synthetic_test_data: pd.DataFrame
    ) -> None:
        """Test constant strategy probabilities."""
        columns = ["trade_price", "bid_ex", "ask_ex"]

        clf = ClassicalClassifier(
            layers=[("nan", "ex")],
            strategy="const",
        )
        clf.fit(synthetic_train_data[columns])
        probabilities = clf.predict_proba(synthetic_test_data[columns])

        assert np.allclose(probabilities, 0.5)


class TestAllClassificationRules:
    """Tests for all classification rules."""

    @pytest.mark.parametrize("rule", ["tick"])
    @pytest.mark.parametrize("subset", ["ex", "all"])
    def test_tick_rules(
        self,
        rule: str,
        subset: str,
        synthetic_train_data: pd.DataFrame,
    ) -> None:
        """Test tick rules."""
        columns = ["trade_price", f"price_{subset}_lag"]

        clf = ClassicalClassifier(layers=[(rule, subset)], random_state=42)
        clf.fit(synthetic_train_data[columns])
        predictions = clf.predict(synthetic_train_data[columns])

        assert len(predictions) == len(synthetic_train_data)

    @pytest.mark.parametrize("rule", ["rev_tick"])
    @pytest.mark.parametrize("subset", ["ex", "all"])
    def test_rev_tick_rules(
        self,
        rule: str,
        subset: str,
        synthetic_train_data: pd.DataFrame,
    ) -> None:
        """Test reverse tick rules."""
        columns = ["trade_price", f"price_{subset}_lead"]

        clf = ClassicalClassifier(layers=[(rule, subset)], random_state=42)
        clf.fit(synthetic_train_data[columns])
        predictions = clf.predict(synthetic_train_data[columns])

        assert len(predictions) == len(synthetic_train_data)

    @pytest.mark.parametrize("rule", ["quote"])
    @pytest.mark.parametrize("subset", ["ex", "best"])
    def test_quote_rule(
        self,
        rule: str,
        subset: str,
        synthetic_train_data: pd.DataFrame,
    ) -> None:
        """Test quote rule."""
        columns = ["trade_price", f"bid_{subset}", f"ask_{subset}"]

        clf = ClassicalClassifier(layers=[(rule, subset)], random_state=42)
        clf.fit(synthetic_train_data[columns])
        predictions = clf.predict(synthetic_train_data[columns])

        assert len(predictions) == len(synthetic_train_data)

    @pytest.mark.parametrize("rule", ["lr", "emo", "clnv"])
    @pytest.mark.parametrize("subset", ["ex", "best"])
    def test_quote_tick_rules(
        self,
        rule: str,
        subset: str,
        synthetic_train_data: pd.DataFrame,
    ) -> None:
        """Test rules that combine quote and tick."""
        columns = ["trade_price", f"bid_{subset}", f"ask_{subset}", f"price_{subset}_lag"]

        clf = ClassicalClassifier(layers=[(rule, subset)], random_state=42)
        clf.fit(synthetic_train_data[columns])
        predictions = clf.predict(synthetic_train_data[columns])

        assert len(predictions) == len(synthetic_train_data)

    @pytest.mark.parametrize("rule", ["rev_lr", "rev_emo", "rev_clnv"])
    @pytest.mark.parametrize("subset", ["ex", "best"])
    def test_quote_rev_tick_rules(
        self,
        rule: str,
        subset: str,
        synthetic_train_data: pd.DataFrame,
    ) -> None:
        """Test rules that combine quote and reverse tick."""
        columns = ["trade_price", f"bid_{subset}", f"ask_{subset}", f"price_{subset}_lead"]

        clf = ClassicalClassifier(layers=[(rule, subset)], random_state=42)
        clf.fit(synthetic_train_data[columns])
        predictions = clf.predict(synthetic_train_data[columns])

        assert len(predictions) == len(synthetic_train_data)

    def test_trade_size_rule(self, synthetic_train_data: pd.DataFrame) -> None:
        """Test trade size rule."""
        columns = ["trade_size", "bid_size_ex", "ask_size_ex"]

        clf = ClassicalClassifier(layers=[("trade_size", "ex")], random_state=42)
        clf.fit(synthetic_train_data[columns])
        predictions = clf.predict(synthetic_train_data[columns])

        assert len(predictions) == len(synthetic_train_data)

    def test_depth_rule(self, synthetic_train_data: pd.DataFrame) -> None:
        """Test depth rule."""
        columns = ["trade_price", "bid_ex", "ask_ex", "bid_size_ex", "ask_size_ex"]

        clf = ClassicalClassifier(layers=[("depth", "ex")], random_state=42)
        clf.fit(synthetic_train_data[columns])
        predictions = clf.predict(synthetic_train_data[columns])

        assert len(predictions) == len(synthetic_train_data)

    def test_nan_rule(self, synthetic_train_data: pd.DataFrame) -> None:
        """Test nan rule (no classification, fallback only)."""
        columns = ["trade_price"]

        clf = ClassicalClassifier(layers=[("nan", "ex")], random_state=42)
        clf.fit(synthetic_train_data[columns])
        predictions = clf.predict(synthetic_train_data[columns])

        assert len(predictions) == len(synthetic_train_data)


class TestLayerStacking:
    """Tests for layer stacking behavior."""

    def test_layer_override(self, synthetic_train_data: pd.DataFrame) -> None:
        """Test that later layers don't override earlier classifications."""
        columns = ["trade_price", "bid_ex", "ask_ex", "price_ex_lag"]

        x_test = pd.DataFrame(
            {
                "trade_price": [100.0, 101.0],
                "bid_ex": [99.5, 100.5],
                "ask_ex": [100.5, 101.5],
                "price_ex_lag": [99.0, 102.0],
            }
        )

        clf = ClassicalClassifier(
            layers=[("quote", "ex"), ("tick", "ex")],
            random_state=42,
        )
        clf.fit(synthetic_train_data[columns])
        predictions = clf.predict(x_test)

        assert len(predictions) == 2

    def test_empty_layer_fallback(self, synthetic_train_data: pd.DataFrame) -> None:
        """Test fallback when no layers classify."""
        columns = ["trade_price"]

        clf = ClassicalClassifier(
            layers=[],
            strategy="random",
            random_state=42,
        )
        clf.fit(synthetic_train_data[columns])
        predictions = clf.predict(synthetic_train_data[columns])

        assert all(p in [-1, 1] for p in predictions)


class TestSklearnCompatibility:
    """Tests for sklearn API compatibility."""

    def test_get_params(self, fitted_classifier: ClassicalClassifier) -> None:
        """Test get_params method."""
        params = fitted_classifier.get_params()

        assert "layers" in params
        assert "random_state" in params
        assert "strategy" in params

    def test_set_params(self, fitted_classifier: ClassicalClassifier) -> None:
        """Test set_params method."""
        new_random_state = 123
        fitted_classifier.set_params(random_state=new_random_state)

        assert fitted_classifier.random_state == new_random_state

    def test_fit_predict_flow(
        self, synthetic_train_data: pd.DataFrame, synthetic_test_data: pd.DataFrame
    ) -> None:
        """Test standard sklearn fit-predict workflow."""
        columns = ["trade_price", "bid_ex", "ask_ex"]

        clf = ClassicalClassifier(layers=[("quote", "ex")], random_state=42)
        clf.fit(synthetic_train_data[columns])
        predictions = clf.predict(synthetic_test_data[columns])

        assert len(predictions) == len(synthetic_test_data)

    def test_fit_transform_flow(
        self, synthetic_train_data: pd.DataFrame
    ) -> None:
        """Test fit-transform workflow if applicable."""
        columns = ["trade_price", "bid_ex", "ask_ex"]

        clf = ClassicalClassifier(layers=[("quote", "ex")], random_state=42)
        clf.fit(synthetic_train_data[columns])

        assert hasattr(clf, "classes_")


class TestPerformance:
    """Performance-related tests."""

    def test_large_dataset(self) -> None:
        """Test with larger dataset for performance."""
        n_samples = 1000
        np.random.seed(42)

        df = pd.DataFrame(
            {
                "trade_price": np.random.uniform(99, 101, n_samples),
                "bid_ex": np.random.uniform(98.5, 99.5, n_samples),
                "ask_ex": np.random.uniform(100.5, 101.5, n_samples),
            }
        )

        clf = ClassicalClassifier(layers=[("quote", "ex")], random_state=42)
        clf.fit(df)
        predictions = clf.predict(df)

        assert len(predictions) == n_samples

    def test_prediction_speed(self, synthetic_train_data: pd.DataFrame) -> None:
        """Test prediction is reasonably fast."""
        import time

        columns = ["trade_price", "bid_ex", "ask_ex"]

        clf = ClassicalClassifier(layers=[("quote", "ex")], random_state=42)
        clf.fit(synthetic_train_data[columns])

        start = time.time()
        for _ in range(100):
            clf.predict(synthetic_train_data[columns])
        elapsed = time.time() - start

        assert elapsed < 10.0
