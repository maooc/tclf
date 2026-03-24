"""Implements classical trade classification rules with a sklearn-like interface.

Both simple rules like quote rule or tick test or hybrids are included.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, get_args

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.utils.validation import (
    _check_sample_weight,
    check_array,
    check_is_fitted,
)

from tclf.types import ArrayLike, MatrixLike

ALLOWED_FUNC_LITERALS = Literal[
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
ALLOWED_FUNC_STR: tuple[ALLOWED_FUNC_LITERALS, ...] = get_args(ALLOWED_FUNC_LITERALS)

ImputeStrategy = Literal["mean", "median", "most_frequent", "constant"]


@dataclass
class FeatureSpec:
    """Specification for a logical feature used by classification rules.

    Attributes:
        name: Logical name of the feature (e.g., 'trade_price', 'bid_ex')
        required_by: List of rule names that require this feature
        description: Human-readable description
    """
    name: str
    required_by: list[str]
    description: str = ""


LOGICAL_FEATURES: dict[str, FeatureSpec] = {
    "trade_price": FeatureSpec(
        name="trade_price",
        required_by=["tick", "rev_tick", "quote", "lr", "rev_lr", "emo", "rev_emo", "clnv", "rev_clnv", "depth"],
        description="Trade execution price",
    ),
    "trade_size": FeatureSpec(
        name="trade_size",
        required_by=["trade_size"],
        description="Trade size/volume",
    ),
    "bid_ex": FeatureSpec(
        name="bid_ex",
        required_by=["quote", "lr", "rev_lr", "emo", "rev_emo", "clnv", "rev_clnv", "depth"],
        description="Exchange bid quote",
    ),
    "ask_ex": FeatureSpec(
        name="ask_ex",
        required_by=["quote", "lr", "rev_lr", "emo", "rev_emo", "clnv", "rev_clnv", "depth"],
        description="Exchange ask quote",
    ),
    "bid_best": FeatureSpec(
        name="bid_best",
        required_by=["quote", "lr", "rev_lr", "emo", "rev_emo", "clnv", "rev_clnv", "depth"],
        description="NBBO best bid",
    ),
    "ask_best": FeatureSpec(
        name="ask_best",
        required_by=["quote", "lr", "rev_lr", "emo", "rev_emo", "clnv", "rev_clnv", "depth"],
        description="NBBO best ask",
    ),
    "bid_size_ex": FeatureSpec(
        name="bid_size_ex",
        required_by=["trade_size", "depth"],
        description="Exchange bid size",
    ),
    "ask_size_ex": FeatureSpec(
        name="ask_size_ex",
        required_by=["trade_size", "depth"],
        description="Exchange ask size",
    ),
    "bid_size_best": FeatureSpec(
        name="bid_size_best",
        required_by=["trade_size", "depth"],
        description="NBBO best bid size",
    ),
    "ask_size_best": FeatureSpec(
        name="ask_size_best",
        required_by=["trade_size", "depth"],
        description="NBBO best ask size",
    ),
    "price_ex_lag": FeatureSpec(
        name="price_ex_lag",
        required_by=["tick"],
        description="Previous trade price on exchange",
    ),
    "price_ex_lead": FeatureSpec(
        name="price_ex_lead",
        required_by=["rev_tick"],
        description="Next trade price on exchange",
    ),
    "price_best_lag": FeatureSpec(
        name="price_best_lag",
        required_by=["tick"],
        description="Previous best price",
    ),
    "price_best_lead": FeatureSpec(
        name="price_best_lead",
        required_by=["rev_tick"],
        description="Next best price",
    ),
    "price_all_lag": FeatureSpec(
        name="price_all_lag",
        required_by=["tick"],
        description="Previous trade price (all trades)",
    ),
    "price_all_lead": FeatureSpec(
        name="price_all_lead",
        required_by=["rev_tick"],
        description="Next trade price (all trades)",
    ),
}


class FeatureMapping:
    """Flexible feature mapping mechanism to decouple column names from algorithm logic.

    This class provides a mapping between user-provided column names and the
    logical feature names used internally by the classification rules.

    Examples:
        >>> mapping = FeatureMapping({
        ...     "price": "trade_price",
        ...     "bid": "bid_ex",
        ...     "ask": "ask_ex",
        ... })
        >>> mapping.get_column_index("trade_price", {"trade_price": 0, "bid": 1, "ask": 2})
        0
    """

    DEFAULT_MAPPINGS: dict[str, str] = {
        "trade_price": "trade_price",
        "trade_size": "trade_size",
        "bid_ex": "bid_ex",
        "ask_ex": "ask_ex",
        "bid_best": "bid_best",
        "ask_best": "ask_best",
        "bid_size_ex": "bid_size_ex",
        "ask_size_ex": "ask_size_ex",
        "bid_size_best": "bid_size_best",
        "ask_size_best": "ask_size_best",
        "price_ex_lag": "price_ex_lag",
        "price_ex_lead": "price_ex_lead",
        "price_best_lag": "price_best_lag",
        "price_best_lead": "price_best_lead",
        "price_all_lag": "price_all_lag",
        "price_all_lead": "price_all_lead",
    }

    def __init__(
        self,
        column_mapping: dict[str, str] | None = None,
        *,
        validate: bool = True,
    ):
        """Initialize the feature mapping.

        Args:
            column_mapping: Mapping from user column names to logical feature names.
                If None, uses default 1:1 mapping.
            validate: Whether to validate the mapping during fit.
        """
        self.column_mapping = column_mapping or {}
        self.validate = validate
        self._index_mapping_: dict[str, int] | None = None
        self._reverse_mapping_: dict[str, str] | None = None

    def fit(
        self,
        columns: list[str] | None,
        n_features: int | None = None,
    ) -> "FeatureMapping":
        """Fit the mapping to the provided column names.

        Args:
            columns: List of column names from the input data.
            n_features: Number of features (used when columns is None).

        Returns:
            self
        """
        if columns is None:
            if n_features is None:
                raise ValueError("Either columns or n_features must be provided")
            columns = [str(i) for i in range(n_features)]

        self._index_mapping_ = {col: idx for idx, col in enumerate(columns)}
        self._reverse_mapping_ = {}

        for user_col, logical_name in self.column_mapping.items():
            if user_col in self._index_mapping_:
                self._reverse_mapping_[logical_name] = user_col

        for logical_name in self.DEFAULT_MAPPINGS:
            if logical_name in self._index_mapping_ and logical_name not in self._reverse_mapping_:
                self._reverse_mapping_[logical_name] = logical_name

        return self

    def get_column_index(self, logical_name: str) -> int:
        """Get the column index for a logical feature name.

        Args:
            logical_name: The logical feature name.

        Returns:
            The column index.

        Raises:
            KeyError: If the logical feature is not found.
        """
        if self._index_mapping_ is None or self._reverse_mapping_ is None:
            raise RuntimeError("FeatureMapping must be fitted before use")

        user_col = self._reverse_mapping_.get(logical_name, logical_name)
        if user_col not in self._index_mapping_:
            raise KeyError(f"Column for logical feature '{logical_name}' not found")
        return self._index_mapping_[user_col]

    def get_required_features(self, rules: list[tuple[str, str]]) -> set[str]:
        """Get the set of required logical features for the given rules.

        Args:
            rules: List of (rule_name, subset) tuples.

        Returns:
            Set of required logical feature names.
        """
        required = set()

        RULE_FEATURE_MAPPING: dict[str, dict[str, list[str]]] = {
            "tick": {
                "ex": ["trade_price", "price_ex_lag"],
                "best": ["trade_price", "price_best_lag"],
                "all": ["trade_price", "price_all_lag"],
            },
            "rev_tick": {
                "ex": ["trade_price", "price_ex_lead"],
                "best": ["trade_price", "price_best_lead"],
                "all": ["trade_price", "price_all_lead"],
            },
            "quote": {
                "ex": ["trade_price", "bid_ex", "ask_ex"],
                "best": ["trade_price", "bid_best", "ask_best"],
            },
            "lr": {
                "ex": ["trade_price", "bid_ex", "ask_ex", "price_ex_lag"],
                "best": ["trade_price", "bid_best", "ask_best", "price_best_lag"],
            },
            "rev_lr": {
                "ex": ["trade_price", "bid_ex", "ask_ex", "price_ex_lead"],
                "best": ["trade_price", "bid_best", "ask_best", "price_best_lead"],
            },
            "emo": {
                "ex": ["trade_price", "bid_ex", "ask_ex", "price_ex_lag"],
                "best": ["trade_price", "bid_best", "ask_best", "price_best_lag"],
            },
            "rev_emo": {
                "ex": ["trade_price", "bid_ex", "ask_ex", "price_ex_lead"],
                "best": ["trade_price", "bid_best", "ask_best", "price_best_lead"],
            },
            "clnv": {
                "ex": ["trade_price", "bid_ex", "ask_ex", "price_ex_lag"],
                "best": ["trade_price", "bid_best", "ask_best", "price_best_lag"],
            },
            "rev_clnv": {
                "ex": ["trade_price", "bid_ex", "ask_ex", "price_ex_lead"],
                "best": ["trade_price", "bid_best", "ask_best", "price_best_lead"],
            },
            "trade_size": {
                "ex": ["trade_size", "bid_size_ex", "ask_size_ex"],
                "best": ["trade_size", "bid_size_best", "ask_size_best"],
            },
            "depth": {
                "ex": ["trade_price", "bid_ex", "ask_ex", "bid_size_ex", "ask_size_ex"],
                "best": ["trade_price", "bid_best", "ask_best", "bid_size_best", "ask_size_best"],
            },
            "nan": {},
        }

        for rule_name, subset in rules:
            if rule_name in RULE_FEATURE_MAPPING:
                subset_features = RULE_FEATURE_MAPPING[rule_name].get(subset, [])
                required.update(subset_features)

        return required

    def validate_features(self, rules: list[tuple[str, str]]) -> list[str]:
        """Validate that all required features are available.

        Args:
            rules: List of (rule_name, subset) tuples.

        Returns:
            List of missing feature names.

        Raises:
            RuntimeError: If the mapping has not been fitted.
        """
        if self._index_mapping_ is None:
            raise RuntimeError("FeatureMapping must be fitted before validation")

        required = self.get_required_features(rules)
        missing = []

        for logical_name in required:
            try:
                self.get_column_index(logical_name)
            except KeyError:
                missing.append(logical_name)

        return missing


class FeatureExtractor:
    """Efficient feature extraction from numpy arrays using pre-computed indices.

    This class provides O(1) access to features by their logical names,
    avoiding the overhead of DataFrame operations.
    """

    def __init__(self, X: npt.NDArray, mapping: FeatureMapping):
        """Initialize the feature extractor.

        Args:
            X: Input feature matrix (n_samples, n_features).
            mapping: Fitted feature mapping.
        """
        self._X = X
        self._mapping = mapping
        self._cache: dict[str, npt.NDArray] = {}

    def get(self, logical_name: str) -> npt.NDArray:
        """Get a feature array by its logical name.

        Args:
            logical_name: The logical feature name.

        Returns:
            1D numpy array of the feature values.
        """
        if logical_name in self._cache:
            return self._cache[logical_name]

        idx = self._mapping.get_column_index(logical_name)
        result = self._X[:, idx]
        self._cache[logical_name] = result
        return result

    def get_with_subset(self, base_name: str, subset: str) -> npt.NDArray:
        """Get a feature array with subset suffix.

        Args:
            base_name: Base feature name (e.g., 'bid', 'ask').
            subset: Subset identifier (e.g., 'ex', 'best').

        Returns:
            1D numpy array of the feature values.
        """
        feature_name = f"{base_name}_{subset}"
        return self.get(feature_name)

    def get_price_lag(self, subset: str) -> npt.NDArray:
        """Get the lagged price for a subset.

        Args:
            subset: Subset identifier.

        Returns:
            1D numpy array of lagged prices.
        """
        return self.get(f"price_{subset}_lag")

    def get_price_lead(self, subset: str) -> npt.NDArray:
        """Get the lead price for a subset.

        Args:
            subset: Subset identifier.

        Returns:
            1D numpy array of lead prices.
        """
        return self.get(f"price_{subset}_lead")


class NaNHandler(TransformerMixin, BaseEstimator):
    """Configurable NaN handling transformer.

    This transformer provides flexible strategies for handling missing values,
    integrating with sklearn's impute module.
    """

    def __init__(
        self,
        strategy: ImputeStrategy = "mean",
        fill_value: float | None = None,
        copy: bool = True,
        add_indicator: bool = False,
    ):
        """Initialize the NaN handler.

        Args:
            strategy: Imputation strategy ('mean', 'median', 'most_frequent', 'constant').
            fill_value: Fill value when strategy='constant'.
            copy: Whether to create a copy of the input array.
            add_indicator: Whether to add missing value indicators.
        """
        self.strategy = strategy
        self.fill_value = fill_value
        self.copy = copy
        self.add_indicator = add_indicator
        self._imputer: SimpleImputer | None = None

    def fit(self, X: MatrixLike, y: ArrayLike | None = None) -> "NaNHandler":
        """Fit the imputer on the training data.

        Args:
            X: Training data.
            y: Ignored, present for API compatibility.

        Returns:
            self
        """
        self._imputer = SimpleImputer(
            strategy=self.strategy,
            fill_value=self.fill_value,
            copy=self.copy,
            add_indicator=self.add_indicator,
        )
        self._imputer.fit(X)
        return self

    def transform(self, X: MatrixLike) -> npt.NDArray:
        """Transform the data by imputing missing values.

        Args:
            X: Data to transform.

        Returns:
            Transformed data with imputed values.
        """
        if self._imputer is None:
            raise RuntimeError("NaNHandler must be fitted before transform")
        return self._imputer.transform(X)

    def fit_transform(self, X: MatrixLike, y: ArrayLike | None = None) -> npt.NDArray:
        """Fit and transform in one step.

        Args:
            X: Training data.
            y: Ignored, present for API compatibility.

        Returns:
            Transformed data.
        """
        self._imputer = SimpleImputer(
            strategy=self.strategy,
            fill_value=self.fill_value,
            copy=self.copy,
            add_indicator=self.add_indicator,
        )
        return self._imputer.fit_transform(X)


class ClassicalClassifier(ClassifierMixin, BaseEstimator):
    """ClassicalClassifier implements several trade classification rules.

    This classifier provides an industrial-grade implementation of classical
    trade classification algorithms with the following features:

    - **Feature Mapping**: Flexible mapping between user column names and
      internal logical feature names, eliminating hard-coded dependencies.
    - **Pipeline Compatibility**: Full compatibility with sklearn.pipeline.Pipeline,
      including support for numpy array inputs without column names.
    - **Advanced Preprocessing**: Configurable NaN handling strategies via
      sklearn.impute integration.
    - **High Performance**: Pure NumPy operations for efficient processing
      of large datasets.

    Supported Rules:
        - Tick test
        - Reverse tick test
        - Quote rule
        - LR algorithm (Lee-Ready)
        - EMO algorithm (Ellis-Michaely-O'Hara)
        - CLNV algorithm (Chakrabarty-Li-Nguyen-Van-Ness)
        - Trade size rule
        - Depth rule

    Args:
        layers: List of (rule_name, subset) tuples defining the classification
            stack. Rules are applied in order until a classification is made.
        feature_mapping: Mapping from user column names to logical feature names.
            If None, uses default 1:1 mapping.
        features: List of feature names in column order. Required when input
            is a numpy array without column names.
        random_state: Random seed for stochastic classification.
        strategy: Strategy for unclassified trades ('random' or 'const').
        impute_strategy: Strategy for handling NaN values ('mean', 'median',
            'most_frequent', 'constant', or None to disable).
        impute_fill_value: Fill value when impute_strategy='constant'.
        scale_features: Whether to scale features (not recommended for rule-based).
    """

    _parameter_constraints: dict[str, Any] = {}

    def __init__(
        self,
        layers: list[tuple[ALLOWED_FUNC_LITERALS, str]] | None = None,
        *,
        feature_mapping: dict[str, str] | None = None,
        features: list[str] | None = None,
        random_state: float | int | None = 42,
        strategy: Literal["random", "const"] = "random",
        impute_strategy: ImputeStrategy | None = None,
        impute_fill_value: float | None = None,
        scale_features: bool = False,
    ):
        self.layers = layers
        self.feature_mapping = feature_mapping
        self.features = features
        self.random_state = random_state
        self.strategy = strategy
        self.impute_strategy = impute_strategy
        self.impute_fill_value = impute_fill_value
        self.scale_features = scale_features

    def _more_tags(self) -> dict[str, bool | dict[str, str]]:
        """Set tags for sklearn estimator checks."""
        return {
            "allow_nan": True,
            "binary_only": True,
            "requires_y": False,
            "poor_score": True,
            "_xfail_checks": {
                "check_classifiers_classes": "Disabled due to partly random classification.",
                "check_classifiers_train": "No check, as unsupervised classifier.",
                "check_classifiers_one_label": "Disabled due to partly random classification.",
                "check_methods_subset_invariance": "No check, as unsupervised classifier.",
                "check_methods_sample_order_invariance": "No check, as unsupervised classifier.",
                "check_supervised_y_no_nan": "No check, as unsupervised classifier.",
                "check_supervised_y_2d": "No check, as unsupervised classifier.",
                "check_classifiers_regression_target": "No check, as unsupervised classifier.",
            },
        }

    def _validate_layers(self) -> None:
        """Validate the layers configuration.

        Raises:
            ValueError: If any rule name is invalid.
        """
        for func_str, _ in self._layers:
            if func_str not in ALLOWED_FUNC_STR:
                raise ValueError(
                    f"Unknown function string: {func_str}, "
                    f"expected one of {ALLOWED_FUNC_STR}."
                )

    def _validate_features(self, columns: list[str] | None, n_features: int) -> list[str]:
        """Validate that all required features are available.

        Args:
            columns: List of column names.
            n_features: Number of features in the input.

        Returns:
            List of missing feature names.
        """
        self._feature_mapping.fit(columns, n_features)
        return self._feature_mapping.validate_features(self._layers)

    def _build_func_mapping(self) -> dict[str, callable]:
        """Build the mapping from rule names to implementation functions.

        Returns:
            Dictionary mapping rule names to callable implementations.
        """
        funcs = (
            self._tick,
            self._rev_tick,
            self._quote,
            self._lr,
            self._rev_lr,
            self._emo,
            self._rev_emo,
            self._clnv,
            self._rev_clnv,
            self._trade_size,
            self._depth,
            self._nan,
        )
        return dict(zip(ALLOWED_FUNC_STR, funcs))

    def fit(
        self,
        X: MatrixLike,
        y: ArrayLike | None = None,
        sample_weight: npt.NDArray | None = None,
    ) -> "ClassicalClassifier":
        """Fit the classifier.

        This method validates the input data, sets up feature mappings,
        and prepares the classifier for prediction.

        Args:
            X: Feature matrix. Can be a numpy array, pandas DataFrame, or scipy sparse matrix.
            y: Ignored, present for API consistency.
            sample_weight: Sample weights (currently not used).

        Returns:
            Fitted classifier instance.

        Raises:
            ValueError: If required features are missing or invalid configuration.
        """
        _check_sample_weight(sample_weight, X)

        self._layers = self.layers if self.layers is not None else []
        self._validate_layers()

        columns = None
        if hasattr(X, "columns"):
            columns = list(X.columns)
        elif self.features is not None:
            columns = self.features

        X_validated = check_array(
            X,
            dtype=[np.float64, np.float32],
            accept_sparse=False,
            ensure_all_finite=False,
            ensure_2d=True,
        )

        self.n_features_in_ = X_validated.shape[1]
        self.classes_ = np.array([-1, 1])

        self._feature_mapping = FeatureMapping(self.feature_mapping)
        missing = self._validate_features(columns, self.n_features_in_)

        if missing:
            raise ValueError(
                f"Missing required features: {sorted(missing)}. "
                f"Please provide these columns or configure feature_mapping. "
                f"See: https://karelze.github.io/tclf/naming_conventions/"
            )

        self.columns_ = columns if columns is not None else [str(i) for i in range(self.n_features_in_)]
        self.func_mapping_ = self._build_func_mapping()

        self._preprocessor_: Pipeline | None = self._build_preprocessor()
        if self._preprocessor_ is not None:
            self._preprocessor_.fit(X_validated)

        return self

    def _build_preprocessor(self) -> Pipeline | None:
        """Build the preprocessing pipeline.

        The pipeline ensures correct order: impute -> scale.
        This guarantees that scaling works on data without NaN values.

        Returns:
            sklearn Pipeline with preprocessing steps, or None if no preprocessing.
        """
        steps = []

        if self.impute_strategy is not None:
            steps.append((
                "imputer",
                SimpleImputer(
                    strategy=self.impute_strategy,
                    fill_value=self.impute_fill_value,
                ),
            ))

        if self.scale_features:
            steps.append(("scaler", StandardScaler()))

        if steps:
            return Pipeline(steps)
        return None

    def _prepare_input(self, X: MatrixLike) -> npt.NDArray:
        """Prepare input data for prediction.

        Args:
            X: Input feature matrix.

        Returns:
            Validated and optionally preprocessed numpy array.
        """
        X_validated = check_array(
            X,
            dtype=[np.float64, np.float32],
            accept_sparse=False,
            ensure_all_finite=False,
            ensure_2d=True,
        )

        if self._preprocessor_ is not None:
            X_validated = self._preprocessor_.transform(X_validated)

        return X_validated

    def predict(self, X: MatrixLike) -> npt.NDArray:
        """Perform classification on test vectors.

        Args:
            X: Feature matrix to classify.

        Returns:
            Array of predicted class labels (-1 or 1).
        """
        check_is_fitted(self)

        X_prepared = self._prepare_input(X)
        extractor = FeatureExtractor(X_prepared, self._feature_mapping)

        pred = self._predict_with_extractor(extractor)

        mask = np.isnan(pred)
        if mask.any():
            rs = check_random_state(self.random_state)
            if self.strategy == "random":
                pred[mask] = rs.choice(self.classes_, pred.shape)[mask]
            else:
                pred[mask] = 0

        return pred

    def _predict_with_extractor(self, extractor: FeatureExtractor) -> npt.NDArray:
        """Execute the classification rule stack.

        Args:
            extractor: Feature extractor for accessing data columns.

        Returns:
            Array of predictions (may contain NaN for unclassified samples).
        """
        n_samples = extractor._X.shape[0]
        pred = np.full(n_samples, np.nan)

        for func_str, subset in self._layers:
            func = self.func_mapping_[func_str]
            mask = np.isnan(pred)
            if mask.any():
                result = func(extractor, subset)
                pred = np.where(mask, result, pred)

        return pred

    def _tick(self, extractor: FeatureExtractor, subset: str) -> npt.NDArray:
        """Tick rule: classify based on price change from previous trade.

        Classifies as buy (1) if price increased, sell (-1) if decreased.
        """
        trade_price = extractor.get("trade_price")
        price_lag = extractor.get_price_lag(subset)

        return np.where(
            trade_price > price_lag,
            1,
            np.where(trade_price < price_lag, -1, np.nan),
        )

    def _rev_tick(self, extractor: FeatureExtractor, subset: str) -> npt.NDArray:
        """Reverse tick rule: classify based on price change to next trade.

        Classifies as sell (-1) if price will increase, buy (1) if will decrease.
        """
        trade_price = extractor.get("trade_price")
        price_lead = extractor.get_price_lead(subset)

        return np.where(
            price_lead > trade_price,
            -1,
            np.where(price_lead < trade_price, 1, np.nan),
        )

    def _mid(self, extractor: FeatureExtractor, subset: str) -> npt.NDArray:
        """Calculate the midpoint of bid-ask spread.

        Returns NaN if bid > ask (invalid spread).
        """
        bid = extractor.get_with_subset("bid", subset)
        ask = extractor.get_with_subset("ask", subset)

        return np.where(
            ask >= bid,
            0.5 * (ask + bid),
            np.nan,
        )

    def _quote(self, extractor: FeatureExtractor, subset: str) -> npt.NDArray:
        """Quote rule: classify based on position relative to midspread.

        Classifies as buy (1) if above mid, sell (-1) if below mid.
        """
        trade_price = extractor.get("trade_price")
        mid = self._mid(extractor, subset)

        return np.where(
            trade_price > mid,
            1,
            np.where(trade_price < mid, -1, np.nan),
        )

    def _lr(self, extractor: FeatureExtractor, subset: str) -> npt.NDArray:
        """Lee-Ready algorithm: quote rule with tick test for midspread trades."""
        quote_result = self._quote(extractor, subset)
        return np.where(~np.isnan(quote_result), quote_result, self._tick(extractor, subset))

    def _rev_lr(self, extractor: FeatureExtractor, subset: str) -> npt.NDArray:
        """Lee-Ready with reverse tick test for midspread trades."""
        quote_result = self._quote(extractor, subset)
        return np.where(~np.isnan(quote_result), quote_result, self._rev_tick(extractor, subset))

    def _is_at_ask_xor_bid(self, extractor: FeatureExtractor, subset: str) -> npt.NDArray:
        """Check if trade price is exactly at ask XOR bid."""
        trade_price = extractor.get("trade_price")
        bid = extractor.get_with_subset("bid", subset)
        ask = extractor.get_with_subset("ask", subset)

        at_ask = np.isclose(trade_price, ask, atol=1e-4)
        at_bid = np.isclose(trade_price, bid, atol=1e-4)
        return at_ask ^ at_bid

    def _is_at_upper_xor_lower_quantile(
        self, extractor: FeatureExtractor, subset: str, quantiles: float = 0.3
    ) -> npt.NDArray:
        """Check if trade is in upper or lower quantile of the spread."""
        trade_price = extractor.get("trade_price")
        bid = extractor.get_with_subset("bid", subset)
        ask = extractor.get_with_subset("ask", subset)

        in_upper = (
            (1.0 - quantiles) * ask + quantiles * bid <= trade_price
        ) & (trade_price <= ask)
        in_lower = (bid <= trade_price) & (
            trade_price <= quantiles * ask + (1.0 - quantiles) * bid
        )
        return in_upper ^ in_lower

    def _emo(self, extractor: FeatureExtractor, subset: str) -> npt.NDArray:
        """EMO algorithm: quote rule at bid/ask, tick test elsewhere."""
        at_quote = self._is_at_ask_xor_bid(extractor, subset)
        return np.where(
            at_quote,
            self._quote(extractor, subset),
            self._tick(extractor, subset),
        )

    def _rev_emo(self, extractor: FeatureExtractor, subset: str) -> npt.NDArray:
        """EMO with reverse tick test for off-quote trades."""
        at_quote = self._is_at_ask_xor_bid(extractor, subset)
        return np.where(
            at_quote,
            self._quote(extractor, subset),
            self._rev_tick(extractor, subset),
        )

    def _clnv(self, extractor: FeatureExtractor, subset: str) -> npt.NDArray:
        """CLNV algorithm: quote rule in spread deciles, tick test elsewhere."""
        in_quantile = self._is_at_upper_xor_lower_quantile(extractor, subset)
        return np.where(
            in_quantile,
            self._quote(extractor, subset),
            self._tick(extractor, subset),
        )

    def _rev_clnv(self, extractor: FeatureExtractor, subset: str) -> npt.NDArray:
        """CLNV with reverse tick test for middle deciles."""
        in_quantile = self._is_at_upper_xor_lower_quantile(extractor, subset)
        return np.where(
            in_quantile,
            self._quote(extractor, subset),
            self._rev_tick(extractor, subset),
        )

    def _trade_size(self, extractor: FeatureExtractor, subset: str) -> npt.NDArray:
        """Trade size rule: classify if trade size matches bid or ask size."""
        trade_size = extractor.get("trade_size")
        bid_size = extractor.get_with_subset("bid_size", subset)
        ask_size = extractor.get_with_subset("ask_size", subset)

        bid_eq_ask = np.isclose(ask_size, bid_size, atol=1e-4)
        ts_eq_bid = np.isclose(trade_size, bid_size, atol=1e-4) & ~bid_eq_ask
        ts_eq_ask = np.isclose(trade_size, ask_size, atol=1e-4) & ~bid_eq_ask

        return np.where(ts_eq_bid, 1, np.where(ts_eq_ask, -1, np.nan))

    def _depth(self, extractor: FeatureExtractor, subset: str) -> npt.NDArray:
        """Depth rule: classify midspread trades by relative quote sizes."""
        trade_price = extractor.get("trade_price")
        mid = self._mid(extractor, subset)
        bid_size = extractor.get_with_subset("bid_size", subset)
        ask_size = extractor.get_with_subset("ask_size", subset)

        at_mid = np.isclose(mid, trade_price, atol=1e-4)

        return np.where(
            at_mid & (ask_size > bid_size),
            1,
            np.where(at_mid & (ask_size < bid_size), -1, np.nan),
        )

    def _nan(self, extractor: FeatureExtractor, subset: str) -> npt.NDArray:
        """Placeholder rule that never classifies (returns all NaN)."""
        return np.full(extractor._X.shape[0], np.nan)

    def predict_proba(self, X: MatrixLike) -> npt.NDArray:
        """Predict class probabilities for X.

        Probabilities are either 0 or 1 depending on the class.
        For strategy 'const', probabilities are (0.5, 0.5) for unclassified.

        Args:
            X: Feature matrix.

        Returns:
            Array of shape (n_samples, 2) with class probabilities.
        """
        prob = np.full((len(X), 2), 0.5)

        preds = self.predict(X)
        mask = np.flatnonzero(preds)

        if mask.size > 0:
            indices = np.nonzero(preds[mask, None] == self.classes_[None, :])[1]
            n_classes = np.max(self.classes_) + 1
            prob[mask] = np.identity(n_classes)[indices]

        return prob

    def get_feature_mapping(self) -> dict[str, int]:
        """Get the fitted feature mapping.

        Returns:
            Dictionary mapping logical feature names to column indices.

        Raises:
            NotFittedError: If the classifier has not been fitted.
        """
        check_is_fitted(self)

        mapping = {}
        for logical_name in LOGICAL_FEATURES:
            try:
                mapping[logical_name] = self._feature_mapping.get_column_index(logical_name)
            except KeyError:
                pass

        return mapping

    def get_required_features_for_layers(self) -> list[str]:
        """Get the list of features required by the configured layers.

        Returns:
            Sorted list of required feature names.

        Raises:
            NotFittedError: If the classifier has not been fitted.
        """
        check_is_fitted(self)
        return sorted(self._feature_mapping.get_required_features(self._layers))
