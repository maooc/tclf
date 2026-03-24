"""Implements classical trade classification rules with a sklearn-like interface.

Both simple rules like quote rule or tick test or hybrids are included.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Literal, get_args

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
    validate_data,
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


class FeatureMapper:
    """Maps logical feature names to actual column names/indices in the data.
    
    This class provides a flexible mechanism to decouple algorithm logic from
    specific column naming conventions in the input data.
    
    Args:
        mapping: Dictionary mapping logical feature names to actual column names or indices.
                If None, uses default naming convention.
    """
    
    # Default feature names used by the classification rules
    DEFAULT_FEATURES = {
        'trade_price': 'trade_price',
        'trade_size': 'trade_size',
        'bid_ex': 'bid_ex',
        'ask_ex': 'ask_ex',
        'bid_best': 'bid_best',
        'ask_best': 'ask_best',
        'bid_size_ex': 'bid_size_ex',
        'ask_size_ex': 'ask_size_ex',
        'bid_size_best': 'bid_size_best',
        'ask_size_best': 'ask_size_best',
        'price_ex_lag': 'price_ex_lag',
        'price_ex_lead': 'price_ex_lead',
        'price_best_lag': 'price_best_lag',
        'price_best_lead': 'price_best_lead',
        'price_all_lag': 'price_all_lag',
        'price_all_lead': 'price_all_lead',
    }
    
    def __init__(self, mapping: dict[str, str | int] | None = None):
        """Initialize feature mapper.
        
        Args:
            mapping: Optional custom mapping from logical to actual feature names/indices.
        """
        self.mapping = mapping or {}
        self._resolved_mapping: dict[str, int] = {}
        
    def fit(self, X: MatrixLike, feature_names: list[str] | None = None) -> 'FeatureMapper':
        """Fit the mapper to the data schema.
        
        Args:
            X: Input data (DataFrame or ndarray)
            feature_names: Optional list of feature names for ndarray input
            
        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            self._column_names = X.columns.tolist()
        elif feature_names is not None:
            self._column_names = feature_names
        else:
            self._column_names = None
            
        # Resolve all logical features to indices
        for logical_name, default_name in self.DEFAULT_FEATURES.items():
            actual_name = self.mapping.get(logical_name, default_name)
            
            if self._column_names is not None:
                # Map column name to index
                if isinstance(actual_name, int):
                    self._resolved_mapping[logical_name] = actual_name
                elif actual_name in self._column_names:
                    self._resolved_mapping[logical_name] = self._column_names.index(actual_name)
                else:
                    # Column not found - will be handled during validation
                    self._resolved_mapping[logical_name] = -1
            else:
                # No column names, assume indices
                self._resolved_mapping[logical_name] = actual_name if isinstance(actual_name, int) else -1
                
        return self
    
    def get_index(self, logical_name: str) -> int:
        """Get the column index for a logical feature name.
        
        Args:
            logical_name: Logical feature name (e.g., 'trade_price', 'bid_ex')
            
        Returns:
            Column index in the data array
            
        Raises:
            KeyError: If logical_name is not recognized
        """
        if logical_name not in self._resolved_mapping:
            raise KeyError(f"Unknown logical feature: {logical_name}")
        return self._resolved_mapping[logical_name]
    
    def get_feature_names(self) -> list[str]:
        """Get list of all logical feature names.
        
        Returns:
            List of logical feature names
        """
        return list(self.DEFAULT_FEATURES.keys())
    
    def validate_required_features(self, required: list[str]) -> list[str]:
        """Validate that required features are available.
        
        Args:
            required: List of required logical feature names
            
        Returns:
            List of missing feature names (empty if all present)
        """
        missing = []
        for feat in required:
            if feat not in self._resolved_mapping or self._resolved_mapping[feat] == -1:
                missing.append(feat)
        return missing


class ClassicalPreprocessor(TransformerMixin, BaseEstimator):
    """Preprocessor for classical trade classification features.
    
    Handles missing value imputation, scaling, and outlier detection
    in a sklearn-compatible transformer.
    
    Args:
        impute_strategy: Strategy for imputing missing values.
                        Options: 'mean', 'median', 'most_frequent', 'constant', 'none'
        scale: Whether to apply standard scaling
        outlier_method: Method for outlier detection ('none', 'iqr', 'zscore')
        outlier_threshold: Threshold for outlier detection
    """
    
    def __init__(
        self,
        impute_strategy: Literal['mean', 'median', 'most_frequent', 'constant', 'none'] = 'none',
        scale: bool = False,
        outlier_method: Literal['none', 'iqr', 'zscore'] = 'none',
        outlier_threshold: float = 3.0,
    ):
        self.impute_strategy = impute_strategy
        self.scale = scale
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        
    def fit(self, X: MatrixLike, y: ArrayLike | None = None) -> 'ClassicalPreprocessor':
        """Fit the preprocessor to the data.
        
        Args:
            X: Feature matrix
            y: Target values (ignored)
            
        Returns:
            self
        """
        X_arr = self._to_array(X)
        
        if self.impute_strategy != 'none':
            self.imputer_ = SimpleImputer(strategy=self.impute_strategy)
            self.imputer_.fit(X_arr)
        else:
            self.imputer_ = None
            
        if self.scale:
            self.scaler_ = StandardScaler()
            # Fit on non-NaN values only
            mask = ~np.isnan(X_arr).any(axis=1)
            if mask.any():
                self.scaler_.fit(X_arr[mask])
        else:
            self.scaler_ = None
            
        if self.outlier_method != 'none':
            self._fit_outlier_detector(X_arr)
            
        return self
    
    def transform(self, X: MatrixLike) -> np.ndarray:
        """Transform the data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed numpy array
        """
        check_is_fitted(self)
        X_arr = self._to_array(X).copy()
        
        if self.imputer_ is not None:
            X_arr = self.imputer_.transform(X_arr)
            
        if self.scaler_ is not None:
            X_arr = self.scaler_.transform(X_arr)
            
        if self.outlier_method != 'none':
            X_arr = self._apply_outlier_treatment(X_arr)
            
        return X_arr
    
    def _to_array(self, X: MatrixLike) -> np.ndarray:
        """Convert input to numpy array.
        
        Args:
            X: Input data
            
        Returns:
            Numpy array
        """
        if isinstance(X, pd.DataFrame):
            return X.to_numpy()
        return np.asarray(X)
    
    def _fit_outlier_detector(self, X: np.ndarray) -> None:
        """Fit outlier detection parameters.
        
        Args:
            X: Input array
        """
        mask = ~np.isnan(X).any(axis=1)
        X_clean = X[mask]
        
        if self.outlier_method == 'iqr':
            self.outlier_q1_ = np.percentile(X_clean, 25, axis=0)
            self.outlier_q3_ = np.percentile(X_clean, 75, axis=0)
            self.outlier_iqr_ = self.outlier_q3_ - self.outlier_q1_
        elif self.outlier_method == 'zscore':
            self.outlier_mean_ = np.mean(X_clean, axis=0)
            self.outlier_std_ = np.std(X_clean, axis=0)
            
    def _apply_outlier_treatment(self, X: np.ndarray) -> np.ndarray:
        """Apply outlier detection and treatment.
        
        Args:
            X: Input array
            
        Returns:
            Array with outliers treated (clipped)
        """
        if self.outlier_method == 'iqr':
            lower = self.outlier_q1_ - self.outlier_threshold * self.outlier_iqr_
            upper = self.outlier_q3_ + self.outlier_threshold * self.outlier_iqr_
            return np.clip(X, lower, upper)
        elif self.outlier_method == 'zscore':
            z_scores = np.abs((X - self.outlier_mean_) / (self.outlier_std_ + 1e-10))
            outlier_mask = z_scores > self.outlier_threshold
            # Clip outliers to threshold
            X_clipped = X.copy()
            for i in range(X.shape[1]):
                col_outliers = outlier_mask[:, i]
                if col_outliers.any():
                    direction = np.sign(X[col_outliers, i] - self.outlier_mean_[i])
                    X_clipped[col_outliers, i] = (
                        self.outlier_mean_[i] + 
                        direction * self.outlier_threshold * self.outlier_std_[i]
                    )
            return X_clipped
        return X


class ClassicalClassifier(ClassifierMixin, BaseEstimator):
    """ClassicalClassifier implements several trade classification rules.

    Including:
    Tick test,
    Reverse tick test,
    Quote rule,
    LR algorithm,
    EMO algorithm,
    CLNV algorithm,
    Trade size rule,
    Depth rule,
    and nan

    This refactored version provides:
    - Flexible feature mapping to decouple from hardcoded column names
    - Full sklearn Pipeline compatibility
    - Advanced preprocessing options
    - Optimized memory usage and performance
    - Early validation during fit

    Args:
        layers: Layers of classical rule and subset name. Supported rules: 
               "tick", "rev_tick", "quote", "lr", "rev_lr", "emo", "rev_emo", 
               "trade_size", "depth", and "nan".
        feature_mapping: Dictionary mapping logical feature names to actual column 
                        names or indices. If None, uses default naming convention.
        features: List of feature names in order of columns. Used when input is 
                 numpy array or when Pipeline transformers strip column names.
        random_state: Random seed for unclassified trades.
        strategy: Strategy to fill unclassified trades ("random" or "const").
        preprocessor: Optional preprocessor instance. If None, no preprocessing is applied.
                     Can be 'default' to use ClassicalPreprocessor with default settings.
        validate_on_fit: Whether to validate all required features during fit.
    """

    def __init__(
        self,
        layers: list[
            tuple[
                ALLOWED_FUNC_LITERALS,
                str,
            ]
        ]
        | None = None,
        *,
        feature_mapping: dict[str, str | int] | None = None,
        features: list[str] | None = None,
        random_state: int | None = 42,
        strategy: Literal["random", "const"] = "random",
        preprocessor: ClassicalPreprocessor | Literal['default'] | None = None,
        validate_on_fit: bool = True,
    ):
        """Initialize a ClassicalClassifier.

        Examples:
            >>> X = pd.DataFrame(
            ... [
            ...     [1.5, 1, 3],
            ...     [2.5, 1, 3],
            ...     [1.5, 3, 1],
            ...     [2.5, 3, 1],
            ...     [1, np.nan, 1],
            ...     [3, np.nan, np.nan],
            ... ],
            ... columns=["trade_price", "bid_ex", "ask_ex"],
            ... )
            >>> clf = ClassicalClassifier(layers=[("quote", "ex")], strategy="const")
            >>> clf.fit(X)
            ClassicalClassifier(layers=[('quote', 'ex')], strategy='const')
            >>> pred = clf.predict_proba(X)

        Args:
            layers: Layers of classical rule and subset name.
            feature_mapping: Mapping from logical to actual feature names/indices.
            features: List of feature names for numpy array input.
            random_state: Random seed.
            strategy: Strategy to fill unclassified ("random" or "const").
            preprocessor: Preprocessor instance, 'default', or None.
            validate_on_fit: Whether to validate features during fit.
        """
        self.layers = layers
        self.feature_mapping = feature_mapping
        self.features = features
        self.random_state = random_state
        self.strategy = strategy
        self.preprocessor = preprocessor
        self.validate_on_fit = validate_on_fit

    def _more_tags(self) -> dict[str, bool | dict[str, str]]:
        """Set tags for sklearn.

        See: https://scikit-learn.org/stable/developers/develop.html#estimator-tags

        Returns:
            dict with tags
        """
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

    def _get_feature_values(self, logical_name: str, X: np.ndarray) -> np.ndarray:
        """Get feature values by logical name from numpy array.
        
        Args:
            logical_name: Logical feature name
            X: Feature matrix as numpy array
            
        Returns:
            Feature values as 1D array
        """
        idx = self.feature_mapper_.get_index(logical_name)
        if idx < 0 or idx >= X.shape[1]:
            raise ValueError(
                f"Feature '{logical_name}' (index {idx}) not found in data with "
                f"{X.shape[1]} columns. Check your feature_mapping configuration."
            )
        return X[:, idx]

    def _tick(self, subset: str, X: np.ndarray) -> npt.NDArray:
        """Classify a trade as a buy (sell) if its trade price is above (below) 
        the closest different price of a previous trade.

        Args:
            subset: subset i.e., 'all' or 'ex'.
            X: Feature matrix

        Returns:
            result of tick rule. Can be np.NaN.
        """
        trade_price = self._get_feature_values("trade_price", X)
        price_lag = self._get_feature_values(f"price_{subset}_lag", X)
        
        result = np.full(X.shape[0], np.nan)
        
        # Use np.where for vectorized comparison
        valid = ~np.isnan(price_lag)
        gt_mask = valid & (trade_price > price_lag)
        lt_mask = valid & (trade_price < price_lag)
        
        result[gt_mask] = 1
        result[lt_mask] = -1
        
        return result

    def _rev_tick(self, subset: str, X: np.ndarray) -> npt.NDArray:
        """Classify a trade as a sell (buy) if its trade price is below (above) 
        the closest different price of a subsequent trade.

        Args:
            subset: subset i.e.,'all' or 'ex'.
            X: Feature matrix

        Returns:
            result of reverse tick rule. Can be np.NaN.
        """
        trade_price = self._get_feature_values("trade_price", X)
        price_lead = self._get_feature_values(f"price_{subset}_lead", X)
        
        result = np.full(X.shape[0], np.nan)
        
        valid = ~np.isnan(price_lead)
        gt_mask = valid & (price_lead > trade_price)
        lt_mask = valid & (price_lead < trade_price)
        
        result[gt_mask] = -1
        result[lt_mask] = 1
        
        return result

    def _mid(self, subset: str, X: np.ndarray) -> npt.NDArray:
        """Calculate the midpoint of the bid and ask spread.

        Midpoint is calculated as the average of the bid and ask spread 
        if the spread is positive. Otherwise, np.NaN is returned.

        Args:
            subset: subset i.e., 'ex' or 'best'
            X: Feature matrix

        Returns:
            midpoints. Can be np.NaN.
        """
        bid = self._get_feature_values(f"bid_{subset}", X)
        ask = self._get_feature_values(f"ask_{subset}", X)
        
        result = np.full(X.shape[0], np.nan)
        valid_mask = (ask >= bid) & ~np.isnan(ask) & ~np.isnan(bid)
        result[valid_mask] = 0.5 * (ask[valid_mask] + bid[valid_mask])
        
        return result

    def _quote(self, subset: str, X: np.ndarray) -> npt.NDArray:
        """Classify a trade as a buy (sell) if its trade price is above (below) 
        the midpoint of the bid and ask spread. Trades executed at the 
        midspread are not classified.

        Args:
            subset: subset i.e., 'ex' or 'best'.
            X: Feature matrix

        Returns:
            result of quote rule. Can be np.NaN.
        """
        trade_price = self._get_feature_values("trade_price", X)
        mid = self._mid(subset, X)
        
        result = np.full(X.shape[0], np.nan)
        valid_mask = ~np.isnan(mid)
        
        gt_mask = valid_mask & (trade_price > mid)
        lt_mask = valid_mask & (trade_price < mid)
        
        result[gt_mask] = 1
        result[lt_mask] = -1
        
        return result

    def _lr(self, subset: str, X: np.ndarray) -> npt.NDArray:
        """Classify a trade as a buy (sell) if its price is above (below) the 
        midpoint (quote rule), and use the tick test to classify midspread trades.

        Adapted from Lee and Ready (1991).

        Args:
            subset: subset i.e., 'ex' or 'best'.
            X: Feature matrix

        Returns:
            result of the lee and ready algorithm with tick rule. Can be np.NaN.
        """
        q_r = self._quote(subset, X)
        result = q_r.copy()
        nan_mask = np.isnan(q_r)
        
        if nan_mask.any():
            tick_result = self._tick(subset, X)
            result[nan_mask] = tick_result[nan_mask]
            
        return result

    def _rev_lr(self, subset: str, X: np.ndarray) -> npt.NDArray:
        """Classify a trade as a buy (sell) if its price is above (below) the 
        midpoint (quote rule), and use the reverse tick test to classify midspread trades.

        Adapted from Lee and Ready (1991).

        Args:
            subset: subset i.e.,'ex' or 'best'.
            X: Feature matrix

        Returns:
            result of the lee and ready algorithm with reverse tick rule. Can be np.NaN.
        """
        q_r = self._quote(subset, X)
        result = q_r.copy()
        nan_mask = np.isnan(q_r)
        
        if nan_mask.any():
            rev_tick_result = self._rev_tick(subset, X)
            result[nan_mask] = rev_tick_result[nan_mask]
            
        return result

    def _is_at_ask_xor_bid(self, subset: str, X: np.ndarray) -> np.ndarray:
        """Check if the trade price is at the ask xor bid.

        Args:
            subset: subset i.e., 'ex' or 'best'.
            X: Feature matrix

        Returns:
            boolean array with result.
        """
        trade_price = self._get_feature_values("trade_price", X)
        ask = self._get_feature_values(f"ask_{subset}", X)
        bid = self._get_feature_values(f"bid_{subset}", X)
        
        at_ask = np.isclose(trade_price, ask, atol=1e-4)
        at_bid = np.isclose(trade_price, bid, atol=1e-4)
        return at_ask ^ at_bid

    def _is_at_upper_xor_lower_quantile(
        self, subset: str, X: np.ndarray, quantiles: float = 0.3
    ) -> np.ndarray:
        """Check if the trade price is at the ask xor bid.

        Args:
            subset: subset i.e., 'ex'.
            quantiles: percentage of quantiles. Defaults to 0.3.
            X: Feature matrix

        Returns:
            boolean array with result.
        """
        trade_price = self._get_feature_values("trade_price", X)
        ask = self._get_feature_values(f"ask_{subset}", X)
        bid = self._get_feature_values(f"bid_{subset}", X)
        
        in_upper = (
            ((1.0 - quantiles) * ask + quantiles * bid <= trade_price) & 
            (trade_price <= ask)
        )
        in_lower = (
            (bid <= trade_price) & 
            (trade_price <= quantiles * ask + (1.0 - quantiles) * bid)
        )
        return in_upper ^ in_lower

    def _emo(self, subset: str, X: np.ndarray) -> npt.NDArray:
        """Classify a trade as a buy (sell) if the trade takes place at the 
        ask (bid) quote, and use the tick test to classify all other trades.

        Adapted from Ellis et al. (2000).

        Args:
            subset: subset i.e., 'ex' or 'best'.
            X: Feature matrix

        Returns:
            result of the emo algorithm with tick rule. Can be np.NaN.
        """
        at_ask_xor_bid = self._is_at_ask_xor_bid(subset, X)
        quote_result = self._quote(subset, X)
        tick_result = self._tick(subset, X)
        
        return np.where(at_ask_xor_bid, quote_result, tick_result)

    def _rev_emo(self, subset: str, X: np.ndarray) -> npt.NDArray:
        """Classify a trade as a buy (sell) if the trade takes place at the 
        ask (bid) quote, and use the reverse tick test to classify all other trades.

        Adapted from Grauer et al. (2022).

        Args:
            subset: subset i.e., 'ex' or 'best'.
            X: Feature matrix

        Returns:
            result of the emo algorithm with reverse tick rule. Can be np.NaN.
        """
        at_ask_xor_bid = self._is_at_ask_xor_bid(subset, X)
        quote_result = self._quote(subset, X)
        rev_tick_result = self._rev_tick(subset, X)
        
        return np.where(at_ask_xor_bid, quote_result, rev_tick_result)

    def _clnv(self, subset: str, X: np.ndarray) -> npt.NDArray:
        """Classify a trade based on deciles of the bid and ask spread.

        Spread is divided into ten deciles and trades are classified as follows:
        - use quote rule for at ask until 30 % below ask (upper 3 deciles)
        - use quote rule for at bid until 30 % above bid (lower 3 deciles)
        - use tick rule for all other trades (±2 deciles from midpoint; outside
        bid or ask).

        Adapted from Chakrabarty et al. (2007).

        Args:
            subset: subset i.e.,'ex' or 'best'.
            X: Feature matrix

        Returns:
            result of the emo algorithm with tick rule. Can be np.NaN.
        """
        at_quantile = self._is_at_upper_xor_lower_quantile(subset, X)
        quote_result = self._quote(subset, X)
        tick_result = self._tick(subset, X)
        
        return np.where(at_quantile, quote_result, tick_result)

    def _rev_clnv(self, subset: str, X: np.ndarray) -> npt.NDArray:
        """Classify a trade based on deciles of the bid and ask spread.

        Spread is divided into ten deciles and trades are classified as follows:
        - use quote rule for at ask until 30 % below ask (upper 3 deciles)
        - use quote rule for at bid until 30 % above bid (lower 3 deciles)
        - use reverse tick rule for all other trades (±2 deciles from midpoint;
        outside bid or ask).

        Similar to extension of emo algorithm proposed Grauer et al. (2022).

        Args:
            subset: subset i.e., 'ex' or 'best'.
            X: Feature matrix

        Returns:
            result of the emo algorithm with tick rule. Can be np.NaN.
        """
        at_quantile = self._is_at_upper_xor_lower_quantile(subset, X)
        quote_result = self._quote(subset, X)
        rev_tick_result = self._rev_tick(subset, X)
        
        return np.where(at_quantile, quote_result, rev_tick_result)

    def _trade_size(self, subset: str, X: np.ndarray) -> npt.NDArray:
        """Classify a trade as a buy (sell) the trade size matches exactly 
        either the bid (ask) quote size.

        Adapted from Grauer et al. (2022).

        Args:
            subset: subset i.e., 'ex' or 'best'.
            X: Feature matrix

        Returns:
            result of the trade size rule. Can be np.NaN.
        """
        trade_size = self._get_feature_values("trade_size", X)
        ask_size = self._get_feature_values(f"ask_size_{subset}", X)
        bid_size = self._get_feature_values(f"bid_size_{subset}", X)
        
        bid_eq_ask = np.isclose(ask_size, bid_size, atol=1e-4)
        ts_eq_bid = np.isclose(trade_size, bid_size, atol=1e-4) & ~bid_eq_ask
        ts_eq_ask = np.isclose(trade_size, ask_size, atol=1e-4) & ~bid_eq_ask
        
        return np.where(ts_eq_bid, 1, np.where(ts_eq_ask, -1, np.nan))

    def _depth(self, subset: str, X: np.ndarray) -> npt.NDArray:
        """Classify midspread trades as buy (sell), if the ask size (bid size) 
        exceeds the bid size (ask size).

        Adapted from Grauer et al. (2022).

        Args:
            subset: subset i.e., 'ex' or 'best'.
            X: Feature matrix

        Returns:
            result of depth rule. Can be np.NaN.
        """
        trade_price = self._get_feature_values("trade_price", X)
        ask_size = self._get_feature_values(f"ask_size_{subset}", X)
        bid_size = self._get_feature_values(f"bid_size_{subset}", X)
        mid = self._mid(subset, X)
        
        at_mid = np.isclose(mid, trade_price, atol=1e-4)
        
        result = np.full(X.shape[0], np.nan)
        
        buy_mask = at_mid & (ask_size > bid_size)
        sell_mask = at_mid & (ask_size < bid_size)
        
        result[buy_mask] = 1
        result[sell_mask] = -1
        
        return result

    def _nan(self, subset: str, X: np.ndarray) -> npt.NDArray:
        """Classify nothing. Fast forward results from previous classifier.

        Args:
            subset: Ignored
            X: Feature matrix

        Returns:
            Array filled with np.NaN.
        """
        return np.full(X.shape[0], np.nan)

    def _get_required_features(self, layers: list[tuple[str, str]]) -> set[str]:
        """Determine required features based on layer configuration.
        
        Args:
            layers: List of (function_name, subset) tuples
            
        Returns:
            Set of required logical feature names
        """
        required = set()
        
        for func_str, subset in layers:
            # Skip feature requirements for 'nan' rule (it doesn't classify anything)
            if func_str == 'nan':
                continue
                
            # trade_size rule only needs trade_size and size columns
            if func_str == 'trade_size':
                required.add('trade_size')
                required.add(f'bid_size_{subset}')
                required.add(f'ask_size_{subset}')
                continue
                
            # All other rules need trade_price
            required.add('trade_price')
            
            if func_str in ('tick', 'lr', 'emo', 'clnv'):
                required.add(f'price_{subset}_lag')
            if func_str in ('rev_tick', 'rev_lr', 'rev_emo', 'rev_clnv'):
                required.add(f'price_{subset}_lead')
            if func_str in ('quote', 'lr', 'rev_lr', 'emo', 'rev_emo', 'clnv', 'rev_clnv', 'depth'):
                required.add(f'bid_{subset}')
                required.add(f'ask_{subset}')
            if func_str == 'depth':
                required.add(f'bid_size_{subset}')
                required.add(f'ask_size_{subset}')
                
        return required

    def _validate_and_build_mapper(
        self, 
        X: MatrixLike, 
        feature_names: list[str] | None
    ) -> FeatureMapper:
        """Validate input and build feature mapper.
        
        Args:
            X: Input data
            feature_names: Optional feature names
            
        Returns:
            Fitted FeatureMapper
            
        Raises:
            ValueError: If required features are missing
        """
        # Determine effective feature names
        if isinstance(X, pd.DataFrame):
            effective_names = X.columns.tolist()
        elif feature_names is not None:
            effective_names = feature_names
        else:
            effective_names = None
            
        # Create and fit feature mapper
        mapper = FeatureMapper(self.feature_mapping)
        mapper.fit(X, effective_names)
        
        # Validate required features
        if self.validate_on_fit and self._layers:
            required = self._get_required_features(self._layers)
            missing = mapper.validate_required_features(list(required))
            
            if missing:
                # Try to provide helpful error message
                missing_str = ', '.join(sorted(missing))
                actual_cols = effective_names if effective_names else f"{X.shape[1]} columns (unnamed)"
                raise ValueError(
                    f"Missing required features: [{missing_str}]. "
                    f"Available columns: {actual_cols}. "
                    f"Use feature_mapping parameter to map your column names to expected features. "
                    f"See documentation for expected feature names."
                )
                
        return mapper

    def fit(
        self,
        X: MatrixLike,
        y: ArrayLike | None = None,
        sample_weight: npt.NDArray | None = None,
    ) -> 'ClassicalClassifier':
        """Fit the classifier.

        Args:
            X: features
            y: ignored, present here for API consistency by convention.
            sample_weight: Sample weights. Defaults to None.

        Raises:
            ValueError: Unknown subset e. g., 'ise'
            ValueError: Unknown function string e. g., 'lee-ready'
            ValueError: Multi output is not supported.
            ValueError: Required features are missing.

        Returns:
            Instance of itself.
        """
        _check_sample_weight(sample_weight, X)

        # Build function mapping
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
        self.func_mapping_: dict[str, Callable] = dict(zip(ALLOWED_FUNC_STR, funcs))

        # Validate and process layers
        self._layers = self.layers if self.layers is not None else []
        for func_str, _ in self._layers:
            if func_str not in ALLOWED_FUNC_STR:
                raise ValueError(
                    f"Unknown function string: {func_str},"
                    f"expected one of {ALLOWED_FUNC_STR}."
                )

        # Validate data and get feature names
        X_validated = validate_data(
            self,
            X,
            y="no_validation",
            dtype=[np.float64, np.float32],
            accept_sparse=False,
            ensure_all_finite=False,
        )

        self.classes_ = np.array([-1, 1])
        self.n_features_in_ = X_validated.shape[1]
        
        # Determine effective feature names for validation
        effective_names = self.features
        if isinstance(X, pd.DataFrame):
            effective_names = X.columns.tolist()
        elif effective_names is None:
            # Generate default names for validation
            effective_names = [str(i) for i in range(X_validated.shape[1])]

        # Validate features and build mapper (early validation)
        self.feature_mapper_ = self._validate_and_build_mapper(X, effective_names)
        
        # Store feature names for predict (use different name to avoid sklearn conflicts)
        if isinstance(X, pd.DataFrame):
            self._feature_names_in = X.columns.tolist()
        elif self.features is not None:
            self._feature_names_in = self.features
        else:
            self._feature_names_in = [str(i) for i in range(X_validated.shape[1])]

        # Setup preprocessor
        if self.preprocessor == 'default':
            self.preprocessor_ = ClassicalPreprocessor(impute_strategy='none')
        elif self.preprocessor is not None:
            self.preprocessor_ = self.preprocessor
        else:
            self.preprocessor_ = None
            
        if self.preprocessor_ is not None:
            self.preprocessor_.fit(X_validated)

        return self

    def _transform_X(self, X: MatrixLike) -> np.ndarray:
        """Transform input to numpy array, applying preprocessor if available.
        
        Args:
            X: Input data (DataFrame or ndarray)
            
        Returns:
            Processed numpy array
        """
        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy()
        else:
            X_arr = np.asarray(X)
            
        if self.preprocessor_ is not None:
            X_arr = self.preprocessor_.transform(X_arr)
            
        return X_arr

    def predict(self, X: MatrixLike) -> npt.NDArray:
        """Perform classification on test vectors `X`.

        Args:
            X: feature matrix.

        Returns:
            Predicted target values for X.
        """
        check_is_fitted(self)
        
        # Validate input shape (reset=False to check consistency with fit)
        X = validate_data(
            self,
            X,
            dtype=[np.float64, np.float32],
            accept_sparse=False,
            ensure_all_finite=False,
            reset=False,
        )

        # Transform input (convert to numpy, apply preprocessing)
        X_arr = self._transform_X(X)
        
        # Predict using numpy operations only
        rs = check_random_state(self.random_state)
        pred = self._predict_numpy(X_arr)

        # Fill NaNs randomly with -1 and 1 or with constant zero
        mask = np.isnan(pred)
        if self.strategy == "random":
            pred[mask] = rs.choice(self.classes_, pred.shape)[mask]
        else:
            pred[mask] = 0

        return pred

    def _predict_numpy(self, X: np.ndarray) -> npt.NDArray:
        """Predict with rule stack using numpy operations.

        Args:
            X: Feature matrix as numpy array

        Returns:
            prediction
        """
        pred = np.full(X.shape[0], np.nan)
        
        for func_str, subset in self._layers:
            func = self.func_mapping_[func_str]
            layer_result = func(subset=subset, X=X)
            
            # Update predictions where current prediction is NaN
            nan_mask = np.isnan(pred)
            pred[nan_mask] = layer_result[nan_mask]
            
        return pred

    def predict_proba(self, X: MatrixLike) -> npt.NDArray:
        """Predict class probabilities for X.

        Probabilities are either 0 or 1 depending on the class.

        For strategy 'constant' probabilities are (0.5,0.5) for unclassified classes.

        Args:
            X: feature matrix

        Returns:
            probabilities
        """
        check_is_fitted(self)
        
        X_arr = validate_data(
            self,
            X,
            dtype=[np.float64, np.float32],
            accept_sparse=False,
            ensure_all_finite=False,
            reset=False,
        )
        
        n_samples = X_arr.shape[0]
        
        # assign 0.5 to all classes. Required for strategy 'constant'.
        prob = np.full((n_samples, 2), 0.5)

        # Class can be assumed to be -1 or 1 for strategy 'random'.
        # Class might be zero though for strategy constant. Mask non-zeros.
        preds = self.predict(X)
        mask = np.flatnonzero(preds)

        if len(mask) > 0:
            # get index of predicted class and one-hot encode it
            indices = np.nonzero(preds[mask, None] == self.classes_[None, :])[1]
            n_classes = np.max(self.classes_) + 1

            # overwrite defaults with one-hot encoded classes.
            prob[mask] = np.identity(n_classes)[indices]
            
        return prob
    
    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        """Get output feature names for transformation.
        
        Args:
            input_features: Input features (ignored, present for API consistency)
            
        Returns:
            Array of feature names
        """
        check_is_fitted(self)
        return np.array(self._feature_names_in)
