"""Preprocessing pipeline definitions using Scikit-Learn ColumnTransformer and Pipeline."""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES


def create_numerical_pipeline(impute_strategy: str = "median") -> Pipeline:
    """Create preprocessing sub-pipeline for continuous numerical features.

    Steps:
    1. SimpleImputer (default: 'median') to safeguard against missing values.
    2. StandardScaler to scale continuous features to zero mean and unit variance.

    Args:
        impute_strategy: Strategy for SimpleImputer ('median', 'mean', etc.).

    Returns:
        Pipeline: Numerical transformation pipeline.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=impute_strategy)),
            ("scaler", StandardScaler()),
        ]
    )


def create_categorical_pipeline(
    handle_unknown: str = "ignore",
    drop: Optional[str] = None,
) -> Pipeline:
    """Create preprocessing sub-pipeline for categorical features.

    Steps:
    1. SimpleImputer ('most_frequent') to handle any missing category values.
    2. OneHotEncoder with handle_unknown='ignore' to gracefully handle unseen
       categories at test/inference time without breaking.

    Args:
        handle_unknown: Strategy for handling unseen categories ('ignore' recommended).
        drop: Category drop strategy (None, 'first', or 'if_binary').

    Returns:
        Pipeline: Categorical transformation pipeline.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown=handle_unknown,
                    sparse_output=False,
                    drop=drop,
                ),
            ),
        ]
    )


def build_preprocessor(
    numerical_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    handle_unknown: str = "ignore",
    drop: Optional[str] = None,
) -> ColumnTransformer:
    """Construct full preprocessing ColumnTransformer combining numerical and categorical pipelines.

    Args:
        numerical_cols: List of numerical column names. Defaults to NUMERICAL_FEATURES.
        categorical_cols: List of categorical column names. Defaults to CATEGORICAL_FEATURES.
        handle_unknown: Setting for OneHotEncoder handle_unknown ('ignore').
        drop: Setting for OneHotEncoder drop strategy.

    Returns:
        ColumnTransformer: Unfitted preprocessor ready for fitting on training data.
    """
    num_cols = numerical_cols if numerical_cols is not None else NUMERICAL_FEATURES
    cat_cols = categorical_cols if categorical_cols is not None else CATEGORICAL_FEATURES

    num_pipeline = create_numerical_pipeline()
    cat_pipeline = create_categorical_pipeline(handle_unknown=handle_unknown, drop=drop)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def get_feature_names(fitted_preprocessor: ColumnTransformer) -> List[str]:
    """Retrieve output feature names from a fitted ColumnTransformer.

    Args:
        fitted_preprocessor: A fitted ColumnTransformer.

    Returns:
        List[str]: Clean list of feature names.
    """
    try:
        return list(fitted_preprocessor.get_feature_names_out())
    except Exception:
        # Fallback if get_feature_names_out() fails on older custom transforms
        names = []
        for name, transformer, cols in fitted_preprocessor.transformers_:
            if name == "num":
                names.extend(cols)
            elif name == "cat":
                encoder = transformer.named_steps["encoder"]
                cat_names = encoder.get_feature_names_out(cols)
                names.extend(cat_names)
        return list(names)


def transform_to_dataframe(
    fitted_preprocessor: ColumnTransformer,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Transform an input DataFrame and return a pandas DataFrame with aligned column names.

    Args:
        fitted_preprocessor: Fitted ColumnTransformer instance.
        X: Input feature DataFrame.

    Returns:
        pd.DataFrame: Transformed data with original index and feature names.
    """
    transformed_array = fitted_preprocessor.transform(X)
    feature_names = get_feature_names(fitted_preprocessor)
    return pd.DataFrame(
        transformed_array,
        columns=feature_names,
        index=X.index,
    )
