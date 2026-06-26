
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.constants import (
    DERIVED_CATEGORICAL_COLUMNS,
    DERIVED_FLAG_COLUMNS,
    ESCMAE2010_ORDINAL_COLUMN,
    HISTGB_PARAMS,
    KOTELCHUCK_ORDINAL_COLUMN,
    NUMERIC_COLUMNS,
    PASSTHROUGH_COLUMNS,
    RANDOM_STATE,
)
from src.transformers import FeatureEngineer


def build_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    numeric_columns = list(NUMERIC_COLUMNS) + [
        ESCMAE2010_ORDINAL_COLUMN,
        KOTELCHUCK_ORDINAL_COLUMN,
    ]
    categorical_columns = list(DERIVED_CATEGORICAL_COLUMNS)
    passthrough_columns = list(PASSTHROUGH_COLUMNS) + list(DERIVED_FLAG_COLUMNS)

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
            ("passthrough", "passthrough", passthrough_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    classifier = HistGradientBoostingClassifier(random_state=random_state, **HISTGB_PARAMS)

    return Pipeline(
        [
            ("feature_engineer", FeatureEngineer()),
            ("preprocessor", preprocessor),
            ("histgradientboostingclassifier", classifier),
        ]
    )
