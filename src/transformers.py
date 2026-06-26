from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.constants import (
    ESCMAE2010_COLUMN,
    ESCMAE2010_MAP,
    FAIXAETAMAE_BINS_LABELS,
    FAIXAETAMAE_COLUMN,
    IDADEMAE_COLUMN,
    KOTELCHUCK_COLUMN,
    KOTELCHUCK_MAP,
    MESPRENAT_COLUMN,
    PRIMIPARA_COLUMN,
    PNTARDIO_COLUMN,
    QTDPARTCES_COLUMN,
    QTDPARTNOR_COLUMN,
    QTDFILMORT_COLUMN,
    HISTPERDAFETAL_COLUMN,
)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless feature engineering transformer.

    Produces these derived features (if source columns exist):
    - PNTARDIO: MESPRENAT > 4
    - HISTPERDAFETAL: QTDFILMORT > 0
    - PRIMIPARA: QTDPARTNOR + QTDPARTCES == 0
    - FAIXAETAMAE: binned IDADEMAE (categorical labels)
    - ESCMAE2010_ORDINAL: mapped from ESCMAE2010 via ESCMAE2010_MAP
    - KOTELCHUCK_ORDINAL: mapped from KOTELCHUCK via KOTELCHUCK_MAP

    The transformer is intentionally stateless: ``fit`` returns self.
    """

    def __init__(self, idade_bins: Iterable[int] | None = None, idade_labels: Iterable[str] | None = None):
        if idade_bins is not None and idade_labels is not None:
            self.idade_bins = list(idade_bins)
            self.idade_labels = list(idade_labels)
        else:
            self.idade_bins = list(FAIXAETAMAE_BINS_LABELS.keys())
            self.idade_labels = list(FAIXAETAMAE_BINS_LABELS.values())

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineer expects a pandas DataFrame as input")

        out = X.copy()

        # PNTARDIO: MESPRENAT > 4
        if MESPRENAT_COLUMN in out.columns:
            out[PNTARDIO_COLUMN] = (out[MESPRENAT_COLUMN] > 4).astype(int)

        # HISTPERDAFETAL: QTDFILMORT > 0
        if QTDFILMORT_COLUMN in out.columns:
            out[HISTPERDAFETAL_COLUMN] = (out[QTDFILMORT_COLUMN].fillna(0) > 0).astype(int)

        # PRIMIPARA: QTDPARTNOR + QTDPARTCES == 0
        if QTDPARTNOR_COLUMN in out.columns or QTDPARTCES_COLUMN in out.columns:
            qn = out.get(QTDPARTNOR_COLUMN)
            qc = out.get(QTDPARTCES_COLUMN)
            s = pd.Series(0, index=out.index, dtype=float)
            if qn is not None:
                s = s.add(qn.fillna(0), fill_value=0)
            if qc is not None:
                s = s.add(qc.fillna(0), fill_value=0)
            out[PRIMIPARA_COLUMN] = (s == 0).astype(int)

        # FAIXAETAMAE: binned IDADEMAE
        if IDADEMAE_COLUMN in out.columns:
            try:
                out[FAIXAETAMAE_COLUMN] = pd.cut(
                    out[IDADEMAE_COLUMN],
                    bins=self.idade_bins,
                    labels=self.idade_labels,
                    include_lowest=True,
                )
            except Exception:
                out[FAIXAETAMAE_COLUMN] = pd.Series(pd.NA, index=out.index)

        # ESCMAE2010_ORDINAL
        if ESCMAE2010_COLUMN in out.columns:
            out["ESCMAE2010_ORDINAL"] = out[ESCMAE2010_COLUMN].map(ESCMAE2010_MAP)

        # KOTELCHUCK_ORDINAL
        if KOTELCHUCK_COLUMN in out.columns:
            out["KOTELCHUCK_ORDINAL"] = out[KOTELCHUCK_COLUMN].map(KOTELCHUCK_MAP)

        return out

    # Make get_feature_names_out available to be used by pipelines that rely on it.
    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> list[str]:
        base = list(input_features) if input_features is not None else []
        derived = [
            PNTARDIO_COLUMN,
            HISTPERDAFETAL_COLUMN,
            PRIMIPARA_COLUMN,
            FAIXAETAMAE_COLUMN,
            "ESCMAE2010_ORDINAL",
            "KOTELCHUCK_ORDINAL",
        ]
        # preserve order but avoid duplicates
        out = base + [c for c in derived if c not in base]
        return out
