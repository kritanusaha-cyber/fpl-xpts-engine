"""Phase 1 -- minutes model.

Target is the ordered categorical {0 min, 1-59, 60+}. Implemented as two chained
binary logits rather than a single ordered logit:

    P(appear)          -- over all rows
    P(>=60 | appear)   -- over rows where the player appeared

The chained form is used because the two decisions have genuinely different
drivers (selection vs. in-match usage), so forcing them through one set of
coefficients with a shared linear index costs calibration. It also degrades
more gracefully: P(appear) stays meaningful when the conditional model is thin.

Calibration is what matters here, not accuracy. Everything downstream multiplies
by these probabilities, so a miscalibrated minutes model biases every component.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fpl.features.minutes import FEATURES


@dataclass
class MinutesModel:
    """Chained P(appear) and P(>=60 | appear)."""

    appear: Pipeline
    play60: Pipeline
    features: list[str]

    @staticmethod
    def _pipeline() -> Pipeline:
        return Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, C=1.0)),
        ])

    @classmethod
    def fit(cls, train: pd.DataFrame, features: list[str] = FEATURES) -> "MinutesModel":
        X = train[features].fillna(0.0)

        appear = cls._pipeline().fit(X, train["appeared"])

        played = train[train["appeared"] == 1]
        play60 = cls._pipeline().fit(played[features].fillna(0.0), played["played_60"])

        return cls(appear=appear, play60=play60, features=features)

    def predict(self, test: pd.DataFrame) -> pd.DataFrame:
        X = test[self.features].fillna(0.0)
        p_appear = self.appear.predict_proba(X)[:, 1]
        p_60_given = self.play60.predict_proba(X)[:, 1]

        p_60 = p_appear * p_60_given
        p_cameo = p_appear * (1.0 - p_60_given)

        return pd.DataFrame({
            "p_appear": p_appear,
            "p_60": p_60,
            "p_cameo": p_cameo,
            "p_none": 1.0 - p_appear,
            # Expected minutes, for components that need a rate rather than a
            # threshold. Cameo midpoint of 30 is a placeholder to be fit later.
            "exp_minutes": p_60 * 75.0 + p_cameo * 30.0,
        }, index=test.index)
