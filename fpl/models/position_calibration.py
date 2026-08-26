"""Positional bias correction for the assembled projection.

Walk-forward over four seasons the projection is not positionally neutral.
Forwards come in 0.29 points a gameweek low and midfielders 0.21, while
defenders run 0.04 high; among players the model expects to start the gap
widens to 0.75 for forwards.

The direction says where it comes from. Attacking returns are under-projected
and defensive ones are not, and 48.5% of a forward's points arrive in returns
of eight or more. A simulation that generates too little mass in that tail
lands near the median of a right-skewed distribution rather than its mean,
which is exactly the observed signature -- forwards are predicted 2.73 against
a median of 2 and a mean of 4.34.

Fixing the tail properly means revisiting the attacking-shares draw. This is
the smaller, honest intervention in the meantime: a multiplicative correction
per position, fitted only on gameweeks already played, applied to gameweeks
not yet played. It is calibration, in the same sense the isotonic step is
calibration for DefCon -- it does not claim to know why the model is low, only
that it reliably is, and by how much.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Below this many observations a positional factor is noise, so the correction
# stays at 1.0 and the projection is left alone.
MIN_OBS = 1500
# Corrections outside this band mean something is wrong that a multiplier
# should not be quietly hiding.
CLAMP = (0.80, 1.35)

# Fitted on 2022-23 through 2025-26 and applied to the live season, which has
# no completed history of its own to fit on. Held out, the correction was worth
# +57, +31 and +42 points across three seasons -- three from three -- and the
# factors barely move between fits, which is what a structural bias looks like
# as opposed to a noisy one.
SEASON_FACTORS = {"GKP": 1.20, "DEF": 0.98, "MID": 1.23, "FWD": 1.31}


class PositionCalibrator:
    """Per-position multiplicative correction, fitted on realised outcomes."""

    def __init__(self, min_obs: int = MIN_OBS) -> None:
        self.factors: dict[str, float] = {}
        self.min_obs = min_obs

    def fit(self, d: pd.DataFrame, pred: str = "xpts",
            actual: str = "total_points") -> "PositionCalibrator":
        g = d.dropna(subset=[pred, actual])
        for pos, sub in g.groupby("position"):
            if len(sub) < self.min_obs:
                continue
            p = sub[pred].sum()
            if p <= 0:
                continue
            # Ratio of totals, not mean of ratios: a player projected at 0.02
            # who returns 2 would otherwise dominate the estimate.
            self.factors[pos] = float(np.clip(sub[actual].sum() / p, *CLAMP))
        return self

    def transform(self, d: pd.DataFrame, pred: str = "xpts") -> pd.Series:
        f = d["position"].map(self.factors).fillna(1.0)
        return pd.to_numeric(d[pred], errors="coerce").fillna(0.0) * f
