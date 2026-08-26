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

# How fast the running season's own bias should override the fitted prior.
# Same empirical-Bayes form as everywhere else: with n player-gameweeks of
# evidence the running season carries n / (n + PRIOR_N) of the weight.
# PRIOR_N is set so that a full gameweek of ~600 rows moves the factor about
# 5%, and half a season is needed before the live data dominates.
PRIOR_N = 12000.0


def season_factors(live: pd.DataFrame | None = None,
                   pred: str = "xpts", actual: str = "total_points") -> dict:
    """Positional factors for the running season.

    Returns the fitted priors before a ball is kicked, and blends the running
    season in as it accumulates. Without this the factors are frozen at last
    season's values for the whole campaign -- correct in August and stale by
    December, with nothing to signal that it has gone wrong.
    """
    if live is None or live.empty:
        return dict(SEASON_FACTORS)
    out = dict(SEASON_FACTORS)
    g = live.dropna(subset=[pred, actual])
    for pos, sub in g.groupby("position"):
        p = sub[pred].sum()
        if p <= 0:
            continue
        obs = float(np.clip(sub[actual].sum() / p, *CLAMP))
        w = len(sub) / (len(sub) + PRIOR_N)
        out[pos] = float(np.clip(w * obs + (1 - w) * SEASON_FACTORS.get(pos, 1.0), *CLAMP))
    return out

class LevelCalibrator:
    """Isotonic calibration per position, which a flat multiplier is not.

    A single multiplier per position is the wrong shape. Fitted on totals it is
    dominated by the many near-zero rows, where the actual-to-projected ratio
    is 1.45, and then applied to the handful of high projections where the true
    ratio is 1.10. The result inflates precisely the top of the list a squad is
    picked from -- the shipped flat factors put the optimal squad at 64.1 points
    a gameweek against a displayed per-gameweek sum of 54.9 and an FPL average
    of 50.

    Isotonic keeps the ordering, so nothing about selection is disturbed, while
    letting the correction differ by level. Same machinery already used to
    calibrate DefCon.
    """

    def __init__(self, min_obs: int = MIN_OBS) -> None:
        self.models: dict = {}
        self.min_obs = min_obs

    def fit(self, d: pd.DataFrame, pred: str = "xpts",
            actual: str = "total_points") -> "LevelCalibrator":
        from sklearn.isotonic import IsotonicRegression
        g = d.dropna(subset=[pred, actual])
        for pos, sub in g.groupby("position"):
            if len(sub) < self.min_obs:
                continue
            iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
            iso.fit(sub[pred].to_numpy(float), sub[actual].to_numpy(float))
            self.models[pos] = iso
        return self

    def transform(self, d: pd.DataFrame, pred: str = "xpts") -> pd.Series:
        out = pd.to_numeric(d[pred], errors="coerce").fillna(0.0).copy()
        for pos, iso in self.models.items():
            m = (d["position"] == pos).to_numpy()
            if m.any():
                out.loc[m] = iso.predict(out[m].to_numpy(float))
        return out
