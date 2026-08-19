"""Monotonic recalibration for the DefCon threshold model.

The negative binomial ranks players well but its probabilities are distorted in
a specific, repeatable way. Measured over 8,626 walk-forward predictions, with
binomial confidence intervals so the tail is not dismissed as small-sample noise:

    predicted 0.05  ->  realised 0.034   CI [0.029, 0.039]   over-predicts
    predicted 0.26  ->  realised 0.308   CI [0.291, 0.325]   under-predicts
    predicted 0.39  ->  realised 0.376   CI [0.326, 0.428]   calibrated
    predicted 0.57  ->  realised 0.280   CI [0.135, 0.473]   over-predicts
    predicted 0.77  ->  realised 0.091   CI [0.010, 0.353]   badly over-predicts

The intervals exclude the predictions in four of five bands, so this is a real
distortion, not sampling noise. The shape is an S: too confident at both ends,
honest in the middle.

Why it happens: the negative binomial's tail is doing work the data cannot
support. A player whose recent rate implies a high threshold probability is
usually one who had an unusually busy few matches -- against a possession side,
or in a game his team spent defending. The model reads that as a durable rate and
extrapolates; the next fixture regresses. The count model is right about the
ordering and wrong about the extremity.

Isotonic regression fixes exactly this: it is monotonic, so the ranking the model
gets right is preserved, and it is non-parametric, so it can flatten the tail
without assuming a functional form. Fitted walk-forward -- the map comes only
from gameweeks already played.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


# Isotonic repairs the bulk of the distribution but cannot learn a tail its own
# training history barely contains -- after recalibration the p > 0.5 band still
# read 0.606 predicted against 0.188 realised (n = 16). So the tail is capped as
# well. The cap is not a fudge: on walk-forward data every cap from 0.60 down to
# 0.40 improves both Brier and log-loss monotonically, which says the model
# simply should not be emitting high-confidence DefCon calls at all.
#
# It matters where it counts. The 0.72% of players above raw p = 0.45 averaged
# 0.562 predicted against 0.306 realised -- about 0.51 xPts per match of
# overstatement, applied to exactly the defenders a DefCon strategy would buy.
TAIL_CAP = 0.40


class DefConCalibrator:
    """Monotonic probability map, fitted on realised outcomes, with a tail cap."""

    def __init__(self, min_samples: int = 800, cap: float = TAIL_CAP):
        self.iso: IsotonicRegression | None = None
        self.min_samples = min_samples
        self.cap = cap

    def fit(self, p: np.ndarray, y: np.ndarray) -> "DefConCalibrator":
        p, y = np.asarray(p, float), np.asarray(y, float)
        ok = np.isfinite(p) & np.isfinite(y)
        if ok.sum() < self.min_samples:
            self.iso = None
            return self
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.iso.fit(p[ok], y[ok])
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, float)
        if self.iso is None:
            return np.clip(p, 0.0, self.cap)
        out = self.iso.predict(np.clip(p, 0, 1))
        return np.clip(out, 0.0, self.cap)
