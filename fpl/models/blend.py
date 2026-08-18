"""Precision-weighted blend of the team model with the betting market.

The market is a strong, well-calibrated, free prior. The model's job is to add
the residual it does not price -- squad-level information, rotation, and the
team-strength drift the market is slow on -- not to beat it outright.

Blending happens in log-rate space, where both sources are approximately
Gaussian and a fixed weight behaves sensibly across the range of totals:

    log lambda_blend = w * log lambda_model + (1 - w) * log lambda_market

w is tuned out-of-sample rather than assumed.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson


def blend_rates(model_lam: float, model_mu: float,
                mkt_lam: float, mkt_mu: float, w: float) -> tuple[float, float]:
    if not (np.isfinite(mkt_lam) and np.isfinite(mkt_mu)):
        return model_lam, model_mu
    lam = np.exp(w * np.log(max(model_lam, 1e-6)) + (1 - w) * np.log(max(mkt_lam, 1e-6)))
    mu = np.exp(w * np.log(max(model_mu, 1e-6)) + (1 - w) * np.log(max(mkt_mu, 1e-6)))
    return float(lam), float(mu)


def score_matrix(lam: float, mu: float, rho: float, max_goals: int = 10) -> np.ndarray:
    g = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    m[0, 0] *= 1.0 - lam * mu * rho
    m[0, 1] *= 1.0 + lam * rho
    m[1, 0] *= 1.0 + mu * rho
    m[1, 1] *= 1.0 - rho
    return m / m.sum()


def outcome_probs(m: np.ndarray) -> tuple[float, float, float]:
    return (float(np.tril(m, -1).sum()), float(np.trace(m)), float(np.triu(m, 1).sum()))


def clean_sheets(m: np.ndarray) -> tuple[float, float]:
    """P(home CS), P(away CS) -- the Poisson zero."""
    return float(m[:, 0].sum()), float(m[0, :].sum())
