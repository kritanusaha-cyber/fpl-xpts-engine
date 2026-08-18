"""Phase 4 -- DefCon threshold probabilities.

FPL awards 2 points for reaching a threshold count of defensive actions:

    DEF        tackles + CBI                >= 10
    MID / FWD  tackles + CBI + recoveries   >= 12

Both the composition and the thresholds were derived empirically -- see
FINDINGS.md. Defenders exclude recoveries, which is easy to get wrong.

Because the payoff is a threshold, the mean is not sufficient: you need
P(N >= threshold) at the player's PROJECTED minutes, not at 90. Counts are
heavily overdispersed and role-dependent, so a Poisson underfits badly. This
fits a negative binomial with a log-minutes offset and a role-varying
dispersion parameter, and compares against Poisson to show the difference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import nbinom, poisson

THRESHOLDS = {"DEF": 10, "MID": 12, "FWD": 12, "GKP": None}


def defcon_count(d: pd.DataFrame) -> pd.Series:
    """Position-dependent qualifying-action count."""
    t = pd.to_numeric(d["tackles"], errors="coerce").fillna(0)
    c = pd.to_numeric(d["clearances_blocks_interceptions"], errors="coerce").fillna(0)
    r = pd.to_numeric(d["recoveries"], errors="coerce").fillna(0)
    return np.where(d["position"] == "DEF", t + c, t + c + r)


@dataclass
class NegBinDefCon:
    """log(mu) = beta . x + log(minutes/90); dispersion alpha per position."""

    beta: dict
    alpha: dict
    features: list

    @staticmethod
    def _nll(params, X, y, offset):
        k = X.shape[1]
        b, log_alpha = params[:k], params[k]
        alpha = np.exp(log_alpha)
        mu = np.exp(np.clip(X @ b + offset, -10, 10))
        r = 1.0 / alpha
        # NB2 log-likelihood
        ll = (gammaln(y + r) - gammaln(r) - gammaln(y + 1)
              + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu)))
        return -np.sum(ll)

    @classmethod
    def fit(cls, train: pd.DataFrame, features: list) -> "NegBinDefCon":
        beta, alpha = {}, {}
        for pos in ["DEF", "MID", "FWD"]:
            g = train[train.position == pos]
            if len(g) < 200:
                continue
            X = np.column_stack([np.ones(len(g)), g[features].fillna(0).to_numpy()])
            y = g["defcon_n"].to_numpy(dtype=float)
            offset = np.log(np.clip(g["minutes"].to_numpy(), 1, None) / 90.0)
            p0 = np.concatenate([[np.log(max(y.mean(), 0.1))], np.zeros(len(features)), [0.0]])
            res = minimize(cls._nll, p0, args=(X, y, offset), method="L-BFGS-B",
                           options={"maxiter": 400})
            beta[pos] = res.x[:X.shape[1]]
            alpha[pos] = float(np.exp(res.x[-1]))
        return cls(beta=beta, alpha=alpha, features=features)

    def rate(self, d: pd.DataFrame, minutes: np.ndarray | None = None) -> np.ndarray:
        mins = d["minutes"].to_numpy(dtype=float) if minutes is None else minutes
        out = np.full(len(d), np.nan)
        for pos, b in self.beta.items():
            m = (d.position == pos).to_numpy()
            if not m.any():
                continue
            X = np.column_stack([np.ones(m.sum()), d.loc[m, self.features].fillna(0).to_numpy()])
            offset = np.log(np.clip(mins[m], 1, None) / 90.0)
            out[m] = np.exp(np.clip(X @ b + offset, -10, 10))
        return out

    def p_threshold(self, d: pd.DataFrame, minutes: np.ndarray | None = None) -> np.ndarray:
        """P(N >= threshold), negative binomial."""
        mu = self.rate(d, minutes)
        out = np.zeros(len(d))
        for pos, alpha in self.alpha.items():
            m = (d.position == pos).to_numpy() & np.isfinite(mu)
            if not m.any():
                continue
            thr = THRESHOLDS[pos]
            r = 1.0 / alpha
            p = r / (r + mu[m])
            out[m] = 1.0 - nbinom.cdf(thr - 1, r, p)
        return out


def p_threshold_poisson(d: pd.DataFrame, mu: np.ndarray) -> np.ndarray:
    out = np.zeros(len(d))
    for pos in ["DEF", "MID", "FWD"]:
        m = (d.position == pos).to_numpy() & np.isfinite(mu)
        if m.any():
            out[m] = 1.0 - poisson.cdf(THRESHOLDS[pos] - 1, mu[m])
    return out
