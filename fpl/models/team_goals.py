"""Phase 2 -- Dixon-Coles bivariate Poisson team model.

One model, four downstream outputs: P(clean sheet), the full distribution of
goals conceded, projected team xG, and (with possession data) defensive action
volume.

Home and away goals are modelled as Poisson with a low-score dependence
correction:

    lambda = exp(attack_h - defence_a + home_adv)
    mu     = exp(attack_a - defence_h)

    tau(0,0) = 1 - lambda*mu*rho     tau(0,1) = 1 + lambda*rho
    tau(1,0) = 1 + mu*rho            tau(1,1) = 1 - rho

The tau correction exists because 0-0 and 1-1 are observed more often than
independent Poissons predict. Exponential time decay exp(-xi * days_ago)
downweights old matches; xi is tuned on out-of-sample log-loss rather than
assumed.

Clean sheets are read off as the Poisson zero, P(opponent scores 0), NOT as a
linear function of xGA -- that is the whole reason to fit a distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10


def _tau(x, y, lam, mu, rho):
    """Dixon-Coles low-score correction."""
    out = np.ones_like(lam, dtype=float)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    out[m00] = 1.0 - lam[m00] * mu[m00] * rho
    out[m01] = 1.0 + lam[m01] * rho
    out[m10] = 1.0 + mu[m10] * rho
    out[m11] = 1.0 - rho
    return np.clip(out, 1e-10, None)


@dataclass
class DixonColes:
    clubs: np.ndarray
    attack: np.ndarray
    defence: np.ndarray
    home_adv: float
    rho: float
    xi: float
    index: dict = field(default_factory=dict)

    # -- fitting ------------------------------------------------------------
    @classmethod
    def fit(cls, matches: pd.DataFrame, xi: float = 0.003,
            target: str = "goals", ref_time: pd.Timestamp | None = None,
            max_matches: int | None = 1140, warm_start: "DixonColes | None" = None) -> "DixonColes":
        """`matches` is one row per FIXTURE (home perspective).

        Columns required: home_code, away_code, home_goals, away_goals, kickoff_time.
        """
        m = matches.dropna(subset=["home_goals", "away_goals"]).copy()
        # With exponential decay, matches beyond ~3 seasons carry negligible
        # weight, so truncating the window is close to free and makes the fit
        # roughly an order of magnitude cheaper.
        if max_matches is not None and len(m) > max_matches:
            m = m.nlargest(max_matches, "kickoff_time")
        clubs = np.sort(pd.unique(np.concatenate([m.home_code, m.away_code])))
        idx = {c: i for i, c in enumerate(clubs)}
        n = len(clubs)

        hi = m.home_code.map(idx).to_numpy()
        ai = m.away_code.map(idx).to_numpy()
        x = m.home_goals.to_numpy(dtype=float)
        y = m.away_goals.to_numpy(dtype=float)

        ref = ref_time if ref_time is not None else m.kickoff_time.max()
        days = (ref - m.kickoff_time).dt.total_seconds().to_numpy() / 86400.0
        w = np.exp(-xi * np.clip(days, 0, None))

        # params: attack[n-1] (last is -sum for identifiability), defence[n], gamma, rho
        def unpack(p):
            atk = np.empty(n)
            atk[:-1] = p[:n - 1]
            atk[-1] = -atk[:-1].sum()
            dfn = p[n - 1:2 * n - 1]
            return atk, dfn, p[-2], p[-1]

        # Integer tau is only meaningful for goal counts. When fitting xG the
        # correction is dropped rather than applied to non-integers.
        use_tau = target == "goals"

        def nll(p):
            atk, dfn, gamma, rho = unpack(p)
            lam = np.exp(atk[hi] - dfn[ai] + gamma)
            mu = np.exp(atk[ai] - dfn[hi])
            lam = np.clip(lam, 1e-8, 30)
            mu = np.clip(mu, 1e-8, 30)
            ll = x * np.log(lam) - lam + y * np.log(mu) - mu
            if use_tau:
                ll = ll + np.log(_tau(x, y, lam, mu, rho))
            return -np.sum(w * ll)

        p0 = np.concatenate([np.zeros(n - 1), np.zeros(n), [0.25], [-0.05]])
        if warm_start is not None:
            # Reuse the previous week's estimates for clubs we have already seen.
            for c, i in idx.items():
                j = warm_start.index.get(c)
                if j is None:
                    continue
                if i < n - 1:
                    p0[i] = warm_start.attack[j]
                p0[n - 1 + i] = warm_start.defence[j]
            p0[-2], p0[-1] = warm_start.home_adv, warm_start.rho
        bounds = ([(-3, 3)] * (n - 1) + [(-3, 3)] * n + [(-1, 1)] +
                  [(-0.2, 0.2) if use_tau else (0.0, 0.0)])
        res = minimize(nll, p0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500})
        atk, dfn, gamma, rho = unpack(res.x)
        return cls(clubs=clubs, attack=atk, defence=dfn, home_adv=gamma,
                   rho=rho, xi=xi, index=idx)

    # -- prediction ---------------------------------------------------------
    def rates(self, home_code: int, away_code: int) -> tuple[float, float]:
        h, a = self.index[home_code], self.index[away_code]
        lam = float(np.exp(self.attack[h] - self.defence[a] + self.home_adv))
        mu = float(np.exp(self.attack[a] - self.defence[h]))
        return lam, mu

    def score_matrix(self, home_code: int, away_code: int,
                     max_goals: int = MAX_GOALS) -> np.ndarray:
        """Joint P(home=x, away=y), Dixon-Coles corrected and renormalised."""
        lam, mu = self.rates(home_code, away_code)
        g = np.arange(max_goals + 1)
        ph = poisson.pmf(g, lam)
        pa = poisson.pmf(g, mu)
        mat = np.outer(ph, pa)
        mat[0, 0] *= 1.0 - lam * mu * self.rho
        mat[0, 1] *= 1.0 + lam * self.rho
        mat[1, 0] *= 1.0 + mu * self.rho
        mat[1, 1] *= 1.0 - self.rho
        return mat / mat.sum()

    def outcome_probs(self, home_code: int, away_code: int) -> tuple[float, float, float]:
        m = self.score_matrix(home_code, away_code)
        return (float(np.tril(m, -1).sum()),   # home win
                float(np.trace(m)),            # draw
                float(np.triu(m, 1).sum()))    # away win

    def clean_sheet_probs(self, home_code: int, away_code: int) -> tuple[float, float]:
        """P(home keeps CS), P(away keeps CS) -- the Poisson zero, not a proxy."""
        m = self.score_matrix(home_code, away_code)
        return float(m[:, 0].sum()), float(m[0, :].sum())

    def conceded_dist(self, home_code: int, away_code: int) -> tuple[np.ndarray, np.ndarray]:
        """Full distribution of goals conceded by each side.

        Needed because the -1-per-2-conceded term is a step function, so the
        mean is not sufficient.
        """
        m = self.score_matrix(home_code, away_code)
        return m.sum(axis=0), m.sum(axis=1)  # home concedes (away goals), away concedes

    def strength_table(self) -> pd.DataFrame:
        return (pd.DataFrame({"club_code": self.clubs,
                              "attack": self.attack, "defence": self.defence})
                  .sort_values("attack", ascending=False).reset_index(drop=True))
