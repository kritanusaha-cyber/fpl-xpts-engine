"""Phase 6 -- assemble components into an xPts DISTRIBUTION.

The doc is emphatic that this step must simulate rather than sum, and it is
right for a specific reason: the components are correlated. A clean sheet and a
low goals-conceded count are the same event. A goal and bonus co-occur. Summing
marginal expectations reproduces the mean but destroys the variance, and
variance is exactly what captaincy and rank-optimisation need.

The simulation is organised per fixture so that correlation is preserved
structurally rather than modelled:

    1. draw a scoreline from the (blended) score matrix -- ONE draw per fixture
    2. draw each player's minutes class from the Phase 1 probabilities
    3. draw goals/assists conditional on the team's drawn goals and minutes
    4. clean sheets and goals conceded read off the SAME drawn scoreline
    5. draw DefCon counts from the Phase 4 negative binomial at drawn minutes
    6. score every draw with the validated scoring engine

Because steps 3-5 all condition on the draws in steps 1-2, the joint structure
falls out without needing a copula.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fpl.models.defcon import THRESHOLDS

GOAL_PTS = {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4}
CS_PTS = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}
DC_PTS = {"GKP": 0, "DEF": 2, "MID": 2, "FWD": 2}


def simulate_fixture(players: pd.DataFrame, score_matrix: np.ndarray,
                     is_home: bool, n_sims: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate `n_sims` realisations of points for every player in one team.

    `players` needs: position, p_60, p_cameo, xg_share, xa_share, dc_rate, dc_alpha.
    Returns an (n_players, n_sims) array of points.
    """
    n = len(players)
    if n == 0:
        return np.zeros((0, n_sims))

    # --- 1. one scoreline draw per simulation -----------------------------
    flat = score_matrix.ravel()
    flat = flat / flat.sum()
    idx = rng.choice(len(flat), size=n_sims, p=flat)
    hg, ag = np.divmod(idx, score_matrix.shape[1])
    team_goals = hg if is_home else ag
    conceded = ag if is_home else hg

    # --- 2. minutes class --------------------------------------------------
    p60 = players["p_60"].to_numpy()[:, None]
    pcam = players["p_cameo"].to_numpy()[:, None]
    u = rng.random((n, n_sims))
    played_60 = u < p60
    cameo = (u >= p60) & (u < p60 + pcam)
    appeared = played_60 | cameo
    minutes = np.where(played_60, 75.0, np.where(cameo, 30.0, 0.0))

    pos = players["position"].to_numpy()

    # --- 3. goals and assists, conditional on the drawn team goals ---------
    # Each team goal is allocated to a scorer with probability = his xG share,
    # scaled by whether he was on the pitch. This keeps sum(player goals) tied
    # to the drawn scoreline instead of drifting away from it.
    share_g = players["xg_share"].to_numpy()[:, None] * (minutes / 90.0)
    share_a = players["xa_share"].to_numpy()[:, None] * (minutes / 90.0)
    tg = team_goals[None, :]
    goals = rng.binomial(np.broadcast_to(tg, (n, n_sims)), np.clip(share_g, 0, 1))
    assists = rng.binomial(np.broadcast_to(tg, (n, n_sims)), np.clip(share_a, 0, 1))

    # --- 4. clean sheet / conceded read off the SAME scoreline -------------
    # Goals conceded are only charged for time actually on the pitch: FPL docks
    # -1 per 2 conceded WHILE PLAYING, so a player who did not appear must score
    # zero here, and a cameo should not be charged for the full match.
    conc_full = np.broadcast_to(conceded[None, :], (n, n_sims))
    conc_on = rng.binomial(conc_full, np.clip(minutes / 90.0, 0, 1))
    cs = (conc_on == 0) & played_60
    conc_pts = np.where(appeared, -(conc_on // 2), 0)

    # --- 5. DefCon at the DRAWN minutes ------------------------------------
    dc_rate = players["dc_rate"].to_numpy()[:, None] * (minutes / 90.0)
    alpha = players["dc_alpha"].to_numpy()[:, None]
    r = 1.0 / np.clip(alpha, 1e-3, None)
    p = r / (r + np.clip(dc_rate, 1e-6, None))
    dc_n = rng.negative_binomial(np.broadcast_to(r, (n, n_sims)), np.clip(p, 1e-6, 1 - 1e-9))
    thr = np.array([THRESHOLDS.get(x) or 999 for x in pos])[:, None]
    dc_hit = (dc_n >= thr) & appeared

    # --- 6. score ----------------------------------------------------------
    gp = np.array([GOAL_PTS[x] for x in pos])[:, None]
    csp = np.array([CS_PTS[x] for x in pos])[:, None]
    dcp = np.array([DC_PTS[x] for x in pos])[:, None]
    is_def_gk = np.isin(pos, ["GKP", "DEF"])[:, None]

    pts = (np.where(played_60, 2.0, np.where(cameo, 1.0, 0.0))
           + goals * gp + assists * 3
           + cs * csp
           + np.where(is_def_gk, conc_pts, 0)
           + dc_hit * dcp)
    return pts


def summarise(pts: np.ndarray) -> pd.DataFrame:
    """Mean, sd and tail quantiles -- captaincy needs the upside, not just xPts."""
    return pd.DataFrame({
        "xpts": pts.mean(axis=1),
        "sd": pts.std(axis=1),
        "p10": np.percentile(pts, 10, axis=1),
        "p50": np.percentile(pts, 50, axis=1),
        "p90": np.percentile(pts, 90, axis=1),
        "p_haul": (pts >= 10).mean(axis=1),
        "p_blank": (pts <= 2).mean(axis=1),
    })
