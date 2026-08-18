"""Fixture-level joint simulation, both teams together.

Simulating one team at a time cannot produce bonus, because bonus is a
competition across all 22+ players in the fixture. This module draws one
scoreline per simulation, plays out both squads against it, computes a
reduced-form BPS for every player in every draw, and awards 3/2/1 on the
resulting rank -- which is what makes the goal/bonus correlation the doc cares
about fall out structurally rather than being bolted on.

It also adds the two terms the team-at-a-time version omitted: goalkeeper save
points and bonus. Both were missing and both were biasing projections down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fpl.models.defcon import THRESHOLDS

P_TEAM_PENALTY = 0.121      # 92 penalties / (380 fixtures x 2 teams), 2025/26
PENALTY_CONVERSION = 0.79   # standard penalty conversion / xG value

GOAL_PTS = {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4}
CS_PTS = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}
DC_PTS = {"GKP": 0, "DEF": 2, "MID": 2, "FWD": 2}

# Reduced-form BPS weights, fitted in Phase 5. Used for RANKING within a
# fixture only -- they are not the official table (see FINDINGS.md).
BPS_W = {
    "play_60": 6.0, "play_1_59": 2.0, "assist": 11.0, "saves": 2.0,
    "yellow": -4.0, "og": -6.0,
    "goal": {"GKP": 12.0, "DEF": 15.0, "MID": 20.0, "FWD": 24.0},
    "cs": {"GKP": 10.0, "DEF": 13.0, "MID": 0.0, "FWD": 0.0},
    "gc": {"GKP": -5.0, "DEF": -4.0, "MID": 0.0, "FWD": 0.0},
    "defcon_action": 0.8,
}


def _play_side(side: pd.DataFrame, team_goals, conceded, n_sims, rng):
    """Draw minutes, goals, assists, CS, conceded, saves and DefCon for one team."""
    n = len(side)
    p60 = side["p_60"].to_numpy()[:, None]
    pcam = side["p_cameo"].to_numpy()[:, None]
    u = rng.random((n, n_sims))
    played_60 = u < p60
    cameo = (u >= p60) & (u < p60 + pcam)
    appeared = played_60 | cameo
    minutes = np.where(played_60, 75.0, np.where(cameo, 30.0, 0.0))
    mfrac = np.clip(minutes / 90.0, 0, 1)

    tg = np.broadcast_to(team_goals[None, :], (n, n_sims))
    share_g = np.clip(side["xg_share"].to_numpy()[:, None] * mfrac, 0, 1)
    share_a = np.clip(side["xa_share"].to_numpy()[:, None] * mfrac, 0, 1)
    goals = rng.binomial(tg, share_g)
    assists = rng.binomial(tg, share_a)

    # Penalties, modelled separately from open play. xg_share is now non-penalty
    # (penalty xG is stripped in coldstart), so the taker's penalty value has to
    # be added back explicitly -- and it attaches to whoever holds the duty NOW,
    # from the API's penalties_order, not to whoever took them last season.
    #   P(team wins a penalty) = 92 / (380*2) = 0.121 per team-match
    p_pen = P_TEAM_PENALTY * side["pen_duty"].to_numpy()[:, None] * (minutes > 0)
    pen_won = rng.random((n, n_sims)) < p_pen
    pen_scored = pen_won & (rng.random((n, n_sims)) < PENALTY_CONVERSION)
    goals = goals + pen_scored.astype(int)

    conc_on = rng.binomial(np.broadcast_to(conceded[None, :], (n, n_sims)), mfrac)
    cs = (conc_on == 0) & played_60

    pos = side["position"].to_numpy()
    is_gk = (pos == "GKP")[:, None]
    # Saves scale with shots faced; opponent goals are the observable proxy.
    save_rate = side["save_per90"].to_numpy()[:, None] * mfrac
    saves = np.where(is_gk, rng.poisson(np.clip(save_rate, 0, 12)), 0)

    dc_rate = np.clip(side["dc_rate"].to_numpy()[:, None] * mfrac, 1e-6, None)
    alpha = side["dc_alpha"].to_numpy()[:, None]
    r = 1.0 / np.clip(alpha, 1e-3, None)
    p = r / (r + dc_rate)
    dc_n = rng.negative_binomial(np.broadcast_to(r, (n, n_sims)), np.clip(p, 1e-6, 1 - 1e-9))
    thr = np.array([THRESHOLDS.get(x) or 999 for x in pos])[:, None]
    dc_hit = (dc_n >= thr) & appeared

    yellow = (rng.random((n, n_sims)) < (0.11 * mfrac)).astype(int)

    return dict(pos=pos, played_60=played_60, cameo=cameo, appeared=appeared,
                goals=goals, assists=assists, cs=cs, conc_on=conc_on,
                saves=saves, dc_n=dc_n, dc_hit=dc_hit, yellow=yellow)


def _bps(d) -> np.ndarray:
    pos = d["pos"]
    g = np.array([BPS_W["goal"][x] for x in pos])[:, None]
    c = np.array([BPS_W["cs"][x] for x in pos])[:, None]
    gc = np.array([BPS_W["gc"][x] for x in pos])[:, None]
    return (d["played_60"] * BPS_W["play_60"] + d["cameo"] * BPS_W["play_1_59"]
            + d["goals"] * g + d["assists"] * BPS_W["assist"]
            + d["cs"] * c + (d["conc_on"] // 2) * gc
            + d["saves"] * BPS_W["saves"]
            + d["dc_n"] * BPS_W["defcon_action"]
            + d["yellow"] * BPS_W["yellow"])


def _components(d) -> dict:
    """Points broken out by source, so a projection can be explained rather than
    just asserted. Every component is an (n_players, n_sims) array; they sum
    exactly to the total."""
    pos = d["pos"]
    gp = np.array([GOAL_PTS[x] for x in pos])[:, None]
    csp = np.array([CS_PTS[x] for x in pos])[:, None]
    dcp = np.array([DC_PTS[x] for x in pos])[:, None]
    is_def_gk = np.isin(pos, ["GKP", "DEF"])[:, None]
    is_gk = (pos[:, None] == "GKP")
    return {
        "minutes": np.where(d["played_60"], 2.0, np.where(d["cameo"], 1.0, 0.0)),
        "goals": d["goals"] * gp,
        "assists": d["assists"] * 3.0,
        "clean_sheet": d["cs"] * csp,
        "conceded": np.where(is_def_gk & d["appeared"], -(d["conc_on"] // 2), 0).astype(float),
        "saves": np.where(is_gk, d["saves"] // 3, 0).astype(float),
        "defcon": (d["dc_hit"] * dcp).astype(float),
        "cards": d["yellow"] * -1.0,
    }


def _points(d) -> np.ndarray:
    return sum(_components(d).values())


def simulate(home: pd.DataFrame, away: pd.DataFrame, score_matrix: np.ndarray,
             n_sims: int, rng: np.random.Generator,
             return_components: bool = False):
    """Returns (home_points, away_points), each (n_players, n_sims), bonus included."""
    flat = score_matrix.ravel() / score_matrix.sum()
    idx = rng.choice(len(flat), size=n_sims, p=flat)
    hg, ag = np.divmod(idx, score_matrix.shape[1])

    dh = _play_side(home, hg, ag, n_sims, rng)
    da = _play_side(away, ag, hg, n_sims, rng)

    pts_h, pts_a = _points(dh), _points(da)
    bps_all = np.vstack([_bps(dh), _bps(da)])
    # Bonus: 3/2/1 on BPS rank within the fixture, per simulation.
    order = np.argsort(-bps_all, axis=0)
    bonus = np.zeros_like(bps_all, dtype=float)
    rows = np.arange(bps_all.shape[1])
    for slot, pts in enumerate((3.0, 2.0, 1.0)):
        bonus[order[slot], rows] = pts

    nh = len(home)
    if not return_components:
        return pts_h + bonus[:nh], pts_a + bonus[nh:]

    comps = {}
    for side_name, dd, off, n in [("home", dh, 0, nh), ("away", da, nh, len(away))]:
        c = _components(dd)
        c["bonus"] = bonus[off:off + n]
        comps[side_name] = {k: v.mean(axis=1) for k, v in c.items()}
    return pts_h + bonus[:nh], pts_a + bonus[nh:], comps
