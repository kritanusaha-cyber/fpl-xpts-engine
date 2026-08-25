"""Rank optimisation: choosing a squad to finish high, not to score most.

THE SUBTLETY THAT DECIDES THE DESIGN

Write your margin over the field as a sum over every player in the game:

    margin = SUM_i (own_i - EO_i) * points_i

where own_i is 1 if you own him and EO_i is his effective ownership -- the
fraction of the field holding him, counting captaincy twice. Split it:

    margin = SUM_{owned} points_i  -  SUM_all EO_i * points_i

The second term does not depend on your choices. So **in expectation, maximising
rank is identical to maximising expected points.** Any claim that they differ has
to come from somewhere else, and it does: rank is a *non-linear* function of
margin. Finishing top 1% needs a large positive margin, not a positive one.

That is where ownership re-enters. A template squad's margin is pinned near zero
by construction -- you own what everyone owns, so your score tracks the field.
Differentials add variance to the margin, and variance is what buys access to the
upper tail. The cost is symmetric: the same variance can sink you.

So the objective is not "maximise points" and it is not "avoid template players".
It is: maximise a high quantile of the margin distribution. This module implements
a linear surrogate of that, tunable by one parameter:

    rank_value_i = xpts_i * (1 + k * (1 - EO_i))

k = 0 recovers pure expected points. k > 0 pays a premium for the same projected
points held by fewer rivals.

THE TILT LOSES. THE EXACT OBJECTIVE WINS, AND IN THE OTHER DIRECTION

Swept on four seasons, every k > 0 was worse than k = 0, and monotonically so.
Paying for scarcity destroys points.

Replacing the surrogate with the exact objective -- see `rank_value_mv` --
reversed the sign of the answer. Margin variance is linear in the squad choice
with weight (1 - 2*EO), and the best setting is **gamma = -0.05**: penalise
variance in players the field does not own, and *reward* it in players it does.

    season      2022-23  2023-24  2024-25  2025-26   total
    expected pts    -27      -77      +46      +69     +11   (2 of 4)
    gamma = -0.05   +78      +43      +49     +119    +289   (4 of 4)

Four seasons from four, on strictly pre-deadline ownership. The folk wisdom
that differentials win rank is backwards under these constraints. A blank from
a player nobody owns costs rank; a blank from a player everybody owns costs
nothing, because the field blanks with you. **Take risk where the field takes
it with you, and be conservative where you are alone.**

SHIPPED DEFAULT: gamma = -0.05, ownership lagged to the deadline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# A captained player counts twice toward the field's score, so effective
# ownership is ownership plus the captaincy share. Captaincy is not in the
# warehouse, so it is approximated from projected points: the field captains the
# highest-projected players it owns.
CAPTAIN_CONCENTRATION = 3.0


def effective_ownership(gw_pool: pd.DataFrame, own_col: str = "owned",
                        pred_col: str = "xpts") -> pd.Series:
    """EO as a fraction of the field, including the captaincy premium."""
    own = pd.to_numeric(gw_pool[own_col], errors="coerce").fillna(0.0)
    total = max(own.max(), 1.0)          # the most-owned player approximates the field size
    base = (own / total).clip(0, 1)

    # Captaincy concentrates on the best projected players. Model the field's
    # captain choice as a softmax over projection, restricted to what it owns.
    pred = pd.to_numeric(gw_pool[pred_col], errors="coerce").fillna(0.0)
    z = (pred - pred.max()) * CAPTAIN_CONCENTRATION
    w = np.exp(z) * base
    capt_share = w / max(w.sum(), 1e-9)

    return (base + capt_share).clip(0, 2.0)


def rank_value(gw_pool: pd.DataFrame, k: float, pred_col: str = "xpts",
               own_col: str = "owned") -> pd.Series:
    """Linear surrogate for a high quantile of the margin distribution.

    k = 0 is expected points. Larger k pays more for scarcity, which is what
    buys margin variance and therefore upper-tail rank.
    """
    eo = effective_ownership(gw_pool, own_col, pred_col)
    pred = pd.to_numeric(gw_pool[pred_col], errors="coerce").fillna(0.0)
    return pred * (1.0 + k * (1.0 - eo))


def margin_stats(squad_pts: np.ndarray, template_pts: np.ndarray) -> dict:
    """Distribution of your margin over the template, which is what rank reads."""
    d = squad_pts - template_pts
    return {
        "mean": float(d.mean()),
        "sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
        "p_beat": float((d > 0).mean()),
        "q90": float(np.percentile(d, 90)),
        "q10": float(np.percentile(d, 10)),
    }

# --- the exact objective, rather than a surrogate -------------------------

def margin_variance_weight(eo: pd.Series) -> pd.Series:
    """How much owning a player changes the VARIANCE of your margin.

    The margin over the field is SUM_i (own_i - EO_i) * points_i. Treating
    players as independent, its variance is

        Var = SUM_i (own_i - EO_i)^2 * var_i

    which looks quadratic in the decision and therefore outside a linear
    programme. It is not. own_i is binary, so own_i^2 = own_i, and

        (own_i - EO_i)^2 = own_i * (1 - 2*EO_i) + EO_i^2

    The second term is a constant. **Margin variance is linear in the squad
    choice**, with weight (1 - 2*EO_i), and the LP can carry it exactly.

    The sign is the interesting part. A player owned by more than half the field
    has a negative weight: holding him *reduces* your margin variance, because
    his hauls and blanks move you and the field together. A differential carries
    the full weight. This is the hedging structure of the game, stated exactly,
    and it is what the linear tilt was only gesturing at.
    """
    return 1.0 - 2.0 * eo


def rank_value_mv(gw_pool: pd.DataFrame, gamma: float, pred_col: str = "xpts",
                  sd_col: str = "sd", own_col: str = "owned") -> pd.Series:
    """Mean-variance objective on the margin: E[margin] + gamma * Var[margin].

    gamma = 0 is expected points. gamma > 0 buys variance, which is what gets a
    squad into the upper tail of the rank distribution; gamma < 0 sells it,
    which is what protects a lead.
    """
    eo = effective_ownership(gw_pool, own_col, pred_col)
    pred = pd.to_numeric(gw_pool[pred_col], errors="coerce").fillna(0.0)
    sd = pd.to_numeric(gw_pool.get(sd_col, 0.0), errors="coerce").fillna(0.0)
    return pred + gamma * margin_variance_weight(eo) * sd ** 2


def rank_value_plain_var(gw_pool: pd.DataFrame, gamma: float, pred_col: str = "xpts",
                         sd_col: str = "sd") -> pd.Series:
    """Control for the mean-variance objective: variance with no ownership term.

    `rank_value_mv` weights variance by (1 - 2*EO), which is the hedging claim:
    a template player's swings move you and the field together, so they are not
    really risk to your rank. This drops the weight and penalises variance flat.

    If the flat version does as well, the ownership structure is doing nothing
    and the honest claim is the duller one -- prefer consistent players, a
    statement about FPL scoring rather than about the field.
    """
    pred = pd.to_numeric(gw_pool[pred_col], errors="coerce").fillna(0.0)
    sd = pd.to_numeric(gw_pool.get(sd_col, 0.0), errors="coerce").fillna(0.0)
    return pred + gamma * sd ** 2


RANK_GAMMA = -0.05          # fitted on four seasons; see the module docstring


def rank_value_live(d: pd.DataFrame, gamma: float = RANK_GAMMA,
                    pred_col: str = "xpts", sd_col: str = "sd",
                    own_col: str = "selected_by_percent") -> pd.Series:
    """The shipped objective, for the live squad rather than the backtest.

    Differs from `rank_value_mv` only in where ownership comes from. The
    backtest carries a `selected` headcount; the live feed publishes ownership
    already as a percentage of entries, so it needs dividing rather than
    normalising against the most-owned player.

    Captaincy is added the same way: the field concentrates its armband on the
    best projected players it already owns, so the premium is a softmax over
    projection weighted by ownership.
    """
    if own_col not in d.columns:
        raise KeyError(f"{own_col!r} not in frame; rank objective needs ownership")
    own = pd.to_numeric(d[own_col], errors="coerce").fillna(0.0) / 100.0
    pred = pd.to_numeric(d[pred_col], errors="coerce").fillna(0.0)
    if sd_col not in d.columns:
        raise KeyError(f"{sd_col!r} not in frame; rank objective needs a spread")
    sd = pd.to_numeric(d[sd_col], errors="coerce").fillna(0.0)

    z = (pred - pred.max()) * CAPTAIN_CONCENTRATION
    w = np.exp(z) * own
    eo = (own + w / max(w.sum(), 1e-9)).clip(0, 2.0)

    return pred + gamma * margin_variance_weight(eo) * sd ** 2
