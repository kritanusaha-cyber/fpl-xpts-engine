"""Phase 5 -- bonus points.

Bonus is 3/2/1 to the top three BPS scorers in each fixture, with ties sharing.
BPS itself is a deterministic function of the match action log, so in principle
bonus is exactly computable. In practice FPL stopped publishing most of the
action log after 2018/19: key passes, dribbles, crosses, fouls, big chances and
pass completion are all BPS inputs that the current API does not expose.

Measured consequence (2016/17, the last season with the full log):

    R2 0.9893, MAE 0.88   using every published action column
    R2 0.8967, MAE 2.60   using only the columns the API still provides

So the BPS weights recovered here are reduced-form: omitted correlated actions
inflate the coefficients on the actions we can see. They are fit for ranking
players within a fixture, which is what bonus depends on, and not for
reproducing the official table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

POSITIONS = ["GKP", "DEF", "MID", "FWD"]


def design(d: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=d.index)
    X["play_1_59"] = ((d.minutes > 0) & (d.minutes < 60)).astype(float)
    X["play_60"] = (d.minutes >= 60).astype(float)
    for p in POSITIONS:
        X[f"goal_{p}"] = d.goals_scored * (d.position == p)
    X["assist"] = d.assists
    for p in ["GKP", "DEF"]:
        X[f"cs_{p}"] = (d.clean_sheets * (d.position == p) * (d.minutes >= 60)).astype(float)
        X[f"gc_{p}"] = d.goals_conceded * (d.position == p)
    X["saves"] = d.saves
    X["pen_saved"] = d.penalties_saved
    X["pen_missed"] = d.penalties_missed
    X["yellow"] = d.yellow_cards
    X["red"] = d.red_cards
    X["og"] = d.own_goals
    for c in ["tackles", "clearances_blocks_interceptions", "recoveries"]:
        X[c] = pd.to_numeric(d[c], errors="coerce").fillna(0) if c in d.columns else 0.0
    return X.fillna(0.0).astype(float)


def fit_bps(train: pd.DataFrame) -> pd.Series:
    X = design(train)
    coef, *_ = np.linalg.lstsq(X.values, train.bps.astype(float).values, rcond=None)
    return pd.Series(coef, index=X.columns)


def predict_bps(d: pd.DataFrame, coef: pd.Series) -> np.ndarray:
    return design(d)[coef.index].values @ coef.values


def award_bonus(df: pd.DataFrame, bps_col: str) -> pd.Series:
    """Award 3/2/1 by BPS rank within each fixture, honouring FPL's tie rules.

    Ties at the top share the higher award and consume the lower slots: two
    players tied first both get 3 and the next gets 1; two tied second both
    get 2 and nobody gets 1.
    """
    out = pd.Series(0, index=df.index, dtype=int)
    for _, g in df.groupby(["season", "fixture"], observed=True):
        vals = g[bps_col].to_numpy()
        order = np.argsort(-vals)
        ranked = g.index[order]
        sortedv = vals[order]
        awarded, slot = {}, 0
        for points in (3, 2, 1):
            if slot >= len(sortedv):
                break
            top = sortedv[slot]
            tied = [i for i, v in zip(ranked[slot:], sortedv[slot:]) if v == top]
            for i in tied:
                awarded[i] = max(awarded.get(i, 0), points)
            slot += len(tied)
        for i, v in awarded.items():
            out.loc[i] = v
    return out
