"""Full-season squad simulation: the plan's second success criterion.

Per-gameweek accuracy says the projections are good. It does not say a manager
using them finishes well, because a season is a sequence of constrained decisions
-- a fixed budget, one free transfer a week, a four-point charge for the second,
and a squad that has to be carried through blanks.

This plays the season out. At each gameweek the manager sees only what had
happened by then, re-optimises, takes at most the transfers that are worth their
cost, and is scored on what his starting XI actually returned.

Three managers run side by side on identical information:
    engine     picks on the model's projections
    template   picks the most-owned players, which is what following the crowd
               produces and is the benchmark the plan names
    form       picks on recent points per game, the common heuristic

The comparison that matters is engine against template. Everything else is
context.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pulp

SQUAD = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}
BUDGET = 100.0
HIT_COST = 4.0


def pick_squad(pool: pd.DataFrame, pred: str, budget: float = BUDGET,
               locked: set | None = None, max_changes: int | None = None) -> set:
    """MILP squad selection, optionally constrained to stay near a current squad."""
    P = pool.reset_index(drop=True)
    idx = list(P.index)
    prob = pulp.LpProblem("squad", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", idx, cat="Binary")
    s = pulp.LpVariable.dicts("s", idx, cat="Binary")     # starting XI
    val = pd.to_numeric(P[pred], errors="coerce").fillna(0).to_dict()

    prob += pulp.lpSum(s[i] * val[i] for i in idx)
    prob += pulp.lpSum(x[i] for i in idx) == 15
    for pos, n in SQUAD.items():
        mem = [i for i in idx if P.position[i] == pos]
        prob += pulp.lpSum(x[i] for i in mem) == n
    prob += pulp.lpSum(x[i] * P.price[i] for i in idx) <= budget
    for club in P.club_code.dropna().unique():
        mem = [i for i in idx if P.club_code[i] == club]
        prob += pulp.lpSum(x[i] for i in mem) <= 3
    prob += pulp.lpSum(s[i] for i in idx) == 11
    for i in idx:
        prob += s[i] <= x[i]
    for pos in XI_MIN:
        mem = [i for i in idx if P.position[i] == pos]
        prob += pulp.lpSum(s[i] for i in mem) >= XI_MIN[pos]
        prob += pulp.lpSum(s[i] for i in mem) <= XI_MAX[pos]
    if locked is not None and max_changes is not None:
        keep = [i for i in idx if P.element[i] in locked]
        prob += pulp.lpSum(x[i] for i in keep) >= max(0, len(locked) - max_changes)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        return set()
    return {int(P.element[i]) for i in idx if x[i].value() and x[i].value() > 0.5}


def score_xi(pool: pd.DataFrame, squad: set, pred: str) -> float:
    """Points from the best legal XI within the squad, captain doubled.

    The XI is chosen on the PROJECTION (what the manager knew), then scored on
    what happened -- picking it on the outcome would be hindsight.
    """
    s = pool[pool.element.isin(squad)].copy()
    if s.empty:
        return 0.0
    # A prediction column can be entirely missing for a season (early seasons
    # have no ownership, so the template forecast is undefined). Fall back to
    # zero rather than crashing, so one benchmark's gap does not kill the run.
    s[pred] = pd.to_numeric(s[pred], errors="coerce").fillna(0.0)
    s = s.sort_values(pred, ascending=False)
    xi, counts = [], {k: 0 for k in XI_MAX}
    for _, r in s.iterrows():
        if len(xi) >= 11:
            break
        if counts[r.position] < XI_MAX[r.position]:
            xi.append(r); counts[r.position] += 1
    for pos, need in XI_MIN.items():
        while counts[pos] < need:
            cand = s[(s.position == pos) & (~s.element.isin([r.element for r in xi]))]
            if cand.empty:
                break
            xi.append(cand.iloc[0]); counts[pos] += 1
            xi = xi[:11]
    if not xi:
        return 0.0
    df = pd.DataFrame(xi).head(11)
    if df[pred].notna().sum() == 0:
        return float(df.total_points.sum())
    capt = df[pred].idxmax()
    return float(df.total_points.sum() + df.loc[capt, "total_points"])


def run_season(preds: pd.DataFrame, pred_col: str, free_transfers: int = 1,
               start_gw: int | None = None, rank_k: float = 0.0) -> pd.DataFrame:
    """Play the season with one free transfer a week and -4 for extras."""
    gws = sorted(preds.gw.dropna().unique())
    if start_gw:
        gws = [g for g in gws if g >= start_gw]
    squad, rows, banked = None, [], 0
    for gw in gws:
        pool = preds[preds.gw == gw].dropna(subset=["price", "position"]).copy()
        pool = pool.drop_duplicates("element")
        if pool.empty:
            continue
        # Rank mode: value a player by projection discounted for how much of the
        # field already holds him. Selection and captaincy both use it; scoring
        # still uses realised points, so nothing here peeks at the outcome.
        sel_col = pred_col
        if rank_k > 0:
            from fpl.optimize.rank import rank_value
            pool["_rank_val"] = rank_value(pool, rank_k, pred_col=pred_col)
            sel_col = "_rank_val"
        if squad is None:
            squad = pick_squad(pool, sel_col)
            hits = 0
        else:
            allowed = min(free_transfers + banked, 3)
            new = pick_squad(pool, sel_col, locked=squad, max_changes=allowed)
            if not new:
                new = squad
            changed = len(squad - new)
            hits = max(0, changed - (free_transfers + banked))
            banked = max(0, min(4, free_transfers + banked - changed))
            squad = new
        pts = score_xi(pool, squad, sel_col) - hits * HIT_COST
        rows.append({"gw": gw, "points": pts, "hits": hits, "squad_size": len(squad)})
    return pd.DataFrame(rows)
