"""Recover the constraint duals that price a player correctly.

A single budget shadow price is not enough. FPL forces a squad of 2 GKP / 5 DEF
/ 5 MID / 3 FWD, so the positional quota is a binding constraint with its own
dual, and the honest reduced cost is

    surplus = xPts - lambda_budget * price - mu_position

Dropping mu makes the metric structurally unfair between positions: you MUST buy
three forwards, so the relevant question is never "is this forward worth his
price against the whole market" but "is he worth it against the other forwards
you are forced to choose from". Without mu, cheap mandatory positions (keepers)
all look like bargains and expensive ones (forwards) all look like traps -- which
is an artefact of the metric, not a property of the players.

Duals come from the LP relaxation. The integer program has no meaningful dual, so
we relax the binaries to [0,1] and read the shadow prices off that; for a problem
this size the relaxation is tight enough for the prices to be informative.
"""

from __future__ import annotations

import pandas as pd
import pulp

FORMATION_MIN = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}


def squad_duals(players: pd.DataFrame, cfg: dict, xpts_col: str = "xpts") -> dict:
    """LP-relaxation duals for the budget, positional quota and club limit."""
    sq, pos_cfg = cfg["squad"], cfg["positions"]
    P = players.reset_index(drop=True)
    idx = list(P.index)
    xp = P[xpts_col].fillna(0).to_dict()

    prob = pulp.LpProblem("fpl_lp", pulp.LpMaximize)
    pick = pulp.LpVariable.dicts("pick", idx, lowBound=0, upBound=1, cat="Continuous")

    # Objective counts only the 11 that start, approximated in the relaxation by
    # selecting 15 and weighting -- so we solve the XI directly instead.
    prob += pulp.lpSum(pick[i] * xp[i] for i in idx)

    c_budget = pulp.LpConstraint(
        pulp.lpSum(pick[i] * P.price[i] for i in idx), sense=-1,
        name="budget", rhs=sq["budget"] / 10.0)
    prob += c_budget

    for pos, spec in pos_cfg.items():
        members = [i for i in idx if P.position[i] == pos]
        prob += pulp.LpConstraint(pulp.lpSum(pick[i] for i in members), sense=0,
                                  name=f"quota_{pos}", rhs=spec["squad_select"])

    for club in P.club_code.unique():
        members = [i for i in idx if P.club_code[i] == club]
        prob += pulp.LpConstraint(pulp.lpSum(pick[i] for i in members), sense=-1,
                                  name=f"club_{int(club)}", rhs=sq["team_limit"])

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    duals = {n: c.pi for n, c in prob.constraints.items() if c.pi is not None}
    return {
        "status": pulp.LpStatus[prob.status],
        "lambda_budget": duals.get("budget", 0.0),
        "mu": {pos: duals.get(f"quota_{pos}", 0.0) for pos in pos_cfg},
        "club": {k: v for k, v in duals.items() if k.startswith("club_") and abs(v) > 1e-9},
    }


def add_surplus(players: pd.DataFrame, d: dict) -> pd.DataFrame:
    """Attach both the naive and the position-adjusted reduced cost."""
    p = players.copy()
    lam = d["lambda_budget"]
    p["surplus_naive"] = p["xpts"] - lam * p["price"]
    p["surplus"] = p["surplus_naive"] - p["position"].map(d["mu"]).fillna(0.0)
    return p
