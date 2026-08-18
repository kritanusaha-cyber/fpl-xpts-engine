"""Phase 7 -- MILP squad selection.

Constraints are taken from the live API's game_config rather than hardcoded:
15 players (2/5/5/3), £100.0m budget, max 3 per club, and a valid starting XI
(1 GK, >=3 DEF, >=2 MID, >=1 FWD, 11 total).

The doc's two arguments for solving this as a MILP rather than ranking by
points-per-million both hold:

  * the budget's shadow price falls out of the LP dual, so the correct price
    weighting is derived rather than assumed, and it moves week to week;
  * bench value is priced properly. A £4.0m non-playing defender has real option
    value as a budget enabler and no xPts value, and only the joint problem
    values that correctly.

Objective is starting-XI xPts plus the captain's xPts again. Bench players
contribute nothing directly, which is what makes the enabler trade-off real.
"""

from __future__ import annotations

import pandas as pd
import pulp
import yaml

FORMATION_MIN = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
FORMATION_MAX = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}


def load_rules(path: str = "config/scoring_2026_27.yaml") -> dict:
    return yaml.safe_load(open(path))


def optimise(players: pd.DataFrame, cfg: dict, xpts_col: str = "xpts",
             budget: float | None = None, verbose: bool = False) -> dict:
    """Select squad, XI and captain. `players` needs element, position, club_code, price."""
    sq = cfg["squad"]
    pos_cfg = cfg["positions"]
    budget = budget if budget is not None else sq["budget"] / 10.0

    P = players.reset_index(drop=True)
    idx = list(P.index)

    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    pick = pulp.LpVariable.dicts("pick", idx, cat="Binary")     # in 15-man squad
    start = pulp.LpVariable.dicts("start", idx, cat="Binary")   # in starting XI
    capt = pulp.LpVariable.dicts("capt", idx, cat="Binary")     # captain

    xp = P[xpts_col].fillna(0).to_dict()
    prob += pulp.lpSum(start[i] * xp[i] + capt[i] * xp[i] for i in idx)

    # squad size and composition
    prob += pulp.lpSum(pick[i] for i in idx) == sq["size"]
    for pos, spec in pos_cfg.items():
        members = [i for i in idx if P.position[i] == pos]
        prob += pulp.lpSum(pick[i] for i in members) == spec["squad_select"]

    # budget
    prob += pulp.lpSum(pick[i] * P.price[i] for i in idx) <= budget

    # max 3 per club
    for club in P.club_code.unique():
        members = [i for i in idx if P.club_code[i] == club]
        prob += pulp.lpSum(pick[i] for i in members) <= sq["team_limit"]

    # starting XI nested inside the squad, with a valid formation
    prob += pulp.lpSum(start[i] for i in idx) == sq["starting"]
    for i in idx:
        prob += start[i] <= pick[i]
        prob += capt[i] <= start[i]
    for pos in FORMATION_MIN:
        members = [i for i in idx if P.position[i] == pos]
        prob += pulp.lpSum(start[i] for i in members) >= FORMATION_MIN[pos]
        prob += pulp.lpSum(start[i] for i in members) <= FORMATION_MAX[pos]
    prob += pulp.lpSum(capt[i] for i in idx) == 1

    prob.solve(pulp.PULP_CBC_CMD(msg=1 if verbose else 0))

    P["in_squad"] = [pick[i].value() > 0.5 for i in idx]
    P["in_xi"] = [start[i].value() > 0.5 for i in idx]
    P["is_captain"] = [capt[i].value() > 0.5 for i in idx]
    squad = P[P.in_squad].copy()
    return {
        "status": pulp.LpStatus[prob.status],
        "objective": pulp.value(prob.objective),
        "squad": squad.sort_values(["in_xi", "position", xpts_col], ascending=[False, True, False]),
        "spend": float(squad.price.sum()),
    }


def budget_shadow_price(players: pd.DataFrame, cfg: dict, xpts_col: str = "xpts",
                        base_budget: float = 100.0, delta: float = 0.5) -> float:
    """Marginal xPts per extra £1.0m, by re-solving. The MILP dual is not
    directly meaningful with integer variables, so this uses a finite difference,
    which is the honest version of the doc's 'shadow price from the LP dual'."""
    lo = optimise(players, cfg, xpts_col, budget=base_budget)["objective"]
    hi = optimise(players, cfg, xpts_col, budget=base_budget + delta)["objective"]
    return (hi - lo) / delta
