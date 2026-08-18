"""Export the projection as a self-contained JSON payload for the dashboard.

'Implied value' is the LP reduced cost, not points-per-million:

    surplus = xPts - lambda_budget * price - mu_position

Both duals matter. lambda prices the budget; mu prices the positional quota,
because FPL forces a 2/5/5/3 squad and you are therefore never choosing a
forward against the whole market -- only against the other forwards you are
compelled to field.

Using lambda alone (as this first did) makes the metric structurally unfair
between positions: it charged forwards the budget cost of their price without
crediting that a forward slot must be filled. The result was 76% of goalkeepers
showing positive value against 1% of forwards, which said nothing about the
players and everything about the missing term.

Points-per-million has the opposite failure -- it flatters cheap players who
could never make the XI.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fpl.optimize.squad import optimise, load_rules
from fpl.optimize.duals import squad_duals

COMPONENTS = ["minutes", "goals", "assists", "clean_sheet", "defcon",
              "bonus", "saves", "conceded", "cards"]


def build(proj_path: str = "data/features/gw1_projection.parquet") -> dict:
    d = pd.read_parquet(proj_path)
    cfg = load_rules()

    duals = squad_duals(d, cfg)
    lam = duals["lambda_budget"]
    mu = duals["mu"]
    r = optimise(d, cfg)
    picked = set(r["squad"].element)
    xi = set(r["squad"][r["squad"].in_xi].element)
    capt = set(r["squad"][r["squad"].is_captain].element)

    d = d.copy()
    d["surplus_naive"] = d["xpts"] - lam * d["price"]
    d["surplus"] = d["surplus_naive"] - d["position"].map(mu).fillna(0.0)
    d["ppm"] = d["xpts"] / d["price"]
    d["in_squad"] = d.element.isin(picked)
    d["in_xi"] = d.element.isin(xi)
    d["is_captain"] = d.element.isin(capt)

    rows = []
    for _, p in d.iterrows():
        rows.append({
            "id": int(p.element),
            "name": p.web_name,
            "club": p.club_name,
            "pos": p.position,
            "price": round(float(p.price), 1),
            "xpts": round(float(p.xpts), 3),
            "sd": round(float(p.sd), 2),
            "surplus": round(float(p.surplus), 3),
            "surplus_naive": round(float(p.surplus_naive), 3),
            "ppm": round(float(p.ppm), 3),
            "haul": round(float(p.p_haul), 3),
            "blank": round(float(p.p_blank), 3),
            "own": float(p.selected_by_percent or 0),
            "comp": {c: round(float(p.get(f"c_{c}", 0.0)), 3) for c in COMPONENTS},
            "drv": {
                "p60": round(float(p.p_60), 3),
                "xg_share": round(float(p.xg_share), 4),
                "xa_share": round(float(p.xa_share), 4),
                "dc90": round(float(p.dc_per90 or 0), 2),
                "team_goals": round(float(p.fixture_lam), 2),
                "opp_goals": round(float(p.opp_lam), 2),
                "n90": round(float(p.n90), 1),
                "hist": bool(p.has_history),
                "fxg": (round(float(p.foreign_xg_share), 4)
                        if pd.notna(p.get("foreign_xg_share")) else None),
                "pen": int(p.penalties_order) if pd.notna(p.penalties_order) else 0,
            },
            "squad": bool(p.in_squad), "xi": bool(p.in_xi), "capt": bool(p.is_captain),
            "status": p.status,
        })

    # Rank within position, since that is the actual comparison being made:
    # you replace a defender with a defender, never with a forward.
    for pos in d["position"].unique():
        sub = [row for row in rows if row["pos"] == pos]
        for i, row in enumerate(sorted(sub, key=lambda x: -x["surplus"]), 1):
            row["pos_rank"] = i
            row["pos_n"] = len(sub)

    rows.sort(key=lambda x: -x["xpts"])
    return {
        "generated": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
        "gameweek": 1, "season": "2026/27",
        "shadow_price": round(lam, 3),
        "mu": {k: round(v, 3) for k, v in mu.items()},
        "squad_xpts": round(r["objective"], 2),
        "squad_spend": round(r["spend"], 1),
        "n_players": len(rows),
        "components": COMPONENTS,
        "players": rows,
    }


if __name__ == "__main__":
    data = build()
    Path("data/features/dashboard.json").write_text(json.dumps(data))
    print(f"exported {data['n_players']} players")
    print(f"  shadow price λ = {data['shadow_price']} xPts per £1.0m")
    print(f"  optimal squad: £{data['squad_spend']}m, {data['squad_xpts']} xPts")
    print(f"  positional duals: {data['mu']}")
    import collections
    pos_pct = collections.defaultdict(list)
    for p in data["players"]:
        pos_pct[p["pos"]].append(p["surplus"] > 0)
    for k, v in pos_pct.items():
        print(f"  {k}: {sum(v)}/{len(v)} positive ({sum(v)/len(v):.0%})")
