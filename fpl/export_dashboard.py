"""Export the projection as a self-contained JSON payload for the dashboard.

'Implied value' is the LP reduced cost, not points-per-million:

    surplus = xPts - lambda * price

where lambda is the budget shadow price recovered from the optimiser. This is
the doc's argument made concrete -- the correct price weighting is *derived*
from the budget constraint rather than assumed, and it moves week to week. A
positive surplus means the player earns his price at the current shadow price;
points-per-million, by contrast, systematically flatters cheap players who can
never actually make the starting XI.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fpl.optimize.squad import optimise, load_rules, budget_shadow_price

COMPONENTS = ["minutes", "goals", "assists", "clean_sheet", "defcon",
              "bonus", "saves", "conceded", "cards"]


def build(proj_path: str = "data/features/gw1_projection.parquet") -> dict:
    d = pd.read_parquet(proj_path)
    cfg = load_rules()

    lam = budget_shadow_price(d, cfg)
    r = optimise(d, cfg)
    picked = set(r["squad"].element)
    xi = set(r["squad"][r["squad"].in_xi].element)
    capt = set(r["squad"][r["squad"].is_captain].element)

    d = d.copy()
    d["surplus"] = d["xpts"] - lam * d["price"]
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
                "pen": int(p.penalties_order) if pd.notna(p.penalties_order) else 0,
            },
            "squad": bool(p.in_squad), "xi": bool(p.in_xi), "capt": bool(p.is_captain),
            "status": p.status,
        })

    rows.sort(key=lambda x: -x["xpts"])
    return {
        "generated": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
        "gameweek": 1, "season": "2026/27",
        "shadow_price": round(lam, 3),
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
    top = data["players"][:3]
    for p in top:
        print(f"  {p['name']:<14} xPts {p['xpts']:>5.2f}  surplus {p['surplus']:>+6.2f}")
