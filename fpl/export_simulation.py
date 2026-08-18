"""Export the horizon simulation for the graphs dashboard.

Value is expressed three ways, because they answer different questions:

    surplus_pos   xPts_H - lambda*price - mu_position
                  "does he earn his price given the squad quota"
    surplus_role  the same, then centred within his ROLE cluster
                  "is he better than others doing the same job"
    fair_price    the price at which surplus_role would be zero
                  "what should he cost", in pounds rather than points

fair_price is the most legible of the three: a £6.0m player whose fair price is
£8.4m is underpriced by £2.4m, which is a sentence you can act on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fpl.optimize.duals import squad_duals
from fpl.optimize.squad import load_rules


def build() -> dict:
    tot = pd.read_parquet("data/features/horizon_roles.parquet")
    gw = pd.read_parquet("data/features/horizon_by_gw.parquet")
    cfg = load_rules()

    d = tot.rename(columns={"xpts_h": "xpts"}).copy()
    duals = squad_duals(d, cfg)
    lam, mu = duals["lambda_budget"], duals["mu"]

    d["surplus_pos"] = d["xpts"] - lam * d["price"] - d["position"].map(mu).fillna(0)

    # Centre within role against REPLACEMENT LEVEL, not the role median.
    # Most members of a role cluster are squad filler who will not play; taking
    # the median over all of them puts the baseline near zero and makes every
    # genuine starter look enormously underpriced (a 6.5m holding midfielder
    # came out with an 18.9m "fair price"). Replacement is the median among
    # players who project to actually start.
    starters = d["starts60"].fillna(0) >= 0.5
    base = (d[starters].groupby("role")["surplus_pos"].median()
              .rename("role_base").reset_index())
    d = d.merge(base, on="role", how="left")
    d["role_base"] = d["role_base"].fillna(d.groupby("role")["surplus_pos"].transform("median"))
    d["surplus_role"] = d["surplus_pos"] - d["role_base"]
    # Invert to a price. lambda is xPts per GBP1.0m, so surplus/lambda is the
    # pounds of price the player's output would justify.
    d["fair_price"] = (d["price"] + d["surplus_role"] / max(lam, 1e-6)).clip(3.5, 20.0)
    d["mispricing"] = d["fair_price"] - d["price"]

    # Rank and count only among plausible starters, for the same reason.
    d["is_starter"] = starters
    d["role_rank"] = (d.where(starters).groupby("role")["surplus_role"]
                        .rank(ascending=False, method="min"))
    d["role_rank"] = d["role_rank"].fillna(0).astype(int)
    d["role_n"] = d.groupby("role")["is_starter"].transform("sum").astype(int)

    gw = gw.copy()
    for c in ("xpts", "team_goals", "opp_goals"):
        gw[c] = gw[c].round(2)
    gw["is_home"] = gw["is_home"].astype(int)
    runs = {int(e): g.sort_values("gw")[["gw", "xpts", "team_goals", "opp_goals", "is_home"]]
                     .to_dict("records")
            for e, g in gw.groupby("element")}

    players = []
    for _, p in d.iterrows():
        players.append({
            "id": int(p.element), "name": p.web_name, "club": p.club_name,
            "pos": p.position, "role": p.role, "price": round(float(p.price), 1),
            "xpts": round(float(p.xpts), 2), "sd": round(float(p.sd_h), 2),
            "q": [round(float(p[f"q{q}"]), 1) for q in (5, 25, 50, 75, 95)],
            "hist": [int(x) for x in p["hist"]],
            "surplus": round(float(p.surplus_role), 2),
            "fair": round(float(p.fair_price), 1),
            "mis": round(float(p.mispricing), 1),
            "rank": int(p["role_rank"]), "role_n": int(p["role_n"]),
            "own": float(p.selected_by_percent or 0),
            "hist_pl": bool(p.has_history), "starter": bool(p["is_starter"]),
            "runs": runs.get(int(p.element), []),
        })
    players.sort(key=lambda x: -x["xpts"])

    return {
        "generated": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
        "horizon": int(gw.gw.nunique()), "n_sims": 8000,
        "lambda": round(lam, 3), "mu": {k: round(v, 2) for k, v in mu.items()},
        "hist_edges": [int(x) for x in tot.hist_edges.iloc[0]],
        "roles": sorted(d.role.unique().tolist()),
        "players": players,
    }


if __name__ == "__main__":
    data = build()
    Path("data/features/simulation.json").write_text(json.dumps(data))
    print(f"horizon H={data['horizon']}, {len(data['players'])} players")
    print(f"  lambda={data['lambda']}  roles={len(data['roles'])}")
    top = sorted(data["players"], key=lambda x: -x["mis"])[:6]
    print("\n  most underpriced (fair price - actual):")
    for p in top:
        print(f"    {p['name']:<14}{p['role']:<24}"
              f"£{p['price']:>4.1f}m -> £{p['fair']:>4.1f}m  ({p['mis']:+.1f})")
