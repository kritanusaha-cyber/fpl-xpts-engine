"""Single payload for the combined dashboard.

Merges the two views that were separate pages:

  * the GW1 projection, whose value is the per-component breakdown -- what a
    given number of expected points is actually made of;
  * the six-gameweek simulation, whose value is the distribution, the fixture
    run, and enough separation between players for pricing to mean anything.

They belong together: the component breakdown explains *why* a player projects
well, and the horizon explains whether that survives his fixtures. Splitting
them across two pages meant answering half a question in each.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fpl.optimize.duals import squad_duals
from fpl.optimize.squad import optimise, load_rules

COMPONENTS = ["minutes", "goals", "assists", "clean_sheet", "defcon",
              "bonus", "saves", "conceded", "cards"]


def build() -> dict:
    gw1 = pd.read_parquet("data/features/gw1_projection.parquet")
    hz = pd.read_parquet("data/features/horizon_roles.parquet")
    runs_df = pd.read_parquet("data/features/horizon_by_gw.parquet")
    cfg = load_rules()

    # Horizon drives valuation: one gameweek cannot separate players.
    d = hz.rename(columns={"xpts_h": "xpts"}).copy()
    duals = squad_duals(d, cfg)
    lam, mu = duals["lambda_budget"], duals["mu"]

    d["surplus_pos"] = d["xpts"] - lam * d["price"] - d["position"].map(mu).fillna(0)
    starters = d["starts60"].fillna(0) >= 0.5
    base = (d[starters].groupby("role")["surplus_pos"].median()
              .rename("role_base").reset_index())
    d = d.merge(base, on="role", how="left")
    d["role_base"] = d["role_base"].fillna(d.groupby("role")["surplus_pos"].transform("median"))
    d["surplus_role"] = d["surplus_pos"] - d["role_base"]
    d["fair_price"] = (d["price"] + d["surplus_role"] / max(lam, 1e-6)).clip(3.5, 20.0)
    d["mispricing"] = d["fair_price"] - d["price"]
    d["is_starter"] = starters
    d["role_rank"] = (d.where(starters).groupby("role")["surplus_role"]
                        .rank(ascending=False, method="min")).fillna(0).astype(int)
    d["role_n"] = d.groupby("role")["is_starter"].transform("sum").astype(int)

    # GW1 component breakdown, keyed by element
    g1 = gw1.set_index("element")
    comp = {int(e): {c: round(float(g1.loc[e].get(f"c_{c}", 0.0)), 2) for c in COMPONENTS}
            for e in g1.index}
    gw1_xpts = g1["xpts"].to_dict()

    runs_df = runs_df.copy()
    for c in ("xpts", "team_goals", "opp_goals"):
        runs_df[c] = runs_df[c].round(2)
    runs_df["is_home"] = runs_df["is_home"].astype(int)
    runs = {int(e): g.sort_values("gw")[["gw", "xpts", "team_goals", "opp_goals", "is_home"]]
                     .to_dict("records") for e, g in runs_df.groupby("element")}

    # Optimal squad over the horizon
    r = optimise(d, cfg)
    picked = set(r["squad"].element)
    xi = set(r["squad"][r["squad"].in_xi].element)
    capt = set(r["squad"][r["squad"].is_captain].element)

    players = []
    for _, p in d.iterrows():
        e = int(p.element)
        players.append({
            "id": e, "name": p.web_name, "club": p.club_name,
            "pos": p.position, "role": p.role, "price": round(float(p.price), 1),
            "xpts": round(float(p.xpts), 2), "sd": round(float(p.sd_h), 2),
            "gw1": round(float(gw1_xpts.get(e, 0.0)), 2),
            "q": [round(float(p[f"q{q}"]), 1) for q in (5, 25, 50, 75, 95)],
            "hist": [int(x) for x in p["hist"]],
            "surplus": round(float(p.surplus_role), 2),
            "fair": round(float(p.fair_price), 1),
            "mis": round(float(p.mispricing), 1),
            "rank": int(p["role_rank"]), "role_n": int(p["role_n"]),
            "own": float(p.selected_by_percent or 0),
            "starter": bool(p.is_starter),
            "hist_pl": bool(p.has_history),
            "fxg": (round(float(p.foreign_xg_share), 4)
                    if pd.notna(p.get("foreign_xg_share")) else None),
            "status": p.status,
            "squad": e in picked, "xi": e in xi, "capt": e in capt,
            "comp": comp.get(e, {}),
            "runs": runs.get(e, []),
            "drv": {
                "p60": round(float(p.starts60), 3),
                "xg_share": round(float(p.xg_share), 4),
                "xa_share": round(float(p.xa_share), 4),
                "dc90": round(float(p.dc_per90 or 0), 2),
                "n90": round(float(p.n90), 1),
                "pen": int(p.penalties_order) if pd.notna(p.penalties_order) else 0,
            },
        })
    players.sort(key=lambda x: -x["xpts"])

    return {
        "generated": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
        "season": "2026/27", "horizon": int(runs_df.gw.nunique()), "n_sims": 8000,
        "lambda": round(lam, 3), "mu": {k: round(v, 2) for k, v in mu.items()},
        "squad_xpts": round(r["objective"], 1), "squad_spend": round(r["spend"], 1),
        "hist_edges": [int(x) for x in hz.hist_edges.iloc[0]],
        "roles": sorted(d.role.unique().tolist()),
        "components": COMPONENTS,
        "n_players": len(players),
        "players": players,
    }


if __name__ == "__main__":
    data = build()
    Path("data/features/combined.json").write_text(json.dumps(data))
    print(f"{data['n_players']} players, H={data['horizon']}, "
          f"lambda={data['lambda']}, {len(data['roles'])} roles")
    print(f"  optimal squad {data['squad_xpts']} xPts over H, "
          f"spend GBP{data['squad_spend']}m")
    print(f"  payload {len(json.dumps(data))//1024} KB")
