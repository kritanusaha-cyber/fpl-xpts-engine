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
from fpl.models.price_curve import fit as fit_price_curve
from fpl.optimize.squad import optimise, load_rules

COMPONENTS = ["minutes", "goals", "assists", "clean_sheet", "defcon",
              "bonus", "saves", "conceded", "cards"]


def build() -> dict:
    gw1 = pd.read_parquet("data/features/gw1_projection.parquet")
    hz = pd.read_parquet("data/features/horizon_roles.parquet")
    runs_df = pd.read_parquet("data/features/horizon_by_gw.parquet")
    horizon_n = int(runs_df.gw.nunique())
    cfg = load_rules()

    # Horizon drives valuation: one gameweek cannot separate players.
    d = hz.rename(columns={"xpts_h": "xpts"}).copy()
    duals = squad_duals(d, cfg)
    lam, mu = duals["lambda_budget"], duals["mu"]

    d["surplus_pos"] = d["xpts"] - lam * d["price"] - d["position"].map(mu).fillna(0)
    # A "starter" must be BOTH likely to start and available. Using last
    # season's start rate alone counted 42 injured or unavailable players (12%
    # of the set) as starters projecting exactly 0.00 xPts. That polluted the
    # overpriced list -- the most "overpriced" players were simply injured --
    # and dragged the role replacement baseline down by up to 1.1 xPts.
    available = ~d["status"].isin(["i", "s", "u", "n"])
    # "Starter" = would plausibly make his club's XI, defined as being among the
    # top N at his club and position, where N is the slots a typical XI fills.
    # A flat probability threshold is arbitrary once squad-depth normalisation
    # has redistributed minutes: five Arsenal defenders sharing four slots all
    # sit near 0.5, so a 0.5 cut kept or dropped them essentially at random.
    XI_SLOTS = {"GKP": 1, "DEF": 4, "MID": 4, "FWD": 2}
    rank_in_club = (d.where(available)
                     .groupby(["club_code", "position"])["starts60"]
                     .rank(ascending=False, method="first"))
    starters = available & (rank_in_club <= d["position"].map(XI_SLOTS).fillna(1))
    base = (d[starters].groupby("role")["surplus_pos"].median()
              .rename("role_base").reset_index())
    d = d.merge(base, on="role", how="left")
    d["role_base"] = d["role_base"].fillna(d.groupby("role")["surplus_pos"].transform("median"))
    d["surplus_role"] = d["surplus_pos"] - d["role_base"]
    d["available"] = available

    # Fair price comes from the fitted market curve, NOT from surplus/lambda.
    # lambda is the marginal rate at a constrained optimum (1.11 xPts per GBP1m
    # over six gameweeks); the market's realised gradient for defenders is ~3.2.
    # Dividing by a number three times too small inflated every gap threefold and
    # gave a GBP4.5m defender a GBP13.3m fair price. The curve answers the
    # market question directly: at what price has FPL historically delivered this
    # rate of output, for this position?
    curve = fit_price_curve()
    d["ppg_proj"] = d["xpts"] / max(horizon_n, 1)
    d["fair_price"] = [curve.price_for(pos, v)
                       for pos, v in zip(d["position"], d["ppg_proj"])]
    d["mispricing"] = d["fair_price"] - d["price"]
    d["is_starter"] = starters
    d["role_rank"] = (d.where(starters).groupby("role")["surplus_role"]
                        .rank(ascending=False, method="min")).fillna(0).astype(int)
    d["role_n"] = d.groupby("role")["is_starter"].transform("sum").astype(int)

    # Shot-zone grids, keyed by FPL element via the stable code.
    grids = {}
    gp = Path("data/features/pitch_grids.parquet")
    if gp.exists():
        from fpl.resolve.players import resolve as _res
        from fpl.ingest.fbref import manual_overrides
        g = pd.read_parquet(gp)
        # Attach the source team id so the resolver can disambiguate on club.
        _st = pd.read_parquet("data/raw/fotmob/player_match_stats.parquet")
        g["team_id"] = g["player_id"].map(_st.groupby("player_id")["team_id"].first())
        g = _res(g, overrides=manual_overrides()).dropna(subset=["code"])
        g["code"] = g["code"].astype(int)
        code_to_el = dict(zip(d["code"], d["element"]))
        for _, r in g.drop_duplicates("code").iterrows():
            el = code_to_el.get(int(r["code"]))
            if el is None:
                continue
            grids[int(el)] = {
                "tier": [int(v) for v in r["grid_tier"]],
                "shots": int(r["shots"]),
                "z14": round(float(r["zone14_share"]), 3),
                "box": round(float(r["box_share"]), 3),
                "centre": round(float(r["centre_share"]), 3),
                "half": round(float(r["halfspace_share"]), 3),
                "wing": round(float(r["wing_share"]), 3),
            }

    # Goalkeeper shot-stopping, keyed by element via the stable code.
    keepers = {}
    kp = Path("data/features/keeper_stats.parquet")
    if kp.exists():
        kk = pd.read_parquet(kp)
        c2e = dict(zip(d["code"], d["element"]))
        for _, r in kk.iterrows():
            el = c2e.get(int(r["code"]))
            if el is None:
                continue
            keepers[int(el)] = {
                "faced": int(r["faced_np"]),
                "psxg": round(float(r["psxg_np"]), 1),
                "conceded": int(r["conceded_np"]),
                "prevented": round(float(r["goals_prevented"]), 2),
                "save_oe": round(float(r["save_pct_oe"]), 3),
                "save_pct": round(float(r["save_pct"]), 3),
                "matches": int(r["matches"]),
                # Quality of the shots he faces, which is a read on the defence
                # in front of him rather than on him.
                "psxg_per_sot": round(float(r["psxg_np"]) / max(int(r["faced_np"]), 1), 3),
                # GSAA measures him against the flat league save rate. Goals
                # prevented measures him against the shots he actually faced.
                # The gap between them is the defence.
                "gsaa": round(float(r["gsaa"]), 2),
                "diff_effect": round(float(r["difficulty_effect"]), 2),
                "lg_save": round(float(r["league_save_rate"]), 3),
                "thin": bool(r.get("low_sample", False)),
            }

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

    # Optimal squad over the horizon, under the rank objective rather than raw
    # expected points. gamma = -0.05 beat the template in four seasons of four
    # in walk-forward simulation; expected points alone won two. See rank.py.
    #
    # The parameter transfers from the weekly backtest to this H-gameweek
    # horizon without rescaling. Expected points and variance both scale
    # linearly in H for independent gameweeks, so the objective is H times the
    # weekly one and the argmax is unchanged.
    # Positional bias correction, applied to SELECTION only and never to the
    # displayed projection.
    #
    # The factors are worth +130 points across three held-out seasons, but they
    # are the wrong shape to use as a projection. Fitted on totals they are
    # dominated by the many near-zero rows, where actual-over-projected is
    # 1.45, and then applied to the few high projections where the true ratio
    # is 1.10. Applied to xpts they put the optimal squad at 64.1 points a
    # gameweek against a per-gameweek sum of 54.9 and an FPL average of 50 --
    # and they disagreed with the per-gameweek figures on the same page,
    # because those were left uncorrected.
    #
    # What the factors actually encode is that the model under-rates attacking
    # returns relative to defensive ones. That is a statement about which
    # players to pick, not about how many points they will score, so it belongs
    # in the objective beside the rank term and nowhere else.
    from fpl.models.position_calibration import season_factors
    _live = None
    _log = Path("data/features/projection_log.parquet")
    if _log.exists():
        try:
            _live = pd.read_parquet(_log)
        except Exception:
            _live = None
    _fac = season_factors(_live)

    from fpl.optimize.rank import RANK_GAMMA, rank_value_live
    d["rank_val"] = rank_value_live(d, RANK_GAMMA, sd_col="sd_h") \
                    * d["position"].map(_fac).fillna(1.0)
    r = optimise(d, cfg, xpts_col="rank_val")
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
            "fair": round(float(p.fair_price), 1), "ppg": round(float(p.ppg_proj), 2),
            "mis": round(float(p.mispricing), 1),
            "rank": int(p["role_rank"]), "role_n": int(p["role_n"]),
            "own": float(p.selected_by_percent or 0),
            "starter": bool(p.is_starter), "avail": bool(p.available),
            "hist_pl": bool(p.has_history),
            "fxg": (round(float(p.foreign_xg_share), 4)
                    if pd.notna(p.get("foreign_xg_share")) else None),
            "status": p.status,
            "squad": e in picked, "xi": e in xi, "capt": e in capt,
            "comp": comp.get(e, {}),
            "runs": runs.get(e, []),
            # Only midfielders and forwards render a shot grid; shipping one for
            # a keeper or a defender is dead weight in a payload this size.
            "grid": (grids.get(int(p.element))
                     if p.position in ("MID", "FWD") else None),
            "gk": keepers.get(int(p.element)),
            "zon": ({
                "six": round(float(p.six_yard_share), 3) if pd.notna(p.get("six_yard_share")) else None,
                "box_t": round(float(p.box_touches_p90), 2) if pd.notna(p.get("box_touches_p90")) else None,
                "cross": round(float(p.crosses_p90), 2) if pd.notna(p.get("crosses_p90")) else None,
                "ft": round(float(p.passes_ft_p90), 2) if pd.notna(p.get("passes_ft_p90")) else None,
                "sp": round(float(p.sp_share), 3) if pd.notna(p.get("sp_share")) else None,
                "thin": bool(p.get("zon_low_sample", False)),
                "mins": (int(p["zon_minutes"]) if pd.notna(p.get("zon_minutes")) else None),
            } if pd.notna(p.get("box_touches_p90")) else None),
            "drv": {
                "p60": round(float(p.starts60), 3),
                "xg_share": round(float(p.xg_share), 4),
                "xa_share": round(float(p.xa_share), 4),
                "dc90": round(float(p.dc_per90 or 0), 2),
                "n90": round(float(p.n90), 1),
                "pen": int(p.penalties_order) if pd.notna(p.penalties_order) else 0,
            },
        })
    # Percentile ranks within position, so a zonal bar reads as "high for a
    # defender" rather than as a raw number the reader must calibrate himself.
    zon_cols = ["six_yard_share", "box_touches_p90", "crosses_p90", "passes_ft_p90"]
    pct = {}
    for c in zon_cols:
        if c in d.columns:
            pct[c] = d.groupby("position")[c].rank(pct=True)
    for row in players:
        i = d.index[d["element"] == row["id"]]
        if row.get("zon") and len(i):
            row["zon"]["pct"] = {k: (round(float(v.loc[i[0]]), 2)
                                     if pd.notna(v.loc[i[0]]) else None)
                                 for k, v in pct.items()}

    # Touch heatmap zones. Attached last because it keys on the stable FPL code
    # rather than the element id, which is reassigned every season.
    hm = Path("data/features/heatmap_zones.parquet")
    if hm.exists():
        from fpl.ingest.fotmob_heatmap import POSITION_ZONES
        z = pd.read_parquet(hm)
        code_of = dict(zip(d["element"], d["code"])) if "code" in d.columns else {}
        pos_of = dict(zip(d["element"], d["position"]))
        z["pos"] = z["code"].map({v: pos_of.get(k) for k, v in code_of.items()})
        zc = [c for c in z.columns if c.startswith("z_")]
        for c in zc:
            z["p" + c[1:]] = z.groupby("pos")[c].rank(pct=True)
        z = z.set_index("code")
        for row in players:
            code = code_of.get(row["id"])
            if code is None or code not in z.index:
                continue
            r0 = z.loc[code]
            if isinstance(r0, pd.DataFrame):
                r0 = r0.iloc[0]
            keys = POSITION_ZONES.get(row["pos"], [])
            row["heat"] = {
                "n": int(r0["touches"]),
                "thin": bool(r0["low_sample"]),
                "x_mean": round(float(r0["x_mean"]), 1),
                "z": {k: [round(float(r0[f"z_{k}"]) * 100, 1),
                          (round(float(r0[f"p_{k}"]) * 100) if pd.notna(r0.get(f"p_{k}")) else None)]
                      for k in keys if f"z_{k}" in r0.index},
            }

    players.sort(key=lambda x: -x["xpts"])

    return {
        "generated": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
        "season": "2026/27", "horizon": int(runs_df.gw.nunique()), "n_sims": 8000,
        "lambda": round(lam, 3), "mu": {k: round(v, 2) for k, v in mu.items()},
        # The LP objective is now the rank value, which includes a variance
        # term. Report the squad's actual expected points instead -- calling
        # the objective "xPts" would overstate the projection by whatever the
        # variance term contributed.
        "squad_xpts": round(float(
            d.loc[d.element.isin(xi), "xpts"].sum()
            + d.loc[d.element.isin(capt), "xpts"].sum()), 1),
        "squad_obj": round(r["objective"], 1),
        "squad_spend": round(r["spend"], 1),
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
