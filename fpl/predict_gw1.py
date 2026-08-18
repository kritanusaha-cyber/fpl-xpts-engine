"""End-to-end GW1 2026/27 projection and squad.

Chains every phase: cold-start priors (Phase 0/1/3/4) -> Dixon-Coles team rates
carried across the season boundary (Phase 2) -> joint simulation (Phase 6) ->
MILP (Phase 7).

Everything here runs on priors alone, because 2026/27 has zero played matches.
That is the honest state of the world three days before the season starts, and
the projections should be read with that in mind.
"""

from __future__ import annotations

import gzip
import glob
import json

import numpy as np
import pandas as pd

from fpl.features.fixtures import fixture_frame
from fpl.models.team_goals import DixonColes
from fpl.models.assemble import summarise
from fpl.models.assemble_fixture import simulate
from fpl.models.coldstart import build as build_coldstart, promoted_club_prior
from fpl.optimize.squad import optimise, load_rules, budget_shadow_price

N_SIMS = 20_000
DEFAULT_ALPHA = {"GKP": 0.5, "DEF": 0.35, "MID": 0.45, "FWD": 0.6}


def gw1_fixtures() -> pd.DataFrame:
    path = sorted(glob.glob("data/raw/snapshots/fixtures/date=*/*.json.gz"))[-1]
    fx = pd.DataFrame(json.load(gzip.open(path, "rt")))
    boot = sorted(glob.glob("data/raw/snapshots/bootstrap/date=*/*.json.gz"))[-1]
    b = json.load(gzip.open(boot, "rt"))
    codes = {t["id"]: t["code"] for t in b["teams"]}
    fx = fx[fx.event == 1].copy()
    fx["home_code"] = fx["team_h"].map(codes)
    fx["away_code"] = fx["team_a"].map(codes)
    return fx[["id", "home_code", "away_code", "kickoff_time"]]


def team_model() -> tuple[DixonColes, dict]:
    """Fit on all completed history; club identity is the stable club_code."""
    f = fixture_frame()
    f = f.dropna(subset=["home_xg", "away_xg"]).assign(
        home_goals=lambda d: d.home_xg, away_goals=lambda d: d.away_xg)
    model = DixonColes.fit(f, xi=0.003, target="xg",
                           ref_time=pd.Timestamp.now(tz="UTC"))
    return model, promoted_club_prior()


def rates_for(model: DixonColes, prior: dict, home: int, away: int) -> tuple[float, float]:
    """Rates for a fixture, substituting the promoted-club prior where a club
    has no Premier League history at all."""
    league_mean = 1.45
    known_h, known_a = home in model.index, away in model.index
    if known_h and known_a:
        return model.rates(home, away)
    atk = lambda c: (np.exp(model.attack[model.index[c]]) if c in model.index
                     else prior["attack_ratio"])
    dfn = lambda c: (np.exp(-model.defence[model.index[c]]) if c in model.index
                     else prior["defence_ratio"])
    lam = league_mean * atk(home) * dfn(away) * np.exp(model.home_adv)
    mu = league_mean * atk(away) * dfn(home)
    return float(np.clip(lam, 0.2, 5)), float(np.clip(mu, 0.2, 5))


def main() -> None:
    rng = np.random.default_rng(7)
    players = build_coldstart()
    model, prior = team_model()
    fx = gw1_fixtures()

    from fpl.models.blend import score_matrix

    promoted = sorted(set(players.club_code) - set(model.index))
    if promoted:
        names = players[players.club_code.isin(promoted)].club_name.unique()
        print(f"promoted clubs on prior only: {', '.join(names)}\n")

    def prep(code):
        side = players[players.club_code == code].copy()
        side["p_60"] = side["starts60"].clip(0, 0.97)
        side["p_cameo"] = ((1 - side["p_60"]) * 0.45).clip(0, 0.5)
        side["dc_rate"] = side["dc_per90"].fillna(2.0)
        side["dc_alpha"] = side["position"].map(DEFAULT_ALPHA)
        side["save_per90"] = side["save_per90"].fillna(0.0)
        # Official penalty duty from the live API: order 1 is the designated
        # taker, 2 the backup. Better than inferring duty from last season.
        order = pd.to_numeric(side["penalties_order"], errors="coerce")
        side["pen_duty"] = np.where(order == 1, 1.0, np.where(order == 2, 0.15, 0.0))
        unavailable = side["status"].isin(["i", "s", "u"]) | (
            side["chance_of_playing_next_round"].fillna(100) < 25)
        side.loc[unavailable, ["p_60", "p_cameo"]] = 0.0
        return side

    out = []
    for _, f in fx.iterrows():
        lam, mu = rates_for(model, prior, f.home_code, f.away_code)
        m = score_matrix(lam, mu, model.rho)
        h, a = prep(f.home_code), prep(f.away_code)
        if not len(h) or not len(a):
            continue
        ph, pa = simulate(h, a, m, N_SIMS, rng)
        for side, pts, own, opp in [(h, ph, lam, mu), (a, pa, mu, lam)]:
            s = summarise(pts)
            s.index = side.index
            res = pd.concat([side, s], axis=1)
            res["fixture_lam"], res["opp_lam"] = own, opp
            out.append(res)

    proj = pd.concat(out, ignore_index=True)
    proj.to_parquet("data/features/gw1_projection.parquet", index=False)

    print(f"projected {len(proj)} players over {len(fx)} GW1 fixtures "
          f"({N_SIMS:,} sims each)\n")
    print("top 15 by xPts:")
    cols = ["web_name", "club_name", "position", "price", "xpts", "sd", "p_haul", "p_blank"]
    print(proj.nlargest(15, "xpts")[cols].to_string(index=False,
          float_format=lambda v: f"{v:.2f}"))

    cfg = load_rules()
    r = optimise(proj, cfg)
    print(f"\n=== optimal GW1 squad ===  status {r['status']}  "
          f"spend £{r['spend']:.1f}m  XI+capt xPts {r['objective']:.2f}")
    sq = r["squad"]
    for pos in ["GKP", "DEF", "MID", "FWD"]:
        g = sq[sq.position == pos]
        for _, p in g.iterrows():
            mark = "C" if p.is_captain else ("*" if p.in_xi else " ")
            print(f"  {mark} {pos}  {p.web_name:<16}{p.club_name:<16}"
                  f"£{p.price:>4.1f}m  {p.xpts:>5.2f}")
    sp = budget_shadow_price(proj, cfg)
    print(f"\nbudget shadow price: {sp:.3f} xPts per extra £1.0m")


if __name__ == "__main__":
    main()
