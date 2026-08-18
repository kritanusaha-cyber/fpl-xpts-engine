"""Multi-gameweek projection over an H-gameweek horizon.

A single gameweek compresses everything: over one match the gap between the best
and worst pick is a couple of points, most of which is noise. The doc specifies
H = 5-8 for the optimiser precisely because that is the scale on which decisions
are actually made -- and it is also where genuine separation appears, since
fixture runs compound. A team with three home games against promoted sides pulls
away from one facing the top four.

Each gameweek is simulated independently and accumulated, which is correct for
the mean and slightly understates the variance (it ignores serial correlation in
form and rotation). Stated rather than hidden.
"""

from __future__ import annotations

import gzip
import glob
import json

import numpy as np
import pandas as pd

from fpl.models.assemble_fixture import simulate
from fpl.models.blend import score_matrix
from fpl.models.coldstart import build as build_coldstart, promoted_club_prior
from fpl.predict_gw1 import team_model, rates_for, DEFAULT_ALPHA

N_SIMS = 8_000
HORIZON = 6


def fixtures_for(gws: list[int]) -> pd.DataFrame:
    fx = pd.DataFrame(json.load(gzip.open(
        sorted(glob.glob("data/raw/snapshots/fixtures/date=*/*.json.gz"))[-1], "rt")))
    b = json.load(gzip.open(
        sorted(glob.glob("data/raw/snapshots/bootstrap/date=*/*.json.gz"))[-1], "rt"))
    codes = {t["id"]: t["code"] for t in b["teams"]}
    fx = fx[fx.event.isin(gws)].copy()
    fx["home_code"] = fx["team_h"].map(codes)
    fx["away_code"] = fx["team_a"].map(codes)
    return fx[["id", "event", "home_code", "away_code", "kickoff_time"]]


def prep_side(players: pd.DataFrame, code: int) -> pd.DataFrame:
    side = players[players.club_code == code].copy()
    side["p_60"] = side["starts60"].clip(0, 0.97)
    side["p_cameo"] = ((1 - side["p_60"]) * 0.45).clip(0, 0.5)
    side["dc_rate"] = side["dc_per90"].fillna(2.0)
    side["dc_alpha"] = side["position"].map(DEFAULT_ALPHA)
    side["save_per90"] = side["save_per90"].fillna(0.0)
    order = pd.to_numeric(side["penalties_order"], errors="coerce")
    side["pen_duty"] = np.where(order == 1, 1.0, np.where(order == 2, 0.15, 0.0))
    unavailable = side["status"].isin(["i", "s", "u"]) | (
        side["chance_of_playing_next_round"].fillna(100) < 25)
    side.loc[unavailable, ["p_60", "p_cameo"]] = 0.0
    return side


def run(horizon: int = HORIZON, n_sims: int = N_SIMS) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(11)
    players = build_coldstart()
    model, prior = team_model()
    gws = list(range(1, horizon + 1))
    fx = fixtures_for(gws)

    per_gw = []
    totals = {}          # element -> accumulated per-sim points
    for _, f in fx.iterrows():
        lam, mu = rates_for(model, prior, f.home_code, f.away_code)
        m = score_matrix(lam, mu, model.rho)
        h, a = prep_side(players, f.home_code), prep_side(players, f.away_code)
        if not len(h) or not len(a):
            continue
        ph, pa = simulate(h, a, m, n_sims, rng)
        for side, pts, own, opp in [(h, ph, lam, mu), (a, pa, mu, lam)]:
            for i, el in enumerate(side.element.to_numpy()):
                totals.setdefault(el, np.zeros(n_sims))
                totals[el] += pts[i]
            per_gw.append(pd.DataFrame({
                "element": side.element.to_numpy(), "gw": int(f.event),
                "xpts": pts.mean(axis=1), "sd": pts.std(axis=1),
                "team_goals": own, "opp_goals": opp,
                "opponent": f.away_code if own == lam else f.home_code,
                "is_home": own == lam,
            }))

    gw_df = pd.concat(per_gw, ignore_index=True)
    els = list(totals)
    arr = np.vstack([totals[e] for e in els])
    qs = [5, 10, 25, 50, 75, 90, 95]
    pct = np.percentile(arr, qs, axis=1)
    tot = pd.DataFrame({
        "element": els,
        "xpts_h": arr.mean(axis=1), "sd_h": arr.std(axis=1),
        **{f"q{q}": pct[i] for i, q in enumerate(qs)},
        "p10_h": pct[1], "p90_h": pct[5],
        "n_fixtures": gw_df.groupby("element").size().reindex(els).values,
    })
    # Coarse histogram of the simulated horizon total, for plotting the shape
    # rather than only its summary statistics -- the distributions are skewed
    # and a mean plus sd misrepresents them.
    edges = np.arange(0, 61, 4)
    tot["hist"] = [np.histogram(row, bins=edges)[0].tolist() for row in arr]
    tot["hist_edges"] = [edges.tolist()] * len(tot)
    tot = tot.merge(players, on="element", how="left")
    return tot, gw_df


if __name__ == "__main__":
    tot, gw = run()
    tot.to_parquet("data/features/horizon_projection.parquet", index=False)
    gw.to_parquet("data/features/horizon_by_gw.parquet", index=False)
    print(f"horizon H={HORIZON}: {len(tot)} players, {gw.gw.nunique()} gameweeks\n")
    print(f"  xPts spread over 1 GW  : {gw[gw.gw==1].xpts.max():.2f}")
    print(f"  xPts spread over H GWs : {tot.xpts_h.max():.2f}")
    print()
    print(tot.nlargest(10, "xpts_h")[
        ["web_name","club_name","position","price","xpts_h","sd_h","p10_h","p90_h","n_fixtures"]
    ].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
