"""Walk-forward DefCon evaluation within 2025/26.

2025/26 is the only season played under the DefCon rule that also has the
component counts, so the model is trained GW1..t and tested on t+1 within it.
2016/17-2018/19 also carry the counts but predate the rule by seven seasons and
a considerable tactical shift, so they are used only as a prior check.

Tests three things the doc asserts:
  1. Negative binomial beats Poisson (counts are overdispersed).
  2. The threshold probability must be evaluated at projected minutes, not 90.
  3. Clean sheets and DefCon are negatively correlated across fixtures, so
     modelling them independently misprices the defender archetype.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from fpl.models.defcon import (NegBinDefCon, defcon_count, p_threshold_poisson,
                               THRESHOLDS)
from fpl.backtest.walkforward import brier, calibration

FEATURES = ["dc_ewm5", "dc_ewm10", "opp_strength", "is_home"]


def load() -> pd.DataFrame:
    con = duckdb.connect("data/fpl.duckdb")
    d = con.execute("""
        SELECT p.season, p.gw, p.element, p.fixture, p.position, p.minutes,
               p.tackles, p.clearances_blocks_interceptions, p.recoveries,
               p.was_home, p.kickoff_time, p.club_code_x AS club_code,
               t.club_code, t.xg_against, t.clean_sheet
        FROM player_gw p
        JOIN team_match t ON p.season=t.season AND p.fixture=t.fixture AND p.team_id=t.team_id
        WHERE p.minutes > 0 AND p.position IN ('DEF','MID','FWD')
          AND p.tackles IS NOT NULL
    """).df() if False else con.execute("""
        SELECT p.season, p.gw, p.element, p.fixture, p.position, p.minutes,
               p.tackles, p.clearances_blocks_interceptions, p.recoveries,
               p.was_home, p.kickoff_time, t.club_code, t.xg_against, t.clean_sheet
        FROM player_gw p
        JOIN team_match t ON p.season=t.season AND p.fixture=t.fixture AND p.team_id=t.team_id
        WHERE p.minutes > 0 AND p.position IN ('DEF','MID','FWD')
          AND p.tackles IS NOT NULL
    """).df()
    con.close()
    d["kickoff_time"] = pd.to_datetime(d["kickoff_time"], utc=True)
    d["defcon_n"] = defcon_count(d)
    d["threshold"] = d["position"].map(THRESHOLDS)
    d["hit"] = (d["defcon_n"] >= d["threshold"]).astype(int)
    d["dc_per90"] = d["defcon_n"] / d["minutes"].clip(lower=1) * 90
    d["is_home"] = d["was_home"].astype(float)
    return d.sort_values(["season", "element", "kickoff_time"]).reset_index(drop=True)


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby(["season", "element"], observed=True)["dc_per90"]
    for hl in (5, 10):
        d[f"dc_ewm{hl}"] = (g.shift(1)
                            .groupby([d["season"], d["element"]], observed=True)
                            .transform(lambda s: s.ewm(halflife=hl, adjust=False,
                                                       ignore_na=True).mean()))
    # Opponent strength proxy: xG conceded by this player's team in the fixture.
    # Real spec wants projected opponent possession share, which needs FBref.
    d["opp_strength"] = d["xg_against"].fillna(d["xg_against"].mean())
    return d


def main() -> None:
    d = add_features(load())
    cur = d[d.season == "2025-26"].copy()

    rows = []
    for gw in sorted(cur.gw.dropna().unique()):
        train = cur[cur.gw < gw].dropna(subset=FEATURES)
        test = cur[cur.gw == gw].dropna(subset=FEATURES)
        if len(train) < 1500 or not len(test):
            continue
        model = NegBinDefCon.fit(train, FEATURES)
        if not model.beta:
            continue
        t = test.copy()
        t["p_nb"] = model.p_threshold(test)
        mu = model.rate(test)
        t["p_pois"] = p_threshold_poisson(test, mu)
        # Same model evaluated at a flat 90 minutes, to isolate the minutes effect
        t["p_nb_at90"] = model.p_threshold(test, minutes=np.full(len(test), 90.0))
        rows.append(t)

    r = pd.concat(rows, ignore_index=True)
    y = r["hit"].to_numpy()
    print(f"walk-forward within 2025/26: {len(r):,} player-matches, "
          f"realised hit rate {y.mean():.3f}\n")

    print(f'{"":34}{"Brier":>9}{"pred":>8}{"real":>8}{"bias":>9}')
    for name, col in [("negative binomial @ projected", "p_nb"),
                      ("Poisson @ projected", "p_pois"),
                      ("negative binomial @ flat 90", "p_nb_at90"),
                      ("baseline: base rate", None)]:
        p = np.full(len(r), y.mean()) if col is None else r[col].to_numpy()
        print(f'{name:34}{brier(y,p):>9.4f}{p.mean():>8.3f}{y.mean():>8.3f}{p.mean()-y.mean():>+9.4f}')

    print("\nby position (negative binomial):")
    print(f'{"pos":6}{"n":>8}{"Brier":>9}{"pred":>8}{"real":>8}{"bias":>9}')
    for pos, g in r.groupby("position"):
        yy, pp = g.hit.to_numpy(), g.p_nb.to_numpy()
        print(f'{pos:6}{len(g):>8,}{brier(yy,pp):>9.4f}{pp.mean():>8.3f}{yy.mean():>8.3f}{pp.mean()-yy.mean():>+9.4f}')

    print("\nreliability (negative binomial):")
    print(calibration(y, r.p_nb.to_numpy(), bins=8).to_string(index=False))

    print("\nDefCon vs clean sheet -- the hedge the doc describes:")
    dd = r[r.position == "DEF"]
    print(f"  corr(P(DefCon), clean sheet)    {np.corrcoef(dd.p_nb, dd.clean_sheet)[0,1]:+.4f}")
    print(f"  corr(DefCon hit, clean sheet)   {np.corrcoef(dd.hit, dd.clean_sheet)[0,1]:+.4f}")
    print(f"  corr(opponent strength, hit)    {np.corrcoef(dd.opp_strength, dd.hit)[0,1]:+.4f}")


if __name__ == "__main__":
    main()
