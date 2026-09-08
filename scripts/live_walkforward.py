#!/usr/bin/env python3
"""Score the running season honestly: project each gameweek without it.

The projection log was measuring a circle. `coldstart` folds played gameweeks
into the priors -- deliberately, because after a gameweek the team sheets are
the best evidence there is -- and `predict_horizon` then projects gameweeks 1
onward. So gameweek 1 was being projected by a model that already knew who
started gameweek 1, and the resulting "out-of-sample" accuracy was nothing of
the kind.

This restricts the live blend to gameweeks strictly before the one being
scored, which is the same walk-forward discipline the historical backtest uses
and which the live path had quietly escaped.

Gameweek 1 cannot be scored this way at all: there is nothing before it, so its
projection is the pure cold start. That is the honest version of the first
gameweek's number and it is reported as such.
"""
import sys

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import fpl.models.coldstart as cs
from fpl.predict_horizon import run

SEASON = "2026-27"


def played_gameweeks() -> list[int]:
    con = duckdb.connect("data/fpl.duckdb", read_only=True)
    gws = con.execute(f"SELECT DISTINCT gw FROM player_gw WHERE season='{SEASON}' "
                      "ORDER BY gw").df().gw.tolist()
    con.close()
    return [int(g) for g in gws]


def actuals() -> pd.DataFrame:
    con = duckdb.connect("data/fpl.duckdb", read_only=True)
    d = con.execute(f"SELECT gw, element, position, minutes, total_points "
                    f"FROM player_gw WHERE season='{SEASON}'").df()
    con.close()
    return d


def main() -> int:
    gws = played_gameweeks()
    if not gws:
        print("no completed gameweeks")
        return 0
    act = actuals()
    real_blend = cs.live_season
    rows = []

    for g in gws:
        # priors may see gameweeks strictly before g, and nothing else
        def restricted(db="data/fpl.duckdb", season=SEASON, _cut=g):
            d = real_blend(db, season)
            if d.empty:
                return d
            con = duckdb.connect(db, read_only=True)
            keep = con.execute(
                f"SELECT element, count(*) gws, sum(minutes) mins, "
                f"avg(CASE WHEN minutes>=60 THEN 1.0 ELSE 0.0 END) starts60, "
                f"sum(total_points) pts, sum(CASE WHEN minutes>0 THEN 1 ELSE 0 END) apps, "
                f"sum(defcon) defcon, sum(saves) saves "
                f"FROM player_gw WHERE season='{season}' AND gw < {_cut} "
                f"GROUP BY element").df()
            con.close()
            if keep.empty:
                return keep
            keep["code"] = keep["element"].map(cs._element_to_code())
            keep = keep.dropna(subset=["code"])
            keep["code"] = keep["code"].astype(int)
            keep["ppg_start"] = np.where(keep.apps > 0, keep.pts / keep.apps.clip(lower=1), np.nan)
            keep["dc_per90"] = keep.defcon / (keep.mins / 90).clip(lower=0.1)
            keep["save_per90"] = keep.saves / (keep.mins / 90).clip(lower=0.1)
            keep["position"] = keep["element"].map(
                dict(zip(act.element, act.position)))
            return keep

        cs.live_season = restricted
        try:
            # Project exactly the gameweek being scored, starting there. This
            # used to lean on run()'s default start, which was gameweek 1 --
            # once the horizon began at the next unplayed gameweek instead, the
            # projection covered gameweeks 4 onward while the scoring loop asked
            # for 1 to 3, and every join came back empty.
            _, by_gw = run(horizon=1, start=g, n_sims=4000)
        finally:
            cs.live_season = real_blend

        p = by_gw[by_gw.gw == g][["element", "xpts"]]
        m = p.merge(act[act.gw == g], on="element", how="inner")
        rows.append(m.assign(gw=g))
        print(f"  GW{g}: projected {m.xpts.sum():.0f}  actual {m.total_points.sum():.0f}  "
              f"ratio {m.total_points.sum()/max(m.xpts.sum(),1):.3f}", flush=True)

    d = pd.concat(rows, ignore_index=True)
    d.to_parquet("data/features/live_walkforward.parquet", index=False)
    print(f"\npooled n={len(d):,}  MAE {np.abs(d.xpts-d.total_points).mean():.3f}  "
          f"corr {np.corrcoef(d.xpts, d.total_points)[0,1]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
