#!/usr/bin/env python3
"""Sweep the margin mean-variance parameter across four seasons.

The earlier sweep tested a linear tilt that pays a premium for scarcity. This
tests the exact objective instead: E[margin] + gamma * Var[margin], where the
variance weight (1 - 2*EO) is negative for template players and positive for
differentials.

Negative gamma is included deliberately. If buying variance does not work, the
symmetric question is whether selling it does -- a squad that hugs the field
more tightly than expected points alone would choose.
"""
import pickle

import numpy as np
import pandas as pd

from fpl.backtest.season_sim import run_season

GAMMAS = [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.40]


def add_sd(d: pd.DataFrame) -> pd.DataFrame:
    """Per-player point standard deviation, from his own past only.

    Shifted before expanding, so a gameweek never contributes to the spread
    used to pick it. Players with too little history fall back to the
    positional spread, which is what an unknown player's variance actually is.
    """
    d = d.sort_values(["element", "gw"]).copy()
    g = d.groupby("element")["total_points"]
    d["sd"] = g.transform(lambda s: s.shift(1).expanding(min_periods=4).std())
    pos = d.groupby("position")["total_points"].transform("std")
    d["sd"] = d["sd"].fillna(pos).fillna(2.0)
    return d


def main() -> None:
    pools = {s: add_sd(p) for s, p in
             pickle.load(open("data/features/sim_pools.pkl", "rb")).items()}
    seasons = list(pools)
    tem = {s: run_season(pools[s], "pred_template").points.values for s in seasons}
    print("template baselines done", flush=True)

    hdr = f'{"gamma":>7}' + "".join(f"{s[-5:]:>10}" for s in seasons) + f'{"total":>10}{"wins":>7}'
    print(hdr, flush=True)
    rows = []
    for gm in GAMMAS:
        per, tot, wins = {}, 0.0, 0
        for s in seasons:
            r = run_season(pools[s], "xpts", gamma=gm).points.values
            d = float(r.sum() - tem[s].sum())
            per[s] = d
            tot += d
            wins += d > 0
        rows.append({"gamma": gm, **per, "total": tot, "wins": wins})
        print(f'{gm:7.2f}' + "".join(f"{per[s]:+10.0f}" for s in seasons)
              + f"{tot:+10.0f}{wins:7d}", flush=True)
    pd.DataFrame(rows).to_csv("data/features/rank_sweep_mv.csv", index=False)


if __name__ == "__main__":
    main()
