#!/usr/bin/env python3
"""Do chips pay, and how patient should the rule be?

THRESHOLD controls how much better than a typical gameweek a chip play must
look before it is taken. A low bar spends chips early and cheaply; a high bar
holds out for a big week and risks the window closing first. Swept together,
since all three chips share the same patience parameter.
"""
import pickle

import pandas as pd

import fpl.optimize.chips as chips
from fpl.backtest.season_sim import run_season_managed
from scripts.rank_sweep_mv import add_sd

LEVELS = [1.00, 1.15, 1.30, 1.45, 1.60, 1.80]


def main() -> None:
    pools = {s: add_sd(p) for s, p in
             pickle.load(open("data/features/sim_pools.pkl", "rb")).items()}
    seasons = list(pools)

    base = {s: run_season_managed(pools[s], "xpts", gamma=-0.05,
                                  chips=False).points.sum() for s in seasons}
    print("no-chip baselines:", {k: round(v) for k, v in base.items()}, flush=True)

    print(f'\n{"bar":>6}' + "".join(f"{s[-5:]:>10}" for s in seasons)
          + f'{"total":>10}{"seasons up":>12}', flush=True)
    rows = []
    for lv in LEVELS:
        chips.THRESHOLD = {k: lv for k in chips.THRESHOLD}
        gains = []
        for s in seasons:
            r = run_season_managed(pools[s], "xpts", gamma=-0.05, chips=True)
            gains.append(r.points.sum() - base[s])
        rows.append({"bar": lv, **dict(zip(seasons, gains)), "total": sum(gains)})
        print(f'{lv:6.2f}' + "".join(f"{g:+10.0f}" for g in gains)
              + f"{sum(gains):+10.0f}{sum(g > 0 for g in gains):12d}", flush=True)
    pd.DataFrame(rows).to_csv("data/features/chip_sweep.csv", index=False)


if __name__ == "__main__":
    main()
