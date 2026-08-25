#!/usr/bin/env python3
"""Leave-one-season-out on the margin variance parameter.

gamma = -0.05 was chosen by sweeping four seasons and reading off the best
total. That is fitting on the test set. The honest question is whether a gamma
chosen without seeing a season still wins that season.

For each held-out season, gamma is picked on the other three and applied to the
one withheld. If the choice is stable the picks agree; if -0.05 is an artefact
of one good season, the held-out results will say so.
"""
import pickle

import numpy as np
import pandas as pd

from fpl.backtest.season_sim import run_season
from scripts.rank_sweep_mv import add_sd

GAMMAS = [-0.20, -0.15, -0.10, -0.05, -0.02, 0.0]


def main() -> None:
    pools = {s: add_sd(p) for s, p in
             pickle.load(open("data/features/sim_pools.pkl", "rb")).items()}
    seasons = list(pools)
    tem = {s: run_season(pools[s], "pred_template").points.values.sum() for s in seasons}

    # margin for every (gamma, season) once, then slice
    grid = {}
    for gm in GAMMAS:
        for s in seasons:
            grid[(gm, s)] = run_season(pools[s], "xpts", gamma=gm).points.values.sum() - tem[s]
        print(f"  swept gamma={gm}", flush=True)

    print(f'\n{"held out":10}{"gamma chosen":>14}{"on other 3":>12}{"held-out result":>17}')
    held = []
    for s in seasons:
        others = [x for x in seasons if x != s]
        best = max(GAMMAS, key=lambda g: sum(grid[(g, o)] for o in others))
        tr = sum(grid[(best, o)] for o in others)
        te = grid[(best, s)]
        held.append(te)
        print(f'{s:10}{best:14.2f}{tr:+12.0f}{te:+17.0f}')
    print(f'\nout-of-sample total {sum(held):+.0f} over {len(held)} seasons, '
          f'{sum(v > 0 for v in held)} winning')
    print(f'in-sample total at a fixed -0.05: '
          f'{sum(grid[(-0.05, s)] for s in seasons):+.0f}')

    print("\n=== full grid, for reference ===")
    t = pd.DataFrame([{"gamma": g, **{s: grid[(g, s)] for s in seasons},
                       "total": sum(grid[(g, s)] for s in seasons)} for g in GAMMAS])
    print(t.to_string(index=False, float_format=lambda v: f"{v:+.0f}"))


if __name__ == "__main__":
    main()
