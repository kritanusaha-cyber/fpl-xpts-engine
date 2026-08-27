#!/usr/bin/env python3
"""Leave-one-season-out on the chip threshold.

The threshold was swept on the same four seasons the +139 is reported on,
which is fitting on the test set. This picks it on three and applies it blind
to the fourth.
"""
import pickle
import numpy as np
import pandas as pd
import fpl.optimize.chips as chips
from fpl.backtest.season_sim import run_season_managed
from scripts.rank_sweep_mv import add_sd

LEVELS = [1.00, 1.15, 1.30, 1.45, 1.60]

pools = {s: add_sd(p) for s, p in
         pickle.load(open("data/features/sim_pools.pkl", "rb")).items()}
seasons = list(pools)
base = {s: run_season_managed(pools[s], "xpts", gamma=-0.05, chips=False).points.sum()
        for s in seasons}
print("baselines done", flush=True)

grid = {}
for lv in LEVELS:
    chips.THRESHOLD = {k: lv for k in chips.THRESHOLD}
    for s in seasons:
        grid[(lv, s)] = (run_season_managed(pools[s], "xpts", gamma=-0.05,
                                            chips=True).points.sum() - base[s])
    print(f"  swept {lv}", flush=True)

print(f'\n{"held out":10}{"chosen":>8}{"on other 3":>12}{"held-out":>11}')
held = []
for s in seasons:
    others = [x for x in seasons if x != s]
    best = max(LEVELS, key=lambda l: sum(grid[(l, o)] for o in others))
    tr = sum(grid[(best, o)] for o in others)
    te = grid[(best, s)]
    held.append(te)
    print(f'{s:10}{best:8.2f}{tr:+12.0f}{te:+11.0f}')
print(f'\nout of sample {sum(held):+.0f} over {len(held)} seasons, '
      f'{sum(v > 0 for v in held)} winning')
print(f'in sample at a fixed 1.15: {sum(grid[(1.15, s)] for s in seasons):+.0f}')
