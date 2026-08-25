#!/usr/bin/env python3
"""Re-run the winning objective on strictly pre-deadline ownership.

`owned` moves during a gameweek as managers transfer players in, so it is
contemporaneous and correlates a little more with that week's points than the
lagged value does (0.336 against 0.318). That is small, but a result that beats
the template four seasons from four has to survive the strict version.
"""
import pickle
import pandas as pd
from fpl.backtest.season_sim import run_season
from scripts.rank_sweep_mv import add_sd

pools = {s: add_sd(p) for s, p in
         pickle.load(open("data/features/sim_pools.pkl", "rb")).items()}
seasons = list(pools)
tem = {s: run_season(pools[s], "pred_template").points.values.sum() for s in seasons}

lagged = {}
for s, p in pools.items():
    q = p.copy()
    q["owned"] = q["own_prev"]        # strictly pre-deadline
    lagged[s] = q

print(f'{"ownership used":30}' + "".join(f"{s[-5:]:>9}" for s in seasons)
      + f'{"total":>9}{"wins":>6}', flush=True)
for lab, src in [("contemporaneous (owned)", pools), ("lagged (own_prev)", lagged)]:
    per = [run_season(src[s], "xpts", gamma=-0.05).points.values.sum() - tem[s]
           for s in seasons]
    print(f'{lab:30}' + "".join(f"{v:+9.0f}" for v in per)
          + f"{sum(per):+9.0f}{sum(v > 0 for v in per):6d}", flush=True)
