#!/usr/bin/env python3
"""Does the ownership weighting earn its place, or is it just risk aversion?"""
import pickle
import pandas as pd
from fpl.backtest.season_sim import run_season
from scripts.rank_sweep_mv import add_sd

pools={s:add_sd(p) for s,p in pickle.load(open("data/features/sim_pools.pkl","rb")).items()}
seasons=list(pools)
tem={s:run_season(pools[s],"pred_template").points.values.sum() for s in seasons}
print(f'{"objective":34}'+"".join(f"{s[-5:]:>9}" for s in seasons)+f'{"total":>9}{"wins":>6}',flush=True)
for lab,kw in [("mean-variance, EO-weighted",dict(gamma=-0.05)),
               ("variance only, no ownership",dict(gamma=-0.05,plain_var=True)),
               ("expected points (baseline)",dict())]:
    per=[run_season(pools[s],"xpts",**kw).points.values.sum()-tem[s] for s in seasons]
    print(f'{lab:34}'+"".join(f"{v:+9.0f}" for v in per)
          +f"{sum(per):+9.0f}{sum(v>0 for v in per):6d}",flush=True)
