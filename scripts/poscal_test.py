#!/usr/bin/env python3
"""Does correcting positional bias produce better squads?

MAE gets worse under the correction and bias gets better, which is the
median-versus-mean tradeoff on a right-skewed distribution. Squad points are a
sum of eleven players, and sums care about means, so the decision criterion is
season points rather than per-player error.

Factors are fitted on seasons already played and applied to the season being
simulated, never on itself.
"""
import pickle
import pandas as pd
from fpl.backtest.season_sim import run_season_managed
from fpl.models.position_calibration import PositionCalibrator
from scripts.rank_sweep_mv import add_sd

pools = {s: add_sd(p) for s, p in
         pickle.load(open("data/features/sim_pools.pkl", "rb")).items()}
seasons = sorted(pools)
rows = []
for i, s in enumerate(seasons):
    if i == 0:
        continue
    train = pd.concat([pools[x] for x in seasons[:i]], ignore_index=True)
    cal = PositionCalibrator().fit(train)
    p = pools[s].copy()
    p["xpts_cal"] = cal.transform(p)
    a = run_season_managed(p, "xpts", gamma=-0.05, chips=True).points.sum()
    b = run_season_managed(p, "xpts_cal", gamma=-0.05, chips=True).points.sum()
    rows.append({"season": s, "raw": a, "calibrated": b, "gain": b - a,
                 "factors": {k: round(v, 2) for k, v in cal.factors.items()}})
    print(f"  {s}: raw {a:.0f} -> calibrated {b:.0f} ({b-a:+.0f})", flush=True)

t = pd.DataFrame(rows)
print(f"\ntotal {t.gain.sum():+.0f} over {len(t)} held-out seasons, "
      f"{(t.gain > 0).sum()} improved")
for r in rows:
    print(f"  {r['season']} factors {r['factors']}")
