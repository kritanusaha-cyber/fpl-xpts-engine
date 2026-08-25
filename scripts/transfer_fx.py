#!/usr/bin/env python3
"""Does valuing transfers against the fixture schedule beat a flat hold?"""
import pickle
import pandas as pd
from fpl.backtest.season_sim import run_season_managed
from scripts.rank_sweep_mv import add_sd

pools = {s: add_sd(p) for s, p in
         pickle.load(open("data/features/sim_pools.pkl", "rb")).items()}
rows = []
for s, p in pools.items():
    a = run_season_managed(p, "xpts", gamma=-0.05, chips=True, fixture_aware=False)
    b = run_season_managed(p, "xpts", gamma=-0.05, chips=True, fixture_aware=True)
    rows.append({"season": s, "flat_hold": a.points.sum(),
                 "fixture_aware": b.points.sum(),
                 "gain": b.points.sum() - a.points.sum()})
    print(f"  {s} done", flush=True)
t = pd.DataFrame(rows)
print("\n" + t.to_string(index=False, float_format=lambda v: f"{v:.0f}"))
print(f"\ntotal {t.gain.sum():+.0f}, {(t.gain > 0).sum()} of {len(t)} seasons improved")
