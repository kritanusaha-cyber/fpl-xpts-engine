#!/usr/bin/env python3
"""All four chip types across four seasons, against the same squad without them."""
import pickle
import pandas as pd
from fpl.backtest.season_sim import run_season_managed
from scripts.rank_sweep_mv import add_sd

pools = {s: add_sd(p) for s, p in
         pickle.load(open("data/features/sim_pools.pkl", "rb")).items()}
rows, logs = [], []
for s, p in pools.items():
    a = run_season_managed(p, "xpts", gamma=-0.05, chips=False)
    b = run_season_managed(p, "xpts", gamma=-0.05, chips=True)
    played = b[b.chip.notna()]
    rows.append({"season": s, "no_chips": a.points.sum(), "with_chips": b.points.sum(),
                 "gain": b.points.sum() - a.points.sum(), "n_chips": len(played)})
    for _, r in played.iterrows():
        logs.append({"season": s, "gw": int(r.gw), "chip": r.chip, "gain": r.chip_gain})
    print(f"  {s} done", flush=True)

t = pd.DataFrame(rows)
print("\n" + t.to_string(index=False, float_format=lambda v: f"{v:.0f}"))
print(f"\ntotal gain {t.gain.sum():+.0f} over four seasons, "
      f"{(t.gain > 0).sum()} seasons improved, mean {t.gain.mean():+.1f}")
l = pd.DataFrame(logs)
print("\nby chip type:")
print(l.groupby("chip").agg(times=("gain", "size"), total=("gain", "sum"),
                            mean=("gain", "mean")).to_string(float_format=lambda v: f"{v:.1f}"))
l.to_csv("data/features/chip_log.csv", index=False)
