#!/usr/bin/env python3
"""Sweep the rank-tilt parameter k across four seasons of walk-forward data."""
import json, pickle, sys
import numpy as np
from fpl.backtest.season_sim import run_season

KS = [0.0, 0.15, 0.30, 0.50, 0.75, 1.00]

def main():
    pools = pickle.load(open("data/features/sim_pools.pkl", "rb"))
    seasons = list(pools)
    tem = {s: run_season(pools[s], "pred_template").points.values for s in seasons}
    print("template baselines done", flush=True)

    out = {}
    hdr = f'{"k":>6}' + "".join(f"{s[-5:]:>10}" for s in seasons) + f'{"total":>10}{"wins":>7}'
    print(hdr, flush=True)
    for k in KS:
        per, tot, wins = {}, 0.0, 0
        for s in seasons:
            r = run_season(pools[s], "xpts", rank_k=k).points.values
            per[s] = [float(x) for x in r]
            d = float(r.sum() - tem[s].sum())
            tot += d
            wins += int(d > 0)
        out[str(k)] = per
        row = f"{k:>6.2f}" + "".join(
            f"{np.sum(per[s]) - tem[s].sum():>+10.0f}" for s in seasons)
        print(row + f"{tot:>+10.0f}{wins:>7}", flush=True)

    json.dump(out, open("data/features/rank_sweep.json", "w"))
    json.dump({s: [float(x) for x in v] for s, v in tem.items()},
              open("data/features/template_baseline.json", "w"))
    print("saved", flush=True)

if __name__ == "__main__":
    main()
