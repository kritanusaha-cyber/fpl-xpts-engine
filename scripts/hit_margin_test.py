#!/usr/bin/env python3
"""Is hit-taking unprofitable because of the optimiser's curse?

If the losses come from acting on inflated gain estimates, then demanding a
larger margin should recover them -- and the recovery should be monotonic in the
margin until hits stop being taken at all.
"""
import json, pickle
import numpy as np
from fpl.backtest.season_sim import run_season, run_season_managed

def main():
    pools = pickle.load(open("data/features/sim_pools.pkl", "rb"))
    seasons = list(pools)
    tem = {s: run_season(pools[s], "pred_template").points.values for s in seasons}
    print(f'{"hit margin":14}' + "".join(f"{s[-5:]:>9}" for s in seasons)
          + f'{"total":>9}{"hits":>7}', flush=True)
    out = {}
    for mg in [1.0, 1.5, 2.0, 3.0, 99.0]:
        cells, tot, hits = [], 0.0, 0
        for s in seasons:
            r = run_season_managed(pools[s], "xpts", hold_weeks=4.0, hit_margin=mg)
            d = float(r.points.sum() - tem[s].sum())
            cells.append(d); tot += d; hits += int(r.hits.sum())
        out[str(mg)] = {"cells": cells, "total": tot, "hits": hits}
        lab = "no hits" if mg > 50 else f"{mg:.1f}x"
        print(f'{lab:14}' + "".join(f"{c:>+9.0f}" for c in cells)
              + f"{tot:>+9.0f}{hits:>7}", flush=True)
    json.dump(out, open("data/features/hit_margin.json", "w"))
    print("saved", flush=True)

if __name__ == "__main__":
    main()
