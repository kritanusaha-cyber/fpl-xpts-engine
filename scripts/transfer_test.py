#!/usr/bin/env python3
"""Does real transfer economics beat greedy weekly re-picking?"""
import json, pickle
import numpy as np
from fpl.backtest.season_sim import run_season, run_season_managed

def main():
    pools = pickle.load(open("data/features/sim_pools.pkl", "rb"))
    seasons = list(pools)
    tem = {s: run_season(pools[s], "pred_template").points.values for s in seasons}
    print("template baselines done", flush=True)

    variants = {
        "greedy (current)": lambda p: run_season(p, "xpts"),
        "managed h=3": lambda p: run_season_managed(p, "xpts", hold_weeks=3.0),
        "managed h=4": lambda p: run_season_managed(p, "xpts", hold_weeks=4.0),
        "managed h=6": lambda p: run_season_managed(p, "xpts", hold_weeks=6.0),
        "managed h=4 no hits": lambda p: run_season_managed(p, "xpts", hold_weeks=4.0, max_hits=0),
    }
    print(f'{"variant":24}' + "".join(f"{s[-5:]:>9}" for s in seasons)
          + f'{"total":>9}{"wins":>6}{"hits":>7}', flush=True)
    out = {}
    for lab, fn in variants.items():
        tot, wins, hits, per = 0.0, 0, 0, {}
        cells = []
        for s in seasons:
            r = fn(pools[s])
            d = float(r.points.sum() - tem[s].sum())
            per[s] = [float(x) for x in r.points]
            hits += int(r.hits.sum()) if "hits" in r else 0
            tot += d; wins += int(d > 0); cells.append(d)
        out[lab] = per
        print(f'{lab:24}' + "".join(f"{c:>+9.0f}" for c in cells)
              + f"{tot:>+9.0f}{wins:>6}{hits:>7}", flush=True)
    json.dump(out, open("data/features/transfer_test.json", "w"))
    print("saved", flush=True)

if __name__ == "__main__":
    main()
