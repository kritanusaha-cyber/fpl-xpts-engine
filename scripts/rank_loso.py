#!/usr/bin/env python3
"""Leave-one-season-out test of the rank tilt.

A sweep that spikes at k = 0.05 and reverts by k = 0.10 is not a dose-response,
it is noise -- and the bootstrap that flattered it resamples gameweeks from only
four seasons, so it measures week-sampling error rather than whether the
parameter generalises.

The honest test: choose k on three seasons, score it on the fourth. If the tilt
is real it survives; if it was fitted to one season's quirks it collapses.
"""
import json, pickle
import numpy as np
from fpl.backtest.season_sim import run_season

KS = [0.0, 0.05, 0.10, 0.15]


def main():
    pools = pickle.load(open("data/features/sim_pools.pkl", "rb"))
    seasons = list(pools)
    tem = {s: run_season(pools[s], "pred_template").points.values for s in seasons}

    # margin per season per k, computed once
    margin = {}
    for k in KS:
        for s in seasons:
            r = run_season(pools[s], "xpts", rank_k=k).points.values
            margin[(k, s)] = float(r.sum() - tem[s].sum())
        print(f"k={k} done", flush=True)

    print()
    print(f'{"held out":10}{"k chosen on other 3":>22}{"margin on held-out":>21}'
          f'{"vs k=0 there":>15}')
    rows = []
    for held in seasons:
        others = [s for s in seasons if s != held]
        best_k = max(KS, key=lambda k: np.mean([margin[(k, s)] for s in others]))
        got = margin[(best_k, held)]
        base = margin[(0.0, held)]
        rows.append((held, best_k, got, base))
        print(f"{held:10}{best_k:>22.2f}{got:>21.0f}{got - base:>+15.0f}")

    diff = [r[2] - r[3] for r in rows]
    print()
    print(f"tuned tilt vs plain expected points, across held-out seasons: "
          f"{np.mean(diff):+.1f} points/season")
    print(f"  seasons improved: {sum(1 for d in diff if d > 0)} of {len(diff)}")
    json.dump({"rows": [[r[0], r[1], r[2], r[3]] for r in rows]},
              open("data/features/rank_loso.json", "w"))


if __name__ == "__main__":
    main()
