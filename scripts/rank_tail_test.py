#!/usr/bin/env python3
"""Does a small rank tilt buy upper-tail access? Test the thing rank depends on.

A points test cannot answer this: tilting deliberately trades expected points for
variance. What matters is the probability of a LARGE positive margin, since that
is what a high finish requires.

With four seasons there are not enough season-level replications to read a tail
directly, so seasons are bootstrapped: resample gameweeks with replacement to
build synthetic 31-week seasons, and read the distribution of season margin.
"""
import json, pickle
import numpy as np
from fpl.backtest.season_sim import run_season

KS = [0.0, 0.05, 0.10, 0.15, 0.20]
N_BOOT = 4000
RNG = np.random.default_rng(7)


def main():
    pools = pickle.load(open("data/features/sim_pools.pkl", "rb"))
    seasons = list(pools)
    tem = {s: run_season(pools[s], "pred_template").points.values for s in seasons}

    print(f'{"k":>6}{"mean/season":>13}{"P(margin>0)":>13}'
          f'{"P(>+50)":>10}{"P(>+100)":>11}{"q90 season":>12}', flush=True)
    out = {}
    for k in KS:
        weekly = []
        for s in seasons:
            r = run_season(pools[s], "xpts", rank_k=k).points.values
            weekly.extend(r - tem[s])
        weekly = np.array(weekly)
        # bootstrap synthetic seasons of 31 gameweeks
        boot = RNG.choice(weekly, size=(N_BOOT, 31), replace=True).sum(axis=1)
        out[str(k)] = {"mean": float(boot.mean()),
                       "p_pos": float((boot > 0).mean()),
                       "p50": float((boot > 50).mean()),
                       "p100": float((boot > 100).mean()),
                       "q90": float(np.percentile(boot, 90))}
        o = out[str(k)]
        print(f'{k:>6.2f}{o["mean"]:>13.1f}{o["p_pos"]:>13.3f}'
              f'{o["p50"]:>10.3f}{o["p100"]:>11.3f}{o["q90"]:>12.1f}', flush=True)
    json.dump(out, open("data/features/rank_tail.json", "w"))
    print("saved", flush=True)


if __name__ == "__main__":
    main()
