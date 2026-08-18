"""Tune the Dixon-Coles time-decay xi on out-of-sample log-loss.

The doc suggests starting at xi = 0.003/day and tuning. This does the tuning
walk-forward: for each fixture, fit on everything that kicked off strictly
earlier and score the 1X2 outcome. Refits are batched by gameweek for speed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fpl.features.fixtures import fixture_frame
from fpl.models.team_goals import DixonColes


def evaluate(xi: float, test_seasons: list[str], min_train: int = 380,
             target: str = "goals") -> dict:
    f = fixture_frame()
    ll, briers, n = [], [], 0
    prev = None

    for season in test_seasons:
        gws = sorted(f[f.season == season].gw.dropna().unique())
        for gw in gws:
            test = f[(f.season == season) & (f.gw == gw)]
            if not len(test):
                continue
            cutoff = test.kickoff_time.min()
            train = f[f.kickoff_time < cutoff]
            if target == "xg":
                # Fit the rate parameters on xG, but always SCORE against the
                # realised result -- otherwise the comparison is meaningless.
                train = (train.dropna(subset=["home_xg", "away_xg"])
                              .assign(home_goals=lambda d: d.home_xg,
                                      away_goals=lambda d: d.away_xg))
            if len(train) < min_train:
                continue
            try:
                model = DixonColes.fit(train, xi=xi, target=target, ref_time=cutoff,
                                       warm_start=prev)
                prev = model
            except Exception:
                continue
            for _, r in test.iterrows():
                if r.home_code not in model.index or r.away_code not in model.index:
                    continue
                ph, pd_, pa = model.outcome_probs(r.home_code, r.away_code)
                probs = np.clip([ph, pd_, pa], 1e-9, 1)
                res = 0 if r.home_goals > r.away_goals else (1 if r.home_goals == r.away_goals else 2)
                ll.append(-np.log(probs[res] / probs.sum()))
                cs_h, cs_a = model.clean_sheet_probs(r.home_code, r.away_code)
                briers.append((cs_h - (r.away_goals == 0)) ** 2)
                briers.append((cs_a - (r.home_goals == 0)) ** 2)
                n += 1
    return {"xi": xi, "n": n, "logloss": float(np.mean(ll)),
            "cs_brier": float(np.mean(briers))}


if __name__ == "__main__":
    seasons = ["2024-25", "2025-26"]
    print(f"tuning xi, walk-forward over {seasons}\n")
    print(f'{"xi":>8}{"n":>7}{"1X2 logloss":>14}{"CS Brier":>11}')
    rows = []
    for xi in [0.0, 0.001, 0.002, 0.003, 0.005, 0.008]:
        r = evaluate(xi, seasons)
        rows.append(r)
        print(f'{xi:>8.4f}{r["n"]:>7}{r["logloss"]:>14.4f}{r["cs_brier"]:>11.4f}')
    best = min(rows, key=lambda r: r["logloss"])
    print(f'\nbest xi = {best["xi"]} (logloss {best["logloss"]:.4f})')
    print(f'benchmark: uniform 1/3 = {np.log(3):.4f}')
