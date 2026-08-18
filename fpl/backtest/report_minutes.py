"""Walk-forward evaluation of the minutes model.

Calibration is reported before and separately from aggregate error, and per
position, because a model can post a decent overall Brier score while being
badly broken for one position in a way that cancels out in the aggregate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fpl.backtest.walkforward import brier, log_loss, calibration
from fpl.models.minutes import MinutesModel
from fpl.backtest.walkforward import walk_forward

TEST_SEASONS = ["2023-24", "2024-25", "2025-26"]


def run(features_path: str = "data/features/minutes.parquet") -> pd.DataFrame:
    f = pd.read_parquet(features_path)
    f = f[f["position"].notna()]
    return walk_forward(f, MinutesModel, test_seasons=TEST_SEASONS, refit_every=2)


def report(preds: pd.DataFrame, features_path: str = "data/features/minutes.parquet") -> None:
    f = pd.read_parquet(features_path)[
        ["season", "gw", "element", "fixture", "ewm_start_5", "prev_played_60"]]
    m = preds.merge(f, on=["season", "gw", "element", "fixture"], how="left")
    y = m["played_60"].values

    print(f"walk-forward over {TEST_SEASONS}: {len(m):,} player-match predictions\n")

    base = np.full(len(m), y.mean())
    rows = [
        ("model P(>=60)", m["p_60"].values),
        ("baseline: base rate", base),
        ("baseline: persistence", m["prev_played_60"].fillna(0).values),
        ("baseline: ewm start rate", m["ewm_start_5"].fillna(0).values),
    ]
    b0 = brier(y, base)
    print(f'{"":28}{"Brier":>8}{"LogLoss":>10}{"skill":>9}')
    for name, p in rows:
        print(f"{name:28}{brier(y,p):>8.4f}"
              f"{log_loss(y,np.clip(p,1e-6,1-1e-6)):>10.4f}{1-brier(y,p)/b0:>8.1%}")

    print("\nBy position, P(>=60):")
    print(f'{"pos":6}{"n":>9}{"Brier":>9}{"pred":>8}{"real":>8}{"bias":>8}')
    for pos, g in m.groupby("position"):
        yy, pp = g["played_60"].values, g["p_60"].values
        print(f"{pos:6}{len(g):>9,}{brier(yy,pp):>9.4f}"
              f"{pp.mean():>8.3f}{yy.mean():>8.3f}{pp.mean()-yy.mean():>8.4f}")

    print("\nReliability, P(>=60):")
    print(calibration(y, m["p_60"].values).to_string(index=False))


if __name__ == "__main__":
    p = run()
    p.to_parquet("data/features/minutes_preds.parquet", index=False)
    report(p)
