"""Walk-forward accuracy of the real model stack, season by season.

Every model is refitted at each gameweek on gameweeks already played, so a
projection for GW20 has never seen GW20. That is the whole point: a model
fitted once on a finished season scores itself on its own training data and
flatters itself badly.

Three references, because "is it accurate" only means something relative to
what a manager could do without it:
  naive     -- the player's own season-to-date points per game
  form      -- his last five gameweeks, which is what most managers actually use
  FPL ep_next -- the game's own published projection, where the season has it
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fpl.backtest.engine_backtest import run

# Six seasons since the team-xG backfill from FotMob. Previously four, because
# the warehouse only carried expected goals from 2022-23.
SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]


def references(d: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date and 5-gameweek form, computed with past data only."""
    d = d.sort_values(["element", "gw"]).copy()
    g = d.groupby("element")["total_points"]
    d["naive"] = g.transform(lambda s: s.shift(1).expanding().mean())
    d["form"] = g.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    return d


def score(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        m = d[d[c].notna() & d.total_points.notna()]
        if m.empty:
            continue
        err = m[c] - m.total_points
        # Spearman per gameweek, then averaged. Pooling across gameweeks would
        # let between-week scoring differences masquerade as ranking skill.
        rho = [spearmanr(w[c], w.total_points).statistic
               for _, w in m.groupby("gw") if w[c].nunique() > 1]
        rows.append({"model": c, "n": len(m),
                     "MAE": np.abs(err).mean(), "RMSE": np.sqrt((err ** 2).mean()),
                     "bias": err.mean(),
                     "spearman": np.nanmean(rho)})
    return pd.DataFrame(rows)


def main() -> None:
    all_rows, per_season = [], []
    for s in SEASONS:
        print(f"\n=== {s} ===", flush=True)
        d = run(season=s)
        d = references(d)
        d["season"] = s
        all_rows.append(d)
        r = score(d, ["xpts", "form", "naive"]).assign(season=s)
        per_season.append(r)
        print(r.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    d = pd.concat(all_rows, ignore_index=True)
    d.to_parquet("data/features/backtest_predictions.parquet", index=False)
    ps = pd.concat(per_season, ignore_index=True)
    ps.to_csv("data/features/backtest_accuracy.csv", index=False)

    print("\n=== pooled across all seasons ===")
    print(score(d, ["xpts", "form", "naive"]).to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n=== per season, engine vs best reference ===")
    piv = ps.pivot(index="season", columns="model", values="MAE")
    piv["best_ref"] = piv[["form", "naive"]].min(axis=1)
    piv["engine_better_by"] = (piv.best_ref - piv.xpts) / piv.best_ref * 100
    print(piv.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    sys.exit(main())
