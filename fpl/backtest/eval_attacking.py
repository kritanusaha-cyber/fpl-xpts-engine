"""Test the doc's central Phase 3 claim: shares beat rates.

Two structures for predicting a player's xG in an unseen match:

    rate  : E[xG] = ewm(player xG per 90)              * minutes/90
    share : E[xG] = ewm(player share of team xG) * team_xG * minutes/90

The share form is meant to avoid double-counting team quality. Whether it
actually predicts better is an empirical question, so it is measured here.

Two team-xG variants are run:
  * oracle    -- the realised team xG, which isolates the STRUCTURAL question
  * projected -- market-implied team goals, which is what you can actually use

Actual minutes are used throughout so that error from the Phase 1 minutes model
does not contaminate the comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fpl.features.attacking import (load, add_shares, add_history, price_tier,
                                    empirical_bayes_weight)

TEST_SEASONS = ["2024-25", "2025-26"]


def _market_team_xg(d: pd.DataFrame) -> pd.Series:
    """Market-implied goal expectation for each player's own team."""
    odds = pd.read_parquet("data/raw/odds/football_data.parquet")
    names = pd.read_parquet("data/features/club_names.parquet")
    o = (odds.merge(names.rename(columns={"name": "home_name", "club_code": "home_code"}),
                    on=["season", "home_name"], how="left")
             .merge(names.rename(columns={"name": "away_name", "club_code": "away_code"}),
                    on=["season", "away_name"], how="left")
             .dropna(subset=["home_code", "away_code"]))
    long = pd.concat([
        o[["season", "home_code", "date", "mkt_home_goals"]].rename(
            columns={"home_code": "club_code", "mkt_home_goals": "mkt_team_goals"}),
        o[["season", "away_code", "date", "mkt_away_goals"]].rename(
            columns={"away_code": "club_code", "mkt_away_goals": "mkt_team_goals"}),
    ])
    long["club_code"] = long["club_code"].astype(int)
    long["date"] = long["date"].dt.date
    # Join per MATCH, not per season. Averaging a club's market goals over the
    # season throws away the fixture-difficulty signal, which is the only reason
    # to consult the market in the first place.
    d = d.copy()
    d["date"] = d["kickoff_time"].dt.date
    merged = d.merge(long[["season", "club_code", "date", "mkt_team_goals"]],
                     on=["season", "club_code", "date"], how="left")
    return merged["mkt_team_goals"]


def build_eval(hl: int = 5) -> pd.DataFrame:
    d = add_history(add_shares(load()))
    d["tier"] = price_tier(d)
    d["group"] = d["position"].astype(str) + "_" + d["tier"].astype(str)

    # Empirical-Bayes shrunk share. The prior is computed on the training
    # seasons only, so it does not see the evaluation period.
    train_mask = ~d["season"].isin(TEST_SEASONS)
    prior_tbl = (d[train_mask].groupby("group")[f"xg_share_ewm{hl}"]
                   .mean().rename("prior_share").reset_index())
    d = d.merge(prior_tbl, on="group", how="left")

    w, _ = empirical_bayes_weight(d[f"xg_share_ewm{hl}"].fillna(0),
                                  d["n90"].fillna(0), d["group"])
    d["w_shrink"] = w
    d["shrunk_share"] = (d["w_shrink"] * d[f"xg_share_ewm{hl}"].fillna(0)
                         + (1 - d["w_shrink"]) * d["prior_share"].fillna(0))

    d["mkt_team_goals"] = _market_team_xg(d)
    return d


def evaluate(d: pd.DataFrame, hl: int = 5) -> pd.DataFrame:
    t = d[d["season"].isin(TEST_SEASONS)].copy()
    t = t.dropna(subset=[f"xg_share_ewm{hl}", f"xg_per90_ewm{hl}", "xg", "team_xg"])
    mins = t["minutes"] / 90.0
    y = t["xg"].to_numpy()

    preds = {
        "rate (per-90 EWMA)":            t[f"xg_per90_ewm{hl}"] * mins,
        "share x team xG (oracle)":      t[f"xg_share_ewm{hl}"] * t["team_xg"] * mins,
        "shrunk share x team xG (oracle)": t["shrunk_share"] * t["team_xg"] * mins,
        "share x market goals":          t[f"xg_share_ewm{hl}"] * t["mkt_team_goals"] * mins,
        "shrunk share x market goals":   t["shrunk_share"] * t["mkt_team_goals"] * mins,
    }
    rows = []
    for name, p in preds.items():
        p = pd.Series(p).fillna(0).to_numpy()
        rows.append({
            "method": name, "n": len(y),
            "MAE": float(np.mean(np.abs(p - y))),
            "RMSE": float(np.sqrt(np.mean((p - y) ** 2))),
            "correl": float(np.corrcoef(p, y)[0, 1]),
            "bias": float(np.mean(p - y)),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    d = build_eval()
    d.to_parquet("data/features/attacking.parquet", index=False)
    r = evaluate(d)
    print(f"predicting player xG, walk-forward features, {TEST_SEASONS}\n")
    print(f'{"method":34}{"MAE":>9}{"RMSE":>9}{"correl":>8}{"bias":>9}')
    for _, x in r.iterrows():
        print(f'{x.method:34}{x.MAE:>9.4f}{x.RMSE:>9.4f}{x.correl:>8.4f}{x.bias:>+9.4f}')
    print(f'\nn = {int(r.n.iloc[0]):,} player-matches')
