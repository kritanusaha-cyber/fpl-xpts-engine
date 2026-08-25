#!/usr/bin/env python3
"""Does the methodology hold across different kinds of Premier League?

A result averaged over four seasons can hide a method that works only in one
kind of environment. These seasons are not interchangeable: goals per team-game
range from 1.35 to 1.64, clean-sheet rate from 0.21 to 0.30, and the spread of
team possession from 10.7 to 14.2 points. A threshold-heavy scoring system
should be most sensitive to exactly those things.

Each slice is a different question:
  scoring       -- do the projections survive a goal-rich league, where
                   thresholds trip more often and variance rises?
  clean sheets  -- do they survive a mean league, where they trip rarely?
  fixture size  -- blank and double gameweeks change the pool the optimiser
                   chooses from, sometimes drastically
  season phase  -- early gameweeks run on priors; late ones on data
  congestion    -- midweek rounds, where rotation dominates minutes
"""
import numpy as np
import pandas as pd
import duckdb


def load() -> pd.DataFrame:
    d = pd.read_parquet("data/features/backtest_predictions.parquet")
    d = d[d.xpts.notna() & d.form.notna() & d.naive.notna() & d.total_points.notna()].copy()

    con = duckdb.connect("data/fpl.duckdb", read_only=True)
    tm = con.execute("""SELECT season, gw, avg(goals_for) AS gpg,
                          avg(CASE WHEN clean_sheet=1 THEN 1.0 ELSE 0.0 END) AS cs,
                          count(*) AS teams, min(kickoff_time) AS first_ko
                        FROM team_match WHERE season >= '2022-23'
                        GROUP BY season, gw""").df()
    con.close()
    tm["dow"] = pd.to_datetime(tm.first_ko, utc=True).dt.dayofweek
    return d.merge(tm, on=["season", "gw"], how="left")


def score(g: pd.DataFrame) -> dict:
    """Engine advantage over the better of the two references, plus top-10 precision."""
    e = np.abs(g.xpts - g.total_points).mean()
    ref = min(np.abs(g.form - g.total_points).mean(),
              np.abs(g.naive - g.total_points).mean())
    top = [w.nlargest(10, "xpts").total_points.mean()
           for _, w in g.groupby(["season", "gw"]) if len(w) >= 10]
    topref = [w.nlargest(10, "naive").total_points.mean()
              for _, w in g.groupby(["season", "gw"]) if len(w) >= 10]
    return {"n": len(g), "engine_MAE": e, "best_ref_MAE": ref,
            "MAE_gain_%": (ref - e) / ref * 100,
            "top10": np.mean(top) if top else np.nan,
            "top10_ref": np.mean(topref) if topref else np.nan}


def report(d: pd.DataFrame, col: str, title: str) -> None:
    print(f"\n=== {title} ===")
    rows = []
    for k, g in d.groupby(col, observed=True):
        if len(g) < 500:
            continue
        rows.append({col: k, **score(g)})
    t = pd.DataFrame(rows)
    t["top10_gain"] = t.top10 - t.top10_ref
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def main() -> None:
    d = load()
    print(f"{len(d):,} player-gameweeks across {d.season.nunique()} seasons")

    d["scoring_env"] = pd.qcut(d.gpg, 3, labels=["low goals", "mid", "high goals"])
    d["cs_env"] = pd.qcut(d.cs, 3, labels=["few clean sheets", "mid", "many clean sheets"])
    d["fixture_size"] = pd.cut(d.teams, [0, 18, 20, 99],
                               labels=["blank (<=18 teams)", "normal (20)", "double (>20)"])
    d["phase"] = pd.cut(d.gw, [0, 12, 26, 99], labels=["early (8-12)", "mid (13-26)", "late (27-38)"])
    d["midweek"] = np.where(d.dow.isin([1, 2, 3]), "midweek round", "weekend round")

    for col, title in [("season", "by season"),
                       ("scoring_env", "by scoring environment (league goals that gameweek)"),
                       ("cs_env", "by clean-sheet environment"),
                       ("fixture_size", "by fixture availability"),
                       ("phase", "by season phase"),
                       ("midweek", "by congestion")]:
        report(d, col, title)

    print("\n=== by position within the extreme scoring environments ===")
    for env in ["low goals", "high goals"]:
        sub = d[d.scoring_env == env]
        r = []
        for pos, g in sub.groupby("position"):
            s = score(g)
            r.append({"env": env, "pos": pos, "n": s["n"],
                      "MAE_gain_%": s["MAE_gain_%"]})
        print(pd.DataFrame(r).to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
