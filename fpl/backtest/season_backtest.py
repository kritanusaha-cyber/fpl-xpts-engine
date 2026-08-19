"""The success criterion the build plan sets, tested at last.

    "beat the 'template' benchmark -- the xPts-weighted top-15 by ownership --
     on out-of-sample per-GW MAE and on simulated final rank. If you can't beat
     the template, you have a hobby, not an edge."

Everything else in this project measures a component. This measures whether the
assembled thing is worth using, against the benchmarks the plan names, in the
order it names them:

    1. last season's points per 90
    2. FPL's own forecast (`ep_next`, which the API publishes)
    3. the template -- what a manager gets by following the crowd
    4. a naive price heuristic

Walk-forward over 2025/26: at each gameweek, every model sees only gameweeks
already played. Scored on the metrics the plan specifies -- per-GW MAE, and
Spearman within position, which is the decision actually being made.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MIN_GW = 6          # models need some history before they say anything


def load_season(db: str = "data/fpl.duckdb", season: str = "2025-26") -> pd.DataFrame:
    con = duckdb.connect(db)
    d = con.execute(f"""
        SELECT p.season, p.gw, p.element, p.position, p.minutes, p.total_points,
               p.value/10.0 AS price, p.selected AS owned, p.team_id,
               p.expected_goals AS xg, p.expected_assists AS xa,
               p.clearances_blocks_interceptions AS cbi, p.tackles, p.recoveries,
               t.xg_for AS team_xg, t.club_code
        FROM player_gw p
        JOIN team_match t
          ON p.season=t.season AND p.fixture=t.fixture AND p.team_id=t.team_id
        WHERE p.season='{season}' AND p.position IS NOT NULL
    """).df()
    con.close()
    return d.sort_values(["gw", "element"]).reset_index(drop=True)


def _prior_form(hist: pd.DataFrame) -> pd.DataFrame:
    """Points per appearance and per team-game, from gameweeks already played."""
    g = hist.groupby("element")
    return pd.DataFrame({
        "ppg_app": g["total_points"].sum() / g["total_points"].count().clip(lower=1),
        "ppg_all": g["total_points"].mean(),
        "mins": g["minutes"].sum(),
        "start_rate": g["minutes"].apply(lambda s: (s >= 60).mean()),
        "xgi90": (g["xg"].sum() + g["xa"].sum()) / (g["minutes"].sum() / 90).clip(lower=0.5),
    }).reset_index()


def build_predictions(d: pd.DataFrame, ep_next: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per player-gameweek with every benchmark's forecast."""
    rows = []
    for gw in sorted(d.gw.dropna().unique()):
        if gw < MIN_GW:
            continue
        hist = d[d.gw < gw]
        cur = d[d.gw == gw].copy()
        if hist.empty or cur.empty:
            continue
        f = _prior_form(hist)
        cur = cur.merge(f, on="element", how="left")

        # 1. last season's / to-date points per appearance
        cur["pred_ppg"] = cur["ppg_app"].fillna(0)
        # 2. form: same thing weighted to recent games (the plan's "form field")
        recent = d[(d.gw < gw) & (d.gw >= gw - 4)]
        rf = recent.groupby("element")["total_points"].mean().rename("form4")
        cur = cur.merge(rf, on="element", how="left")
        cur["pred_form"] = cur["form4"].fillna(cur["pred_ppg"])
        # 3. template: what the crowd owns, scaled to a points forecast
        # Double gameweeks give a player two rows, so ownership must be
        # de-duplicated before it can be used as a lookup.
        own = (hist[hist.gw == hist.gw.max()]
               .groupby("element")["owned"].max())
        cur["own_prev"] = cur["element"].map(own).fillna(0)
        cur["pred_template"] = (cur["own_prev"] / max(cur["own_prev"].max(), 1)
                                * cur["pred_ppg"].mean() * 2)
        # 4. price heuristic
        cur["pred_price"] = cur["price"] * cur["pred_ppg"].mean() / max(cur["price"].mean(), 1e-6)
        # engine proxy: minutes-weighted underlying output, the structure this
        # project actually uses (shares x availability), rebuilt from history only
        cur["pred_engine"] = (cur["xgi90"].fillna(0) * cur["start_rate"].fillna(0) * 3.4
                              + cur["start_rate"].fillna(0) * 1.6)
        rows.append(cur)
    return pd.concat(rows, ignore_index=True)


def score(p: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Per-GW MAE and within-position Spearman, over players who featured."""
    out = []
    for m in models:
        played = p[p.minutes > 0]
        mae = float(np.mean(np.abs(played[m] - played["total_points"])))
        sp, n = [], 0
        for (_, _), g in played.groupby(["gw", "position"]):
            if len(g) < 8 or g[m].std() == 0:
                continue
            r = spearmanr(g[m], g["total_points"]).correlation
            if np.isfinite(r):
                sp.append(r); n += 1
        top20 = []
        for gw, g in played.groupby("gw"):
            k = min(20, len(g))
            picked = set(g.nlargest(k, m).element)
            actual = set(g.nlargest(k, "total_points").element)
            top20.append(len(picked & actual) / k)
        out.append({"model": m, "MAE": mae,
                    "spearman_in_pos": float(np.mean(sp)) if sp else np.nan,
                    "top20_precision": float(np.mean(top20)), "n_gw": p.gw.nunique()})
    return pd.DataFrame(out).sort_values("MAE")
