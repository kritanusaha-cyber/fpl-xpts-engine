"""Cold start for players with no Premier League history.

The default fallback -- a position x price-tier prior -- throws away two things
that are actually known about a new signing:

  1. THE ROLE HE IS STEPPING INTO. A club's share structure is far more stable
     than its personnel. Brighton's centre-backs take a similar share of team xG
     and rack up similar defensive-action counts season to season, whoever is
     wearing the shirt. So the best prior for Vuskovic is what Brighton's
     centre-backs did last season -- including Van Hecke, who has since left.
     The role persists even when the player does not.

  2. WHAT HE DID ABROAD. Output in another league, discounted for league
     strength, is a genuine signal that the price tier only crudely proxies.

This module builds (1) from data already in the warehouse and blends it with (2)
where foreign stats are available. Weights are fit, not assumed -- see
fpl/backtest/eval_transfers.py, which scores each prior against what newcomers
actually went on to do.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

METRICS = ["xg_share", "xa_share", "dc_per90", "starts60"]


def club_role_profiles(db: str = "data/fpl.duckdb", season: str = "2025-26") -> pd.DataFrame:
    """Per club x position, the profile of whoever actually held that role.

    Weighted by minutes so a fringe player does not define the role, and
    restricted to players who cleared a meaningful minutes floor.
    """
    con = duckdb.connect(db)
    d = con.execute(f"""
        SELECT p.element, p.position, t.club_code, p.minutes,
               p.expected_goals AS xg, p.expected_assists AS xa,
               p.tackles, p.recoveries, p.clearances_blocks_interceptions AS cbi,
               t.xg_for AS team_xg
        FROM player_gw p
        JOIN team_match t
          ON p.season=t.season AND p.fixture=t.fixture AND p.team_id=t.team_id
        WHERE p.season='{season}' AND p.minutes>0 AND t.xg_for IS NOT NULL
    """).df()
    con.close()

    d["dc_n"] = np.where(d.position == "DEF",
                         d.tackles.fillna(0) + d.cbi.fillna(0),
                         d.tackles.fillna(0) + d.cbi.fillna(0) + d.recoveries.fillna(0))

    per = (d.groupby(["club_code", "position", "element"])
             .agg(mins=("minutes", "sum"), xg=("xg", "sum"), xa=("xa", "sum"),
                  dc=("dc_n", "sum"), team_xg=("team_xg", "sum"),
                  starts=("minutes", lambda s: (s >= 60).mean()))
             .reset_index())
    per = per[per.mins >= 450]                       # five 90s floor
    per["xg_share"] = per.xg / per.team_xg.clip(lower=0.1)
    per["xa_share"] = per.xa / per.team_xg.clip(lower=0.1)
    per["dc_per90"] = per.dc / per.mins.clip(lower=1) * 90
    per["starts60"] = per.starts

    def wavg(g, col):
        return np.average(g[col], weights=g["mins"]) if len(g) else np.nan

    prof = (per.groupby(["club_code", "position"])
              .apply(lambda g: pd.Series({m: wavg(g, m) for m in METRICS}
                                         | {"n_players": len(g), "mins": g.mins.sum()}),
                     include_groups=False)
              .reset_index())
    return prof


def league_profiles(prof: pd.DataFrame) -> pd.DataFrame:
    """League-wide position profile -- the fallback when a club has no analogue
    (a promoted side, or a position nobody held for 450+ minutes)."""
    return (prof.groupby("position")[METRICS].mean().reset_index()
              .rename(columns={m: f"lg_{m}" for m in METRICS}))


def role_prior(players: pd.DataFrame, prof: pd.DataFrame) -> pd.DataFrame:
    """Attach the club x position role profile to each player."""
    lg = league_profiles(prof)
    out = players.merge(prof, on=["club_code", "position"], how="left")
    out = out.merge(lg, on="position", how="left")
    for m in METRICS:
        out[f"role_{m}"] = out[m].fillna(out[f"lg_{m}"])
        out = out.drop(columns=[m])
    return out
