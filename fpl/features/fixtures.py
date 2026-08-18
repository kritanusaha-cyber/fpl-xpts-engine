"""One row per fixture, home perspective -- the input the team model fits on."""

from __future__ import annotations

import duckdb
import pandas as pd


def fixture_frame(db: str = "data/fpl.duckdb") -> pd.DataFrame:
    con = duckdb.connect(db)
    t = con.execute("""
        SELECT season, fixture, gw, kickoff_time,
               club_code AS home_code, opponent_code AS away_code,
               goals_for AS home_goals, goals_against AS away_goals,
               xg_for AS home_xg, xg_against AS away_xg
        FROM team_match WHERE was_home = TRUE
    """).df()
    con.close()
    t["kickoff_time"] = pd.to_datetime(t["kickoff_time"], utc=True)
    return t.sort_values("kickoff_time").reset_index(drop=True)
