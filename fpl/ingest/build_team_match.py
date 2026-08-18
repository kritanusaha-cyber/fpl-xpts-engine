"""Build the team_match fact table -- one row per team per fixture.

Aggregated from player_gw rather than sourced separately, so team-level numbers
are guaranteed consistent with the player-level ones feeding Phases 3-5. Team xG
is the sum of player expected_goals, which is only defined from 2022/23; goals
are available for all ten seasons.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def _club_code_lookup(raw_dir: Path = Path("data/raw")) -> pd.DataFrame:
    """(season, team_id) -> stable club_code, from players_raw."""
    rows = []
    for path in sorted((raw_dir / "vaastav" / "players_raw").glob("season=*.parquet")):
        season = path.stem.split("=", 1)[1]
        pl = pd.read_parquet(path)
        if not {"team", "team_code"} <= set(pl.columns):
            continue
        m = pl[["team", "team_code"]].drop_duplicates()
        m["season"] = season
        rows.append(m.rename(columns={"team": "team_id", "team_code": "club_code"}))
    return pd.concat(rows, ignore_index=True)


def _attach_club_code(agg: pd.DataFrame) -> pd.DataFrame:
    lut = _club_code_lookup()
    agg = agg.merge(lut, on=["season", "team_id"], how="left")
    opp = lut.rename(columns={"team_id": "opponent_id", "club_code": "opponent_code"})
    agg = agg.merge(opp, on=["season", "opponent_id"], how="left")
    return agg


def build(db_path: Path = Path("data/fpl.duckdb")) -> pd.DataFrame:
    con = duckdb.connect(str(db_path))
    p = con.execute("""
        SELECT season, fixture, gw, team_id, opponent_team, was_home,
               kickoff_time, team_h_score, team_a_score,
               expected_goals, expected_goals_conceded, minutes
        FROM player_gw
        WHERE team_id IS NOT NULL AND fixture IS NOT NULL
    """).df()

    p["was_home"] = p["was_home"].astype(bool)

    agg = (p.groupby(["season", "fixture", "team_id"], as_index=False)
             .agg(gw=("gw", "first"),
                  opponent_id=("opponent_team", "first"),
                  was_home=("was_home", "first"),
                  kickoff_time=("kickoff_time", "first"),
                  team_h_score=("team_h_score", "first"),
                  team_a_score=("team_a_score", "first"),
                  xg_for=("expected_goals", "sum"),
                  minutes=("minutes", "sum")))

    agg["goals_for"] = np.where(agg["was_home"], agg["team_h_score"], agg["team_a_score"])
    agg["goals_against"] = np.where(agg["was_home"], agg["team_a_score"], agg["team_h_score"])

    # xG against is the opponent's xG for the same fixture -- self-join rather
    # than using player expected_goals_conceded, which is a per-player share and
    # does not sum to a team total.
    opp = agg[["season", "fixture", "team_id", "xg_for"]].rename(
        columns={"team_id": "opponent_id", "xg_for": "xg_against"})
    agg = agg.merge(opp, on=["season", "fixture", "opponent_id"], how="left")

    # xG is only published from 2022/23 -- and within 2022/23 only from GW16, when
    # FPL began publishing it at the World Cup break. Those 272 rows come through
    # as exactly 0.000, which is absence, not a goalless expectation: the smallest
    # genuine team xG observed anywhere is 0.02. Left as zeros they enter the fit
    # as real observations and corrupt every share computed from them.
    no_xg = (agg["season"] < "2022-23") | (agg["xg_for"] == 0) | (agg["xg_against"] == 0)
    agg.loc[no_xg, ["xg_for", "xg_against"]] = np.nan

    # FPL reassigns team_id every season (alphabetical), so it cannot be used to
    # follow a club across seasons -- id 2 is Bournemouth in 2016/17 and Aston
    # Villa in 2025/26. `team_code` IS stable (Arsenal is 3 throughout, including
    # in the live 2026/27 API), so it is the identity used for cross-season
    # priors and for carrying strength into a new season.
    agg = _attach_club_code(agg)

    agg["kickoff_time"] = pd.to_datetime(agg["kickoff_time"], errors="coerce", utc=True)
    agg["clean_sheet"] = (agg["goals_against"] == 0).astype(int)
    agg = agg.drop(columns=["team_h_score", "team_a_score"])

    con.execute("DROP TABLE IF EXISTS team_match")
    con.execute("CREATE TABLE team_match AS SELECT * FROM agg")
    con.close()
    return agg


if __name__ == "__main__":
    t = build()
    print(f"team_match: {len(t):,} rows")
    print(f"  seasons {t.season.nunique()}  fixtures {t.fixture.nunique()}")
    print(f"  goals/match {t.goals_for.mean():.3f}  home adv "
          f"{t[t.was_home].goals_for.mean() - t[~t.was_home].goals_for.mean():+.3f}")
    print(f"  club_code nulls {t.club_code.isna().sum()}  distinct clubs {t.club_code.nunique()}")
    print(f"  clean sheet rate {t.clean_sheet.mean():.3f}  xG coverage {t.xg_for.notna().mean():.1%}")
