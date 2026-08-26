#!/usr/bin/env python3
"""Rebuild the simulation pools, extending them back to 2020-21.

The pools stopped at 2022-23 because they need team expected goals and the
warehouse only carries those from 2022-23. FotMob now supplies six seasons of
shots, so the two missing seasons are a data problem rather than a modelling
one -- and every headline result in this project currently rests on four
seasons.

What each earlier season needs and where it comes from:
  ownership, price, minutes, points   already in the warehouse for all six
  team expected goals                 summed from FotMob shots
  player expected goals and assists    FotMob per-match player stats
  defensive counts                     FotMob components

The join from FotMob player ids to FPL codes is done per season, because a
player's FPL element id is reassigned each year while his code is stable, and
because the squads differ.
"""
import pickle
import sys

import duckdb
import numpy as np
import pandas as pd

from fpl.resolve.players import resolve
from fpl.ingest.fbref import manual_overrides

SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]


def team_xg_by_fixture() -> pd.DataFrame:
    """Team xG per club per match date, from FotMob, keyed to FPL club codes."""
    tm = pd.read_parquet("data/raw/fotmob/team_match_fotmob.parquet")
    con = duckdb.connect("data/fpl.duckdb", read_only=True)
    wh = con.execute("SELECT season, club_code, kickoff_time FROM team_match "
                     "WHERE season >= '2020-21'").df()
    con.close()
    tm["d"] = pd.to_datetime(tm.date).dt.date
    wh["d"] = pd.to_datetime(wh.kickoff_time, utc=True).dt.date
    j = wh.merge(tm[["d", "team_id", "season"]], on=["d", "season"])
    pair = j.groupby(["club_code", "team_id"]).size().rename("n").reset_index()
    best = pair.sort_values("n", ascending=False).drop_duplicates("club_code")
    mp = dict(zip(best.team_id, best.club_code))
    tm["club_code"] = tm.team_id.map(mp)
    return tm.dropna(subset=["club_code"])[["season", "d", "club_code", "xg"]]


def fotmob_players(season: str) -> pd.DataFrame:
    """Per-match player stats for a season, resolved to the FPL code."""
    st = pd.read_parquet("data/raw/fotmob/player_match_stats.parquet")
    tm = pd.read_parquet("data/raw/fotmob/team_match_fotmob.parquet")
    st = st[st.season == season]
    if st.empty:
        return pd.DataFrame()
    st = st.drop(columns=["season"]).merge(
        tm[["match_id", "team_id", "date", "season"]], on=["match_id", "team_id"])
    names = st.groupby(["player_id", "player_name"], as_index=False).minutes.sum()
    r = resolve(names.rename(columns={"player_name": "name"}), name_col="name",
                team_col=None, season=season, overrides=manual_overrides())
    r = r.dropna(subset=["code"])
    st["code"] = st.player_id.map(dict(zip(r.player_id, r.code.astype(int))))
    st["d"] = pd.to_datetime(st.date).dt.date
    return st.dropna(subset=["code"])


def main() -> None:
    txg = team_xg_by_fixture()
    con = duckdb.connect("data/fpl.duckdb", read_only=True)
    pools = {}
    for s in SEASONS:
        base = con.execute(f"""
            SELECT p.season, p.gw, p.element, p.position, p.minutes, p.total_points,
                   p.value/10.0 AS price, p.selected AS owned, p.team_id,
                   t.club_code, t.kickoff_time
            FROM player_gw p
            JOIN team_match t ON p.season=t.season AND p.fixture=t.fixture
                             AND p.team_id=t.team_id
            WHERE p.season='{s}' AND p.position IS NOT NULL""").df()
        if base.empty:
            continue
        pl = pd.read_parquet(f"data/raw/vaastav/players_raw/season={s}.parquet")[["id", "code"]]
        base = base.merge(pl, left_on="element", right_on="id", how="left").drop(columns="id")
        base["d"] = pd.to_datetime(base.kickoff_time, utc=True).dt.date

        base = base.merge(txg[txg.season == s][["d", "club_code", "xg"]]
                          .rename(columns={"xg": "team_xg"}),
                          on=["d", "club_code"], how="left")

        fm = fotmob_players(s)
        if not fm.empty:
            cols = {"xg": "xg", "xa": "xa", "tackles": "tackles",
                    "recoveries": "recoveries"}
            keep = ["code", "d"] + [c for c in cols if c in fm.columns]
            g = fm[keep].groupby(["code", "d"], as_index=False).sum()
            g["code"] = g["code"].astype(int)
            base["code"] = pd.to_numeric(base["code"], errors="coerce")
            base = base.merge(g, on=["code", "d"], how="left")
            for a, b, c in [("clearances", "blocks", "interceptions")]:
                if all(x in fm.columns for x in (a, b, c)):
                    cbi = fm.groupby(["code", "d"], as_index=False)[[a, b, c]].sum()
                    cbi["cbi"] = cbi[a] + cbi[b] + cbi[c]
                    cbi["code"] = cbi["code"].astype(int)
                    base = base.merge(cbi[["code", "d", "cbi"]], on=["code", "d"], how="left")
        pools[s] = base.drop(columns=["kickoff_time"])
        print(f"  {s}: {len(base):,} rows, team_xg on "
              f"{base.team_xg.notna().mean()*100:.0f}%, "
              f"player xg on {base.get('xg', pd.Series(dtype=float)).notna().mean()*100:.0f}%",
              flush=True)
    con.close()
    pickle.dump(pools, open("data/features/sim_pools_raw.pkl", "wb"))
    print(f"\nwrote {len(pools)} seasons to data/features/sim_pools_raw.pkl")


if __name__ == "__main__":
    sys.exit(main())
