"""Ingest vaastav/Fantasy-Premier-League historical per-GW CSVs.

The upstream schema drifts substantially across seasons: FPL published detailed
defensive counts (CBI/tackles/recoveries) in the early seasons, dropped them
around 2018/19, and reintroduced them plus `defensive_contribution` in 2025/26.
Expected-goals columns only appear from 2020/21. We therefore ingest the union
of all columns and record a per-season coverage matrix rather than forcing an
intersection schema, which would throw away most of the useful signal.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

RAW = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

SEASONS = [
    "2016-17", "2017-18", "2018-19", "2019-20", "2020-21",
    "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
]

# Columns we treat as the stable spine. Every season must supply these or the
# ingest fails loudly -- silently dropping them would corrupt the fact table.
SPINE = [
    "element", "fixture", "minutes", "total_points", "goals_scored", "assists",
    "clean_sheets", "goals_conceded", "own_goals", "penalties_saved",
    "penalties_missed", "yellow_cards", "red_cards", "saves", "bonus", "bps",
    "influence", "creativity", "threat", "ict_index", "value", "was_home",
    "opponent_team", "kickoff_time", "selected", "transfers_in",
    "transfers_out", "round",
]

POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def _get(url: str, retries: int = 3) -> bytes:
    for attempt in range(retries):
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            return r.content
        if attempt == retries - 1:
            r.raise_for_status()
        time.sleep(2 ** attempt)
    raise RuntimeError(f"unreachable: {url}")


def fetch_season_gw(season: str) -> pd.DataFrame:
    """Per-GW player rows for one season."""
    df = pd.read_csv(io.BytesIO(_get(f"{RAW}/{season}/gws/merged_gw.csv")),
                     encoding_errors="replace")
    df.columns = [c.strip().strip('"') for c in df.columns]

    missing = [c for c in SPINE if c not in df.columns]
    if missing:
        raise ValueError(f"{season}: missing spine columns {missing}")

    # `GW` is the canonical gameweek; `round` can diverge in rescheduled seasons
    # (2019/20 COVID restart in particular), so keep both and prefer GW.
    df["gw"] = df["GW"] if "GW" in df.columns else df["round"]
    df["season"] = season
    return df


def fetch_season_players(season: str) -> pd.DataFrame | None:
    """players_raw.csv -- carries element_type (position) and season-level totals.

    Position is a per-season attribute, not a player attribute. Storing it any
    other way makes historical DefCon thresholds wrong.
    """
    try:
        raw = _get(f"{RAW}/{season}/players_raw.csv")
    except requests.HTTPError:
        return None
    df = pd.read_csv(io.BytesIO(raw), encoding_errors="replace")
    df["season"] = season
    if "element_type" in df.columns:
        df["position"] = df["element_type"].map(POSITION_MAP)
    return df


def ingest_all(out_dir: Path, seasons: list[str] = SEASONS) -> pd.DataFrame:
    """Write one Parquet per season per table; return the coverage matrix."""
    gw_dir = out_dir / "vaastav" / "player_gw"
    pl_dir = out_dir / "vaastav" / "players_raw"
    gw_dir.mkdir(parents=True, exist_ok=True)
    pl_dir.mkdir(parents=True, exist_ok=True)

    coverage = {}
    for season in seasons:
        gw = fetch_season_gw(season)
        gw.to_parquet(gw_dir / f"season={season}.parquet", index=False)
        coverage[season] = {c: True for c in gw.columns}
        print(f"  {season}  player_gw: {len(gw):>6,} rows x {len(gw.columns):>2} cols")

        players = fetch_season_players(season)
        if players is not None:
            players.to_parquet(pl_dir / f"season={season}.parquet", index=False)
            print(f"  {season}  players_raw: {len(players):>5,} rows")

    matrix = pd.DataFrame(coverage).fillna(False).sort_index()
    matrix.to_csv(out_dir / "vaastav" / "column_coverage.csv")
    return matrix


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
    print(f"Ingesting {len(SEASONS)} seasons -> {out}")
    m = ingest_all(out)
    print(f"\nUnion schema: {len(m)} distinct columns across {len(m.columns)} seasons")
