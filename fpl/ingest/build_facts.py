"""Build the player_gw fact table in DuckDB from the raw season Parquet.

Key design points:

* The primary key is (season, element, gw), NOT element alone. FPL reassigns
  element ids every season.
* Position is resolved per season from players_raw for the pre-2020/21 seasons
  whose per-GW files omit it. Position is a per-season attribute -- treating it
  as a player attribute makes historical DefCon thresholds wrong.
* `defcon` is reconstructed for every season that carries the component counts,
  using the position-dependent definition verified against 2025/26 (defenders
  exclude recoveries). Seasons without components get NULL, not 0 -- the
  distinction between "no defensive actions" and "not recorded" matters.
* Every row carries `as_of_gw` semantics implicitly via `gw`; feature code must
  only ever read gw < as_of_gw.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import yaml

DEFCON_COMPONENTS = {
    "DEF": ["tackles", "clearances_blocks_interceptions"],
    "MID": ["tackles", "clearances_blocks_interceptions", "recoveries"],
    "FWD": ["tackles", "clearances_blocks_interceptions", "recoveries"],
    "GKP": [],
}
POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

# vaastav labels goalkeepers "GK" from 2020/21 and "GKP" earlier -- and 2021/22
# carries BOTH within the same season. Left unnormalised this silently drops
# every goalkeeper from position-keyed scoring lookups.
POSITION_ALIASES = {"GK": "GKP", "GKP": "GKP", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}


def _resolve_positions(gw: pd.DataFrame, season: str, raw_dir: Path) -> pd.DataFrame:
    """Fill position from players_raw where the per-GW file lacks it."""
    if "position" in gw.columns and gw["position"].notna().any():
        return gw
    pl_path = raw_dir / "vaastav" / "players_raw" / f"season={season}.parquet"
    if not pl_path.exists():
        gw["position"] = pd.NA
        return gw
    pl = pd.read_parquet(pl_path)
    if "element_type" not in pl.columns:
        gw["position"] = pd.NA
        return gw
    lut = dict(zip(pl["id"], pl["element_type"].map(POSITION_MAP)))
    gw["position"] = gw["element"].map(lut)
    return gw


def _normalise_positions(gw: pd.DataFrame) -> pd.DataFrame:
    gw["position"] = gw["position"].map(POSITION_ALIASES).astype("object")
    return gw


def _add_defcon(gw: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct DefCon counts where components exist."""
    have = {c for c in ["tackles", "clearances_blocks_interceptions", "recoveries"]
            if c in gw.columns}
    if "defensive_contribution" in gw.columns:
        gw["defcon"] = gw["defensive_contribution"]
        gw["defcon_source"] = "native"
        return gw
    if not have:
        gw["defcon"] = pd.NA
        gw["defcon_source"] = "unavailable"
        return gw

    out = pd.Series(pd.NA, index=gw.index, dtype="Float64")
    for pos, comps in DEFCON_COMPONENTS.items():
        usable = [c for c in comps if c in have]
        if len(usable) != len(comps):
            continue  # incomplete components -> leave NULL rather than undercount
        mask = gw["position"] == pos
        if mask.any():
            out.loc[mask] = gw.loc[mask, usable].sum(axis=1) if usable else 0
    gw["defcon"] = out
    gw["defcon_source"] = "reconstructed"
    return gw


# Columns carried into the fact table, in a stable order.
FACT_COLS = [
    "season", "element", "gw", "position", "team", "opponent_team", "fixture",
    "was_home", "kickoff_time", "minutes", "starts", "total_points", "xP",
    "goals_scored", "assists", "clean_sheets", "goals_conceded", "own_goals",
    "penalties_saved", "penalties_missed", "yellow_cards", "red_cards", "saves",
    "bonus", "bps", "influence", "creativity", "threat", "ict_index",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "tackles", "clearances_blocks_interceptions",
    "recoveries", "defcon", "defcon_source", "value", "selected",
    "transfers_in", "transfers_out", "name",
]


def build(raw_dir: Path = Path("data/raw"),
          db_path: Path = Path("data/fpl.duckdb")) -> pd.DataFrame:
    gw_dir = raw_dir / "vaastav" / "player_gw"
    frames = []
    for path in sorted(gw_dir.glob("season=*.parquet")):
        season = path.stem.split("=", 1)[1]
        gw = pd.read_parquet(path)
        gw["season"] = season
        gw = _resolve_positions(gw, season, raw_dir)
        gw = _normalise_positions(gw)
        gw = _add_defcon(gw)
        for c in FACT_COLS:
            if c not in gw.columns:
                gw[c] = pd.NA
        frames.append(gw[FACT_COLS])

    facts = pd.concat(frames, ignore_index=True)
    facts["gw"] = pd.to_numeric(facts["gw"], errors="coerce").astype("Int64")
    facts["element"] = pd.to_numeric(facts["element"], errors="coerce").astype("Int64")

    # DefCon became a scoring rule in 2025/26. Reconstructed counts for earlier
    # seasons are a valid model covariate but must never be scored -- doing so
    # awards points that were not available at the time.
    facts["defcon_scoring_active"] = facts["season"] >= "2025-26"

    # vaastav carries a handful of exact duplicate rows (10 in 2025/26). They are
    # byte-identical, so deduping is safe -- but report the count rather than
    # dropping silently, because a growing number would signal upstream breakage.
    key = ["season", "element", "gw", "fixture"]
    before = len(facts)
    facts = facts.drop_duplicates(subset=key, keep="first").reset_index(drop=True)
    dropped = before - len(facts)
    if dropped:
        print(f"  deduped {dropped} exact-duplicate rows on {key}")
    if dropped > 100:
        raise ValueError(f"unexpected duplicate volume ({dropped}) -- inspect upstream")

    con = duckdb.connect(str(db_path))
    con.execute("DROP TABLE IF EXISTS player_gw")
    con.execute("CREATE TABLE player_gw AS SELECT * FROM facts")
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS pk_player_gw "
                "ON player_gw(season, element, gw, fixture)")
    con.close()
    return facts


if __name__ == "__main__":
    f = build()
    print(f"player_gw: {len(f):,} rows, {len(f.columns)} cols")
    print(f"  seasons: {f.season.nunique()}  players: {f.groupby('season').element.nunique().sum():,} player-seasons")
