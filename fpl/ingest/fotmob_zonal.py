"""Zonal and positional player data from FotMob match detail.

Three layers the shot-only ingest could not provide:

  ZONAL SHOOTING. Shots carry pitch coordinates, so they can be bucketed by the
  zone they were taken from. A striker's six-yard-box share is a direct read on
  whether he gets on the end of things or shoots from range -- two players with
  the same xG can have very different shot maps, and the six-yard-box player is
  the one whose returns survive a drop in service.

  BOX PRESENCE AND PROGRESSION. `Touches in opposition box` and `Passes into
  final third` are the territorial measures the model otherwise has no access to.
  For a midfielder, arriving in the box is a different skill from passing into
  it, and both are separable here. For a full-back, box touches and crosses are
  the attacking-output proxy that price alone does not capture.

  REAL POSITIONS. `usualPosition` gives RB / LW / CM rather than FPL's four
  buckets, which is a better basis for like-for-like comparison than clusters
  inferred from output.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fpl.ingest.fotmob import _get, API, PAUSE

# Pitch is 105 x 68 metres in FotMob's coordinates, attacking toward x = 105.
PITCH_L, PITCH_W = 105.0, 68.0
SIX_YARD_X, PEN_X = PITCH_L - 5.5, PITCH_L - 16.5
SIX_YARD_Y = (PITCH_W / 2 - 9.16, PITCH_W / 2 + 9.16)
PEN_Y = (PITCH_W / 2 - 20.16, PITCH_W / 2 + 20.16)

# The cache already holds roughly sixty fields per player per match. These are
# the ones with a use: DefCon components first, since those are modelled rather
# than displayed, then duels and the keeper set.
STAT_KEYS = {
    "Touches": "touches",
    "Touches in opposition box": "box_touches",
    "Passes into final third": "passes_final_third",
    "Accurate crosses": "crosses",
    "Chances created": "chances_created",
    "Big chances created": "big_chances_created",
    "Expected assists (xA)": "xa",
    "Accurate passes": "passes",
    "Minutes played": "minutes",
    # DefCon components. FotMob's own "Defensive actions" is a rollup whose
    # definition need not match FPL's, so the parts are kept separately and
    # summed under FPL's rules rather than trusted wholesale.
    "Tackles": "tackles",
    "Interceptions": "interceptions",
    "Clearances": "clearances",
    "Blocks": "blocks",
    "Recoveries": "recoveries",
    "Defensive actions": "fotmob_def_actions",
    "Headed clearance": "headed_clearances",
    "Blocked shots": "blocked_shots",
    "Dribbled past": "dribbled_past",
    # Duels and carrying
    "Ground duels won": "ground_duels_won",
    "Aerial duels won": "aerial_duels_won",
    "Duels won": "duels_won",
    "Duels lost": "duels_lost",
    "Successful dribbles": "dribbles_won",
    "Dispossessed": "dispossessed",
    "Was fouled": "fouled",
    "Fouls committed": "fouls",
    # Shooting
    "Total shots": "shots",
    "Shots on target": "shots_on_target",
    "Expected goals (xG)": "xg",
    "xG Non-penalty": "xg_np",
    "Expected goals on target (xGOT)": "xgot",
    "Big chances missed": "big_chances_missed",
    "Goals": "goals",
    "Assists": "assists",
    "Accurate long balls": "long_balls",
    "Corners": "corners",
    "Offsides": "offsides",
    "FotMob rating": "rating",
    # Keeper
    "Saves": "saves",
    "Goals conceded": "conceded",
    "Saves inside box": "saves_inside_box",
    "Diving save": "diving_saves",
    "Punches": "punches",
    "High claim": "high_claims",
    "Acted as sweeper": "sweeper_actions",
    "Throws": "gk_throws",
    "xGOT faced": "xgot_faced",
    "Goals prevented": "goals_prevented",
    # Rare but decisive
    "Error led to goal": "errors_led_to_goal",
    "Penalties won": "pens_won",
    "Conceded penalty": "pens_conceded",
    "Last man tackle": "last_man_tackles",
    "Clearance off the line": "line_clearances",
    "Own goal": "own_goals",
}


def zone_of(x: float, y: float) -> str:
    """Which shooting zone a shot came from."""
    if x is None or y is None:
        return "unknown"
    if x >= SIX_YARD_X and SIX_YARD_Y[0] <= y <= SIX_YARD_Y[1]:
        return "six_yard"
    if x >= PEN_X and PEN_Y[0] <= y <= PEN_Y[1]:
        return "penalty_area"
    return "outside_box"


def _num(v):
    """FotMob stat values arrive as numbers, or strings like '3 (75%)' or '1.23'."""
    if v is None:
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    m = re.match(r"\s*([\d.]+)", str(v))
    return float(m.group(1)) if m else np.nan


def parse_player_stats(content: dict) -> list[dict]:
    rows = []
    for pid, p in (content.get("playerStats") or {}).items():
        rec = {
            "player_id": p.get("id") or pid,
            "player_name": p.get("name"),
            "team_id": p.get("teamId"),
            # positionId is a pitch-slot code, not a position enum -- decoding it
            # was attempted and abandoned (84% unmapped). usualPosition collapses
            # to the same four lines FPL already gives, so granular positions are
            # not available here; role clusters remain the basis for comparison.
            "line": {0: "GK", 1: "DEF", 2: "MID", 3: "FWD"}.get(p.get("usualPosition")),
            "is_gk": bool(p.get("isGoalkeeper")),
        }
        for grp in (p.get("stats") or []):
            for label, val in (grp.get("stats") or {}).items():
                key = STAT_KEYS.get(label)
                if key:
                    # Values nest as {"key":..., "stat": {"value": N, "type": ...}}.
                    # Reading val["value"] returns None for every stat, silently
                    # producing a table of zeros rather than an error.
                    v = val
                    if isinstance(v, dict):
                        v = v.get("stat", v)
                    if isinstance(v, dict):
                        v = v.get("value")
                    rec[key] = _num(v)
        rows.append(rec)
    return rows


def fetch(match_ids: list[int], cache: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache.mkdir(parents=True, exist_ok=True)
    stats, shots = [], []
    for i, mid in enumerate(match_ids, 1):
        f = cache / f"{mid}.json"
        if f.exists():
            c = json.loads(f.read_text())
        else:
            d = _get(f"{API}/matchDetails?matchId={mid}")
            c = d.get("content") or {}
            c = {"playerStats": c.get("playerStats") or {}, "shotmap": c.get("shotmap") or {}}
            f.write_text(json.dumps(c))
            time.sleep(PAUSE)
        for r in parse_player_stats(c):
            r["match_id"] = mid
            stats.append(r)
        for s in ((c.get("shotmap") or {}).get("shots") or []):
            shots.append({
                "match_id": mid, "player_id": s.get("playerId"),
                "player_name": s.get("playerName"), "x": s.get("x"), "y": s.get("y"),
                "xg": s.get("expectedGoals"), "situation": s.get("situation"),
                "event_type": s.get("eventType"), "own_goal": bool(s.get("isOwnGoal")),
                "inside_box": bool(s.get("isFromInsideBox")),
            })
        if i % 60 == 0:
            print(f"  {i}/{len(match_ids)}")
    return pd.DataFrame(stats), pd.DataFrame(shots)


def build(out: Path = Path("data/raw/fotmob"),
          cache: Path = Path("data/raw/fotmob/detail")) -> pd.DataFrame:
    from fpl.ingest.fotmob import season_match_ids
    ids = season_match_ids()
    st, sh = fetch(ids, cache)
    sh["zone"] = [zone_of(x, y) for x, y in zip(sh["x"], sh["y"])]
    st.to_parquet(out / "player_match_stats.parquet", index=False)
    sh.to_parquet(out / "shots_zoned.parquet", index=False)
    return st


if __name__ == "__main__":
    st = build()
    print(f"player-match rows: {len(st):,}")
