"""Reparse the whole match-detail cache into season-labelled fact tables.

The cache is keyed by match id alone, so the season has to come from the
manifest the ingest job writes. Without it every season collapses into one
undated pile and any persistence test becomes impossible.
"""
import json
from pathlib import Path

import pandas as pd

from fpl.ingest.fotmob_zonal import parse_player_stats

CACHE = Path("data/raw/fotmob/detail")
OUT = Path("data/raw/fotmob")


def main() -> None:
    manifest = json.loads(Path("data/raw/fotmob/season_matches.json").read_text())
    season_of = {int(m): s.replace("/", "-")[:4] + "-" + s[-2:]
                 for s, ids in manifest.items() for m in ids}

    stats, shots = [], []
    files = sorted(CACHE.glob("*.json"))
    for n, f in enumerate(files, 1):
        mid = int(f.stem)
        c = json.loads(f.read_text())
        season = season_of.get(mid)
        for r in parse_player_stats(c):
            r["match_id"] = mid
            r["season"] = season
            stats.append(r)
        for s in ((c.get("shotmap") or {}).get("shots") or []):
            shots.append({
                "match_id": mid, "season": season,
                "player_id": s.get("playerId"), "player_name": s.get("playerName"),
                "team_id": s.get("teamId"), "keeper_id": s.get("keeperId"),
                "x": s.get("x"), "y": s.get("y"),
                "xg": s.get("expectedGoals"), "psxg": s.get("expectedGoalsOnTarget"),
                "on_target": bool(s.get("isOnTarget")), "blocked": bool(s.get("isBlocked")),
                "situation": s.get("situation"), "event_type": s.get("eventType"),
                "own_goal": bool(s.get("isOwnGoal")),
                "inside_box": bool(s.get("isFromInsideBox")),
                "goal_y": s.get("goalCrossedY"), "goal_z": s.get("goalCrossedZ"),
            })
        if n % 400 == 0:
            print(f"  {n}/{len(files)} matches parsed", flush=True)

    st = pd.DataFrame(stats)
    sh = pd.DataFrame(shots)
    st.to_parquet(OUT / "player_match_stats.parquet", index=False)
    sh.to_parquet(OUT / "shots_all.parquet", index=False)
    print(f"\nplayer-matches : {len(st):,}  ({st.season.nunique()} seasons)")
    print(f"shots          : {len(sh):,}")
    print(st.groupby("season").agg(player_matches=("player_id", "size"),
                                   players=("player_id", "nunique")).to_string())


if __name__ == "__main__":
    main()
