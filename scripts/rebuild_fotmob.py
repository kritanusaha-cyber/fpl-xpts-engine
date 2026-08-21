"""Reparse the whole match-detail cache into season-labelled fact tables.

The cache is keyed by match id alone, so the season has to come from the
manifest the ingest job writes. Without it every season collapses into one
undated pile and any persistence test becomes impossible.
"""
import json
from pathlib import Path

import pandas as pd

from fpl.ingest.fotmob_zonal import parse_player_stats, _num

CACHE = Path("data/raw/fotmob/detail")
OUT = Path("data/raw/fotmob")


def main() -> None:
    manifest = json.loads(Path("data/raw/fotmob/season_matches.json").read_text())
    season_of = {int(m): s.replace("/", "-")[:4] + "-" + s[-2:]
                 for s, ids in manifest.items() for m in ids}

    stats, shots, teams = [], [], []
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
        # Team block. Ball possession is the covariate the DefCon model has
        # been missing; the build plan had it down as FBref-only.
        ts = ((c.get("teamStats") or {}).get("Periods") or {}).get("All") or {}
        meta = c.get("meta") or {}
        vals = {}
        for grp in (ts.get("stats") or []):
            for st in (grp.get("stats") or []):
                v = st.get("stats")
                if isinstance(v, list) and len(v) == 2:
                    vals[st.get("title")] = v
        if vals and meta.get("home"):
            for side, idx in (("home", 0), ("away", 1)):
                other = 1 - idx
                rec = {"match_id": mid, "season": season, "date": meta.get("date"),
                       "round": meta.get("round"), "is_home": side == "home",
                       "team_id": (meta.get(side) or {}).get("id"),
                       "team": (meta.get(side) or {}).get("name"),
                       "opp_id": (meta.get("away" if side == "home" else "home") or {}).get("id")}
                for k, col in [("Ball possession", "possession"),
                               ("Expected goals (xG)", "xg"),
                               ("xG on target (xGOT)", "xgot"),
                               ("Total shots", "shots"),
                               ("Shots on target", "sot"),
                               ("Touches in opposition box", "box_touches"),
                               ("Big chances", "big_chances"),
                               ("Passes", "passes"),
                               ("Tackles", "tackles"),
                               ("Interceptions", "interceptions"),
                               ("Clearances", "clearances"),
                               ("Keeper saves", "saves"),
                               ("Corners", "corners"),
                               ("Fouls committed", "fouls")]:
                    v = vals.get(k)
                    if v is None:
                        continue
                    rec[col] = _num(v[idx])
                    rec[f"opp_{col}"] = _num(v[other])
                teams.append(rec)
        if n % 400 == 0:
            print(f"  {n}/{len(files)} matches parsed", flush=True)

    st = pd.DataFrame(stats)
    sh = pd.DataFrame(shots)
    tm = pd.DataFrame(teams)
    st.to_parquet(OUT / "player_match_stats.parquet", index=False)
    sh.to_parquet(OUT / "shots_all.parquet", index=False)
    tm.to_parquet(OUT / "team_match_fotmob.parquet", index=False)
    print(f"team-matches   : {len(tm):,}  possession present "
          f"{tm.possession.notna().mean()*100:.1f}%")
    print(f"\nplayer-matches : {len(st):,}  ({st.season.nunique()} seasons)")
    print(f"shots          : {len(sh):,}")
    print(st.groupby("season").agg(player_matches=("player_id", "size"),
                                   players=("player_id", "nunique")).to_string())


if __name__ == "__main__":
    main()
