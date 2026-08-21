"""Pull every Premier League season FotMob serves into the match-detail cache.

One request per match. The detail payload carries the player stat block, the
shotmap, the team stat block and the fixture metadata, so there is nothing to
fetch twice.

The team block is the reason this stores more than it first did. It carries
ball possession, which is the covariate the DefCon model has been missing --
the build plan had it down as unavailable outside FBref. A file cached before
that was noticed is re-fetched rather than left short.

Resumable: a match already on disk is skipped, so an interrupted run costs
nothing. The job stops itself after ten consecutive failures rather than
grinding through a rate limit -- a burst of refusals means the host has had
enough, and the polite response is to stop, not to retry harder.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

from fpl.ingest.fotmob import _get, API, PAUSE, season_match_ids

CACHE = Path("data/raw/fotmob/detail")
MANIFEST = Path("data/raw/fotmob/season_matches.json")
SEASONS = ["2025/2026", "2024/2025", "2023/2024", "2022/2023", "2021/2022", "2020/2021"]


def _complete(f: Path) -> bool:
    """A cached file counts as complete only if it holds the team block."""
    if not f.exists():
        return False
    try:
        return bool(json.loads(f.read_text()).get("teamStats"))
    except Exception:
        return False


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    fetched = skipped = failed = 0
    streak = 0

    for season in SEASONS:
        if season not in manifest:
            manifest[season] = season_match_ids(season)
            MANIFEST.write_text(json.dumps(manifest))
            time.sleep(PAUSE)
        ids = manifest[season]
        todo = [m for m in ids if not _complete(CACHE / f"{m}.json")]
        print(f"{season}: {len(ids)} matches, {len(todo)} to fetch", flush=True)

        for i, mid in enumerate(todo, 1):
            try:
                d = _get(f"{API}/matchDetails?matchId={mid}")
                c = d.get("content") or {}
                g = d.get("general") or {}
                (CACHE / f"{mid}.json").write_text(json.dumps(
                    {"playerStats": c.get("playerStats") or {},
                     "shotmap": c.get("shotmap") or {},
                     "teamStats": (c.get("stats") or {}),
                     "meta": {"date": g.get("matchTimeUTCDate"),
                              "round": g.get("matchRound"),
                              "home": g.get("homeTeam"),
                              "away": g.get("awayTeam")}}))
                fetched += 1
                streak = 0
            except Exception as e:
                failed += 1
                streak += 1
                print(f"  fail {mid}: {type(e).__name__} {str(e)[:60]}", flush=True)
                if streak >= 10:
                    print("ten consecutive failures -- stopping", flush=True)
                    return 1
            time.sleep(PAUSE)
            if i % 50 == 0:
                print(f"  {season} {i}/{len(todo)} (total fetched {fetched})", flush=True)
        skipped += len(ids) - len(todo)

    print(f"done: {fetched} fetched, {skipped} already cached, {failed} failed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
