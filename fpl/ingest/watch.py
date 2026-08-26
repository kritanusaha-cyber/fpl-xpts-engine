"""Detect when the FPL feed has changed, and rebuild only when it has.

The browser cannot do this. fantasy.premierleague.com sends no CORS headers, so
a static page has no way to fetch its own live data -- the check has to run
somewhere that is not a browser, and the rebuild has to happen before the page
is served rather than inside it.

Polling is cheap; rebuilding is not. A full refresh re-simulates 38 gameweeks.
So this fetches two small endpoints, reduces them to a fingerprint, and does
nothing at all unless the fingerprint moved.

What counts as a change, and why each one matters:

  gameweek finished and checked   the only signal that new results exist, and
                                  the one that shifts every projection
  bonus added                     points are provisional until it is, so
                                  rebuilding earlier bakes in numbers that
                                  are about to change
  price changes                   move roughly 01:30 UTC and change what is
                                  affordable, which changes the optimal squad
                                  without changing a single projection
  availability and news           an injury flag is the fastest-moving thing
                                  in the feed and the most consequential

Ownership is deliberately NOT a trigger. It drifts continuously, it would fire
every poll, and the rank objective is the only thing that reads it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

API = "https://fantasy.premierleague.com/api"
STATE = Path("data/raw/watch_state.json")
LOG = Path("data/raw/watch.log")


def _get(url: str, retries: int = 3) -> dict:
    for i in range(retries):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(2 ** i)
    return {}


def fingerprint() -> dict:
    """A small, stable summary of everything worth rebuilding for."""
    b = _get(f"{API}/bootstrap-static/")
    if not b:
        return {}
    ev = [{"id": e["id"], "finished": e["finished"],
           "checked": e["data_checked"]} for e in b.get("events", [])]
    # Prices and availability, hashed rather than stored: 600 players is too
    # much to keep in a state file and the only question asked of it is
    # "different from last time".
    el = sorted((e["id"], e["now_cost"], e.get("status"),
                 e.get("chance_of_playing_next_round"))
                for e in b.get("elements", []))
    st = _get(f"{API}/event-status/")
    bonus = [(s.get("event"), s.get("bonus_added"), s.get("points"))
             for s in (st.get("status") or [])]
    return {
        "events": hashlib.sha1(json.dumps(ev, sort_keys=True).encode()).hexdigest()[:16],
        "squad": hashlib.sha1(json.dumps(el, sort_keys=True).encode()).hexdigest()[:16],
        "bonus": hashlib.sha1(json.dumps(bonus, sort_keys=True).encode()).hexdigest()[:16],
        "checked_gws": [e["id"] for e in b.get("events", [])
                        if e["finished"] and e["data_checked"]],
        "seen": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def changed(now: dict, before: dict) -> list[str]:
    if not before:
        return ["first run"]
    out = []
    for k, label in [("events", "a gameweek finished or was checked"),
                     ("bonus", "bonus points were applied"),
                     ("squad", "prices or availability moved")]:
        if now.get(k) != before.get(k):
            out.append(label)
    new = set(now.get("checked_gws", [])) - set(before.get("checked_gws", []))
    if new:
        out.append(f"new completed gameweek(s): {sorted(new)}")
    return out


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} {msg}"
    with LOG.open("a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def main(argv: list[str]) -> int:
    deploy = "--deploy" in argv
    force = "--force" in argv
    dry = "--check-only" in argv

    now = fingerprint()
    if not now:
        log("feed unreachable, doing nothing")
        return 0
    before = json.loads(STATE.read_text()) if STATE.exists() else {}
    why = changed(now, before) if not force else ["forced"]

    if not why:
        log("no change")
        return 0
    log("change detected: " + "; ".join(why))
    if dry:
        return 0

    # A rebuild that fails must not advance the fingerprint, or the change is
    # lost and the next poll sees nothing to do.
    #
    # `auto`, not `deploy`. `deploy` republishes the page from whatever is
    # already computed -- correct after a template edit, wrong here. A feed
    # change means the model has to be refitted, and pointing the watcher at
    # `deploy` produced a page that looked freshly built while carrying
    # yesterday's projections: dashboard.html rewritten, horizon_projection
    # untouched.
    target = "auto" if deploy else "refresh"
    r = subprocess.run(["make", target], capture_output=True, text=True)
    if r.returncode != 0:
        log(f"make {target} FAILED, state not advanced\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
        return 1

    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(now, indent=1))
    log(f"make {target} ok, state advanced")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
