"""Live-season gameweek results from the FPL API.

vaastav publishes a season after it finishes, so the running season has to come
from the game's own feed. `event/{gw}/live/` gives every player's stat line for
one gameweek; the bootstrap gives the directory needed to attach team, position
and the stable `code`.

Rows land in `player_gw` under the live season, in the same shape the historical
loader produces, so everything downstream reads one table and does not care
where a season came from.
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests

API = "https://fantasy.premierleague.com/api"
SEASON = "2026-27"

# The live feed and the warehouse disagree on two names.
RENAME = {"defensive_contribution": "defcon", "element": "element"}


def _get(url: str, retries: int = 3) -> dict:
    for i in range(retries):
        r = requests.get(url, timeout=45)
        if r.status_code == 200:
            return r.json()
        time.sleep(2 ** i)
    r.raise_for_status()
    return {}


def finished_gameweeks() -> list[int]:
    b = _get(f"{API}/bootstrap-static/")
    return [e["id"] for e in b["events"] if e["finished"] and e["data_checked"]]


def fixtures() -> pd.DataFrame:
    """The fixture list, which the live endpoint does not carry.

    `event/{gw}/live/` gives a player's stat line and the fixture id, and
    nothing about the fixture itself -- no opponent, no venue, no scoreline, no
    kickoff. Those are exactly the columns `team_match` is built from, so
    without them the live season lands in the warehouse looking complete and
    breaks the next step with a null it cannot cast.
    """
    f = pd.DataFrame(_get(f"{API}/fixtures/"))
    keep = ["id", "event", "team_h", "team_a", "team_h_score", "team_a_score",
            "kickoff_time", "finished"]
    return f[[c for c in keep if c in f.columns]]


def attach_fixture(d: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    """Venue, opponent, scoreline and kickoff, from the player's fixture id."""
    if fx.empty or "fixture" not in d.columns:
        return d
    m = d.merge(fx.rename(columns={"id": "fixture"}), on="fixture", how="left")
    home = m["team"] == m["team_h"]
    m["was_home"] = home.fillna(False).astype(bool)
    m["opponent_team"] = np.where(home, m["team_a"], m["team_h"])
    return m


def gameweek(gw: int, boot: dict | None = None) -> pd.DataFrame:
    b = boot or _get(f"{API}/bootstrap-static/")
    el = pd.DataFrame(b["elements"])
    pos = {t["id"]: t["singular_name_short"] for t in b["element_types"]}
    team = {t["id"]: t["code"] for t in b["teams"]}

    live = _get(f"{API}/event/{gw}/live/")["elements"]
    rows = []
    for e in live:
        s = dict(e["stats"])
        s["element"] = e["id"]
        # A player can have two fixtures in a double gameweek; `explain` lists
        # one entry per fixture, so its length is the count of games played.
        s["fixtures_played"] = len(e.get("explain") or [])
        s["fixture"] = ((e.get("explain") or [{}])[0]).get("fixture")
        rows.append(s)
    d = pd.DataFrame(rows).rename(columns=RENAME)

    meta = el[["id", "code", "team", "element_type", "web_name", "now_cost"]]
    d = d.merge(meta, left_on="element", right_on="id", how="left").drop(columns=["id"])
    d["position"] = d["element_type"].map(pos)
    d["team_id"] = d["team"]
    d["club_code"] = d["team"].map(team)
    d["value"] = d["now_cost"]
    d["name"] = d["web_name"]
    d["season"] = SEASON
    d["gw"] = gw
    return d


def build(db: str = "data/fpl.duckdb") -> pd.DataFrame:
    b = _get(f"{API}/bootstrap-static/")
    gws = [e["id"] for e in b["events"] if e["finished"] and e["data_checked"]]
    if not gws:
        print("no finished gameweeks yet")
        return pd.DataFrame()
    d = pd.concat([gameweek(g, b) for g in gws], ignore_index=True)
    d = attach_fixture(d, fixtures())

    out = Path("data/raw/live")
    out.mkdir(parents=True, exist_ok=True)
    d.to_parquet(out / f"{SEASON}.parquet", index=False)

    con = duckdb.connect(db)
    cols = [r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name='player_gw'").fetchall()]
    keep = [c for c in cols if c in d.columns]
    reg = d[keep].copy()
    for c in cols:
        if c not in reg.columns:
            reg[c] = None
    reg = reg[cols]
    con.register("live_rows", reg)
    con.execute(f"DELETE FROM player_gw WHERE season='{SEASON}'")
    con.execute("INSERT INTO player_gw SELECT * FROM live_rows")
    n = con.execute(f"SELECT count(*) FROM player_gw WHERE season='{SEASON}'").fetchone()[0]
    con.close()
    print(f"{SEASON}: gameweeks {gws}, {n} rows written to player_gw")
    return d


if __name__ == "__main__":
    build()
