"""FotMob shot-level ingest: xGOT, and clean penalty / set-piece separation.

xGOT (expected goals on target) is xG recomputed *after* the shot is struck,
conditioning on where in the goal it ended up. The difference between the two is
placement quality:

    xGOT - xG   for a shooter  = how well he hit his chances
    xG   - xGOT for a keeper   = shot-stopping faced (and saves above expected)

FotMob exposes this per shot, which no other free source here does. The same
payload carries a `situation` tag (RegularPlay / Penalty / SetPiece / FromCorner),
which independently closes the penalty-separation gap that Understat was meant to
fill before Understat restructured.

Cost: there is no league-wide shotmap endpoint, so this is one request per match,
380 per season. Responses are cached to disk and never refetched, and requests are
paced. Run it once.

Caveat carried from the build plan: goals-minus-xG has famously poor
signal-to-noise at Premier League sample sizes. Nothing here should enter a
projection unshrunk -- see fpl/backtest/eval_xgot.py, which tests whether it
predicts anything at all before it is used.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://www.fotmob.com/api/data"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
HEADERS = {"User-Agent": UA, "Referer": "https://www.fotmob.com/"}
PL_LEAGUE_ID = 47
PAUSE = 1.2          # be a polite guest on an undocumented API


def _get(url: str, retries: int = 3) -> dict:
    for i in range(retries):
        r = requests.get(url, headers=HEADERS, timeout=45)
        if r.status_code == 200:
            return r.json()
        time.sleep(2 ** i)
    r.raise_for_status()
    return {}


def season_match_ids(season: str = "2025/2026") -> list[int]:
    d = _get(f"{API}/leagues?id={PL_LEAGUE_ID}&season={season.replace('/', '%2F')}")
    fx = (d.get("fixtures") or {}).get("allMatches") or []
    return [m["id"] for m in fx if (m.get("status") or {}).get("finished")]


def fetch_shotmaps(match_ids: list[int], cache: Path) -> pd.DataFrame:
    """One request per match, cached. Returns one row per shot."""
    cache.mkdir(parents=True, exist_ok=True)
    rows, fetched = [], 0
    for i, mid in enumerate(match_ids, 1):
        f = cache / f"{mid}.json"
        if f.exists():
            d = json.loads(f.read_text())
        else:
            d = _get(f"{API}/matchDetails?matchId={mid}")
            f.write_text(json.dumps((d.get("content") or {}).get("shotmap") or {}))
            d = (d.get("content") or {}).get("shotmap") or {}
            fetched += 1
            time.sleep(PAUSE)
        for s in (d.get("shots") or []):
            rows.append({
                "match_id": mid,
                "player_id": s.get("playerId"),
                "player_name": s.get("playerName"),
                "team_id": s.get("teamId"),
                "xg": s.get("expectedGoals"),
                "xgot": s.get("expectedGoalsOnTarget"),
                "on_target": bool(s.get("isOnTarget")),
                "situation": s.get("situation"),
                "event_type": s.get("eventType"),
                "own_goal": bool(s.get("isOwnGoal")),
                "blocked": bool(s.get("isBlocked")),
            })
        if i % 50 == 0:
            print(f"  {i}/{len(match_ids)} matches ({fetched} fetched, rest cached)")
    return pd.DataFrame(rows)


def aggregate(shots: pd.DataFrame) -> pd.DataFrame:
    """Per player: npxG, xGOT, and the placement residual."""
    s = shots.copy()
    s["xg"] = pd.to_numeric(s["xg"], errors="coerce").fillna(0.0)
    s["xgot"] = pd.to_numeric(s["xgot"], errors="coerce").fillna(0.0)
    s["is_pen"] = s["situation"].eq("Penalty")
    s["is_setpiece"] = s["situation"].isin(["SetPiece", "FromCorner", "FreeKick"])
    s["is_goal"] = s["event_type"].eq("Goal") & ~s["own_goal"]

    g = s.groupby(["player_id", "player_name"], dropna=False)
    out = g.apply(lambda d: pd.Series({
        "shots": len(d),
        "goals": int(d.is_goal.sum()),
        "xg_total": d.xg.sum(),
        "npxg": d.loc[~d.is_pen, "xg"].sum(),
        "pen_shots": int(d.is_pen.sum()),
        "setpiece_shots": int(d.is_setpiece.sum()),
        # xGOT only exists for shots on target; off-target shots contribute 0,
        # so the placement residual is computed on the on-target subset only.
        "shots_on_target": int(d.on_target.sum()),
        "xgot_total": d.loc[d.on_target, "xgot"].sum(),
        "xg_on_target": d.loc[d.on_target, "xg"].sum(),
    }), include_groups=False).reset_index()

    # Placement: how much better the shot became once struck.
    out["placement"] = out["xgot_total"] - out["xg_on_target"]
    # Finishing: goals over expectation, non-penalty.
    out["finishing"] = out["goals"] - out["xg_total"]
    return out


def build(season: str = "2025/2026",
          out: Path = Path("data/raw/fotmob"),
          cache: Path = Path("data/raw/fotmob/shotmaps")) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    ids = season_match_ids(season)
    print(f"{season}: {len(ids)} finished matches")
    shots = fetch_shotmaps(ids, cache)
    tag = season.replace("/", "_")
    shots.to_parquet(out / f"shots_{tag}.parquet", index=False)
    agg = aggregate(shots)
    agg.to_parquet(out / f"player_shooting_{tag}.parquet", index=False)
    return agg


if __name__ == "__main__":
    a = build()
    print(f"\nplayers with shots: {len(a)}")
    print(f"  total shots {int(a.shots.sum()):,}  on target {int(a.shots_on_target.sum()):,}")
    print(f"  penalties {int(a.pen_shots.sum())}  set pieces {int(a.setpiece_shots.sum())}")
    print()
    print(a.nlargest(8, "xgot_total")[
        ["player_name", "shots", "goals", "npxg", "xgot_total", "placement", "finishing"]
    ].to_string(index=False, float_format=lambda v: f"{v:.2f}"))


def resolve_to_fpl(agg: pd.DataFrame, season: str = "2025-26") -> pd.DataFrame:
    """Match FotMob players to the stable FPL `code`.

    Same conservative policy as the FBref resolver: exact normalised full name,
    then a unique-surname fallback, then the shared manual override table. An
    unmatched player stays unmatched rather than being fuzzily attached to
    someone else -- a misattributed set-piece role moves two players at once.
    """
    from fpl.ingest.fbref import normalise, manual_overrides

    pl = pd.read_parquet(f"data/raw/vaastav/players_raw/season={season}.parquet")
    pl["full"] = (pl["first_name"].fillna("") + " " + pl["second_name"].fillna("")).map(normalise)
    pl["surname"] = pl["second_name"].fillna("").map(normalise)

    a = agg.copy()
    a["norm"] = a["player_name"].map(normalise)
    m = a.merge(pl[["code", "full"]].rename(columns={"full": "norm"}), on="norm", how="left")

    counts = pl["surname"].value_counts()
    uniq = pl[pl["surname"].isin(counts[counts == 1].index)]
    lut = dict(zip(uniq["surname"], uniq["code"]))
    for token in (-1, 0):
        miss = m["code"].isna()
        if miss.any():
            m.loc[miss, "code"] = m.loc[miss, "norm"].str.split().str[token].map(lut)

    ov = manual_overrides()
    if ov:
        miss = m["code"].isna()
        m.loc[miss, "code"] = m.loc[miss, "player_name"].map(ov)
    return m


def setpiece_priors(season_tag: str = "2025_2026") -> pd.DataFrame:
    """Per-player set-piece involvement, resolved to FPL codes.

    Reported as a SHARE of the player's own shots rather than a volume, because
    share is the part that persists (r = 0.78 half-to-half) and it transfers with
    the player: set-piece duty is a role, and roles survive a transfer better
    than the corner count of the club he left.
    """
    shots = pd.read_parquet(f"data/raw/fotmob/shots_{season_tag}.parquet")
    s = shots.copy()
    s["xg"] = pd.to_numeric(s["xg"], errors="coerce").fillna(0.0)
    s["is_sp"] = s["situation"].isin(["SetPiece", "FromCorner", "FreeKick", "ThrowInSetPiece"])
    s["is_pen"] = s["situation"].eq("Penalty")

    g = s.groupby(["player_id", "player_name"], dropna=False)
    agg = g.apply(lambda d: pd.Series({
        "shots": len(d),
        "sp_shots": int(d.is_sp.sum()),
        "sp_xg": d.loc[d.is_sp, "xg"].sum(),
        "op_xg": d.loc[~d.is_sp & ~d.is_pen, "xg"].sum(),
    }), include_groups=False).reset_index()
    agg["sp_shot_share"] = agg["sp_shots"] / agg["shots"].clip(lower=1)
    agg["sp_xg_share_of_own"] = agg["sp_xg"] / (agg["sp_xg"] + agg["op_xg"]).clip(lower=0.01)

    from fpl.resolve.players import resolve as _shared
    from fpl.ingest.fbref import manual_overrides
    # Team id lets the resolver disambiguate on club, which is what carries
    # short-form names like "Joao Pedro" onto the right FPL entry.
    st = pd.read_parquet("data/raw/fotmob/player_match_stats.parquet")
    agg["team_id"] = agg["player_id"].map(st.groupby("player_id")["team_id"].first())
    out = _shared(agg, overrides=manual_overrides())
    out = out.dropna(subset=["code"])
    out["code"] = out["code"].astype(int)
    return out[["code", "player_name", "shots", "sp_shots", "sp_shot_share",
                "sp_xg", "op_xg", "sp_xg_share_of_own"]]
