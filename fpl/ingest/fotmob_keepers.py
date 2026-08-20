"""Goalkeeper metrics from post-shot expected goals.

PsxG (post-shot xG) and xGOT are the same quantity under two names: the value of
a shot recomputed *after* it is struck, conditioning on where it crossed the goal
line. For an outfielder it measures placement. For a goalkeeper it is the right
denominator for shot-stopping, because it asks the only fair question --

    given the shots he actually faced, how many would an average keeper concede?

    goals prevented = SUM PsxG(on-target shots faced)  -  goals conceded

Ordinary save percentage cannot do this. A keeper behind a poor defence faces
better chances and will show a worse save rate while playing better; PsxG divides
that out. Off-target shots are excluded entirely: they carry PsxG of zero and were
never the keeper's problem.

The cached shotmaps carry `keeperId`, so shots are attributed to the goalkeeper
who actually faced them rather than to the defending team. An earlier version
aggregated by team, which conflated two keepers at clubs that rotated or lost
someone to injury, and produced a null result on 20 data points.

`goalCrossedY` and `goalCrossedZ` give the crossing point in the goal mouth,
which supports a placement breakdown: keepers are systematically better in some
areas of the goal than others, and low shots to the corners are the classic
weakness.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Goal mouth: 7.32m wide, 2.44m high. FotMob reports the crossing point in pitch
# coordinates for Y (centred on 34) and metres above ground for Z.
GOAL_HALF_W = 3.66
PITCH_MID_Y = 34.0
GOAL_H = 2.44


def parse_cache(cache: Path = Path("data/raw/fotmob/shotmaps")) -> pd.DataFrame:
    rows = []
    for f in sorted(cache.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        for s in (d.get("shots") or []):
            rows.append({
                "match_id": int(f.stem),
                "keeper_id": s.get("keeperId"),
                "shooter": s.get("playerName"),
                "team_id": s.get("teamId"),
                "xg": s.get("expectedGoals"),
                "psxg": s.get("expectedGoalsOnTarget"),
                "on_target": bool(s.get("isOnTarget")),
                "situation": s.get("situation"),
                "event_type": s.get("eventType"),
                "own_goal": bool(s.get("isOwnGoal")),
                "goal_y": s.get("goalCrossedY"),
                "goal_z": s.get("goalCrossedZ"),
            })
    return pd.DataFrame(rows)


def goal_zone(y, z) -> str:
    """Which sixth of the goal mouth the shot crossed: side x height."""
    if y is None or z is None or not np.isfinite(y) or not np.isfinite(z):
        return "unknown"
    dy = y - PITCH_MID_Y
    side = "left" if dy < -GOAL_HALF_W / 3 else ("right" if dy > GOAL_HALF_W / 3 else "centre")
    height = "low" if z < GOAL_H / 3 else ("mid" if z < 2 * GOAL_H / 3 else "high")
    return f"{height}-{side}"


def keeper_stats(shots: pd.DataFrame, min_faced: int = 20) -> pd.DataFrame:
    """Per keeper: PsxG faced, goals conceded, goals prevented."""
    s = shots.dropna(subset=["keeper_id"]).copy()
    s = s[s["on_target"] & ~s["own_goal"]]
    s["psxg"] = pd.to_numeric(s["psxg"], errors="coerce").fillna(0.0)
    s["is_goal"] = s["event_type"].eq("Goal")
    s["is_pen"] = s["situation"].eq("Penalty")
    s["zone"] = [goal_zone(y, z) for y, z in zip(s["goal_y"], s["goal_z"])]

    g = s.groupby("keeper_id")
    out = g.apply(lambda d: pd.Series({
        "matches": d.match_id.nunique(),
        "faced": len(d),
        "psxg_faced": d.psxg.sum(),
        "conceded": int(d.is_goal.sum()),
        # Penalties are excluded from the headline: facing them is mostly luck of
        # the draw and they carry very high PsxG, which flatters a keeper who
        # happened to face several.
        "faced_np": int((~d.is_pen).sum()),
        "psxg_np": d.loc[~d.is_pen, "psxg"].sum(),
        "conceded_np": int((d.is_goal & ~d.is_pen).sum()),
    }), include_groups=False).reset_index()

    out = out[out.faced_np >= min_faced].copy()
    out["goals_prevented"] = out.psxg_np - out.conceded_np
    out["gp_per_shot"] = out.goals_prevented / out.faced_np.clip(lower=1)
    out["save_pct"] = 1 - out.conceded_np / out.faced_np.clip(lower=1)
    out["expected_save_pct"] = 1 - out.psxg_np / out.faced_np.clip(lower=1)
    out["save_pct_oe"] = out.save_pct - out.expected_save_pct
    return out


def zone_breakdown(shots: pd.DataFrame) -> pd.DataFrame:
    """League-wide conversion by area of the goal, for context on the panel."""
    s = shots.dropna(subset=["keeper_id"]).copy()
    s = s[s["on_target"] & ~s["own_goal"] & ~s["situation"].eq("Penalty")]
    s["psxg"] = pd.to_numeric(s["psxg"], errors="coerce").fillna(0.0)
    s["is_goal"] = s["event_type"].eq("Goal")
    s["zone"] = [goal_zone(y, z) for y, z in zip(s["goal_y"], s["goal_z"])]
    z = s.groupby("zone").agg(shots=("is_goal", "size"), goals=("is_goal", "sum"),
                              psxg=("psxg", "mean")).reset_index()
    z["conversion"] = z.goals / z.shots.clip(lower=1)
    return z.sort_values("conversion", ascending=False)


def resolve_to_fpl(k: pd.DataFrame, season: str = "2025-26") -> pd.DataFrame:
    """Attach the stable FPL `code` so keeper metrics reach the dashboard.

    Candidates are restricted to goalkeepers. Without that constraint the
    surname fallback resolved Emiliano Martinez onto Lisandro Martinez, because
    FPL stores the first as "Martinez Romero" and the second as "Martinez" --
    both unique surnames, so the shorter one won the last-token match and a
    Manchester United defender inherited an Aston Villa keeper's save record.
    A keeper's shot-stopping can only belong to a keeper, so say so.
    """
    from fpl.ingest.fbref import normalise, manual_overrides
    pl = pd.read_parquet(f"data/raw/vaastav/players_raw/season={season}.parquet")
    pl = pl[pl["element_type"] == 1]          # goalkeepers only
    pl["full"] = (pl["first_name"].fillna("") + " " + pl["second_name"].fillna("")).map(normalise)
    pl["surname"] = pl["second_name"].fillna("").map(normalise)
    d = k.copy()
    d["norm"] = d["name"].map(normalise)
    m = d.merge(pl[["code", "full"]].rename(columns={"full": "norm"}), on="norm", how="left")
    counts = pl["surname"].value_counts()
    uniq = pl[pl["surname"].isin(counts[counts == 1].index)]
    lut = dict(zip(uniq["surname"], uniq["code"]))
    for tok in (-1, 0):
        miss = m["code"].isna()
        if miss.any():
            m.loc[miss, "code"] = m.loc[miss, "norm"].str.split().str[tok].map(lut)
    ov = manual_overrides()
    if ov:
        miss = m["code"].isna()
        m.loc[miss, "code"] = m.loc[miss, "name"].map(ov)
    return m
