"""Shot quality faced: PsxG per shot on target.

Goals conceded tells you a defence let goals in. It does not tell you why. Two
defences conceding the same number can be doing completely different jobs:

    a defence that concedes many weak shots
    a defence that concedes few strong ones

PsxG per shot on target separates them. It answers: **when the opposition does
hit the target, how good is the chance?**

    PsxG/SOT = PsxG faced / shots on target faced

A low value means the defence forces bad shots. Attackers get the ball in
dangerous areas rarely, or get there under pressure and scuff it. That is
defending working as intended, and it shows up here before it shows up in the
scoreline.

This also decomposes what a keeper faces into two independent things:

    xG on target faced  =  volume (SOT per match)  x  quality (PsxG/SOT)

The two are different problems for a manager picking an FPL defence. A side
facing few good shots keeps clean sheets. A side facing many weak ones concedes
eventually. Goals conceded mixes them; this pulls them apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def team_shot_quality(shots: pd.DataFrame, exclude_penalties: bool = True) -> pd.DataFrame:
    """Per DEFENDING team per match: shots faced, PsxG faced, quality per shot.

    Two corrections that the first version of this got wrong, both of which
    changed the answer:

    A shot's `team_id` names the side that TOOK it, not the side that faced it.
    Grouping on it measures shots created, not shots conceded. The defending side
    is the other team in the fixture, recovered by pairing the two ids present.

    Roughly half of the shots flagged on target carry PsxG of zero. Those are
    blocked on the way -- the flag records the trajectory, not the arrival, and
    the keeper never dealt with them. Counting them in the denominator dilutes
    quality toward zero and produced a league mean of 0.160 against a true 0.30.
    """
    s = shots.copy()
    s["psxg"] = pd.to_numeric(s["psxg"], errors="coerce").fillna(0.0)
    s["is_goal"] = s["event_type"].eq("Goal") & ~s["own_goal"]
    s = s[s["on_target"] & ~s["own_goal"] & (s["psxg"] > 0)]
    if exclude_penalties:
        # A penalty carries ~0.79 PsxG and says nothing about defensive shape.
        # Leaving it in makes a side that gave away two spot-kicks look like it
        # concedes great chances from open play.
        s = s[~s["situation"].eq("Penalty")]

    # Flip attacker -> defender: within a fixture exactly two team ids appear,
    # so the side that faced a shot is the one that did not take it.
    pairs = s.groupby("match_id")["team_id"].unique()
    lut = {}
    for mid, ids in pairs.items():
        if len(ids) == 2:
            a, b = int(ids[0]), int(ids[1])
            lut[(mid, a)] = b
            lut[(mid, b)] = a
    s["defending_team"] = [lut.get((m, int(t))) for m, t in zip(s.match_id, s.team_id)]
    s = s.dropna(subset=["defending_team"])

    g = s.groupby(["match_id", "defending_team"], as_index=False).agg(
        sot=("is_goal", "size"), psxg=("psxg", "sum"), conceded=("is_goal", "sum"))
    g = g.rename(columns={"defending_team": "team_id"})
    g["psxg_per_sot"] = g.psxg / g.sot.clip(lower=1)
    return g


def season_profile(match_level: pd.DataFrame, min_matches: int = 8) -> pd.DataFrame:
    """Season-level defensive profile, split into volume and quality."""
    t = match_level.groupby("team_id", as_index=False).agg(
        matches=("sot", "size"), sot=("sot", "sum"),
        psxg=("psxg", "sum"), conceded=("conceded", "sum"))
    t = t[t.matches >= min_matches].copy()
    t["sot_per_match"] = t.sot / t.matches            # volume: how often attacked
    t["psxg_per_sot"] = t.psxg / t.sot.clip(lower=1)  # quality: how good the shots
    t["psxg_per_match"] = t.psxg / t.matches          # the product of the two
    t["ga_per_match"] = t.conceded / t.matches
    # Keeper effect: goals prevented relative to the chances faced.
    t["goals_prevented"] = t.psxg - t.conceded
    return t


def clean_sheet_rate(match_level: pd.DataFrame) -> pd.Series:
    """Share of matches with no goal conceded, per team."""
    cs = match_level.assign(cs=(match_level.conceded == 0).astype(int))
    return cs.groupby("team_id")["cs"].mean()
