"""Deterministic FPL scoring engine.

Given a player's realized action counts, this reproduces total_points exactly.
That makes it usable in two directions:

  * forward -- convert simulated action sets into points (Phase 5/6 need this)
  * backward -- validate the scoring config against historical outcomes

The backward direction is the important one right now: it is the only way to
confirm a scoring table without trusting documentation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

CONFIG = Path("config/scoring_2026_27.yaml")


def load_config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text())


def score(df: pd.DataFrame, cfg: dict, with_defcon: bool = True) -> pd.Series:
    """Compute FPL points per row. Requires `position` and action-count columns."""
    s = cfg["scoring"]
    pos = df["position"]

    def per_pos(mapping):
        return pos.map(mapping).astype("float64")

    mins = df["minutes"].fillna(0)
    pts = np.where(mins >= 60, s["long_play"], np.where(mins > 0, s["short_play"], 0)).astype("float64")

    pts += df["goals_scored"].fillna(0) * per_pos(s["goals_scored"])
    pts += df["assists"].fillna(0) * s["assists"]

    # Clean sheets only count for players who reached 60 minutes.
    cs = np.where(mins >= 60, df["clean_sheets"].fillna(0) * per_pos(s["clean_sheets"]), 0)
    pts += cs

    # -1 per 2 conceded, defenders and keepers only. Integer division, not linear.
    conceded_rate = per_pos(s["goals_conceded"]).fillna(0)
    pts += np.where(conceded_rate < 0, -(df["goals_conceded"].fillna(0) // 2), 0)

    # 1 point per 3 saves, keepers only.
    pts += np.where(pos == "GKP", df["saves"].fillna(0) // 3 * s["saves"], 0)

    pts += df["penalties_saved"].fillna(0) * s["penalties_saved"]
    pts += df["penalties_missed"].fillna(0) * s["penalties_missed"]
    pts += df["yellow_cards"].fillna(0) * s["yellow_cards"]
    pts += df["red_cards"].fillna(0) * s["red_cards"]
    pts += df["own_goals"].fillna(0) * s["own_goals"]
    pts += df["bonus"].fillna(0) * s["bonus"]

    if with_defcon and "defcon" in df.columns:
        thr = pos.map(cfg["defcon"]["thresholds"])
        dc_val = per_pos(s["defensive_contribution"]).fillna(0)
        hit = (df["defcon"].astype("Float64") >= thr) & thr.notna()
        pts += np.where(hit.fillna(False), dc_val, 0)

    return pd.Series(pts, index=df.index)
