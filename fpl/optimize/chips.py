"""Chip strategy: when to spend the four one-shot advantages.

FPL grants Bench Boost, Triple Captain, Wildcard and Free Hit, and in the
current rules each is granted twice -- once per half-season window. A chip is
worth whatever the gameweek it is played on is worth, so the whole problem is
timing, and timing under uncertainty about what is still coming.

The trap is that a chip looks best on the gameweek you are currently looking
at. Holding out for a better one costs nothing if a better one arrives and
costs the whole chip if the season ends first. The rule below is therefore a
threshold rule rather than an argmax: play when this gameweek clears a bar set
by what the rest of the window is likely to offer, and play unconditionally
once the window is about to close.

Values are computed from the projection alone. Nothing here reads an outcome.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Two windows under the current rules: chips granted in each half do not carry
# across. Gameweek 20 is the boundary the game uses.
WINDOWS = [(1, 19), (20, 38)]

# Chips that can only add points. The rest replace the squad and can subtract.
ADDITIVE = {"bench_boost", "triple_captain"}

# Free Hit is implemented and works -- it fires on blank gameweeks, where the
# squad cannot field eleven, and is worth +13.5 a play. It is off by default
# anyway, because it competes with Wildcard for gameweeks inside a window and
# crowds it out: across four seasons the chip set scores +139 without Free Hit
# and +109 with it, Wildcard falling from four plays worth +108 to three worth
# +37. The difference is inside the noise of four seasons, so this is a
# preference for the better-measured configuration rather than a finding that
# Free Hit is bad.
ENABLE_FREE_HIT = False

# How much better than the window's typical gameweek a chip play must look
# before it is taken. Swept; see scripts/chip_sweep.py.
# Swept over four seasons: 1.15 scored +116 and 1.00 scored +115, with the
# curve falling away above 1.3. Patience is barely rewarded, because a chip
# held for a better week is a chip at risk of expiring unused.
THRESHOLD = {"bench_boost": 1.15, "triple_captain": 1.15,
             "free_hit": 1.15, "wildcard": 1.15}


def bench_value(pool: pd.DataFrame, squad: set, pred: str, xi: set) -> float:
    """Projected points sitting on the bench -- what Bench Boost would add."""
    b = pool[pool.element.isin(squad - xi)]
    if b.empty:
        return 0.0
    return float(pd.to_numeric(b[pred], errors="coerce").fillna(0.0).sum())


def captain_value(pool: pd.DataFrame, xi: set, pred: str) -> float:
    """Extra points from the third captain multiple, over the usual double."""
    s = pool[pool.element.isin(xi)]
    if s.empty:
        return 0.0
    return float(pd.to_numeric(s[pred], errors="coerce").fillna(0.0).max())


def wildcard_value(pool: pd.DataFrame, squad: set, pred: str,
                   pick_fn, sel_col: str) -> float:
    """What a free re-solve would add: the best legal squad, less the one held.

    Wildcard removes the transfer limit for a week, so its value is the whole
    gap between the squad a manager is stuck with and the squad he would pick
    from scratch. That gap widens through a season as injuries and form drift
    accumulate, which is why the chip is worth holding rather than spending in
    gameweek two.
    """
    proj = pd.to_numeric(pool.set_index("element")[pred],
                         errors="coerce").fillna(0.0)
    have = float(proj.reindex(list(squad)).fillna(0.0).nlargest(11).sum())
    try:
        best = set(pick_fn(pool, sel_col))
    except Exception:
        return 0.0
    want = float(proj.reindex(list(best)).fillna(0.0).nlargest(11).sum())
    return max(0.0, want - have)


def window_of(gw: int) -> tuple[int, int] | None:
    for lo, hi in WINDOWS:
        if lo <= gw <= hi:
            return (lo, hi)
    return None


class ChipPlan:
    """Tracks which chips remain in each window and decides when to play one.

    A chip unused when its window closes is worth nothing, so the threshold
    falls to zero on the final gameweek of the window -- at that point any
    positive value beats letting it expire.
    """

    def __init__(self) -> None:
        self.used: dict[tuple, set] = {w: set() for w in WINDOWS}
        self.log: list[dict] = []

    def available(self, gw: int, chip: str) -> bool:
        w = window_of(gw)
        return w is not None and chip not in self.used[w]

    def _bar(self, gw: int, chip: str, baseline: float) -> float:
        """Points this play must clear.

        The bar collapses on the window's last gameweek for ADDITIVE chips
        only. Bench Boost and Triple Captain can never lose points -- they add
        a multiplier to players already owned -- so on the final gameweek any
        positive value beats letting the chip expire.

        Wildcard and Free Hit are not additive. They replace the squad, and a
        squad re-solved on noisy projections can be worse than the one held:
        forcing a wildcard at the 2025-26 window deadline cost that season 44
        points, with the damage arriving in the weeks after the chip rather
        than on the gameweek it was played. They keep a positive bar to the
        end, and expiring unused is the correct outcome when nothing clears it.
        """
        w = window_of(gw)
        if w is None:
            return np.inf
        if gw >= w[1] and chip in ADDITIVE:
            return 0.0
        return baseline * THRESHOLD[chip]

    def consider(self, gw: int, chip: str, value: float, baseline: float) -> bool:
        if not self.available(gw, chip):
            return False
        if value <= self._bar(gw, chip, baseline):
            return False
        self.used[window_of(gw)].add(chip)
        self.log.append({"gw": gw, "chip": chip, "value": round(value, 2),
                         "baseline": round(baseline, 2)})
        return True
