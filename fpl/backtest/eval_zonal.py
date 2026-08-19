"""Do zonal features earn their place?

Same bar as everything else: a feature is used only if it (a) persists within a
season and (b) adds something a model already holding xG does not have. The
xGOT work is the cautionary case -- an intuitive signal that failed both.

Tested separately by line, because the football claims are different:
  attackers   six-yard-box share  -> gets on the end of service
  midfielders box touches per 90  -> arrives in the box
  full-backs  box touches per 90  -> plays high, therefore assists
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 12 or np.std(x[ok]) < 1e-9 or np.std(y[ok]) < 1e-9:
        return np.nan, np.nan, int(ok.sum())
    r = np.corrcoef(x[ok], y[ok])[0, 1]
    t = r * np.sqrt((ok.sum() - 2) / max(1 - r ** 2, 1e-12))
    return r, t, int(ok.sum())


def halves(stats: pd.DataFrame, shots: pd.DataFrame):
    order = {m: i for i, m in enumerate(sorted(stats["match_id"].unique()))}
    stats = stats.assign(seq=stats["match_id"].map(order))
    shots = shots.assign(seq=shots["match_id"].map(order))
    cut = stats["seq"].median()
    return ((stats[stats.seq <= cut], shots[shots.seq <= cut]),
            (stats[stats.seq > cut], shots[shots.seq > cut]))


def agg(stats: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    from fpl.features.zonal import territory, shot_zones
    t = territory(stats)
    z = shot_zones(shots)
    return t.merge(z, on="player_id", how="left")


def evaluate(stats: pd.DataFrame, shots: pd.DataFrame, min_min: int = 400) -> None:
    (s1, h1), (s2, h2) = halves(stats, shots)
    a, b = agg(s1, h1), agg(s2, h2)
    m = a.merge(b, on="player_id", suffixes=("_1", "_2"))
    m = m[(m.minutes_1 >= min_min) & (m.minutes_2 >= min_min)]
    print(f"players with {min_min}+ minutes in both halves: {len(m)}\n")

    print("1. PERSISTENCE (first half -> second half)")
    print(f'{"":34}{"r":>8}{"t":>7}{"n":>6}')
    for lab, col in [("box touches per 90", "box_touches_p90"),
                     ("passes into final third p90", "passes_ft_p90"),
                     ("crosses per 90", "crosses_p90"),
                     ("six-yard-box shot share", "six_yard_share"),
                     ("xG per shot (control)", "xg_per_shot")]:
        r, t, n = _corr(m[f"{col}_1"], m[f"{col}_2"])
        print(f'{lab:34}{r:>+8.3f}{t:>7.2f}{n:>6}')

    print("\n2. DOES IT PREDICT SECOND-HALF OUTPUT, over and above xG?")
    for tgt_lab, tgt, base_lab, base in [
        ("goals", "goals_2", "npxG", "npxg_2"),
        ("chances created", "chances_2", "final-third passes", "passes_ft_2"),
    ]:
        sub = m.dropna(subset=[tgt, base])
        if len(sub) < 25:
            continue
        y = sub[tgt].to_numpy(float)
        X1 = np.column_stack([np.ones(len(sub)), sub[base].to_numpy(float)])
        b1, *_ = np.linalg.lstsq(X1, y, rcond=None)
        r1 = y - X1 @ b1
        print(f"  predicting {tgt_lab}, controlling for {base_lab}:")
        for lab, col in [("box touches p90 (H1)", "box_touches_p90_1"),
                         ("six-yard share (H1)", "six_yard_share_1"),
                         ("final-third passes p90 (H1)", "passes_ft_p90_1")]:
            extra = np.nan_to_num(sub[col].to_numpy(float))
            X2 = np.column_stack([X1, extra])
            b2, *_ = np.linalg.lstsq(X2, y, rcond=None)
            r2 = y - X2 @ b2
            d = 1 - (r2 ** 2).sum() / max((r1 ** 2).sum(), 1e-9)
            rr, tt, _ = _corr(extra, r1)
            print(f'    {lab:30} extra R2 {d:>+7.4f}   corr w/ residual {rr:>+.3f} (t={tt:>5.2f})')
