"""Does xGOT actually predict anything? Test before use.

The build plan is explicit: "Shrink hard toward xG. Goals-minus-xG has a very low
signal-to-noise ratio at PL sample sizes... do not let xGOT overperformance enter
unshrunk." That is a hypothesis, and it is testable.

Three separate claims, tested separately, because they are not equally likely:

  1. FINISHING PERSISTS. Does a player's first-half-season goals-minus-xG predict
     his second-half? This is the claim the literature is most sceptical of.
  2. PLACEMENT PERSISTS. Does xGOT-minus-xG on target -- pure shot placement,
     stripped of luck about whether the keeper saved it -- persist better than
     raw finishing? It should: it removes the goalkeeper from the measurement.
  3. IT ADDS OVER xG. Even if placement persists, does knowing it improve a
     forecast that already has xG? Only this last one justifies using it.

Split is within-season (first half vs second half) so both halves come from the
same squad, role and manager -- a cross-season split would confound persistence
with transfers and tactical change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_halves(shots: pd.DataFrame, matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign each shot to the first or second half of the season by match order."""
    order = {m: i for i, m in enumerate(matches["match_id"])}
    s = shots.copy()
    s["seq"] = s["match_id"].map(order)
    cut = s["seq"].median()
    return s[s["seq"] <= cut], s[s["seq"] > cut]


def player_agg(s: pd.DataFrame) -> pd.DataFrame:
    s = s.copy()
    s["xg"] = pd.to_numeric(s["xg"], errors="coerce").fillna(0.0)
    s["xgot"] = pd.to_numeric(s["xgot"], errors="coerce").fillna(0.0)
    s["is_pen"] = s["situation"].eq("Penalty")
    s["is_goal"] = s["event_type"].eq("Goal") & ~s["own_goal"]
    g = s.groupby("player_id")
    out = g.apply(lambda d: pd.Series({
        "shots": len(d),
        "npshots": int((~d.is_pen).sum()),
        "npgoals": int((d.is_goal & ~d.is_pen).sum()),
        "npxg": d.loc[~d.is_pen, "xg"].sum(),
        "ot": int(d.on_target.sum()),
        "xgot": d.loc[d.on_target, "xgot"].sum(),
        "xg_ot": d.loc[d.on_target, "xg"].sum(),
    }), include_groups=False).reset_index()
    out["finishing"] = out["npgoals"] - out["npxg"]
    out["placement"] = out["xgot"] - out["xg_ot"]
    # Per-shot rates, so a high-volume player is not confused with a good one.
    out["finishing_per_shot"] = out["finishing"] / out["npshots"].clip(lower=1)
    out["placement_per_ot"] = out["placement"] / out["ot"].clip(lower=1)
    return out


def evaluate(shots: pd.DataFrame, min_shots: int = 15) -> None:
    matches = pd.DataFrame({"match_id": sorted(shots["match_id"].unique())})
    h1, h2 = split_halves(shots, matches)
    a, b = player_agg(h1), player_agg(h2)
    m = a.merge(b, on="player_id", suffixes=("_1", "_2"))
    m = m[(m.npshots_1 >= min_shots) & (m.npshots_2 >= min_shots)]
    n = len(m)
    print(f"players with {min_shots}+ non-penalty shots in BOTH halves: {n}\n")
    if n < 20:
        print("too few to test")
        return

    def corr(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 10 or np.std(x[ok]) < 1e-9:
            return np.nan, np.nan
        r = np.corrcoef(x[ok], y[ok])[0, 1]
        t = r * np.sqrt((ok.sum() - 2) / max(1 - r ** 2, 1e-12))
        return r, t

    print("1. DOES IT PERSIST?  (first half -> second half, same player)")
    print(f'{"":34}{"r":>8}{"t":>8}')
    for lab, c in [("finishing per shot", "finishing_per_shot"),
                   ("placement per shot on target", "placement_per_ot"),
                   ("npxG per shot (control)", None)]:
        if c is None:
            x = m.npxg_1 / m.npshots_1.clip(lower=1)
            y = m.npxg_2 / m.npshots_2.clip(lower=1)
        else:
            x, y = m[f"{c}_1"], m[f"{c}_2"]
        r, t = corr(x, y)
        print(f'{lab:34}{r:>+8.3f}{t:>8.2f}')

    print("\n2. DOES IT PREDICT SECOND-HALF GOALS, over and above xG?")
    y = m.npgoals_2.to_numpy(float)
    base = m.npxg_2.to_numpy(float)          # oracle xG for the period being predicted
    X1 = np.column_stack([np.ones(n), base])
    b1, *_ = np.linalg.lstsq(X1, y, rcond=None)
    r1 = y - X1 @ b1
    for lab, extra in [("+ first-half finishing", m.finishing_per_shot_1.to_numpy(float)),
                       ("+ first-half placement", m.placement_per_ot_1.to_numpy(float))]:
        X2 = np.column_stack([X1, np.nan_to_num(extra)])
        b2, *_ = np.linalg.lstsq(X2, y, rcond=None)
        r2 = y - X2 @ b2
        dr2 = 1 - (r2 ** 2).sum() / (r1 ** 2).sum()
        rr, tt = corr(np.nan_to_num(extra), r1)
        print(f'  {lab:24} extra R2 {dr2:>+7.4f}   corr with xG residual {rr:>+.3f} (t={tt:>5.2f})')
    print("\n  (extra R2 near zero = xGOT adds nothing once you already know xG)")


def evaluate_keepers(shots: pd.DataFrame, min_faced: int = 25) -> None:
    """The other side of xGOT: shot-stopping.

    For a keeper, xGOT faced is how many goals an average keeper concedes from
    the shots he actually faced. Subtract the goals he did concede and you have
    saves above expectation -- a measure with the shooter's finishing luck
    divided out, which is why it is expected to persist better than the outfield
    equivalent. Same within-season split, same power check.
    """
    s = shots.copy()
    s["xgot"] = pd.to_numeric(s["xgot"], errors="coerce").fillna(0.0)
    s["is_goal"] = s["event_type"].eq("Goal") & ~s["own_goal"]
    ot = s[s["on_target"] & ~s["situation"].eq("Penalty")].copy()
    if "keeper_id" not in ot.columns:
        # FotMob shots carry the shooter, not the keeper; attribute by opposing team
        ot["conceding_team"] = ot["team_id"]
    matches = pd.DataFrame({"match_id": sorted(s["match_id"].unique())})
    order = {m: i for i, m in enumerate(matches["match_id"])}
    ot["seq"] = ot["match_id"].map(order)
    cut = ot["seq"].median()

    def agg(d):
        g = d.groupby("conceding_team")
        return g.apply(lambda x: pd.Series({
            "faced": len(x), "xgot_faced": x.xgot.sum(), "conceded": int(x.is_goal.sum()),
        }), include_groups=False).reset_index()

    a, b = agg(ot[ot.seq <= cut]), agg(ot[ot.seq > cut])
    m = a.merge(b, on="conceding_team", suffixes=("_1", "_2"))
    m = m[(m.faced_1 >= min_faced) & (m.faced_2 >= min_faced)]
    for h in ("1", "2"):
        m[f"saa_{h}"] = (m[f"xgot_faced_{h}"] - m[f"conceded_{h}"]) / m[f"faced_{h}"]
    if len(m) < 8:
        print(f"only {len(m)} teams qualify -- too few")
        return
    x, y = m.saa_1.to_numpy(float), m.saa_2.to_numpy(float)
    r = np.corrcoef(x, y)[0, 1]
    t = r * np.sqrt((len(m) - 2) / max(1 - r ** 2, 1e-12))
    print(f"shot-stopping (saves above expected per shot faced), by defending side")
    print(f"  teams: {len(m)}   first half -> second half   r = {r:+.3f}  (t = {t:+.2f})")
