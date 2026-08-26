"""Re-basing a player's rates when his context changes.

A player's record is a record of a player *in a situation*. Move him and part
of it stops applying, and the part that stops applying is not the same for
every kind of scoring.

The case that motivated this: Elliot Anderson at Nottingham Forest was cheap,
returned eight goals and assists, and hit the defensive-contribution threshold
consistently, because Forest sit deep and their midfielders defend a great
deal. At Manchester City he plays behind the ball in a side that holds it.
Carried across unchanged, his old rates make him look like the same pick. He
is not.

Two adjustments, both measured rather than asserted, and both multiplicative on
the specific channel they affect.

DEFENSIVE VOLUME is a property of the team, not the player. Across six seasons
and 120 club-seasons, a club's midfielders make defensive actions at a rate
that correlates -0.46 with the club's possession, and the spread between clubs
is wider than possession alone explains -- 0.78x the league rate at Fulham,
1.10x at Forest, 0.85x at City. So the correction is the ratio of the new
club's measured volume to the old club's.

ATTACKING SHARE is a property of the role, and the role is defined by who else
is in the team. A player's share of his old club's chances says what he did
when the alternatives were his old team-mates. The incumbent in the slot he is
joining says what that slot is worth at the new club. Neither alone is right,
so they are blended, with the weight fitted rather than chosen.

What this is NOT: role inheritance for players with no Premier League history.
That was tested on 244 debutants and rejected -- a position-by-price-tier prior
beat it, because the listing price already encodes the club's own view of the
signing. This module only re-bases players who *have* a record.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

VOLUME = Path("data/features/club_defensive_volume.parquet")

# Weight on the incumbent's attacking share versus the player's own.
#
# Fitted on 55 players who changed club between seasons with 900+ minutes on
# both sides, predicting the share they actually achieved at the new club:
#
#     w      MAE      r
#     0.00   0.0442   0.742     his own old share alone
#     0.35   0.0373   0.783
#     0.50   0.0358   0.788     <- shipped
#     0.70   0.0350   0.776
#     1.00   0.0369   0.717     the incumbent alone
#
# Both extremes lose. His old share ignores that the alternatives around him
# have changed; the incumbent's ignores that he is a different player. 0.50 is
# taken over 0.70's marginally better MAE because it has the best correlation
# and 55 movers is a thin basis for a sharper claim.
INCUMBENT_WEIGHT = 0.50

# A correction outside this band is not a context effect, it is a bad join.
CLAMP = (0.55, 1.60)


def club_volume(season: str = "2025-26") -> pd.DataFrame:
    """Relative defensive volume by club and line, latest available season.

    Keyed on the FPL club code, not the club name. FPL writes "Man City" and
    "Nott'm Forest" where FotMob writes "Manchester City" and "Nottingham
    Forest", so a name join silently matched nothing and left every multiplier
    at 1.0 -- the correction appeared to run and did nothing.
    """
    if not VOLUME.exists():
        return pd.DataFrame()
    v = pd.read_parquet(VOLUME)
    s = season if season in set(v.season) else sorted(v.season)[-1]
    return v[v.season == s][["club_code", "team", "line", "rel", "poss"]]


def defensive_multiplier(old_club, new_club, line: str,
                         vol: pd.DataFrame | None = None) -> float:
    """How much a move changes the defensive actions a player will make.

    Returns 1.0 when either club is unknown -- a promoted side has no Premier
    League record, and inventing one is worse than leaving the rate alone.
    """
    if pd.isna(old_club) or pd.isna(new_club) or old_club == new_club:
        return 1.0
    v = club_volume() if vol is None else vol
    if v.empty:
        return 1.0
    sub = v[v.line == line].set_index("club_code")["rel"]
    if old_club not in sub.index or new_club not in sub.index:
        return 1.0
    old = float(sub[old_club])
    if old <= 0:
        return 1.0
    return float(np.clip(sub[new_club] / old, *CLAMP))


def incumbent(pool: pd.DataFrame, club: str, role: str,
              exclude: int | None = None) -> pd.Series | None:
    """The player who held this role at this club, by minutes.

    "Comparable" means the same job at the same club, not the same position.
    A holding midfielder and a number ten are both MID and share almost
    nothing about where chances come from.
    """
    c = pool[(pool.club_name == club) & (pool.role == role)]
    if exclude is not None:
        c = c[c.code != exclude]
    if c.empty:
        return None
    return c.sort_values("n90", ascending=False).iloc[0]


def rebase(d: pd.DataFrame, prev_club_col: str = "prev_club",
           w: float = INCUMBENT_WEIGHT,
           season: str | None = None, matches: dict | None = None) -> pd.DataFrame:
    """Apply both corrections to a squad frame that knows each player's old club."""
    out = d.copy()
    out["ctx_dc_mult"] = 1.0
    out["ctx_share_mult"] = 1.0
    if prev_club_col not in out.columns:
        return out

    vol = club_volume_blended(season, matches) if season else club_volume()
    line = {"GKP": "GK", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}
    moved = out[prev_club_col].notna() & (out[prev_club_col] != out["club_code"])

    for i in out.index[moved]:
        r = out.loc[i]
        out.at[i, "ctx_dc_mult"] = defensive_multiplier(
            r[prev_club_col], r["club_code"], line.get(r["position"], "MID"), vol)
        # The share blend needs role clusters, which are fitted downstream of
        # this frame. Where they are absent the defensive correction still
        # applies on its own -- it is the better-evidenced of the two and needs
        # only the club. Falling back to FPL position here would compare a
        # holding midfielder against a winger, which is worse than not doing it.
        # The incumbent's share at the new club. `role_xg_share` is the
        # club-by-position profile -- what the man whose place he is taking
        # actually did -- which is the comparable-player proxy in the form the
        # cold start already computes.
        own = r.get("xg_share")
        inc_share = r.get("role_xg_share")
        if pd.notna(inc_share) and pd.notna(own) and own > 0:
            blended = (1 - w) * own + w * float(inc_share)
            out.at[i, "ctx_share_mult"] = float(np.clip(blended / own, *CLAMP))

    out["dc_per90"] = out["dc_per90"] * out["ctx_dc_mult"]
    for c in ("xg_share", "xa_share"):
        if c in out.columns:
            out[c] = out[c] * out["ctx_share_mult"]
    return out

# How fast a club's current-season style should override last season's.
#
# This is what handles a managerial change without needing to know a manager
# changed. A new manager shows up as the club's own numbers diverging from its
# prior-season baseline, and the right response is not to detect the event but
# to trust the new evidence at the rate it earns.
#
# Fitted over six seasons, predicting the remainder of a club's season from
# what it had done so far:
#
#     matches   prior only   current only   best blend w
#        3        0.0775        0.1380          0.20
#        5        0.0796        0.1209          0.20
#        8        0.0785        0.1054          0.30
#       12        0.0783        0.0736          0.55
#       19        0.0829        0.0637          0.75
#
# Current-season style only overtakes last season's at about twelve matches,
# and the curve is close to w = n / (n + 10) -- the same empirical-Bayes form
# the player priors use.
STYLE_K = 10.0

# A club's style can move this far in a season with no transfers at all: the
# largest observed one-season shift is 0.307 of relative volume, against 0.325
# for the Forest-to-City move corrected above. Rare -- 2% of club-seasons shift
# more than 0.20 -- but when it happens it is as large as a transfer.


def club_volume_blended(season: str, matches: dict | None = None) -> pd.DataFrame:
    """Club defensive volume, blending the running season into the prior one.

    `matches` maps club_code to how many matches that club has played this
    season. With none played this returns the prior season unchanged, which is
    the correct starting point.
    """
    if not VOLUME.exists():
        return pd.DataFrame()
    v = pd.read_parquet(VOLUME)
    seasons = sorted(v.season.unique())
    if season not in seasons:
        return club_volume()
    cur = v[v.season == season]
    idx = seasons.index(season)
    if idx == 0 or not matches:
        return cur[["club_code", "team", "line", "rel", "poss"]]
    prev = v[v.season == seasons[idx - 1]].set_index(["club_code", "line"])["rel"]

    out = cur.copy()
    n = out["club_code"].map(matches).fillna(0.0)
    w = n / (n + STYLE_K)
    base = pd.MultiIndex.from_arrays([out["club_code"], out["line"]]).map(prev)
    base = pd.Series(base, index=out.index).astype(float)
    # where the club has no prior season -- promoted -- the current data is all
    # there is, so it carries full weight rather than being pulled to nothing
    out["rel"] = np.where(base.isna(), out["rel"], w * out["rel"] + (1 - w) * base)
    return out[["club_code", "team", "line", "rel", "poss"]]
