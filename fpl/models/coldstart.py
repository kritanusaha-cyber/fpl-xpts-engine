"""Cold start, and the first blend of live-season data into it.

2026/27 has now played a gameweek. One gameweek is a tiny sample, but it is not
a uniformly tiny sample -- it says very different amounts about different
things, and the weights below are measured rather than assumed.

Fitted over six seasons of first gameweeks (`GW1_WEIGHT`), predicting gameweeks
2 to 7:

  * MINUTES. Who started GW1 predicts who starts next better than all of last
    season does -- MAE 0.188 against 0.228. The optimal blend puts **0.65** on
    the single gameweek, and **1.00** for goalkeepers, where the man who starts
    the opener is simply the keeper. This is the pecking order the manager has
    actually chosen, after a summer of signings and a preseason no prior can
    see.

  * SCORING RATE. The opposite. GW1 points alone predict forward points per
    start at MAE 2.45 against the prior season's 1.29, and the optimal blend is
    **0.10**. One match of finishing is noise.

A single weight for "current-season data" would be wrong in both directions at
once: far too timid about the team sheet, far too eager about the goals.

Two distinct cold-start problems:

  * PLAYERS. 78% of the 2026/27 squad appears in 2025/26 under a stable FPL
    `code` (identical coverage to `opta_code`). The other 22% -- promoted-club
    players, overseas signings, youth -- have no history at all and fall back to
    the position x price-tier prior. That is precisely the case empirical-Bayes
    shrinkage already handles: n90 = 0 puts full weight on the prior.

  * CLUBS. Promoted clubs have no Premier League matches. Rather than let the
    Dixon-Coles fit invent parameters for them, they are assigned an explicit
    promoted-club prior estimated from how promoted sides have actually
    performed in this dataset.
"""

from __future__ import annotations

import gzip
import glob
import json

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path

POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def current_squad() -> pd.DataFrame:
    """Latest bootstrap snapshot -> the 2026/27 player universe."""
    path = sorted(glob.glob("data/raw/snapshots/bootstrap/date=*/*.json.gz"))[-1]
    b = json.load(gzip.open(path, "rt"))
    e = pd.DataFrame(b["elements"])
    teams = {t["id"]: t["code"] for t in b["teams"]}
    names = {t["id"]: t["name"] for t in b["teams"]}
    e["club_code"] = e["team"].map(teams)
    e["club_name"] = e["team"].map(names)
    e["position"] = e["element_type"].map(POSITION_MAP)
    e["price"] = e["now_cost"] / 10.0
    keep = ["id", "code", "opta_code", "web_name", "position", "club_code",
            "club_name", "price", "status", "chance_of_playing_next_round",
            "penalties_order", "corners_and_indirect_freekicks_order",
            "direct_freekicks_order", "selected_by_percent"]
    return e[keep].rename(columns={"id": "element"})


def promoted_club_prior(db: str = "data/fpl.duckdb") -> dict:
    """How promoted clubs actually perform, measured rather than guessed.

    A club is 'promoted' in season S if it has team_match rows in S but none in
    S-1. Their first-season attack/defence relative to the league average is the
    prior applied to 2026/27's promoted sides.
    """
    con = duckdb.connect(db)
    t = con.execute("""SELECT season, club_code, avg(goals_for) gf, avg(goals_against) ga
                       FROM team_match GROUP BY season, club_code""").df()
    con.close()
    seasons = sorted(t.season.unique())
    rows = []
    for prev, cur in zip(seasons, seasons[1:]):
        was = set(t[t.season == prev].club_code)
        now = t[t.season == cur]
        newly = now[~now.club_code.isin(was)]
        league = now[["gf", "ga"]].mean()
        for _, r in newly.iterrows():
            rows.append({"gf_ratio": r.gf / league.gf, "ga_ratio": r.ga / league.ga})
    d = pd.DataFrame(rows)
    return {"n": len(d),
            "attack_ratio": float(d.gf_ratio.mean()),
            "defence_ratio": float(d.ga_ratio.mean())}


PENALTY_XG = 0.79

# sigma2_within / sigma2_between for xG share, estimated on 2022/23-2025/26.
# See fpl/backtest/fit_shrinkage.py.
SHRINK_K = {"GKP": 12.0, "DEF": 15.7, "MID": 6.5, "FWD": 15.7}

# Weight on gameweek 1 when blending it against the prior-season estimate.
# Fitted on 2020-21 through 2025-26 by minimising MAE against gameweeks 2-7.
# Keepers sit at 1.0 because the choice is binary and the manager has made it.
GW1_WEIGHT = {"minutes": {"GKP": 1.00, "DEF": 0.65, "MID": 0.60, "FWD": 0.65},
              "rate": 0.10}


def _strip_penalty_xg(hist: pd.DataFrame, season: str) -> pd.DataFrame:
    """Subtract penalty xG from each player's season xG, spread across appearances."""
    path = Path("data/raw/fbref/penalties_2025_26.parquet")
    if not path.exists():
        hist["pen_xg_stripped"] = 0.0
        return hist
    pen = pd.read_parquet(path).dropna(subset=["code"])
    pen = pen[pen.pk_att > 0].groupby("code", as_index=False)["pk_att"].sum()

    pl = pd.read_parquet(f"data/raw/vaastav/players_raw/season={season}.parquet")
    pen = pen.merge(pl[["id", "code"]], on="code", how="left").dropna(subset=["id"])
    lut = dict(zip(pen["id"].astype(int), pen["pk_att"]))

    hist = hist.copy()
    hist["_pk"] = hist["element"].map(lut).fillna(0.0)
    # Allocate the player's penalties across his appearances in proportion to
    # minutes, so the subtraction lands where the xG was recorded.
    mins = hist.groupby("element")["minutes"].transform("sum").clip(lower=1)
    pen_xg = PENALTY_XG * hist["_pk"] * (hist["minutes"] / mins)
    hist["pen_xg_stripped"] = pen_xg
    hist["xg"] = (hist["xg"] - pen_xg).clip(lower=0)
    return hist.drop(columns=["_pk"])


def player_priors(db: str = "data/fpl.duckdb", source_season: str = "2025-26") -> pd.DataFrame:
    """Per-player priors carried from the most recent completed season."""
    con = duckdb.connect(db)
    hist = con.execute(f"""
        SELECT p.element, p.position, p.minutes, p.expected_goals xg,
               p.expected_assists xa, p.tackles, p.recoveries,
               p.clearances_blocks_interceptions cbi, p.saves, t.xg_for team_xg,
               t.club_code, p.fixture
        FROM player_gw p
        JOIN team_match t ON p.season=t.season AND p.fixture=t.fixture AND p.team_id=t.team_id
        WHERE p.season='{source_season}' AND p.minutes>0 AND t.xg_for IS NOT NULL
    """).df()
    con.close()

    # Strip penalty xG. FPL's expected_goals INCLUDES penalties, so a designated
    # taker's share is inflated and his open-play threat overstated. Penalty
    # attempts come from FBref (see fpl/ingest/fbref.py); a penalty is worth
    # ~0.79 xG. Penalty value is added back separately at projection time, where
    # it can be attached to whoever holds the duty NOW rather than last season.
    hist = _strip_penalty_xg(hist, source_season)

    hist["dc_n"] = np.where(hist.position == "DEF",
                            hist.tackles.fillna(0) + hist.cbi.fillna(0),
                            hist.tackles.fillna(0) + hist.cbi.fillna(0) + hist.recoveries.fillna(0))
    # Games the player's CLUB actually played, so a start rate is a share of
    # available games rather than a share of the games he happened to appear in.
    # Computing it over appearances only made every backup a near-certain
    # starter -- a reserve keeper with one 90-minute outing scored 1.00, and
    # Arsenal ended up with three goalkeepers all projecting p60 = 0.99.
    team_games = (hist.groupby("club_code")["fixture"].nunique()
                    if "club_code" in hist.columns else None)

    g = hist.groupby("element")
    agg = pd.DataFrame({
        "mins_total": g["minutes"].sum(),
        "apps": g["minutes"].count(),
        "starts_n": g["minutes"].apply(lambda s: (s >= 60).sum()),
        "xg_share": g.apply(lambda d: (d.xg.sum() / d.team_xg.sum()) if d.team_xg.sum() > 0 else 0,
                            include_groups=False),
        "xa_share": g.apply(lambda d: (d.xa.sum() / d.team_xg.sum()) if d.team_xg.sum() > 0 else 0,
                            include_groups=False),
        "dc_per90": g.apply(lambda d: d.dc_n.sum() / max(d.minutes.sum(), 1) * 90,
                            include_groups=False),
        "save_per90": g.apply(lambda d: d.saves.fillna(0).sum() / max(d.minutes.sum(), 1) * 90,
                              include_groups=False),
    }).reset_index()
    agg["n90"] = agg["mins_total"] / 90.0

    # Denominator: how many league games were available to him at his club.
    club_of = hist.groupby("element")["club_code"].first()
    agg["club_code"] = agg["element"].map(club_of)
    agg["team_games"] = agg["club_code"].map(team_games).fillna(38).clip(lower=1)
    agg["starts60"] = (agg["starts_n"] / agg["team_games"]).clip(0, 1)
    agg = agg.drop(columns=["starts_n", "club_code", "team_games"])

    # element -> stable code, so priors survive the id reshuffle between seasons
    pl = pd.read_parquet(f"data/raw/vaastav/players_raw/season={source_season}.parquet")
    agg = agg.merge(pl[["id", "code"]].rename(columns={"id": "element"}), on="element", how="left")
    return agg.drop(columns=["element"])


def _attach_setpiece(d: pd.DataFrame) -> pd.DataFrame:
    """Split a player's threat into open-play and set-piece components.

    The multiplicative model scales a player's whole xG share by projected team
    xG. That is right for open play and wrong for set pieces: a hard fixture
    suppresses open-play chances far more than it suppresses corners. A
    set-piece specialist therefore keeps more of his threat in bad fixtures than
    the model was giving him, which is part of why attackers -- and aerial
    centre-backs -- were being under-projected.

    Set-piece SHOT share is the input because it is the part that persists
    (r = 0.78 half-to-half, against r = -0.01 for xGOT placement) and because it
    is a role, which survives a transfer better than a raw volume would.
    """
    d = d.copy()
    d["sp_share"] = 0.0
    path = Path("data/raw/fotmob/setpiece_priors.parquet")
    if not path.exists():
        return d
    sp = pd.read_parquet(path)
    sp = sp[sp["shots"] >= 8]          # below this the share is noise
    lut = dict(zip(sp["code"], sp["sp_xg_share_of_own"]))
    d["sp_share"] = d["code"].map(lut).fillna(0.0).clip(0, 0.95)

    # Shrink toward the position mean on shot count, same logic as everywhere
    # else: a player with nine shots has not established a set-piece role.
    n = d["code"].map(dict(zip(sp["code"], sp["shots"]))).fillna(0.0)
    pos_mean = d.groupby("position")["sp_share"].transform("mean")
    w = n / (n + 15.0)
    d["sp_share"] = w * d["sp_share"] + (1 - w) * pos_mean
    return d


def _attach_zonal(d: pd.DataFrame) -> pd.DataFrame:
    """Territorial features, kept only where they were shown to predict.

    Validated on a within-season split (fpl/backtest/eval_zonal.py):

      box_touches_p90   persists at r = 0.894 -- more stable than xG per shot --
                        and adds extra R2 = 0.258 to predicting chances created
                        beyond final-third passes (t = 8.67). This is the
                        arriving-runner signal: a midfielder who gets into the
                        box creates and scores more than his passing suggests.

      crosses_p90       for DEFENDERS this is the assist mechanism, not box
                        presence: crosses predict next-half xA at r = 0.539
                        (t = 6.40) against r = 0.174 for box touches. A full-back
                        assists from wide, not by arriving in the area.

      six_yard_share    persists only weakly (r = 0.376) and adds nothing to
                        goals beyond xG, which already encodes shot location.
                        Carried for display, NOT used in projections.
    """
    d = d.copy()
    for c in ("box_touches_p90", "crosses_p90", "passes_ft_p90", "six_yard_share",
              "box_touch_share"):
        d[c] = np.nan
    path = Path("data/features/zonal.parquet")
    if not path.exists():
        return d
    z = pd.read_parquet(path)

    # Attach the source team id so the resolver can bridge on club, which is what
    # lifts coverage from roughly two thirds to almost all.
    st = pd.read_parquet("data/raw/fotmob/player_match_stats.parquet")
    teams = st.groupby("player_id")["team_id"].first()
    z["team_id"] = z["player_id"].map(teams)

    from fpl.resolve.players import resolve
    from fpl.ingest.fbref import manual_overrides
    z = resolve(z, overrides=manual_overrides())
    z = z.dropna(subset=["code"])
    z["code"] = z["code"].astype(int)
    # No minutes floor. A per-90 rate off 34 minutes is one substitute
    # appearance extrapolated eleven-fold, but hiding it tells the reader
    # nothing either. Show whatever exists and mark the sample thin under 450
    # minutes, which is where the rate stops moving much with one more match.
    z["low_sample"] = z["minutes"] < 450
    z = z.drop_duplicates("code")

    for c in ("box_touches_p90", "crosses_p90", "passes_ft_p90", "six_yard_share",
              "box_touch_share"):
        if c in z.columns:
            d[c] = d["code"].map(dict(zip(z["code"], z[c])))
    d["zon_low_sample"] = d["code"].map(dict(zip(z["code"], z["low_sample"]))).fillna(False)
    d["zon_minutes"] = d["code"].map(dict(zip(z["code"], z["minutes"])))

    # Creativity uplift. Applied to the xA share, which is what chances created
    # feed, and shrunk toward the positional mean so a small sample cannot move a
    # projection far. Deliberately modest: this improves the ranking of creators,
    # it does not license a large level change.
    pos_med = d.groupby("position")["box_touches_p90"].transform("median")
    lift = ((d["box_touches_p90"] - pos_med) / pos_med.clip(lower=0.5)).clip(-0.5, 0.5)
    is_def = d["position"].eq("DEF")
    cross_med = d.groupby("position")["crosses_p90"].transform("median")
    cross_lift = ((d["crosses_p90"] - cross_med) / cross_med.clip(lower=0.2)).clip(-0.5, 0.5)
    lift = np.where(is_def, cross_lift, lift)
    d["xa_share"] = d["xa_share"] * (1.0 + 0.25 * pd.Series(lift, index=d.index).fillna(0.0))
    return d


def _priors_cfg(path: Path = Path("config/transfer_priors.yaml")) -> dict:
    import yaml
    return yaml.safe_load(path.read_text())


def _attach_foreign(d: pd.DataFrame) -> pd.DataFrame:
    """Calibrated Big-5 output for players the Premier League has never seen."""
    cfg = _priors_cfg()["foreign"]
    d["foreign_xg_share"] = np.nan
    d["foreign_starts60"] = np.nan
    path = sorted(Path("data/raw/fbref").glob("big5_*.parquet"))
    if not path:
        return d
    fo = pd.concat([pd.read_parquet(p) for p in path], ignore_index=True)
    fo = fo[fo["minutes"].fillna(0) >= cfg["min_minutes"]]
    if fo.empty:
        return d
    fo = fo.sort_values("minutes").drop_duplicates("norm", keep="last")

    from fpl.ingest.fbref_foreign import normalise
    pl = pd.read_parquet("data/raw/vaastav/players_raw/season=2025-26.parquet")
    names = pd.read_parquet("data/features/coldstart_names.parquet") \
        if Path("data/features/coldstart_names.parquet").exists() else None
    if names is None:
        return d
    d = d.merge(names, on="code", how="left")
    d["norm"] = d["full_name"].map(normalise)
    d = d.merge(fo[["norm", "npg_per90_adj", "start_rate"]], on="norm", how="left")

    raw = d["npg_per90_adj"] / cfg["league_team_goals"]
    d["foreign_xg_share"] = (cfg["slope"] * raw + cfg["intercept"]).where(raw.notna())
    d["foreign_starts60"] = d["start_rate"]
    return d


def _attach_role(d: pd.DataFrame) -> pd.DataFrame:
    """Club x position role profile -- what the man he replaces actually did."""
    d["role_xg_share"] = np.nan
    try:
        from fpl.models.transfers import club_role_profiles
        prof = club_role_profiles()
    except Exception:
        return d
    if prof.empty:
        return d
    d = d.merge(prof[["club_code", "position", "xg_share"]]
                  .rename(columns={"xg_share": "role_xg_share_"}),
                on=["club_code", "position"], how="left")
    d["role_xg_share"] = d["role_xg_share_"]
    return d.drop(columns=["role_xg_share_"])



def _element_to_code() -> dict[int, int]:
    """Current season's element id -> stable FPL code, from the latest snapshot."""
    f = sorted(glob.glob("data/raw/snapshots/bootstrap/date=*/*.json.gz"))[-1]
    b = json.load(gzip.open(f))
    return {e["id"]: e["code"] for e in b["elements"]}


def live_season(db: str = "data/fpl.duckdb", season: str = "2026-27") -> pd.DataFrame:
    """Per-player aggregates from the gameweeks the live season has played.

    Returns an empty frame before the season starts, so the caller keeps
    working on priors alone without a special case.
    """
    con = duckdb.connect(db, read_only=True)
    try:
        d = con.execute(f"""
            SELECT element, position, count(*) AS gws,
                   sum(minutes) AS mins,
                   avg(CASE WHEN minutes >= 60 THEN 1.0 ELSE 0.0 END) AS starts60,
                   sum(total_points) AS pts,
                   sum(CASE WHEN minutes > 0 THEN 1 ELSE 0 END) AS apps,
                   sum(defcon) AS defcon, sum(saves) AS saves
            FROM player_gw WHERE season = '{season}'
            GROUP BY element, position
        """).df()
    except Exception:
        d = pd.DataFrame()
    finally:
        con.close()
    if d.empty:
        return d
    # player_gw keys on the element id, which FPL reassigns every season. The
    # stable code has to come from the current bootstrap, and everything
    # downstream joins on it.
    d["code"] = d["element"].map(_element_to_code())
    d = d.dropna(subset=["code"])
    d["code"] = d["code"].astype(int)
    # Points per start, undefined for a player who has not started. Left null
    # rather than zeroed -- an unused substitute has no scoring rate, and
    # filling zero would drag his blended rate down for not playing, which the
    # minutes model already accounts for separately.
    d["ppg_start"] = np.where(d.apps > 0, d.pts / d.apps.clip(lower=1), np.nan)
    d["dc_per90"] = d.defcon / (d.mins / 90).clip(lower=0.1)
    d["save_per90"] = d.saves / (d.mins / 90).clip(lower=0.1)
    return d


def _blend_live(d: pd.DataFrame, db: str, season: str = "2026-27") -> pd.DataFrame:
    """Fold the played gameweeks into the priors at the fitted weights.

    The weight rises with the number of gameweeks played: one gameweek gets the
    fitted GW1 weight, and by roughly gameweek 8 the live season should carry
    essentially all of it. Interpolating on n/(n+2) reaches 0.8 by GW8 while
    starting at the measured GW1 value.
    """
    live = live_season(db, season)
    d["live_gws"] = 0
    if live.empty:
        return d

    n = int(live.gws.max())
    d = d.merge(live[["code", "starts60", "ppg_start", "dc_per90", "save_per90"]]
                  .rename(columns={c: f"live_{c}" for c in
                                   ("starts60", "ppg_start", "dc_per90", "save_per90")}),
                on="code", how="left")
    d["live_gws"] = n

    # Minutes. The fitted GW1 weight is the floor; more gameweeks only add.
    wm = d["position"].map(GW1_WEIGHT["minutes"]).fillna(0.65)
    if n > 1:
        wm = np.maximum(wm, n / (n + 2.0))
    has = d["live_starts60"].notna()
    d.loc[has, "starts60"] = (wm[has] * d.loc[has, "live_starts60"]
                              + (1 - wm[has]) * d.loc[has, "starts60"])

    # Rates. One gameweek of finishing is noise, so the weight starts at 0.10
    # and climbs on the same schedule.
    wr = GW1_WEIGHT["rate"] if n <= 1 else max(GW1_WEIGHT["rate"], n / (n + 8.0))
    for col in ("dc_per90", "save_per90"):
        h = d[f"live_{col}"].notna() & (d["live_starts60"] > 0)
        d.loc[h, col] = wr * d.loc[h, f"live_{col}"] + (1 - wr) * d.loc[h, col]
    return d


def build(db: str = "data/fpl.duckdb") -> pd.DataFrame:
    squad = current_squad()
    priors = player_priors(db)
    d = squad.merge(priors, on="code", how="left")
    d["has_history"] = d["n90"].notna()
    d["n90"] = d["n90"].fillna(0.0)

    # Position x price-tier prior for everyone, used directly for the 22% with
    # no history and as the shrinkage target for everyone else.
    d["tier"] = (d.groupby("position")["price"]
                   .transform(lambda s: pd.qcut(s.rank(method="first"), 4,
                                                labels=False, duplicates="drop")))
    d["group"] = d["position"] + "_" + d["tier"].astype(str)
    d = _attach_foreign(d)
    d = _attach_role(d)
    d = _attach_setpiece(d)
    d = _attach_zonal(d)
    cfg = _priors_cfg()
    for col in ["xg_share", "xa_share", "dc_per90", "starts60", "save_per90"]:
        grp_mean = d.groupby("group")[col].transform("mean")
        d[f"{col}_prior"] = grp_mean

        # For a player with no PL history the tier mean is not the only thing
        # known about him. Blend in what he actually did abroad (validated) and,
        # for xg_share only, the role he is stepping into (weak but non-zero).
        base = grp_mean.copy()
        if col == "xg_share":
            fw = cfg["foreign"]["weight"]
            has_f = d["foreign_xg_share"].notna()
            base = base.where(~has_f, (1 - fw) * grp_mean + fw * d["foreign_xg_share"])
            rw = cfg["role"]["weight_xg_share"]
            has_r = d["role_xg_share"].notna()
            base = base.where(~has_r, (1 - rw) * base + rw * d["role_xg_share"])
        elif col == "starts60":
            fw = cfg["foreign"]["weight"]
            has_f = d["foreign_starts60"].notna()
            base = base.where(~has_f, (1 - fw) * grp_mean + fw * d["foreign_starts60"])

        # Empirical-Bayes weight, w = n / (n + sigma2_within/sigma2_between).
        # The ratio is estimated per position from the historical variance
        # decomposition rather than hardcoded -- it differs a lot by position:
        # midfielders separate from each other much faster (k=6.5) than
        # defenders or forwards (k~15.7), whose match-to-match noise swamps the
        # between-player signal for far longer.
        k = d["position"].map(SHRINK_K).fillna(10.0)
        w = d["n90"] / (d["n90"] + k)
        d[col] = w * d[col].fillna(base) + (1 - w) * base
    # Live-season data folded in last, so it overrides the priors rather than
    # being averaged into them, and before depth normalisation so the squad
    # constraint applies to the blended numbers.
    d = _blend_live(d, db)
    d = _normalise_squad_depth(d)
    return d


if __name__ == "__main__":
    pp = promoted_club_prior()
    print(f"promoted-club prior from {pp['n']} club-seasons:")
    print(f"  attack  {pp['attack_ratio']:.3f} x league average")
    print(f"  defence {pp['defence_ratio']:.3f} x league average goals conceded")
    d = build()
    print(f"\n2026/27 squad: {len(d)} players")
    print(f"  with 2025/26 history: {d.has_history.sum()} ({d.has_history.mean():.1%})")
    print(f"  cold (prior only)   : {(~d.has_history).sum()}")
    d.to_parquet("data/features/coldstart_2026_27.parquet", index=False)
    print("\ntop 8 by prior xG share:")
    print(d.nlargest(8, "xg_share")[["web_name","club_name","position","price","xg_share","n90"]]
            .to_string(index=False))


# Slots a typical XI fills at each position. Start probabilities within a club
# and position cannot exceed these in expectation -- only one keeper plays.
XI_SLOTS = {"GKP": 1.0, "DEF": 4.0, "MID": 4.0, "FWD": 2.0}
SHARPEN = 3.0


def _normalise_squad_depth(d: pd.DataFrame) -> pd.DataFrame:
    """Scale start probabilities so a club cannot field more players than it can.

    Independent per-player priors ignore competition for places: three Arsenal
    goalkeepers each came out at p60 ~ 0.99. Rescaling each club-position group
    so its probabilities sum to the number of XI slots enforces the constraint
    the priors are blind to, and it redistributes rather than flattens -- the
    established starter keeps most of the mass and the backups lose theirs.

    Only ever scales DOWN. A club genuinely short of options should not have its
    remaining players inflated to fill the quota.
    """
    d = d.copy()
    avail = ~d["status"].isin(["i", "s", "u", "n"])
    d["_p"] = d["starts60"].fillna(0.0) * avail

    # Allocate the slots in proportion to p^SHARPEN rather than to p directly.
    # Straight proportional scaling punishes the genuine first choice for his
    # backups' inflated priors -- Raya fell to 0.57 while being obviously
    # Arsenal's starter. Sharpening concentrates the available minutes on the
    # established player and strips them from the reserves, which is how squads
    # actually behave.
    w = d["_p"] ** SHARPEN
    wsum = d.groupby(["club_code", "position"])["_p"].transform(
        lambda s: (s ** SHARPEN).sum())
    slots = d["position"].map(XI_SLOTS).fillna(1.0)
    alloc = slots * w / wsum.replace(0, np.nan)
    # Never inflate above the standalone prior: this constraint can only remove
    # minutes, never invent them.
    d["starts60"] = np.minimum(alloc.fillna(0.0), d["_p"]).clip(0, 0.97)
    return d.drop(columns=["_p"])
