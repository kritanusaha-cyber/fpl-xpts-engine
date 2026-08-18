"""Does role inheritance beat the price-tier prior for new signings?

Test set: every player who appeared in season S having never appeared in any
earlier season in the warehouse. For each, we predict his season-S profile using
only information available before S, and score it against what he actually did.

Priors compared:
  tier  -- position x price-quartile mean (the current fallback)
  role  -- the club x position profile from S-1, i.e. what the man he is
           replacing actually did
  blend -- a weighted average of the two

If role does not beat tier out of sample, it does not go in.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from fpl.models.transfers import club_role_profiles, league_profiles, METRICS

SEASONS = ["2023-24", "2024-25", "2025-26"]


def season_actuals(db: str, season: str) -> pd.DataFrame:
    con = duckdb.connect(db)
    d = con.execute(f"""
        SELECT p.element, p.position, t.club_code, p.minutes, p.value,
               p.expected_goals AS xg, p.expected_assists AS xa,
               p.tackles, p.recoveries, p.clearances_blocks_interceptions AS cbi,
               t.xg_for AS team_xg
        FROM player_gw p
        JOIN team_match t
          ON p.season=t.season AND p.fixture=t.fixture AND p.team_id=t.team_id
        WHERE p.season='{season}' AND p.minutes>0 AND t.xg_for IS NOT NULL
    """).df()
    con.close()
    d["dc_n"] = np.where(d.position == "DEF",
                         d.tackles.fillna(0) + d.cbi.fillna(0),
                         d.tackles.fillna(0) + d.cbi.fillna(0) + d.recoveries.fillna(0))
    g = (d.groupby(["element", "position", "club_code"])
           .agg(mins=("minutes", "sum"), xg=("xg", "sum"), xa=("xa", "sum"),
                dc=("dc_n", "sum"), team_xg=("team_xg", "sum"),
                starts=("minutes", lambda s: (s >= 60).mean()),
                price=("value", "first")).reset_index())
    g["xg_share"] = g.xg / g.team_xg.clip(lower=0.1)
    g["xa_share"] = g.xa / g.team_xg.clip(lower=0.1)
    g["dc_per90"] = g.dc / g.mins.clip(lower=1) * 90
    g["starts60"] = g.starts
    g["price"] = g.price / 10.0
    return g


def _code_map(season: str) -> pd.DataFrame:
    """element -> stable FPL `code` for one season."""
    pl = pd.read_parquet(f"data/raw/vaastav/players_raw/season={season}.parquet")
    return pl[["id", "code"]].rename(columns={"id": "element"})


def newcomers(db: str, season: str) -> set:
    """Players with no Premier League appearance in any earlier season.

    Must be resolved through the stable `code`: FPL reassigns `element` ids every
    season, so an element-based comparison labels every returning player a new
    signing. That bug silently filled an earlier version of this test with
    players who had been in the league for years.
    """
    con = duckdb.connect(db)
    seasons = [s for s in sorted(
        con.execute("SELECT DISTINCT season FROM player_gw").df().season) if s < season]
    prior_codes = set()
    for s in seasons:
        el = set(con.execute(
            f"SELECT DISTINCT element FROM player_gw WHERE season='{s}' AND minutes>0"
        ).df().element)
        cm = _code_map(s)
        prior_codes |= set(cm[cm.element.isin(el)].code)
    cur_el = set(con.execute(
        f"SELECT DISTINCT element FROM player_gw WHERE season='{season}' AND minutes>0"
    ).df().element)
    con.close()
    cm = _code_map(season)
    cur = cm[cm.element.isin(cur_el)]
    new_codes = set(cur.code) - prior_codes
    return set(cur[cur.code.isin(new_codes)].element)


def evaluate(db: str = "data/fpl.duckdb", min_mins: int = 450) -> pd.DataFrame:
    seasons = sorted({s for s in SEASONS})
    rows = []
    for season in seasons:
        prev = f"{int(season[:4])-1}-{str(int(season[:4]))[2:]}"
        prof = club_role_profiles(db, prev)
        if prof.empty:
            continue
        lg = league_profiles(prof)
        act = season_actuals(db, season)
        act = act[act.mins >= min_mins]
        new = newcomers(db, season)
        act = act[act.element.isin(new)]
        if not len(act):
            continue

        # tier prior, built only from the PREVIOUS season
        prev_act = season_actuals(db, prev)
        prev_act = prev_act[prev_act.mins >= min_mins].copy()
        prev_act["tier"] = (prev_act.groupby("position")["price"]
                              .transform(lambda s: pd.qcut(s.rank(method="first"), 4,
                                                           labels=False, duplicates="drop")))
        tier_prior = prev_act.groupby(["position", "tier"])[METRICS].mean().reset_index()

        act = act.copy()
        # assign each newcomer to a tier using the previous season's price bands
        bands = prev_act.groupby("position")["price"].quantile([.25, .5, .75]).unstack()
        def tier_of(r):
            b = bands.loc[r.position] if r.position in bands.index else None
            if b is None: return 1
            return int(sum(r.price > b.values))
        act["tier"] = act.apply(tier_of, axis=1)
        act = act.merge(tier_prior, on=["position", "tier"], how="left",
                        suffixes=("", "_tier"))
        act = act.merge(prof, on=["club_code", "position"], how="left",
                        suffixes=("", "_role"))
        act = act.merge(lg, on="position", how="left")

        for m in METRICS:
            act[f"{m}_tier"] = act.get(f"{m}_tier", act[f"lg_{m}"]).fillna(act[f"lg_{m}"])
            act[f"{m}_role"] = act[f"{m}_role"].fillna(act[f"lg_{m}"])
        act["season"] = season
        rows.append(act)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def score(d: pd.DataFrame, weights=(0.0, 0.25, 0.5, 0.75, 1.0)) -> None:
    print(f"newcomers scored: {len(d)}  "
          f"({', '.join(f'{s}: {n}' for s, n in d.season.value_counts().sort_index().items())})\n")
    for m in METRICS:
        y = d[m].to_numpy()
        t, r = d[f"{m}_tier"].to_numpy(), d[f"{m}_role"].to_numpy()
        print(f"  {m}")
        best = None
        for w in weights:
            p = w * r + (1 - w) * t
            mae = np.mean(np.abs(p - y))
            corr = np.corrcoef(p, y)[0, 1] if np.std(p) > 1e-9 else float("nan")
            lab = ("tier only" if w == 0 else "role only" if w == 1 else f"blend w_role={w}")
            print(f"    {lab:18} MAE {mae:.4f}   corr {corr:+.3f}")
            if best is None or mae < best[1]:
                best = (w, mae)
        print(f"    -> best w_role = {best[0]}\n")


if __name__ == "__main__":
    d = evaluate()
    if d.empty:
        print("no newcomers found")
    else:
        d.to_parquet("data/features/transfer_eval.parquet", index=False)
        score(d)


# --------------------------------------------------------------------------
# Does the foreign-league prior add anything on top of role + tier?
# --------------------------------------------------------------------------

def foreign_check(db: str = "data/fpl.duckdb", season: str = "2025-26",
                  foreign_season: str = "2024_2025", min_mins: int = 450) -> None:
    """Score last season's Big-5 output against what PL newcomers then did.

    This is the only honest way to weight the foreign prior: it either predicts
    Premier League output out of sample or it does not.
    """
    import unicodedata
    from pathlib import Path

    path = Path(f"data/raw/fbref/big5_{foreign_season}.parquet")
    if not path.exists():
        print(f"missing {path}")
        return
    fo = pd.read_parquet(path)
    fo = fo[fo.minutes.fillna(0) >= min_mins]

    act = season_actuals(db, season)
    act = act[(act.mins >= min_mins) & act.element.isin(newcomers(db, season))]

    pl = pd.read_parquet(f"data/raw/vaastav/players_raw/season={season}.parquet")
    def norm(s):
        s = unicodedata.normalize("NFKD", str(s))
        return " ".join("".join(c for c in s if not unicodedata.combining(c))
                        .lower().replace("-", " ").replace("'", "").split())
    pl["norm"] = (pl.first_name.fillna("") + " " + pl.second_name.fillna("")).map(norm)
    act = act.merge(pl[["id", "norm"]].rename(columns={"id": "element"}),
                    on="element", how="left")

    m = act.merge(fo[["norm", "npg_per90_adj", "ast_per90_adj", "start_rate", "league"]],
                  on="norm", how="inner")
    if not len(m):
        print("no newcomers matched to Big-5 by name")
        return

    # Convert a per-90 rate into the share space the model works in.
    m["foreign_share"] = m.npg_per90_adj / 1.45          # league-average team goals
    print(f"newcomers matched to a Big-5 season: {len(m)} of {len(act)}\n")
    print(f'{"target":12}{"predictor":26}{"corr":>8}{"MAE":>9}')
    for tgt, pred, lab in [
        ("xg_share", "foreign_share", "foreign npG/90 (adj)"),
        ("starts60", "start_rate", "foreign start rate"),
    ]:
        y, x = m[tgt].to_numpy(), m[pred].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 4:
            print(f"{tgt:12}{lab:26}{'n<4':>8}")
            continue
        c = np.corrcoef(x[ok], y[ok])[0, 1]
        print(f'{tgt:12}{lab:26}{c:>+8.3f}{np.mean(np.abs(x[ok]-y[ok])):>9.4f}')
    print()
    print(m[["norm", "league", "npg_per90_adj", "foreign_share", "xg_share",
             "start_rate", "starts60"]].round(3).to_string(index=False))
