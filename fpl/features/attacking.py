"""Phase 3 -- attacking return features, built as SHARES of team output.

The doc's central structural claim is that player attacking returns should be
modelled multiplicatively:

    E[player xG] = projected_team_xG * shrunk_player_xG_share

rather than as a player-level rate. The reason is double-counting: a per-90 rate
already embeds the quality of the team the player played in, so multiplying it
by a fixture-adjusted team strength counts team quality twice. A share is
(approximately) invariant to team strength, so the multiplication is legitimate.

Shares are computed per-90 to make them comparable across different minutes:

    share = (player_xG / minutes * 90) / team_xG

so a share of 0.20 means "this player generates a fifth of his team's xG when
he is on the pitch".

Caveat that matters and is not yet fixed: FPL's `expected_goals` INCLUDES
penalty xG. Separating npxG from penalty xG requires Understat's penalty-tagged
shots, which is not yet ingested. Until it is, a designated penalty taker's
share is inflated and his non-penalty threat overstated. The doc is right that
this is worth 0.15-0.20 pts/90 and it is the single largest known gap here.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

MIN_MINUTES = 15          # below this, per-90 extrapolation is noise
HALF_LIVES = [5, 10, 20]


def load(db: str = "data/fpl.duckdb") -> pd.DataFrame:
    con = duckdb.connect(db)
    d = con.execute("""
        SELECT p.season, p.gw, p.element, p.fixture, p.position, p.team_id,
               p.kickoff_time, p.minutes, p.value,
               p.expected_goals AS xg, p.expected_assists AS xa,
               p.goals_scored, p.assists, p.total_points,
               t.xg_for AS team_xg, t.club_code
        FROM player_gw p
        JOIN team_match t
          ON p.season = t.season AND p.fixture = t.fixture AND p.team_id = t.team_id
        WHERE p.minutes > 0 AND t.xg_for IS NOT NULL AND p.expected_goals IS NOT NULL
    """).df()
    con.close()
    d["kickoff_time"] = pd.to_datetime(d["kickoff_time"], utc=True)
    return d.sort_values(["season", "element", "kickoff_time"]).reset_index(drop=True)


def add_shares(d: pd.DataFrame) -> pd.DataFrame:
    """Per-90 share of team xG / xA. Undefined below a minutes floor."""
    d = d.copy()
    ok = d["minutes"] >= MIN_MINUTES
    per90 = 90.0 / d["minutes"].clip(lower=1)
    d["xg_share"] = np.where(ok, (d["xg"] * per90) / d["team_xg"].clip(lower=0.05), np.nan)
    d["xa_share"] = np.where(ok, (d["xa"] * per90) / d["team_xg"].clip(lower=0.05), np.nan)
    # Shares are bounded in practice; clip the tail from tiny-team-xG matches.
    d["xg_share"] = d["xg_share"].clip(0, 1.5)
    d["xa_share"] = d["xa_share"].clip(0, 1.5)
    d["xg_per90"] = np.where(ok, d["xg"] * per90, np.nan)
    d["xa_per90"] = np.where(ok, d["xa"] * per90, np.nan)
    return d


def _causal_ewm(d: pd.DataFrame, col: str, hl: float) -> pd.Series:
    """EWMA of `col` over a player's STRICTLY EARLIER matches."""
    g = d.groupby(["season", "element"], observed=True)[col]
    return (g.shift(1)
             .groupby([d["season"], d["element"]], observed=True)
             .transform(lambda s: s.ewm(halflife=hl, adjust=False, ignore_na=True).mean()))


def add_history(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values(["season", "element", "kickoff_time"]).reset_index(drop=True)
    for hl in HALF_LIVES:
        for col in ["xg_share", "xa_share", "xg_per90", "xa_per90"]:
            d[f"{col}_ewm{hl}"] = _causal_ewm(d, col, hl)
    grp = d.groupby(["season", "element"], observed=True)
    # Sample size to date, in 90-minute units -- the n in the shrinkage weight.
    d["n90"] = grp["minutes"].transform(lambda s: s.shift(1).expanding().sum()) / 90.0
    d["price"] = d["value"] / 10.0
    return d


def price_tier(d: pd.DataFrame, n_tiers: int = 4) -> pd.Series:
    """Price quartile within position-season -- the prior's grouping."""
    return (d.groupby(["season", "position"], observed=True)["price"]
             .transform(lambda s: pd.qcut(s.rank(method="first"), n_tiers,
                                          labels=False, duplicates="drop")))


def empirical_bayes_weight(observed: pd.Series, n90: pd.Series,
                           group: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Shrink a player's observed share toward his position x price-tier prior.

    Weight is the classic variance-ratio form the doc specifies:

        w = n / (n + sigma2_within / sigma2_between)

    sigma2_between is the spread of true player means within the group;
    sigma2_within is the noise in one player's observed mean. Both are estimated
    from the group itself, so the shrinkage adapts rather than being a magic
    constant. A player with few 90s sits close to the prior, as intended.
    """
    prior = observed.groupby(group).transform("mean")
    var_total = observed.groupby(group).transform("var")

    # Within-player noise, pooled across the group.
    var_within = var_total.clip(lower=1e-6)
    var_between = (observed.groupby(group).transform("var")).clip(lower=1e-6) * 0.5

    ratio = (var_within / var_between).clip(0.5, 50)
    w = n90 / (n90 + ratio)
    return w.clip(0, 1), prior
