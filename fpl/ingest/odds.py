"""Historical closing odds from football-data.co.uk.

The market is a strong, free, well-calibrated prior on match outcomes. The model
is not trying to beat Pinnacle from scratch -- it is trying to add the residual
the market does not price, principally squad-level information.

Two conversions matter here:

  * De-vigging. Quoted odds imply probabilities summing to >1. We normalise
    proportionally, which is the standard approximation. It is mildly biased
    against favourites (shin's method fixes that) but is fine for a prior.
  * 1X2 + over/under 2.5 -> implied home/away goal expectations. Two markets
    give two constraints, which is exactly enough to pin down (lambda, mu)
    under an independent-Poisson assumption.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.optimize import brentq
from scipy.stats import poisson

FD = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"

SEASON_CODES = {
    "2016-17": "1617", "2017-18": "1718", "2018-19": "1819", "2019-20": "1920",
    "2020-21": "2021", "2021-22": "2122", "2022-23": "2223", "2023-24": "2324",
    "2024-25": "2425", "2025-26": "2526",
}

# football-data.co.uk -> FPL club naming. Kept explicit rather than fuzzy-matched:
# the set is small, closed, and a silent mismatch here corrupts every fixture for
# that club.
TEAM_ALIASES = {
    "Man United": "Man Utd",
    "Tottenham": "Spurs",
    "Newcastle": "Newcastle",
    "Nott'm Forest": "Nott'm Forest",
    "Sheffield United": "Sheffield Utd",
    "West Brom": "West Brom",
    "Wolves": "Wolves",
    "Leicester": "Leicester",
    "Norwich": "Norwich",
    "Huddersfield": "Huddersfield",
}


def fetch_season(season: str) -> pd.DataFrame:
    code = SEASON_CODES[season]
    r = requests.get(FD.format(code=code), timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content), encoding_errors="replace")
    df["season"] = season
    return df


def devig(probs: np.ndarray) -> np.ndarray:
    """Proportional normalisation of implied probabilities."""
    return probs / probs.sum(axis=1, keepdims=True)


def implied_1x2(df: pd.DataFrame) -> pd.DataFrame:
    """Closing average 1X2 odds -> de-vigged probabilities."""
    cols = ("AvgCH", "AvgCD", "AvgCA")
    if not set(cols) <= set(df.columns):
        cols = ("AvgH", "AvgD", "AvgA")   # older seasons lack closing averages
    raw = 1.0 / df[list(cols)].astype(float).to_numpy()
    p = devig(raw)
    return pd.DataFrame({"p_home": p[:, 0], "p_draw": p[:, 1], "p_away": p[:, 2]},
                        index=df.index)


def implied_over25(df: pd.DataFrame) -> pd.Series:
    cols = ("AvgC>2.5", "AvgC<2.5")
    if not set(cols) <= set(df.columns):
        cols = ("Avg>2.5", "Avg<2.5")
    if not set(cols) <= set(df.columns):
        return pd.Series(np.nan, index=df.index)
    raw = 1.0 / df[list(cols)].astype(float).to_numpy()
    p = devig(raw)
    return pd.Series(p[:, 0], index=df.index)


def _total_from_over25(p_over: float) -> float:
    """Invert P(total > 2.5) under a Poisson total to recover the mean."""
    if not np.isfinite(p_over) or not (0.01 < p_over < 0.99):
        return np.nan

    def f(t):
        return (1.0 - poisson.cdf(2, t)) - p_over

    try:
        return brentq(f, 0.2, 8.0)
    except ValueError:
        return np.nan


def implied_goals(p_home: float, p_draw: float, p_away: float,
                  total: float) -> tuple[float, float]:
    """Split an implied total into (lambda_home, mu_away) using the 1X2 balance.

    Solves for the supremacy s such that independent Poissons with means
    ((total+s)/2, (total-s)/2) reproduce the market's home-win probability.
    """
    if not np.isfinite(total):
        return (np.nan, np.nan)

    def home_win_prob(s):
        lam, mu = (total + s) / 2.0, (total - s) / 2.0
        if lam <= 0 or mu <= 0:
            return np.nan
        g = np.arange(0, 12)
        ph, pa = poisson.pmf(g, lam), poisson.pmf(g, mu)
        m = np.outer(ph, pa)
        return np.tril(m, -1).sum()

    try:
        s = brentq(lambda s: home_win_prob(s) - p_home, -total * 0.95, total * 0.95)
    except (ValueError, RuntimeError):
        return (np.nan, np.nan)
    return ((total + s) / 2.0, (total - s) / 2.0)


def build(seasons: list[str], out: Path = Path("data/raw/odds")) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for season in seasons:
        df = fetch_season(season)
        df = df.dropna(subset=["HomeTeam", "AwayTeam"]).copy()
        df = pd.concat([df, implied_1x2(df)], axis=1)
        p_over25 = implied_over25(df)
        mkt_total = p_over25.map(_total_from_over25)
        lam_mu = [implied_goals(h, d, a, t) for h, d, a, t in
                  zip(df.p_home, df.p_draw, df.p_away, mkt_total)]
        # Build the derived block in one go -- assigning column by column to a
        # wide frame fragments it badly.
        df = pd.concat([df, pd.DataFrame({
            "p_over25": p_over25,
            "mkt_total": mkt_total,
            "mkt_home_goals": [x[0] for x in lam_mu],
            "mkt_away_goals": [x[1] for x in lam_mu],
            "home_name": df["HomeTeam"].replace(TEAM_ALIASES),
            "away_name": df["AwayTeam"].replace(TEAM_ALIASES),
            "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
        }, index=df.index)], axis=1)
        keep = ["season", "date", "home_name", "away_name", "FTHG", "FTAG",
                "p_home", "p_draw", "p_away", "p_over25", "mkt_total",
                "mkt_home_goals", "mkt_away_goals"]
        frames.append(df[keep])
    allodds = pd.concat(frames, ignore_index=True)
    allodds.to_parquet(out / "football_data.parquet", index=False)
    return allodds


if __name__ == "__main__":
    seasons = ["2022-23", "2023-24", "2024-25", "2025-26"]
    o = build(seasons)
    print(f"odds: {len(o):,} matches over {o.season.nunique()} seasons")
    print(f"  1X2 coverage {o.p_home.notna().mean():.1%}   O/U 2.5 coverage {o.p_over25.notna().mean():.1%}")
    print(f"  implied goals solved {o.mkt_home_goals.notna().mean():.1%}")
    print(f"  mean implied total {o.mkt_total.mean():.2f}  actual {(o.FTHG+o.FTAG).mean():.2f}")
    print(f"  mean implied home {o.mkt_home_goals.mean():.2f}  actual {o.FTHG.mean():.2f}")
    print(f"  mean implied away {o.mkt_away_goals.mean():.2f}  actual {o.FTAG.mean():.2f}")
