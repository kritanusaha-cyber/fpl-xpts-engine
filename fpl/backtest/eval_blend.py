"""Walk-forward evaluation: model alone vs market alone vs blend."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fpl.features.fixtures import fixture_frame
from fpl.models.team_goals import DixonColes
from fpl.models.blend import blend_rates, score_matrix, outcome_probs, clean_sheets

WEIGHTS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]


def _join_odds(f: pd.DataFrame) -> pd.DataFrame:
    odds = pd.read_parquet("data/raw/odds/football_data.parquet")
    con_names = pd.read_parquet("data/features/club_names.parquet")
    o = odds.merge(con_names.rename(columns={"name": "home_name", "club_code": "home_code"}),
                   on=["season", "home_name"], how="left")
    o = o.merge(con_names.rename(columns={"name": "away_name", "club_code": "away_code"}),
                on=["season", "away_name"], how="left")
    o = o.dropna(subset=["home_code", "away_code"])
    o["home_code"] = o["home_code"].astype(int)
    o["away_code"] = o["away_code"].astype(int)
    keep = ["season", "home_code", "away_code", "mkt_home_goals", "mkt_away_goals",
            "p_home", "p_draw", "p_away"]
    return f.merge(o[keep], on=["season", "home_code", "away_code"], how="left")


def run(test_seasons: list[str], xi: float = 0.003, target: str = "xg") -> pd.DataFrame:
    f = _join_odds(fixture_frame())
    rows, prev = [], None

    for season in test_seasons:
        for gw in sorted(f[f.season == season].gw.dropna().unique()):
            test = f[(f.season == season) & (f.gw == gw)]
            if not len(test):
                continue
            cutoff = test.kickoff_time.min()
            train = f[f.kickoff_time < cutoff]
            if target == "xg":
                train = (train.dropna(subset=["home_xg", "away_xg"])
                              .assign(home_goals=lambda d: d.home_xg,
                                      away_goals=lambda d: d.away_xg))
            if len(train) < 380:
                continue
            model = DixonColes.fit(train, xi=xi, target=target,
                                   ref_time=cutoff, warm_start=prev)
            prev = model
            for _, r in test.iterrows():
                if r.home_code not in model.index or r.away_code not in model.index:
                    continue
                mlam, mmu = model.rates(r.home_code, r.away_code)
                rows.append({
                    "season": season, "gw": gw,
                    "model_lam": mlam, "model_mu": mmu,
                    "mkt_lam": r.mkt_home_goals, "mkt_mu": r.mkt_away_goals,
                    "rho": model.rho,
                    "home_goals": r.home_goals, "away_goals": r.away_goals,
                })
    return pd.DataFrame(rows)


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["mkt_lam", "mkt_mu"])
    out = []
    for w in WEIGHTS:
        ll, cs = [], []
        for _, r in df.iterrows():
            lam, mu = blend_rates(r.model_lam, r.model_mu, r.mkt_lam, r.mkt_mu, w)
            m = score_matrix(lam, mu, r.rho)
            ph, pdw, pa = outcome_probs(m)
            probs = np.clip([ph, pdw, pa], 1e-9, 1)
            res = 0 if r.home_goals > r.away_goals else (1 if r.home_goals == r.away_goals else 2)
            ll.append(-np.log(probs[res] / probs.sum()))
            csh, csa = clean_sheets(m)
            cs.append((csh - (r.away_goals == 0)) ** 2)
            cs.append((csa - (r.home_goals == 0)) ** 2)
        out.append({"w_model": w, "n": len(df), "logloss": np.mean(ll),
                    "cs_brier": np.mean(cs)})
    return pd.DataFrame(out)


if __name__ == "__main__":
    d = run(["2024-25", "2025-26"])
    d.to_parquet("data/features/blend_preds.parquet", index=False)
    r = evaluate(d)
    print(f"walk-forward blend, n = {int(r.n.iloc[0])} fixtures")
    print("w_model = 1.0 is model only; 0.0 is market only\n")
    print(f'{"w_model":>9}{"1X2 logloss":>14}{"CS Brier":>11}')
    for _, x in r.iterrows():
        print(f'{x.w_model:>9.1f}{x.logloss:>14.4f}{x.cs_brier:>11.4f}')
    best = r.loc[r.logloss.idxmin()]
    print(f'\nbest w_model = {best.w_model:.1f}  logloss {best.logloss:.4f}')
