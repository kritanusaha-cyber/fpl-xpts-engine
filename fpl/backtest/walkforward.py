"""Walk-forward backtest harness.

Train on everything strictly before the target gameweek, predict it, roll.
Random k-fold on football panel data leaks the future and inflates metrics, so
it is not offered here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Reliability table: predicted vs realised rate, by probability decile."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({
            "bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}",
            "n": int(m.sum()),
            "predicted": float(p[m].mean()),
            "realised": float(y[m].mean()),
            "gap": float(p[m].mean() - y[m].mean()),
        })
    return pd.DataFrame(rows)


def walk_forward(features: pd.DataFrame, model_cls, test_seasons: list[str],
                 refit_every: int = 1, min_train_rows: int = 5000) -> pd.DataFrame:
    """Roll through the test seasons gameweek by gameweek.

    At each step the training set is every row that kicked off strictly before
    the first fixture of the target gameweek -- across all prior seasons too.
    """
    d = features.sort_values(["season", "gw"]).reset_index(drop=True)
    preds = []

    for season in test_seasons:
        gws = sorted(d[d["season"] == season]["gw"].dropna().unique())
        model = None
        for i, gw in enumerate(gws):
            train = d[(d["season"] < season) |
                      ((d["season"] == season) & (d["gw"] < gw))]
            test = d[(d["season"] == season) & (d["gw"] == gw)]
            if len(train) < min_train_rows or not len(test):
                continue
            if model is None or i % refit_every == 0:
                model = model_cls.fit(train)
            out = model.predict(test)
            out["season"], out["gw"] = season, gw
            for c in ["element", "fixture", "position", "appeared", "played_60", "minutes"]:
                out[c] = test[c].values
            preds.append(out)

    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
