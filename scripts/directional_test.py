"""Is the model directionally right about who is underpriced?

At each gameweek the engine fits a price curve -- points against log price,
per position -- and calls the residual mispricing. A player projected well
above what his price implies is "undervalued". This asks whether that call
was borne out, and it is asked strictly forward: the flag uses gameweeks
already played, the outcome uses gameweeks that had not happened yet.

Two different meanings of "goes up", both tested:
  price    -- FPL raises a price when enough managers buy him. This measures
              whether the model saw it before the market did.
  points   -- whether he actually outscored the players priced like him.

The second is the one that matters. A price rise is worth 0.1m of team value;
outscoring your price bracket is worth points.
"""
import numpy as np
import pandas as pd

HORIZON = 6          # gameweeks forward
MIN_PRICE = 4.0


def flag(d: pd.DataFrame) -> pd.DataFrame:
    """Mispricing residual per gameweek, per position, from a log-price curve."""
    out = []
    for (s, gw), g in d.groupby(["season", "gw"]):
        g = g[(g.price >= MIN_PRICE) & g.xpts.notna()].copy()
        if len(g) < 50:
            continue
        g["resid"] = np.nan
        for pos, p in g.groupby("position"):
            if len(p) < 15:
                continue
            x = np.log(p.price.to_numpy())
            y = p.xpts.to_numpy()
            b = np.polyfit(x, y, 1)
            g.loc[p.index, "resid"] = y - np.polyval(b, x)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    d = pd.read_parquet("data/features/backtest_predictions.parquet")
    # forward outcomes, computed per player within a season
    d = d.sort_values(["season", "element", "gw"])
    g = d.groupby(["season", "element"])
    d["price_fwd"] = g.price.shift(-HORIZON)
    d["pts_fwd"] = (g.total_points
                     .transform(lambda s: s.shift(-1).rolling(HORIZON, min_periods=3).sum()))
    f = flag(d).dropna(subset=["resid"])

    # decile of mispricing within each gameweek and position
    f["decile"] = (f.groupby(["season", "gw", "position"])["resid"]
                    .transform(lambda s: pd.qcut(s, 10, labels=False, duplicates="drop")))
    f = f.dropna(subset=["decile"])
    f["rose"] = (f.price_fwd > f.price).astype(float)
    f["fell"] = (f.price_fwd < f.price).astype(float)
    f.loc[f.price_fwd.isna(), ["rose", "fell"]] = np.nan

    # did he outscore the players priced like him, over the same window?
    f["band"] = (f.price // 1.0)
    f["peer_pts"] = f.groupby(["season", "gw", "position", "band"])["pts_fwd"].transform("mean")
    f["beat_peers"] = (f.pts_fwd > f.peer_pts).astype(float)
    f.loc[f.pts_fwd.isna(), "beat_peers"] = np.nan

    print(f"flagged player-gameweeks: {len(f):,}\n")
    t = f.groupby("decile").agg(
        n=("resid", "size"), resid=("resid", "mean"), price=("price", "mean"),
        price_rose=("rose", "mean"), price_fell=("fell", "mean"),
        pts_next6=("pts_fwd", "mean"), beat_peers=("beat_peers", "mean"))
    t.index = [f"{int(i)+1}" for i in t.index]
    t.index.name = "decile (10 = most undervalued)"
    print(t.to_string(float_format=lambda v: f"{v:.3f}"))

    top, bot = f[f.decile == f.decile.max()], f[f.decile == 0]
    print(f"\ntop decile   price rose {top.rose.mean()*100:.1f}%  "
          f"beat price peers {top.beat_peers.mean()*100:.1f}%  "
          f"next-{HORIZON} points {top.pts_fwd.mean():.2f}")
    print(f"bottom decile price rose {bot.rose.mean()*100:.1f}%  "
          f"beat price peers {bot.beat_peers.mean()*100:.1f}%  "
          f"next-{HORIZON} points {bot.pts_fwd.mean():.2f}")

    print("\n=== by season, top decile ===")
    print(f[f.decile == f.decile.max()].groupby("season").agg(
        n=("resid", "size"), price_rose=("rose", "mean"),
        beat_peers=("beat_peers", "mean"), pts=("pts_fwd", "mean")
    ).to_string(float_format=lambda v: f"{v:.3f}"))
    f.to_parquet("data/features/directional_test.parquet", index=False)


if __name__ == "__main__":
    main()
