# FPL xPts Engine

A Monte Carlo model of Fantasy Premier League scoring. Every projection is
20,000 simulated fixtures rather than a formula, because most FPL scoring rules
are thresholds — a clean sheet, a DefCon hit, the −1-per-2-conceded step — and
the mean of a threshold is not the threshold of the mean.

**[Live dashboard →](https://kritanusaha-cyber.github.io/fpl-xpts-engine/)**

## What it does

- Projects points for all 592 players over a six-gameweek horizon
- Prices each player against others **in the same role**, not the same FPL
  position — "MID" holds both wingers and holding midfielders
- Selects an optimal squad by MILP under the real constraints (£100.0m,
  2/5/5/3, max 3 per club, valid formation)

## Results, out of sample

| Component | Result |
|---|---|
| Scoring engine | **100.00%** exact reconstruction over 109,681 player-matches |
| Minutes model | Brier 0.0881 — **21.6%** better than the best single-feature baseline |
| Attacking returns | share structure lifts correlation 0.3752 → **0.4622** |
| Bonus award logic | **100%** of bonus winners recovered from true BPS |

Two findings worth stating because they are negative: blending Dixon-Coles with
bookmaker odds adds nothing significant (t = +0.58), and club-role inheritance
for new signings loses to a plain price-tier prior.

Full methodology, every statistical test, and all six bugs found during the
build are in [FINDINGS.md](FINDINGS.md) and on the dashboard's Methodology tab.

## Running it

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt          # core pipeline
uv pip install -r requirements-scrape.txt   # only for the FBref ingest
make all          # ingest, build facts, validate scoring
make refresh      # full weekly refresh through to the dashboard
make serve        # http://localhost:8733
```

`make validate` must report 100% exact on every season; it is the canary for the
scoring config.

## Caveats

- 2026/27 has no completed gameweeks, so projections run on priors alone.
- Midfielders and forwards carry a measured residual bias of ~−0.3 xPts/match.
- Fair price extrapolates through a **marginal** rate; it orders players well and
  misleads at the extremes.
- DefCon calibration collapses above p = 0.5 — treat those as unreliable.

Data: the official FPL API, [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League),
football-data.co.uk, and FBref via soccerdata.
