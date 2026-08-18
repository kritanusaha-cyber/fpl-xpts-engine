# Validated facts

Everything here was derived from data and re-derivable via `make validate`.
Nothing in this file is taken from documentation.

## Scoring engine reconstructs 10 seasons exactly

`fpl/models/scoring.py` reproduces `total_points` from action counts with
**100.00% exact match on 109,681 player-match rows** across 2016/17–2025/26,
with a single explained exception (below). The scoring config is therefore
proven, not assumed.

## DefCon definition is position-dependent

Derived by exhaustive match against 2025/26 (the only season with native
`defensive_contribution`). 100% exact on all 26,330 rows:

| Position | Definition | Threshold | Points |
|---|---|---|---|
| DEF | `tackles + clearances_blocks_interceptions` | **10** | 2 |
| MID | `tackles + CBI + recoveries` | **12** | 2 |
| FWD | `tackles + CBI + recoveries` | **12** | 2 |
| GKP | n/a | n/a | 0 |

**Defenders exclude recoveries.** Including them inflates defender DefCon by
roughly 50% and would badly misprice the exact archetype the strategy targets.

Thresholds were located by reconstructing every other scoring term and finding
where the residual jumps 0 → 2. Separation is clean: for DEF the highest
non-scoring count is 9 and the lowest scoring count is 10; for FWD, 11 and 12.

## Scoring rules are season-specific — one YAML is not enough

The reconstruction found exactly one mismatch in 109,681 rows: Alisson,
2020/21 GW36, predicted 14 vs actual 10. Cause: **goalkeeper goals were worth 6
points then and are worth 10 now.** The engine detected a real rule change from
data alone.

Consequence for the build: the plan's "single YAML for scoring rules" is right
for live operation but wrong for backtesting. Walk-forward validation across
seasons needs a **per-season scoring config**, or every historical GK goal and
any other changed rule silently corrupts the training target. Defender goal
value was verified stable at 6 across all ten seasons.

## Position labels are inconsistent upstream

vaastav labels goalkeepers `GKP` before 2020/21 and `GK` after — and 2021/22
contains **both** within the same season (24 `GKP`, 739 `GK`). Unnormalised,
this drops every goalkeeper from position-keyed lookups and silently zeroes
their scoring. Normalised in `build_facts.py`.

## Historical column coverage is the binding constraint

| Column group | Seasons available |
|---|---|
| Core scoring (goals, assists, CS, bonus, bps, minutes) | all 10 |
| Position / team / xP | 2020/21 → |
| xG, xA, xGI, xGC, starts | 2022/23 → (4 seasons) |
| tackles, CBI, recoveries | 2016/17–2018/19 **and** 2025/26 only |
| native `defensive_contribution` | 2025/26 only (1 season) |

There is a **six-season gap (2019/20–2024/25) with no defensive action counts**,
and no season before 2025/26 has both defensive counts and xG together.

Implications:
- Phase 4 (DefCon) trains on one season of native target, or four seasons of
  reconstructable counts — none of which overlap with xG except 2025/26.
  Joint DefCon × clean-sheet modelling is therefore effectively single-season
  until live 2026/27 data accrues. FBref backfill is not optional.
- Phase 3 (attacking) has 4 seasons of xG. Adequate.
- Phase 1 (minutes) has all 10 seasons. Strongest-identified component.

## Upstream data quality

- 10 byte-identical duplicate rows in 2025/26. Deduped, with a >100 tripwire.
- Players can change club mid-season (e.g. element 391, Liverpool → Bournemouth),
  so `team` is a per-row attribute, not per player-season.

---

# Phase 1 — minutes model

Walk-forward over 2023/24–2025/26, 86,755 player-match predictions. Training set
at each gameweek is every row that kicked off strictly earlier, prior seasons
included. No k-fold anywhere.

## It beats the honest baseline, not just the trivial one

| | Brier | LogLoss | skill vs base rate |
|---|---|---|---|
| **model P(≥60)** | **0.0881** | **0.2889** | **55.4%** |
| baseline: base rate | 0.1974 | 0.5840 | 0% |
| baseline: persistence (last match) | 0.1208 | 1.6683 | 38.8% |
| baseline: EWMA start rate (hl=5) | 0.1124 | 0.6126 | 43.1% |

Beating the base rate is not evidence of anything — a decayed start-rate average
already gets 43%. The number that matters is that the model is **21.6% better in
Brier than the best single-feature baseline**, which is where the actual edge is.

P(appear) scores Brier 0.0996 / LogLoss 0.3334 against a 0.397 base rate.

## Calibration is acceptable; forwards are the weak spot

| Position | n | Brier | predicted | realised | bias |
|---|---|---|---|---|---|
| GKP | 9,709 | 0.0437 | 0.238 | 0.233 | +0.005 |
| DEF | 28,490 | 0.0982 | 0.317 | 0.315 | +0.002 |
| MID | 38,406 | 0.0941 | 0.281 | 0.260 | +0.021 |
| FWD | 10,150 | 0.0793 | 0.257 | 0.225 | **+0.032** |

Keepers are the best-calibrated, as expected — keeper minutes are close to
deterministic given selection. **Forwards are systematically over-predicted by
3.2 points of probability**, because they are substituted off more often than the
shared feature set implies. Reliability is within ±0.04 across all deciles, with
mild overconfidence in the 0.6–0.8 band.

This is good enough to build on, and the forward bias is a known, bounded
correction rather than a structural break. Fixing it likely needs a
position-interacted substitution term rather than more features.

## Structural notes

* `chance_of_playing_next_round` is **excluded from training** despite being the
  strongest injury signal, because the API exposes no history and using it would
  leak. It enters at prediction time only, from the snapshot table.
* `starts` only exists from 2022/23; earlier seasons fall back to a 60-minute
  proxy so the feature is defined over all ten seasons.
* Leakage is enforced structurally by `@point_in_time` in `fpl/features/base.py`,
  which audits both the input and the returned frame. Tested in
  `fpl/features/test_leakage.py` — including that it catches a function which
  re-joins its way back to the future.
* `team_id` is derived from fixture structure (the two distinct `opponent_team`
  values per fixture identify both sides) rather than joined from players_raw,
  because a player's club is a per-match attribute. This recovered clubs for all
  ten seasons with zero nulls, including the four that have no `team` column.
* `(season, gw, element)` is **not** a unique key — double gameweeks break it.
  The key is `(season, gw, element, fixture)`.

---

# Phase 2 — team model

Dixon-Coles bivariate Poisson, fitted walk-forward: at each gameweek, train on
every fixture that kicked off strictly earlier, predict, roll. 758 fixtures
across 2024/25–2025/26.

## Both of the doc's modelling calls are confirmed

**Time decay ξ = 0.003/day is empirically optimal.** Clean U-shape on
out-of-sample 1X2 log-loss, minimum exactly at the doc's suggested starting
value:

| ξ | 0.000 | 0.001 | 0.002 | **0.003** | 0.005 | 0.008 |
|---|---|---|---|---|---|---|
| 1X2 logloss | 1.0202 | 1.0139 | 1.0104 | **1.0093** | 1.0125 | 1.0250 |

**Fitting on xG beats fitting on goals**, on both metrics, despite xG being
available for only 4 of 10 seasons:

| target | 1X2 logloss | CS Brier |
|---|---|---|
| goals | 1.0093 | 0.1777 |
| **xG** | **0.9978** | **0.1757** |

## The market beats the model, and the blend adds nothing measurable

This is the headline result and it is negative.

| w_model | 1X2 logloss | CS Brier |
|---|---|---|
| 0.0 (market only) | 0.9915 | **0.1732** |
| **0.3 (best blend)** | **0.9905** | 0.1734 |
| 1.0 (model only) | 0.9979 | 0.1757 |

The blend's edge over the market alone is **+0.00101 log-loss, t = +0.58, 95% CI
[−0.0024, +0.0044] — not significant**. On clean sheets the market alone is
*better*, also insignificantly (−0.00022, t = −0.65).

So on 758 fixtures the Dixon-Coles model adds **no measurable value over
de-vigged closing odds** for team-level rates. The doc says the model's job is to
add the residual rather than beat Pinnacle. Measured, the residual is currently
indistinguishable from zero.

### What follows from that

1. **Use the market directly for near-term team rates.** It is free, one API pull
   per gameweek, and better calibrated than the model on clean sheets — the
   output Phase 3 and 4 consume most heavily.
2. **Dixon-Coles still earns its place, but for a different reason than the doc
   gives.** The optimizer runs an H = 5–8 gameweek horizon, and bookmakers do not
   post lines that far ahead. The model is what covers fixtures the market has no
   view on yet. That argues for a *horizon-dependent* weight — market-dominant for
   the next fixture, model-dominant beyond it — rather than one fixed w.
3. **Modelling effort should move to where the market has no view at all**:
   player-level shares, minutes, DefCon, and bonus. Team goal rates are close to
   a solved problem you can buy for free; player-level allocation is not.

Market calibration for reference — implied vs realised over 1,520 matches:
total 2.96 vs 2.95, home 1.61 vs 1.62, away 1.34 vs 1.34.

## Entity resolution

All football-data.co.uk club names resolve to FPL clubs with a 10-row alias table
(`Man United`→`Man Utd`, `Tottenham`→`Spurs`, …). Zero unmatched. Well below the
doc's predicted 40–80 manual overrides, consistent with the Phase 0 finding.

Implied goal expectations are recovered by inverting P(total > 2.5) under a
Poisson total, then solving for the supremacy that reproduces the market's
home-win probability. Two markets, two constraints, exactly identified.

---

# Phase 3 — attacking returns

Predicting a player's xG in an unseen match, 21,485 player-matches over
2024/25–2025/26. Actual minutes are used so that Phase 1 error does not
contaminate the structural comparison.

## Shares beat rates — the doc's central claim holds

| method | MAE | RMSE | corr |
|---|---|---|---|
| rate (per-90 EWMA) | 0.1061 | 0.2110 | 0.3752 |
| share × team xG (oracle) | 0.0985 | 0.1963 | 0.5012 |
| **shrunk share × team xG (oracle)** | **0.0932** | **0.1750** | **0.5717** |
| share × market goals | 0.1074 | 0.2070 | 0.4050 |
| **shrunk share × market goals** (usable) | **0.1028** | **0.1903** | **0.4622** |

The oracle rows use realised team xG and isolate the structural question; the
market rows are what you can actually deploy. On the deployable comparison the
share structure plus shrinkage lifts correlation **0.3752 → 0.4622, +23%**.

Empirical-Bayes shrinkage is doing real work, not decoration: it adds more
(0.4050 → 0.4622) than the share structure alone does (0.3752 → 0.4050).

## It holds on the metric that is actually the decision

The doc is right that within-position rank is the real decision. Spearman ρ
within position, half-life 5:

| | target = xG | | | target = realised goals | | |
|---|---|---|---|---|---|---|
| **pos** | rate | share | lift | rate | share | lift |
| DEF | 0.2276 | 0.2695 | +18.4% | 0.0743 | 0.0805 | +8.4% |
| MID | 0.4268 | 0.4864 | +14.0% | 0.2074 | 0.2355 | +13.5% |
| FWD | 0.4518 | 0.5287 | +17.0% | 0.2714 | 0.3083 | +13.6% |

The improvement survives the step from xG to realised goals, which is the step
that pays points. **GKP is excluded** — keepers essentially never score, the
correlation base is ~0, and the apparent "+248%" there is noise on nothing.

## Half-life 5 beats 10 and 20

| half-life | 5 | 10 | 20 |
|---|---|---|---|
| Spearman | **0.5364** | 0.5262 | 0.5084 |
| MAE | **0.1001** | 0.1028 | 0.1074 |

Attacking shares move faster than the decay used for team strength (ξ = 0.003/day
≈ half-life 230 days). Role changes within a squad are quicker than team-quality
drift, which is what you would expect.

## Known gap: penalties are not separated

FPL's `expected_goals` **includes penalty xG**, and splitting npxG from penalty
xG needs Understat's penalty-tagged shots, which is not yet ingested. Until it
is, a designated taker's share is inflated and his open-play threat overstated.
The doc puts this at 0.15–0.20 pts/90 for a taker; it is the largest known gap
in this phase.

Partial mitigation already available: the live API exposes `penalties_order`
(populated for 65 players), plus `corners_and_indirect_freekicks_order` and
`direct_freekicks_order`. That is official set-piece duty, better than inferring
it, and it removes most of the doc's §5 manual-override burden — but it is
current-season only, so it helps live prediction and not the historical fit.

---

# Phase 5 — bonus points

## The award logic is exactly right; the inputs are not available

Ranking by *true* BPS reproduces bonus almost perfectly, which validates the
tie-handling: **100.00% of bonus winners recovered (760/760 fixtures)**, 99.16%
exact on the full 3/2/1 vector. The residual 0.84% is tie-rule edge cases.

That is the ceiling. The binding constraint is predicting BPS itself.

## BPS is deterministic given the action log — but FPL stopped publishing it

Fitted on 2016/17, the last season with the full action detail:

| feature set | R² | MAE |
|---|---|---|
| every published action column | **0.9893** | **0.88** |
| only columns today's API still provides | 0.8967 | 2.60 |

Key passes, dribbles, crosses, fouls, big chances and pass completion are all
BPS inputs, and all disappeared from the FPL feed after 2018/19. The doc's
"BPS is fully deterministic given the action log, so it's tractable" is true in
principle and **not currently actionable** — the action log is the missing piece,
and closing it needs FBref/Opta, not FPL.

## What that costs, measured

| | exact | MAE | corr |
|---|---|---|---|
| oracle (true BPS rank) | 99.16% | 0.0084 | 0.9907 |
| model (predicted BPS) | 90.48% | 0.1306 | 0.7669 |

Bonus winner recovered in **68.95%** of fixtures (vs 100% oracle). Any-bonus
detection: precision 68.2%, recall 75.9%. Better than chance by a wide margin —
a fixture has 22–28 candidates — but far from deterministic.

The fitted weights are **reduced-form and should not be read as the official
table**: omitted correlated actions inflate the coefficients on the actions that
remain visible (fitted assist ≈ 11 against a published 9, fitted MID goal ≈ 20
against 18). They are fit for ranking within a fixture, which is all bonus
depends on.

## Bonus is worth less than the doc claims

The doc puts bonus at "roughly 15–20% of a top score". Measured on 2025/26:

| bucket | n | avg points | avg bonus | % from bonus |
|---|---|---|---|---|
| elite (150+) | 30 | 174.5 | 19.7 | **11.3%** |
| strong (100–149) | 108 | 120.7 | 9.3 | 7.7% |
| mid (50–99) | 157 | 72.6 | 4.3 | 5.9% |
| fringe (<50) | 241 | 19.5 | 0.6 | 3.2% |

11.3% for elite players, not 15–20%. Combined with 69% winner accuracy, Phase 5's
return on complexity is **lower than the doc assumes**. The doc's own advice —
"measure rather than assume" whether Phases 4 and 5 earn their complexity — is
the right instinct, and on this evidence bonus is the weaker of the two.

Recommendation: keep the reduced-form BPS model as a ranking input to the bonus
simulation, and do not invest in Monte-Carlo BPS simulation over the full action
set until FBref action data is ingested. Without the action log the extra
machinery has nothing to consume.
