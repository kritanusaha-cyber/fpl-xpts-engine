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

---

# Phase 4 — DefCon

Walk-forward within 2025/26 (train GW1..t, test t+1), 8,626 player-matches.
2025/26 is the **only** season played under the DefCon rule that also carries the
component counts; 2016/17–2018/19 have the counts but predate the rule by seven
seasons, so they are not used as training data.

## Negative binomial beats Poisson, as the doc says

| | Brier | predicted | realised | bias |
|---|---|---|---|---|
| **negative binomial @ projected minutes** | **0.0993** | 0.135 | 0.137 | **−0.0016** |
| Poisson @ projected minutes | 0.1010 | 0.109 | 0.137 | −0.0277 |
| negative binomial @ **flat 90 minutes** | 0.1270 | 0.225 | 0.137 | **+0.0884** |
| baseline: base rate | 0.1183 | 0.137 | 0.137 | 0.0000 |

Poisson does not just score slightly worse — it is **systematically biased
downward** (−0.028), exactly the underfit the doc predicts from ignoring
overdispersion. The negative binomial is near-unbiased.

**Evaluating at 90 minutes instead of projected minutes is catastrophic**: Brier
0.1270, *worse than simply predicting the base rate*, with +0.088 of
over-prediction. The doc's insistence on projected minutes is not a refinement,
it is load-bearing — and this is the clearest measured vindication of building
Phase 1 first.

Overall skill over base rate is a modest **16%**.

## The doc was right that this is the worst-calibrated component

| position | n | Brier | predicted | realised | bias |
|---|---|---|---|---|---|
| FWD | 1,178 | 0.0075 | 0.008 | 0.008 | +0.0003 |
| MID | 4,266 | 0.0891 | 0.112 | 0.118 | −0.0053 |
| DEF | 3,182 | 0.1470 | 0.213 | 0.211 | +0.0026 |

Aggregate bias is tiny, but the reliability curve falls apart in the tail:

| predicted bin | n | predicted | realised |
|---|---|---|---|
| 0.0–0.2 | 6,650 | ~0.10 | ~0.09 |
| 0.2–0.4 | 1,734 | 0.295 | 0.348 |
| 0.5–0.6 | 19 | 0.543 | 0.368 |
| 0.6–0.8 | 11 | 0.664 | **0.000** |
| 0.8–1.0 | 6 | 0.829 | **0.167** |

Above ~0.5 the model is badly overconfident. The samples are small (37 rows over
0.5), but the direction is consistent and these are exactly the players a
DefCon-targeting strategy would buy. Treat any predicted DefCon probability
above 0.5 as unreliable until there is more data.

## The DefCon / clean-sheet hedge is not measurable

This is the notable negative result. The doc argues the two should be modelled
jointly because "a hard fixture lowers P(clean sheet) and raises P(DefCon)",
creating a hedge "worth building explicitly". Measured across defenders:

| | correlation |
|---|---|
| P(DefCon) vs clean sheet | **−0.0377** |
| DefCon hit vs clean sheet | **+0.0290** |
| opponent strength vs DefCon hit | +0.0660 |

All ≈ 0, and the first two have *opposite signs*. On this evidence joint
modelling buys close to nothing, and two independent screens would misprice
almost nothing.

**Important caveat before acting on this.** The doc specifies *projected opponent
possession share* as the key covariate; possession is not in the FPL feed, so
the proxy used here is team xG conceded. The relationship may well be real and
simply invisible to this proxy. The honest statement is: **with the best
covariate currently available, the hedge is undetectable** — and FBref possession
data is the way to settle it, not more modelling on FPL columns.

---

# Phases 6 & 7 — assembly and optimiser

End-to-end pipeline runs: cold-start priors → Dixon-Coles carried across the
season boundary → 20,000-draw joint simulation per fixture → MILP squad.
`make gw1` reproduces it.

## Cold start, measured

2026/27 has **zero** played gameweeks, so every blend weight sits at 1.0 on
priors. Two separate problems:

* **Players.** 405 of 592 (68.4%) map to 2025/26 via the stable FPL `code`
  (identical coverage to `opta_code`). The other 187 have no history and fall
  back to the position × price-tier prior — which empirical Bayes already
  handles, since n90 = 0 puts full weight on the prior.
* **Clubs.** Promoted sides have no Premier League matches. Rather than let the
  fit invent parameters, they get a prior measured from 27 promoted club-seasons
  in this dataset: **attack 0.711× league average, defence 1.245×** (they concede
  ~25% more). Coventry City and Hull City are on that prior for 2026/27.

## Simulating jointly is not optional — and revealed two omissions

Simulating one team at a time cannot produce bonus, because bonus is a
competition across all 22+ players in a fixture. Restructuring to draw one
scoreline and play out **both** squads against it fixed that and surfaced two
terms that had been silently missing (bonus, goalkeeper saves). Their absence was
biasing every projection down:

| position | gap before | gap after | realised |
|---|---|---|---|
| GKP | −0.80 | **−0.23** | 3.38 |
| DEF | −0.74 | −0.69 | 3.70 |
| MID | −1.30 | −1.23 | 3.97 |
| FWD | −1.59 | −1.32 | 4.11 |

A separate bug found the same way: goals-conceded points were being charged to
players who never appeared, giving a benched keeper **negative** xPts. Conceded
is now drawn as Binomial(conceded, minutes/90) and gated on appearance, matching
FPL's "while on the pitch" rule.

## Honest status: not yet trustworthy for team selection

**Attackers are still under-projected by ~1.2–1.3 points per match.** The known
cause is the gap already flagged in Phase 3: penalties and set pieces are not
separated, which needs Understat. The doc puts penalty duty alone at 0.15–0.20
pts/90 for a designated taker, and set-piece threat is unmodelled on top.

The symptom shows up in the optimiser. Budget sensitivity:

| budget | £90m | £95m | £98m | £100m | £103m | £110m |
|---|---|---|---|---|---|---|
| xPts | 55.06 | 56.11 | 56.28 | **56.28** | 56.80 | 56.98 |

The curve is **flat from £95m to £100m** — the budget does not bind, and the
shadow price at £100m is ~0. In real FPL the budget always binds. The binding
constraints here are max-3-per-club and a talent pool the model has flattened:
price vs xPts correlation is only **0.343**, when premiums should clearly
out-project cheap players.

So the machinery is right and the inputs are not yet good enough. The doc's own
test — "ship the cut-down version, measure it, then decide whether the later
phases earn their complexity" — is what produced this diagnosis, and the answer
is that **the next unit of effort belongs in Understat ingest, not in more
optimiser features**. Chip scheduling, transfer planning over an H-gameweek
horizon and rank-vs-EV objectives are all deferred until attacking returns
calibrate.

---

# Penalty separation (closing the Phase 3 gap)

## Both of the doc's suggested sources have moved

* **Understat has restructured.** The embedded `playersData` / `shotsData` JSON
  that the classic scrape and the `understat` package both depend on is gone
  from the page — only `BASE_URL` and `THEME` remain. That route is dead.
* **FBref is behind Cloudflare.** Plain HTTP gets a 403 "Just a moment"
  challenge. `soccerdata` gets through by driving an undetected browser, which
  is why it is a hard dependency rather than a convenience.

We pull only penalty attempts from FBref — a small, stable column — rather than
depending on their wider schema. `npxG = FPL_xG − 0.79 × PKatt`.

## Entity resolution needed 6 manual overrides, exactly as the doc predicts

Automatic matching (normalised full name → unique surname → unique first token)
resolved 33/39 penalty takers. The override table closes the rest: **39/39,
92/92 penalty attempts covered.**

The instructive case is **Lucas Paquetá**, whom FPL stores as *"Lucas Tolentino
Coelho de Lima"* — no surname token in common with his FBref name. No fuzzy
matcher recovers that, which is precisely why the doc insists on a manual
override file. Six rows, not the predicted 40–80, consistent with earlier phases.

## Penalty xG was materially distorting shares

| player | xG share before | after | change |
|---|---|---|---|
| Igor Thiago | 0.317 | 0.245 | **−23%** |
| Cole Palmer | 0.201 | 0.141 | **−30%** |
| B. Fernandes | 0.162 | 0.119 | −27% |
| Haaland | 0.348 | 0.308 | −11% |

Thiago was projecting the **second-highest xG share in the league** largely on
the back of 9 penalties. Penalty value is now added back as an explicit term
(P(team penalty) = 92/760 = 0.121 per team-match, conversion 0.79) attached to
whoever holds the duty **now** via the API's `penalties_order`, rather than to
whoever took them last season. That is the correct structure and it moved the
captain pick.

## Correction: the attacker "under-projection" was mostly my own measurement error

Earlier I reported attackers under-projected by ~1.2–1.3 points and attributed it
to penalties. **That was wrong, and the penalty fix barely moved it** (FWD
−1.32 → −1.25). The cause was a conditioning mismatch in the diagnostic itself:
projected xPts is *unconditional* (it includes the probability of not playing),
while the realised benchmark was conditional on having played 60+ minutes.

Compared like-for-like:

| position | mean p60 | unconditional | ≈conditional | realised | gap |
|---|---|---|---|---|---|
| GKP | 0.90 | 3.14 | 3.42 | 3.38 | **+0.04** |
| DEF | 0.76 | 3.01 | 3.78 | 3.70 | **+0.07** |
| MID | 0.70 | 2.75 | 3.63 | 3.97 | −0.34 |
| FWD | 0.68 | 2.86 | 3.89 | 4.11 | −0.22 |

Keepers and defenders are **well calibrated**. The genuine residual bias is
−0.34 for midfielders and −0.22 for forwards, plausibly the still-unmodelled
set-piece threat — not the 1.2+ previously claimed.

## What remains genuinely unresolved

The budget still does not bind in the meaningful range — the objective is flat
from £98m to £100m — and price vs xPts correlation is 0.355. That is a real
finding and it is **not** explained by the conditioning artifact above. The
likely cause is that bench players contribute zero to the objective while the
spread across viable starters is narrow, so the optimiser has little to buy with
the last few million. Worth attacking directly rather than assuming better
projections will fix it.

---

# Cold start for new signings

Two ideas for players with no Premier League history, both tested on **244
genuine PL debutants** across 2023/24-2025/26.

## First, a bug that had been faking the test set

`newcomers()` originally compared `element` ids across seasons. **FPL reassigns
element ids every season**, so every returning player looked like a debutant.
The "new signings" test set was mostly established professionals, for whom club
role profiles trivially matched — because they *were* the role. Resolving through
the stable `code` grew the set from 23 to 244 and **reversed the headline
result**. Any conclusion drawn before that fix was measuring the bug.

## Role inheritance: measured, and rejected

The idea — Vuskovic inherits what Van Hecke did, because the role outlives the
player — is intuitive and does not survive contact with the data:

| target | tier only | blend 0.25 | role only |
|---|---|---|---|
| xg_share MAE | **0.0297** | 0.0303 | 0.0368 |
| xa_share MAE | **0.0209** | 0.0213 | 0.0243 |
| starts60 MAE | **0.1774** | 0.1788 | 0.1965 |

The price tier wins on every metric. Role adds a little *rank* information for
xg_share (corr 0.734 → 0.748 at w=0.25) while being worse on level.

Two concrete reasons it fails:

1. **FPL's price already encodes the role.** The club prices a signing knowing
   whether he starts, so the tier prior has absorbed the information the role
   profile would add.
2. **The role profile is minutes-weighted, so it is the FIRST-CHOICE player.**
   Applied to a squad player it inherits the starter's share — a £5.5m backup
   striker at Aston Villa was picking up 0.324, which is Watkins's number. That
   inflates precisely the cheap enablers the optimiser then buys.

Set to weight 0. Reviving it needs role-*within-depth* (first choice vs
rotation), which the current data does not distinguish.

## Foreign-league output: validated and shipped

Non-penalty output in the Big 5, discounted for league strength, genuinely
predicts Premier League share — **corr +0.656** on 39 matched debutants.

But it needs hard calibration:

    xg_share = 0.400 x (npG90_adj / 1.45) + 0.0254

**A slope of 0.40 means foreign output overstates PL share by ~2.5x even after
the league-strength discount.** Raw MAE 0.0611 → 0.0365 calibrated. Blended
against the tier prior, the optimum is **w_foreign = 0.30** (MAE 0.0286 →
0.0275).

Coverage is the real limit: soccerdata's FBref backend serves the Big 5 only, so
**55 of 187 new signings (29%)** get a foreign prior. Arrivals from the
Eredivisie, Primeira Liga, the Championship or outside Europe fall back to the
tier prior. FBref also leaves `league` blank for all 18 Bundesliga clubs (~500
players), who would silently take the unknown-league discount if not patched.

League-strength coefficients (La Liga/Serie A/Bundesliga 0.85, Ligue 1 0.80) are
**assumptions, not fitted values** — there is no matched-move dataset in the
warehouse yet. They are the first thing to validate as arrivals accumulate.

### Effect

Vuskovic (Brighton, £5.0m, no PL history): xG share prior **0.029 → 0.041**, on
the strength of 0.180 npG/90 over 2,442 minutes at Hamburg. The dashboard now
states each player's prior provenance — PL record, calibrated Big 5, or bare
price tier — rather than presenting all three as equivalent.

**The budget constraint now binds**: optimal spend went £97.0m → **£100.0m**, and
Haaland enters the XI. Better differentiation among new signings was part of what
the flat budget curve had been missing.

---

# Getting real separation into implied value

Three changes, in increasing order of effect.

## 1. The shrinkage constant was hardcoded; it should be fitted per position

The doc specifies w = n / (n + sigma2_within/sigma2_between). I had used a flat
k = 8. Estimated from the historical variance decomposition:

| position | k = within/between |
|---|---|
| DEF | 15.7 |
| MID | **6.5** |
| FWD | 15.7 |

Midfielders separate from one another more than twice as fast as defenders or
forwards, whose match-to-match noise swamps the between-player signal for far
longer. Worth noting this *narrows* DEF and FWD spreads — accuracy and spread
pull in opposite directions here, and accuracy wins.

## 2. One gameweek is too short a horizon to separate anyone

Over a single match the gap between the best and worst pick is a couple of
points, most of it noise. Over the doc's H = 5-8 horizon, fixture runs compound:

| | 1 GW | 6 GW |
|---|---|---|
| top projected xPts | 5.48 | **29.80** |

This is the single largest lever on deviation size, and it is also the horizon on
which transfer decisions are actually made.

## 3. Comparing within FPL position is still too coarse

FPL's four positions are a scoring construct, not a football one. "MID" holds
both Saka and Zubimendi. Ranking a player against everyone sharing his FPL label
compares people who are not doing the same job.

Roles are clustered (k-means) on three axes available for every player — share
of team xG, share of team xA, defensive actions per 90 — then labelled by their
dominant axis:

| position | roles found |
|---|---|
| DEF | high goal threat (37), high creator (34), low creator (124) |
| MID | high goal threat (27), high creator (33), high defensive (139), low involvement (62) |
| FWD | high creator (19), low involvement (51) |

The groups are recognisable: *MID high goal threat* is Rogers, Semenyo,
Gibbs-White, Ndiaye — the wingers. *MID high defensive* is Zubimendi,
Gravenberch, Ampadu. *DEF high goal threat* is Virgil, Tarkowski, Van Hecke —
which is exactly Vuskovic's comparison set.

**An earlier version of the labels was wrong and had to be thrown out.** It
guessed football names from cluster centroids and produced "attacking full-back"
for Virgil and Tarkowski (both centre-backs) and put Haaland in "support
forward". The clusters were real; the names were invented. They are now derived
mechanically from the dominant axis, which claims only what the data supports.

## Value is now measured against replacement level within role

    surplus_role = xPts_H - lambda*price - mu_position - role_replacement
    fair_price   = price + surplus_role / lambda

**Replacement level, not the role median.** Most members of a role cluster are
squad filler who will never play; centring on the median over all of them put the
baseline near zero and made every real starter look enormously underpriced — a
£6.5m holding midfielder came out with an £18.9m "fair price". Replacement is now
the median among players projected to actually start.

Result, across 359 projected starters:

| | p5 | median | p95 |
|---|---|---|---|
| mispricing (£m) | −2.4 | 0.0 | +3.6 |

and every role lands at roughly 50% underpriced / 50% overpriced, with genuine
spread (best +7.8, worst −4.0) rather than one position being uniformly punished.

## Two more element-id bugs, same root cause

Both the role join and the earlier newcomer test were silently broken by joining
on `element` across seasons. FPL reassigns element ids annually, so the role join
attached 2025/26 roles to whichever 2026/27 player inherited the number — it put
Haaland in a centre-back cluster. Both now join on the stable `code`.

Related: a carried-over role whose prefix no longer matches the player's current
position is now discarded and re-derived, because **position is a per-season
attribute** and FPL reclassifies players between seasons.

---

# Auditing the large deviations

Checking the outliers surfaced three real bugs. All three inflated the extremes,
which is where they were least likely to be noticed and most likely to be acted on.

## 1. Injured players were counted as starters

`is_starter` used last season's start rate alone and ignored availability, so 42
of 359 "starters" (12%) were injured or unavailable, projecting exactly 0.00
xPts. Consequences:

* The **most-overpriced list was simply a list of injured players** — J. Timber,
  Ekitike and Kroupi Jr all appeared at the floor price purely because they are out.
* Role replacement baselines were dragged down by up to 1.1 xPts, which inflated
  every *other* player's surplus in those roles.

## 2. Start rates were computed over appearances, not games available

`starts60` was the fraction of a player's *appearances* that reached 60 minutes.
For goalkeepers that is almost always 1.0, so **a reserve keeper with a single
90-minute outing scored 1.00**. Arrizabalaga (n90 = 1.0) came out at p60 = 0.997.

Result: **all 20 clubs projected two or more goalkeepers as near-certain
starters**, and Arsenal projected three. Only one keeper can play.

Fixed by dividing by the club's games rather than the player's appearances.

## 3. Nothing enforced squad competition

Even with the right denominator, per-player priors are independent and ignore
competition for places. Added a squad-depth constraint: within each club and
position, start probabilities are allocated across the slots a typical XI fills
(1 GKP / 4 DEF / 4 MID / 2 FWD), in proportion to `p^3`.

The exponent matters. Straight proportional scaling punished the genuine first
choice for his backups' inflated priors — Raya fell to 0.57 while obviously being
Arsenal's starter. Sharpening concentrates minutes on the established player:

| | before | proportional | sharpened |
|---|---|---|---|
| Raya | 0.99 | 0.57 | **0.90** |
| Arrizabalaga | 0.997 | 0.21 | **0.04** |
| Meslier | 0.997 | 0.22 | **0.05** |

Clubs projecting more than one starting keeper: **20 → 0**.

"Starter" is now also defined structurally — top N at his club and position —
rather than by a probability threshold, which had become arbitrary once depth
normalisation compressed everyone toward 0.5. That yields **218 starters, 10–11
per club**, against 20 × 11 = 220 expected.

## What the surviving outliers actually rest on

The important check, given that DefCon is the worst-calibrated component and its
reliability collapses above p = 0.5:

| | share of projection from DefCon |
|---|---|
| the 8 largest mispricings | **9%** |
| all 218 starters | 10% |

**The headline deviations are not built on the unreliable component.** They are
cheap, nailed starters at defensively sound clubs — Shaw, Milenkovic and O'Brien
draw ~1.2 points of clean sheet each; Raya 2.14. That is a real and well-known
FPL edge rather than an artefact.

## A limit worth stating plainly

Fair price extrapolates linearly through lambda, which is the **marginal** rate
at the optimum, not a global price curve. It orders players sensibly and becomes
misleading at the extremes: a "fair price" of £17m is not a price FPL would ever
set — it is a large surplus expressed in price units. This is now stated on the
methodology tab rather than left for the reader to infer.

---

# Fair price was measured against the wrong denominator

Luke Shaw came out with a £13.3m fair price against a £4.5m listing. The
projection was fine — 20.7 xPts over six gameweeks, 3.45 per gameweek for a
nailed cheap defender, which is realistic. **The price mapping was wrong.**

## Lambda is not the price of output

Fair price was `price + surplus / lambda`, with lambda the budget shadow price
from the optimiser. Lambda is the **marginal rate at a constrained optimum** —
what one more pound buys you *at the margin of a full squad* — not what the
market charges for output.

| | xPts per £1m over 6 GW |
|---|---|
| lambda (LP shadow price) | **1.11** |
| realised market gradient, DEF | **~3.2** |

Dividing by a number roughly three times too small tripled every price gap. That
error is invisible in the middle of the distribution and enormous at the edges,
which is exactly where a reader looks.

## FPL prices have diminishing returns; a linear map cannot

Realised points per 90 by price band, 2022/23–2025/26:

| band | DEF | MID | FWD |
|---|---|---|---|
| ≤£4.5m | 2.86 | 3.23 | 4.42 |
| £5.5–6.5m | 4.34 | 4.54 | 4.90 |
| £8–10m | 5.10 | 5.44 | 5.86 |

The curve flattens: premium players are charged for attacking ceiling, which is
itself capped. A constant points-per-pound is therefore the wrong functional
form regardless of the constant chosen.

Replaced with a per-position curve fitted on players with 900+ minutes, on
**points per team-gameweek** — matching the basis the model projects on, since
xPts already includes the probability of not playing:

    points_per_gameweek = a + b·log(price)     →     fair = exp((points − a) / b)

| | £4.5m buys | £9.0m buys | shape |
|---|---|---|---|
| DEF | 1.86 | 4.73 | steepest — cheap defenders genuinely poor |
| MID | 1.67 | 4.16 | steep |
| FWD | 1.75 | 3.82 | flattest — cheap forwards still score |

## Effect

| player | listed | old fair | **new fair** |
|---|---|---|---|
| Shaw | £4.5m | £13.3m | **£6.6m** (+2.1) |
| Anderson | £6.5m | £17.1m | **£9.1m** (+2.6) |
| Gabriel | £8.0m | £10.3m | **£7.8m** (−0.2) |
| Haaland | £15.5m | £20.0m | **£11.8m** (−3.7) |

Mispricing now spans p5 −1.8 to p95 +1.6, against actual FPL prices of £4.0–15.5.
Every number is a price FPL could plausibly set, which is the minimum bar for the
metric to mean anything.

The R² of the price curve is only 0.23–0.35. That is not a defect — price is a
weak predictor of points, which is *why* mispricing exists. But it does mean fair
price is a central tendency and not a target.

**One honest asymmetry**: the model's highest fair price is £11.8m while FPL
charges up to £15.5m. Some of that gap is the known −0.3 xPts/match bias on
midfielders and forwards, not market error, and it is now stated on the page.

---

# SofaScore: what happened, and the lesson

## The access question, answered twice and wrongly the first time

Initial probe from the sandbox returned 403 on every path including
`robots.txt`, and I concluded SofaScore was refusing this machine. **That was
wrong.** The block was on the sandbox's HTTP egress; the browser on the same
machine was served normally. Correcting it took one test I should have run
before drawing a conclusion.

The API is genuinely the best source examined here: `/event/{id}/player/{pid}/heatmap`
returns real coordinate point clouds, and the lineups carry granular passing
stats (`accurateOppositionHalfPasses`, `keyPass`, `bigChanceCreated`) that no
other free source exposes.

## Then I burned the access

A serial collection made roughly 150 heatmap requests in a burst. SofaScore now
returns:

    {"error": {"code": 403, "reason": "challenge"}}

That is bot detection, not a rate-limit pause with a `retry-after` worth waiting
on — the header reads 0 while the challenge persists. **Getting past it would
mean solving or evading a bot challenge, which is not something to do regardless
of who authorises it.** The user's permission governs their machine; it cannot
authorise passage through a third party's own controls.

Two mistakes, both mine:

1. **Too many requests, too fast.** ~150 requests with a 40ms gap is not a
   browsing pattern. A season needs hours, spread out, not one burst.
2. **State lived in memory.** The partial collection — 154 players, 7,382
   coordinates — was accumulating in a `window` variable and was lost when the
   tab closed. It should have been checkpointed to disk from the first match.

## What exists for a retry

`scripts/sofascore_collect.js` is written and ready but **not run**:

* checkpoints to `localStorage` after every match, so a closed tab or a pause
  costs at most one match, and `start()` resumes exactly where it stopped;
* strictly serial with a 1.5s player delay and a 4s match pause;
* **halts on the first 403** rather than retrying into the block, and reports
  why.

Whether the challenge clears is SofaScore's decision, not something to probe
repeatedly. A single check occasionally is reasonable; a retry loop is not.

## What this does not block

The shot-zone grids shipped and cover 315 players: Athletic-style 6x5 pitch over
the attacking half, three-tier colouring against the positional baseline, with
half-spaces and Zone 14 properly named. SofaScore would have upgraded these from
*shot* locations to *touch* territory — a real improvement, and not a
prerequisite.

---

# The success criterion, finally tested

The build plan sets one bar and is blunt about it:

> "beat the 'template' benchmark — the xPts-weighted top-15 by ownership — on
> out-of-sample per-GW MAE and on simulated final rank. If you can't beat the
> template, you have a hobby, not an edge."

Every other result in this file measures a component. This measures whether the
assembled thing is worth using. Walk-forward over 2025/26, GW8–38, 9,196
player-gameweeks in which the player featured; at each gameweek every model —
minutes, shares, team strength, calibrated DefCon — is refitted on gameweeks
already played.

| model | MAE | Spearman (in position) | top-20 precision |
|---|---|---|---|
| **full engine** | **1.991** | **0.282** | 0.123 |
| price heuristic | 2.045 | 0.164 | **0.165** |
| points per appearance to date | 2.067 | 0.251 | 0.160 |
| FPL "form" (last 4 GW) | 2.253 | 0.241 | 0.132 |
| **template (crowd ownership)** | 2.895 | 0.193 | 0.150 |

**Against the template: MAE 31.2% better, Spearman 46.1% better.** The criterion
is met, and not narrowly. The engine also beats every other baseline on both
accuracy and ranking — including "points per appearance to date", which is the
bar that actually embarrasses most projection models.

## Test the real thing, not a proxy

A first pass used a hand-rolled stand-in for the engine (`xGI/90 × start rate`)
and scored MAE 2.121 — *worse* than a to-date average, which would have been a
damning result. Running the actual stack gives 1.991. **The proxy understated the
engine by more than the entire gap to the benchmark.** Worth remembering before
concluding anything from a simplified reconstruction of your own model.

## Where it loses, and why that is the expected trade

The engine is **last but one on top-20 precision** (0.123 against 0.165 for a
plain price heuristic). That is not a defect, it is the shrinkage doing its job:
the engine regresses extreme projections toward what the evidence supports, which
improves average accuracy and ranking while making it worse at guessing which
specific player explodes this weekend.

For FPL that trade is the right way round — a squad is held over multiple
gameweeks, so valuation and ordering matter more than calling a single haul. But
anyone using this to pick a one-week captain differential should know the model
is deliberately conservative about exactly that.

## What is still untested

The plan's second criterion — **simulated final rank** — is not measured here.
That needs a full-season squad simulation with weekly transfers, hits and chip
usage, which is a larger piece of work than the per-gameweek accuracy test. Per-GW
MAE and Spearman are passed; final rank remains an open claim.

---

# The attacker bias, measured properly

This number has been quoted three times in this file with three different values,
because the first two measurements were confounded. The clean version, taken from
the walk-forward backtest where projection and outcome sit on identical rows:

| position | projected | realised | bias |
|---|---|---|---|
| DEF | 1.21 | 1.21 | **−0.00** |
| GKP | 0.66 | 0.73 | −0.07 |
| MID | 0.96 | 1.15 | −0.18 |
| FWD | 1.00 | 1.30 | **−0.30** |
| all | 1.01 | 1.14 | −0.12 |

**Defenders are essentially unbiased.** Forwards are under-projected by 0.30
points per gameweek, midfielders by 0.18, and the model overall by 0.12 — about
10% of the mean. That is a real but modest bias, and smaller than the −0.83 the
filtered comparison implied.

## The confound, stated plainly so it stops recurring

`xpts` is an **unconditional** expectation: it already includes the probability
that the player does not feature. Comparing it against outcomes **conditional on
having played** compares two different quantities and manufactures a large fake
bias:

    filtered to players who featured:   2.16 projected vs 3.00 realised  (−0.83)
    all rows, the correct comparison:   1.01 projected vs 1.14 realised  (−0.12)

Seven times out of ten the model is projecting a player who may not play at all.
Dropping those rows from the denominator but not the expectation is what produced
the earlier numbers.

## What is not a bias

The error decomposes sharply by outcome:

| realised | n | mean error |
|---|---|---|
| 0 (incl. did not play) | 15,485 | +0.36 |
| 1–2 | 5,347 | +0.58 |
| 3–5 | 1,726 | −1.39 |
| 6–9 | 1,227 | −4.47 |
| 10+ | 419 | −9.55 |

This looks alarming and is not a defect. **An expectation must over-predict low
outcomes and under-predict high ones** whenever outcomes are dispersed — a player
projected at 3.0 who blanks contributes +3.0 of error and the same player hauling
15 contributes −12.0, and both are correct behaviour for E[X]. The distribution
is what the simulation is for; the mean is not supposed to track the tail.

The right test of bias is whether the mean is unbiased overall, which it nearly
is. The right test of the tail is calibration of the distribution, which is what
the per-player outcome histograms show.

---

# The second criterion: playing the season out

Per-gameweek accuracy is not the same claim as finishing well. A season is a
sequence of constrained decisions — a fixed budget, one free transfer a week, −4
for the second, and a squad that must be carried through blanks. So the season
was played out: at each gameweek the manager sees only what had happened,
re-optimises under the real constraints, and is scored on what his XI returned.

Four managers, identical information, GW8–38:

| manager | total | per GW | vs template |
|---|---|---|---|
| **engine** | **1,549** | **49.97** | **+69** |
| points to date | 1,498 | 48.32 | +18 |
| template (crowd) | 1,480 | 47.74 | — |
| form (last 4 GW) | 1,311 | 42.29 | −169 |

The engine wins, beats the template in **19 of 31 gameweeks**, and projects to
about **+85 points over a full 38-gameweek season**.

## But it is not statistically significant, and that matters

| | |
|---|---|
| weekly margin over template | **+2.23** points |
| standard deviation of that margin | 15.18 |
| 95% CI | **[−3.12, +7.57]** — straddles zero |
| paired t-test | t = +0.82, **p = 0.42** |
| effect size (Cohen's d) | 0.147 |
| gameweeks needed for 80% power | **~365, or 9.6 seasons** |

So the two criteria give genuinely different verdicts, and both are true:

* **Projection accuracy: passed decisively.** MAE 31% better than the template,
  Spearman 46% better, measured on 9,196 player-gameweeks. That is a large
  sample and the result is not in doubt.
* **Season points: won, but unproven.** +69 points is a real margin and it is
  the right sign, but weekly FPL variance is so large (sd 15.2 on the margin
  alone) that a single season cannot distinguish it from luck.

**This is the honest state of the project.** The model demonstrably knows more
than the crowd about what a player will score. Whether that knowledge converts
into a rank you could not have got by following the template is not something
one season can answer — and anyone claiming otherwise from a single season's
backtest is reading noise.

The gap between the two results is itself the finding: squad constraints are a
coarse filter. Fifteen players, three per club and a hundred-million budget push
every manager toward overlapping squads, so a large edge in projection quality
compresses into a small edge in points. Better projections are necessary and
nowhere near sufficient.

---

# Four seasons: the projection edge is real, the points edge is not

The single-season result reported earlier (+69 points over the template) was one
draw from a distribution centred on zero. Running all four seasons with xG
coverage settles it, and the two halves of the plan's criterion come apart
completely.

## Accuracy holds everywhere

| season | engine MAE | template MAE | to-date MAE | engine ρ | template ρ |
|---|---|---|---|---|---|
| 2022-23 | **1.885** | 2.730 | 1.975 | **0.291** | 0.229 |
| 2023-24 | **1.935** | 2.694 | 1.976 | **0.273** | 0.219 |
| 2024-25 | **1.863** | 2.636 | 1.922 | **0.272** | 0.244 |
| 2025-26 | **1.991** | 2.895 | 2.067 | **0.282** | 0.193 |

**4 of 4 seasons against the template on MAE. 4 of 4 against a to-date average.
4 of 4 on Spearman.** Roughly 95,000 player-gameweeks. The model knows more about
what a player will score than either benchmark, and it is not a fluke of one
season.

## Season points do not follow

| season | engine | template | diff |
|---|---|---|---|
| 2022-23 | 1,349 | 1,376 | **−27** |
| 2023-24 | 1,495 | 1,572 | **−77** |
| 2024-25 | 1,592 | 1,546 | +46 |
| 2025-26 | 1,549 | 1,480 | +69 |

**2 seasons of 4. Mean +2.8 points per season (sd 67). Pooled across 124
gameweeks: mean weekly margin +0.089, 95% CI [−3.15, +3.33], p = 0.96.**

There is no detectable season-points edge over the template.

## What this actually means

The plan's line was "if you can't beat the template, you have a hobby, not an
edge." The honest answer is that both halves are true at once:

* **On knowledge, the engine wins consistently.** Better MAE and better ranking
  in every season tested, against every benchmark.
* **On points, it does not.** The advantage disappears into squad constraints and
  weekly variance.

The mechanism is not mysterious. Fifteen players, three per club, a fixed budget
and eleven starters force every manager toward overlapping squads — and the
template is not a weak benchmark, it is the aggregated judgement of millions of
managers, which is already close to the ceiling that the constraints allow. A 30%
edge in projection error compresses into a margin smaller than one week's noise.

**This is the correct conclusion to draw and it was not the expected one.**
Earlier entries in this file reported the single-season +69 as though it settled
the second criterion. It did not. Four seasons show it was noise, and reporting
it as an edge would have been wrong.

## What would change the answer

Not better projections — those are already winning 4/4 and the points still do
not move. The binding constraints are structural, so the levers are structural:

* **Rank optimisation rather than expected points.** The plan flags this as a
  decision to take before Phase 7 and it was never taken. Against a field that
  mostly owns the template, effective ownership makes variance an asset; the
  max-EV squad is explicitly not the rank-optimal one. This is the single largest
  untried idea in the project.
* **Chips and transfer timing**, which the simulation handles crudely and which
  are worth real points.
* **Differential selection** — deliberately diverging from the template where the
  model's disagreement is largest, rather than picking the highest xPts squad
  that happens to overlap it heavily.

---

# Build 1: rank optimisation — a qualified, honest result

## The maths first, because it constrains everything

Write your margin over the field as a sum over every player:

    margin = SUM_i (own_i - EO_i) * points_i
           = SUM_{owned} points_i  -  SUM_all EO_i * points_i

The second term is independent of your choices. **So in expectation, maximising
rank is identical to maximising expected points.** Any benefit from differentials
must come from rank being a *non-linear* function of margin: finishing top 1%
needs a large margin, not a positive one, and large margins need variance.

That is the entire theory of the differential. It is also why a points-based test
cannot evaluate it — the strategy deliberately trades points for variance.

## Large tilts are clearly harmful

Sweeping `rank_value = xpts * (1 + k*(1 - EO))` over four seasons:

| k | 22-23 | 23-24 | 24-25 | 25-26 | total | seasons won |
|---|---|---|---|---|---|---|
| 0.00 | −27 | −77 | +46 | +69 | +11 | 2 |
| 0.15 | −132 | −46 | +32 | +5 | −141 | 2 |
| 0.30 | −151 | −228 | −13 | −132 | −524 | 0 |
| 0.50 | −197 | −233 | −26 | −123 | −579 | 0 |
| 1.00 | −300 | −325 | −50 | −181 | −856 | 0 |

Monotonic and unambiguous. The mechanism is visible in the margin distribution:
tilting *does* buy variance (weekly SD 18.4 → 20.6) but the mean cost is far
larger (+0.09 → −6.9 per week), so the upper tail does **not** improve. Ownership
correlates with quality — heavily-owned players are owned *because* they are good,
and systematically avoiding them means systematically holding worse players.

## A small tilt is a different story, but a weak one

Leave-one-season-out — choose k on three seasons, score on the fourth:

| held out | k chosen | vs template at k=0 | vs template at chosen k |
|---|---|---|---|
| 2022-23 | 0.05 | −27 | −20 |
| 2023-24 | 0.05 | −77 | **+10** |
| 2024-25 | 0.05 | +46 | +51 |
| 2025-26 | 0.05 | +69 | +65 |

**All four folds independently chose k = 0.05**, which is meaningful — the
parameter is stable, not fitted to one season. Seasons beating the template goes
from **2/4 to 3/4**, which is the pre-registered criterion.

But the effect is small and badly distributed. Mean improvement +23.8 per season,
**median +6**, carried almost entirely by one season (+87 in 2023-24) while the
other three are +7, +5, −4. Against template: mean +26.5, t = 1.37, **p = 0.27**.

## Verdict

**Meets the letter of the criterion, fails the spirit.** 3 of 4 seasons is what
was asked for, and it is not significant, and the median season gains six points.
Shipping k = 0.05 as a mild default is defensible — every fold selects it, and it
does not hurt. Claiming it as an edge is not.

The larger finding stands: **the strong version of the differential thesis is
wrong.** Deliberately fading the template costs more in expectation than the
variance is worth, at every dose above about 0.05. That is a real answer to the
question the build plan raised and left open, and it is the opposite of the
received wisdom.

## Method note on a test that would have misled

The bootstrap that first suggested k = 0.05 resamples *gameweeks* from four
seasons. It produces tight-looking intervals that describe week-sampling error,
not whether a parameter generalises across seasons — and it showed a spike at
k = 0.05 that reverted by k = 0.10, which is an overfitting signature rather than
a dose-response. Leave-one-season-out is the test that answers the question, and
it downgraded the result substantially.

---

# Goalkeeping, rebuilt on post-shot expected goals

PsxG and xGOT are the same quantity: a shot's value recomputed *after* it is
struck, conditioning on where it crossed the goal line. For a keeper it is the
right denominator, because it asks the only fair question — **given the shots he
actually faced, how many would an average keeper concede?**

    goals prevented = SUM PsxG(on-target shots faced) − goals conceded

Save percentage cannot answer that. A keeper behind a poor defence faces better
chances and posts a worse save rate while playing better; PsxG divides that out.

## Attribution fixed

Shots are now attributed to the goalkeeper via `keeperId` from the cached
shotmaps — no new requests. An earlier version aggregated by defending *team*,
which merged two keepers at any club that rotated or lost someone to injury and
produced a null on 20 data points. **27 keepers, 3,144 on-target shots, all 27
resolved to FPL** (four needed manual aliases: Raya, Sánchez, Bayındır, José Sá).

| keeper | faced | PsxG | conceded | goals prevented |
|---|---|---|---|---|
| Lammens | 111 | 40.1 | 33 | **+7.05** |
| Verbruggen | 140 | 42.6 | 37 | +5.59 |
| Donnarumma | 104 | 34.0 | 29 | +5.04 |
| … | | | | |
| Alisson | 83 | 26.0 | 29 | −3.05 |
| Vicario | 132 | 42.9 | 48 | **−5.07** |

## Goal-mouth placement behaves exactly as football says it should

`goalCrossedY` / `goalCrossedZ` give the crossing point, so shots split by area:

| area | shots | conversion |
|---|---|---|
| high-left | 90 | **50.0%** |
| high-right | 107 | 43.9% |
| low-left | 454 | 39.9% |
| low-centre | 544 | 18.8% |
| mid-centre | 423 | **16.3%** |

Corners convert at two to three times the rate of central shots. That is the
whole of goalkeeping in one table, and it validates the coordinate parsing.

## But it is descriptive, not predictive — and the control proves the test is weak

| | r (first half → second) | t |
|---|---|---|
| goals prevented per shot | −0.203 | −0.95 |
| save % over expected | −0.203 | −0.95 |
| raw save % | −0.409 | −2.06 |
| **PsxG faced per shot (control)** | **−0.116** | −0.53 |

**The control fails.** Difficulty faced is defensive quality and must persist;
it does not, at 23 keepers with roughly 65 shots each per half. So this test
cannot detect persistence in anything, and the honest statement is *not* "keeper
shot-stopping doesn't persist" but "**one season cannot measure it**". The
literature puts stabilisation at two to three seasons, which matches.

Raw save % showing significant *negative* persistence (−0.409, t = −2.06) is
straightforward mean reversion, and is the reason not to use it.

## PsxG faced is a keeper metric, not a defence metric

Tested as a clean-sheet input, past goals conceded beats it:

| predictor of next-half goals conceded | r |
|---|---|
| past goals conceded | **+0.657** |
| past PsxG faced | +0.503 |

Counterintuitive until you notice PsxG counts **only shots on target**. Off-target
and blocked shots carry real defensive information that it discards. For team
defence the right input is xGA over all shots, which the Dixon-Coles model
already uses. PsxG belongs to the keeper, not the back four.

## Shipped as

A goalkeeper panel on 22 players: goals prevented as the headline, PsxG faced,
conceded, and save % over expected — **explicitly marked descriptive and excluded
from the projection**, the same treatment given to xGOT for outfielders and
six-yard-box share for strikers. Good scouting information; not a forecast.

---

# Build 2: transfer economics — and why hits destroy points

The simulation previously re-picked greedily each week. Build 2 adds the three
rules that actually govern transfers: horizon valuation rather than next-fixture
valuation, banked free transfers (up to five), and sell price with FPL's 50%
profit tax, which makes budget path-dependent on what you paid.

## Hits are the story

Testing hit discipline across four seasons, where a hit is taken when
`weekly_gain × hold_weeks > 4 × margin`:

| hit margin | total vs template | hits taken |
|---|---|---|
| 1.0× (the naive rule) | **−300** | 39 |
| 1.5× | −102 | 22 |
| 2.0× | −152 | 17 |
| 3.0× | −71 | 3 |
| **no hits at all** | **+49** | **0** |

Thirty-nine hits cost 156 points directly but did **349 points** of damage — so
more than half the loss came from the squads the hits produced, not the fee.

The recovery is monotonic in the margin and is only complete when hits are
abandoned entirely. That pattern is the signature of the **optimiser's curse**:
the transfer you pick is the one with the largest *estimated* gain, and the
largest estimate in a noisy set is biased upward. Paying four points to act on it
means systematically buying estimation error. No simple threshold fixes it,
because raising the bar filters on the same noisy estimate that created the
problem.

This is why "hits rarely pay" is folk wisdom in FPL. It now has a mechanism and a
number attached.

## The rest of the build is a modest, unproven gain

Best configuration — managed transfers, no hits:

| season | greedy | managed | delta |
|---|---|---|---|
| 2022-23 | −27 | −128 | **−101** |
| 2023-24 | −77 | −45 | +32 |
| 2024-25 | +46 | +83 | +37 |
| 2025-26 | +69 | +139 | +70 |

**3 of 4 seasons improved; mean +9.5, median +34.5 per season; p = 0.82.** Against
the template it lifts the average from +2.8 to +12.2 per season, still on 2 of 4
seasons and still not significant.

So: transfer discipline is worth something and probably worth having, and one bad
season (2022-23, −101) is enough to swamp the average. On four seasons it cannot
be distinguished from noise, which is now the third time that sentence has been
written in this file. The pattern is consistent — **structural improvements to
squad management are worth tens of points, and season variance is worth hundreds.**

## Shipped default

`run_season_managed(..., max_hits=0)` — banked transfers and sell-price
accounting on, hits off. The hit machinery stays in the code with the margin
parameter, because the finding is about *why* it fails, not that the rule is
unimplementable.

---

# Walk-forward accuracy, four seasons

Run 2026-08-20 over 2022-23 to 2025-26, the seasons with team xG. Every model
refits at each gameweek on gameweeks already played, so a GW20 projection has
never seen GW20. 87,007 player-gameweeks where all three models produce a
number — scored on the same rows, because comparing MAE across different row
sets is not a comparison.

References are what a manager could do without the engine: season-to-date
points per game, and last-five-gameweek form.

## Magnitude: the engine wins every season

| season | form | naive | engine | better by | n |
|---|---|---|---|---|---|
| 2022-23 | 1.044 | 1.076 | **0.963** | 7.8% | 17,236 |
| 2023-24 | 0.984 | 0.992 | **0.921** | 6.4% | 23,980 |
| 2024-25 | 1.035 | 1.046 | **0.984** | 4.9% | 21,958 |
| 2025-26 | 1.035 | 1.050 | **0.949** | 8.3% | 23,833 |

MAE on matched rows. By position the gain concentrates where the scoring is:
midfielders 10.1%, forwards 10.6%, keepers 4.3%, defenders 2.0%.

*(Corrected. This table first carried per-season figures in which each model was
scored on whatever rows it could produce — form covered fewer rows in 2022-23
than the engine did — while the surrounding text claimed matched rows. The
2022-23 margin was overstated as 11.2%, and the honest range is 4.9% to 8.3%
rather than 5.2% to 11.2%. The ordering and every conclusion are unchanged.)*

## Ranking: a tie, until you look at the top

| season | form | naive | engine |
|---|---|---|---|
| 2022-23 | 0.687 | 0.652 | **0.689** |
| 2023-24 | **0.686** | 0.663 | 0.676 |
| 2024-25 | **0.698** | 0.671 | 0.691 |
| 2025-26 | **0.720** | 0.694 | 0.709 |

Mean Spearman per gameweek. **The engine loses three of four.**

This is not the contradiction it looks like. 61% of rows are players who did
not appear. Across the whole list the ranking question is mostly "will he
play at all", and a five-game form average answers that nearly as well as a
minutes model. Global rank correlation is therefore dominated by rows nobody
picks from.

## Where decisions are actually made, the engine wins clearly

Mean points scored by the players each model ranked highest, chosen before the
gameweek:

| picked | engine | form | naive | field |
|---|---|---|---|---|
| top 10 | **4.80** | 4.21 | 4.41 | 1.11 |
| top 20 | **4.25** | 3.95 | 4.01 | 1.11 |
| top 50 | **3.68** | 3.57 | 3.65 | 1.11 |
| top 1 (captain) | **6.69** | 5.01 | 6.09 | 1.11 |

**The captaincy edge is the largest single effect measured in this project.**
+0.60 per gameweek over season-to-date form, roughly +23 points a season from
the armband alone, and +1.68 over five-game form.

The engine's skill is concentrated at the top of its own ranking. That is the
useful shape: it is the top of the list you pick from.

## Calibration: the top end is under-projected

| projected | n | actual | gap |
|---|---|---|---|
| 3–4 | 5,354 | 3.66 | **+0.34** |
| 4–5 | 177 | 5.45 | **+1.21** |
| 5+ | 6 | 5.83 | +0.75 |

Two problems. The projections are systematically low at the top, and they are
compressed: across four seasons the engine produced six projections above five
points. It never says a player will haul.

Conditional on a 60-minute appearance, and after dividing out the appearance
probability so the comparison is fair, the residual bias is positional:
forwards −1.62, midfielders −1.11, keepers −0.66, defenders −0.17.

## What this changes

Nothing about the headline. Better projections were already winning on
accuracy and still do not move season points, which remains the finding of
this project. But two things are now specific rather than vague:

1. **The engine's value is captaincy and the top of the ranking**, not the
   full-list ordering, where it loses to five-game form.
2. **Attackers are under-projected by roughly a point and a half a start.**
   That is a concrete defect with a known sign, and it is the same population
   the top-end compression affects.

---

# Is the model directionally right about who is underpriced?

Run 2026-08-20 across 2022-23 to 2025-26. At each gameweek the engine fits
points against log price, per position, on gameweeks already played. A player
projected above what his price implies is called undervalued. The outcome is
measured over the following six gameweeks, which had not happened when the
call was made.

## The first answer was wrong

Uncontrolled, the top decile beat comparably-priced players 94.7% of the time.
That is not a mispricing signal. The flag was tracking appearances:

| decile | 1 | 5 | 10 |
|---|---|---|---|
| mean P(appear) | 0.20 | 0.21 | 0.91 |

The peer group contained benched players scoring zero, so any starter cleared
it. The model was being credited for knowing who plays, which was already
established and is not what was being asked.

## Controlled

Starters only, compared against other starters in the same 0.5m price band:

| quintile | resid | next-6 points | peers | beat peers | price rose |
|---|---|---|---|---|---|
| Q1 | −0.11 | 11.34 | 16.09 | 20.1% | 7.4% |
| Q2 | +0.87 | 15.82 | 16.73 | 40.6% | 13.3% |
| Q3 | +1.26 | 17.11 | 16.44 | 49.3% | 14.2% |
| Q4 | +1.52 | 17.46 | 15.92 | 53.6% | 12.6% |
| Q5 | +1.82 | **19.24** | 15.64 | **63.4%** | **19.4%** |

**Monotonic, and it holds in all four seasons** — 63.1%, 59.0%, 63.2%, 68.4%.

## The signal scales with the size of the call

| flag size | n | hit rate | edge vs peers |
|---|---|---|---|
| negative | 2,105 | 15.4% | −5.90 |
| 0 to 1 | 4,247 | 34.2% | −1.85 |
| 1 to 1.5 | 5,896 | 49.2% | +0.65 |
| 1.5 to 2 | 4,480 | 59.0% | +2.59 |
| 2 to 2.5 | 1,010 | 68.0% | +4.54 |
| 2.5+ | 35 | 77.1% | +7.74 |

A bigger call is a better call, which is what a real signal looks like and
what an artefact usually does not.

By position the flag works best on midfielders (69.2%) and worst on keepers
(56.6%), with defenders 58.8% and forwards 62.9%.

## Size of the effect

A most-undervalued starter returns **+3.60 points over six gameweeks** against
players priced like him — 0.60 a gameweek. Real, and consistent with everything
else in this file: worth tens of points a season, against seasonal variance
worth hundreds. It is also why the price-rise number matters less than it
looks. A rise is worth 0.1m of team value; outscoring the bracket is worth
points.

**Roughly one call in three is wrong.** 63% is an edge, not a rule.

---

# Possession unblocks the DefCon covariate gap

Six seasons of FotMob match detail were ingested 2026-08-20/21: 2,281 matches,
90,621 player-matches, 58,647 shots, 4,560 team-matches. Zero failures.

The team block carries **ball possession**, which the build plan had recorded
as unavailable outside FBref and which left Build 4 blocked. It is present on
100% of team-matches. Sanity holds: the mean is 50.0%, home sides average
51.0% against 49.0% away, and every pair sums to 100.

## Opponent possession predicts defensive contribution

47,585 player-matches of 60 minutes or more:

| line | n | mean DefCon | r vs opponent possession |
|---|---|---|---|
| DEF | 17,692 | 6.71 | **+0.298** |
| MID | 14,268 | 9.46 | +0.114 |
| FWD | 11,078 | 5.24 | +0.180 |

## The effect on the threshold is large

DefCon pays 2 points at 10 actions for a defender. That threshold is reached
almost four times as often against a side that keeps the ball:

| opponent possession | n | mean DefCon | hit rate |
|---|---|---|---|
| under 40% | 4,217 | 5.28 | **8.6%** |
| 40–45% | 2,206 | 6.24 | 15.5% |
| 45–50% | 2,624 | 6.55 | 18.5% |
| 50–55% | 2,396 | 6.98 | 21.7% |
| 55–60% | 2,214 | 7.33 | 25.4% |
| 60% or more | 4,035 | 8.06 | **31.7%** |

**Monotonic, and a 3.7-fold spread on a variable the model does not currently
use.** DefCon is the weakest component in the engine — calibration collapses
above p = 0.5, which is why the tail is capped at 0.40. This is the covariate
the build plan named, and it is now in hand.

A defender at home to a possession side is a materially different DefCon bet
from the same defender away at a team that sits deep. The model presently
treats them alike.

---

# Possession does not improve DefCon — Build 4 closed

The build plan named opponent possession share as the covariate DefCon was
missing, and recorded it as unreachable outside FBref. It is in the FotMob team
block, now ingested for six seasons with 100% coverage and validated: possession
sums to 100 in all 4,560 team-match rows, covering 2,280 of the 2,281 matches
cached.

The raw relationship is exactly as advertised. Over 47,585 player-matches, an
outfielder whose opponent holds 60% or more of the ball reaches the DefCon
threshold **26.6%** of the time; at 40% or less, **11.9%**. The rate more than
doubles.

*(Corrected. The figures first published here, 34.4% and 17.4%, applied a flat
threshold of 10 to every outfielder. FPL's threshold is position-dependent:
10 for defenders, 12 for midfielders and forwards. The ratio is unchanged at
about 2.2x and no conclusion moves, but the levels were overstated.)*

Possession is also forecastable, which goals are not. Team means correlate 0.77
to 0.91 season to season and 0.85 from the first ten matches to the rest. A
rating difference plus a home term predicts a fixture at r = 0.71 walk-forward,
against 10.3 points of error for assuming an even split. Forecast rather than
actual possession still moves the DefCon hit rate from 11.9% to 24.0%.

**And it adds nothing to the model.**

| features | Brier | log-loss | vs no-opponent baseline |
|---|---|---|---|
| form only | 0.14293 | 0.44139 | — |
| + opp_strength *(current)* | **0.14161** | **0.43795** | **+0.92%** |
| + expected opponent possession | 0.14297 | 0.44135 | −0.03% |
| + both | 0.14181 | 0.43838 | +0.78% |

Walk-forward on 2025-26, 6,120 scored player-gameweeks, 60-minute appearances.

Not collinearity — `opp_strength` and expected possession correlate only 0.112.
The information is already in the player's own history. `dc_ewm5` and
`dc_ewm10` are a direct measurement of how much defending this player does; a
player at a low-possession club already carries a high rolling average. A
team-level possession forecast is a noisy proxy for something the player's own
record states outright.

**Build 4 moves from blocked to tested and rejected.** The covariate is real,
the football reasoning was sound, and the model does not want it. The possession
data stays ingested — it cost nothing to keep and may serve clean sheets, where
no equivalent player-level history exists.

---

# Rank optimisation lands — the first squad-level win

Build 1 in the plan, and the only item four seasons of evidence pointed at.

## The surrogate was wrong, and wrong in its sign

The existing module tilted value toward scarce players: `xpts * (1 + k(1 - EO))`.
Swept, **every k > 0 lost, monotonically** — k = 0.15 gave −141 and k = 1.0 gave
−856. Paying for scarcity destroys points.

## The exact objective is linear, which the surrogate existed to avoid

Margin over the field is `SUM_i (own_i - EO_i) * points_i`. Its variance looks
quadratic in the decision and therefore outside a linear programme. It is not.
`own_i` is binary, so `own_i^2 = own_i`, and

    (own_i - EO_i)^2 = own_i * (1 - 2*EO_i) + EO_i^2

The second term is constant. **Margin variance is linear in the squad choice**,
weight `(1 - 2*EO_i)`, and the LP carries it exactly. No surrogate needed.

The sign of that weight is the whole finding. A player owned by more than half
the field carries a *negative* weight: holding him reduces margin variance,
because his hauls and blanks move you and the field together.

## Result

| objective | 22-23 | 23-24 | 24-25 | 25-26 | total | wins |
|---|---|---|---|---|---|---|
| expected points | −27 | −77 | +46 | +69 | +11 | 2 of 4 |
| **gamma = −0.05** | **+78** | **+43** | **+49** | **+119** | **+289** | **4 of 4** |
| gamma = +0.05 | −311 | −324 | −38 | −183 | −856 | 0 of 4 |

Points against the template. Sign test on 4 of 4 gives p = 0.06; mean +72 a
season.

The optimum is a **shallow plateau, not a point**, which is the better outcome:
γ = −0.02 scores +362 and γ = −0.05 scores +357 on contemporaneous ownership,
while γ = −0.10 gives +275 and any positive γ collapses. The negative region is
also somewhat noisy — γ = −0.15 dips to +87 between two better neighbours — so
the defensible claim is that a small negative γ works, not that −0.05
specifically is optimal.

### Leave-one-season-out

Choosing γ by sweeping four seasons and reading off the best total is fitting on
the test set. So γ was re-chosen on three seasons and applied blind to the
fourth:

| held out | γ chosen on the other three | held-out result |
|---|---|---|
| 2022-23 | −0.02 | **+53** |
| 2023-24 | −0.02 | **+63** |
| 2024-25 | −0.05 | **+62** |
| 2025-26 | −0.05 | **+167** |

**+345 out of sample, winning 4 of 4**, against +357 fitted in sample on the
same ownership basis. An overfitting penalty of 12 points across four seasons is
close to nothing, and the two γ values chosen sit next to each other on the
plateau. This is the strongest evidence the result is not an artefact of the
sweep.

## Two controls, because 4 of 4 demands them

**Is it just risk aversion?** Penalising variance flat, with the ownership
weight removed, scores **−711 and wins 1 of 4**. The ownership structure is
doing the work, not a preference for consistent players.

| | total | wins |
|---|---|---|
| mean-variance, EO-weighted | **+357** | 4 |
| variance only, no ownership | −711 | 1 |
| expected points | +11 | 2 |

**Is it leakage?** `owned` moves during a gameweek and correlates a little more
with that week's points than its lag does (0.336 against 0.318). Re-run on
strictly pre-deadline ownership the result holds: **+289, still 4 of 4**. About
68 points of the headline was that contamination; the finding is not.

## What it means in football terms

The folk wisdom is that differentials win rank. Under FPL's constraints it is
backwards. A blank from a player nobody owns costs you rank. A blank from a
player everybody owns costs nothing, because the field blanks alongside you.
**Take your risk where the field takes it with you, and be conservative where
you are alone.**

This is also the first result in this file to beat the template on season points
in more than two seasons. Every previous structural intervention was worth tens
of points against seasonal variance worth hundreds. This one is worth +72 a
season, four times from four.

---

# Does the methodology hold across different Premier Leagues?

A four-season average can hide a method that works in one kind of league. These
seasons are not interchangeable: goals per team-game run 1.35 to 1.64,
clean-sheet rate 0.21 to 0.30, and the spread of team possession 10.7 to 14.2
points. A threshold-heavy scoring system should be most sensitive to exactly
those things. 87,007 player-gameweeks, sliced.

## The projection advantage is stable everywhere

| slice | engine MAE | best reference | gain |
|---|---|---|---|
| low-goal gameweeks | 0.944 | 1.023 | **7.7%** |
| high-goal gameweeks | 0.951 | 1.013 | **6.1%** |
| few clean sheets | 0.942 | 1.011 | **6.8%** |
| many clean sheets | 0.970 | 1.040 | **6.7%** |
| blank gameweeks | 0.923 | 0.982 | **6.0%** |
| double gameweeks | 0.925 | 0.985 | **6.1%** |
| midweek rounds | 0.965 | 1.023 | **5.7%** |
| early season (GW8-12) | 1.003 | 1.109 | **9.5%** |
| late season (GW27-38) | 0.908 | 0.962 | **5.5%** |

**No environment where it fails.** The range is 4.9% to 9.5% and every slice is
positive.

## But the *useful* advantage is not stable

MAE is not what a manager spends. Top-10 selection is. Measured as the points
scored by the ten players the engine ranked highest, against the ten a
points-per-appearance benchmark ranked highest:

| slice | engine top-10 | reference | gain |
|---|---|---|---|
| weekend rounds | 4.93 | 4.48 | **+0.45** |
| normal gameweeks | 4.85 | 4.33 | **+0.51** |
| high-goal gameweeks | 5.28 | 4.82 | **+0.47** |
| few clean sheets | 5.00 | 4.48 | **+0.51** |
| early season | 4.93 | 4.14 | **+0.79** |
| late season | 4.67 | 4.43 | +0.24 |
| double gameweeks | 4.61 | 4.49 | +0.12 |
| **many clean sheets** | 4.43 | 4.37 | **+0.06** |
| **midweek rounds** | 4.02 | 3.98 | **+0.04** |
| **blank gameweeks** | 4.83 | 4.83 | **0.00** |

Three environments where the edge disappears:

**Blank gameweeks.** Exactly zero. When the pool shrinks, the engine's top ten
and the benchmark's top ten are worth the same.

**Midweek rounds.** +0.04 against +0.45 at weekends. Midweek is rotation, and
rotation is a minutes problem the model does not solve — it knows who usually
starts, not who a manager will rest three days before a European tie.

**Clean-sheet-rich gameweeks.** +0.06. When defences dominate, points
concentrate in threshold events the model caps at 0.40 by its own admission.

## Where the advantage lives, by position

| environment | DEF | MID | FWD | GKP |
|---|---|---|---|---|
| low-goal gameweeks | 2.4% | 11.9% | **12.6%** | 3.0% |
| high-goal gameweeks | 1.4% | 8.8% | 8.9% | 6.9% |

Attackers in a mean league is where the model earns its keep, and defenders are
where it barely beats a rolling average anywhere. That is consistent with the
DefCon component being the weakest part of the engine.

## The honest summary

The projections are robust across environments. The *decisions* are not. Anyone
using this should discount it in midweek rounds, in blank gameweeks, and when
defences are on top — which are, unhelpfully, three of the moments a manager
most wants help.

---

# Self-audit, 2026-08-21

Every headline number from this session was recomputed from source. Four
claims survived unchanged; three did not, and are corrected in place above.

## Confirmed exactly

| claim | recomputed |
|---|---|
| Captain pick 6.69 vs 6.09 naive, 5.01 form | 6.69 / 6.09 / 5.01 |
| Top-10 4.80 vs 4.41 / 4.21 | 4.80 / 4.41 / 4.21 |
| Directional Q5 63.4%, Q1 20.1% | 63.4% / 20.1% |
| GW1 minutes weight 0.65 | 0.65 |
| Heatmap zones, xG/90 r = 0.902 | 0.902 (also holds with a season filter) |
| Possession sums to 100 in every match | 100.0% of 2,280 matches |

## Corrected

**DefCon threshold rates were computed against a flat threshold of 10.** FPL's
threshold is position-dependent — 10 for defenders, 12 for midfielders and
forwards. The published 17.4% → 34.4% becomes **11.9% → 26.6%**, and the
forecast-band version 17.5% → 32.1% becomes **11.9% → 24.0%**. The ratio is
unchanged at about 2.2x, so the possession conclusion is unaffected, but the
levels were overstated by roughly a third.

**The MAE table was published on unmatched rows** while the surrounding text
claimed matched rows. Form covered fewer 2022-23 rows than the engine did, and
that season's margin was inflated from 7.8% to 11.2%. The honest range is
**4.9% to 8.3%**, not 5.2% to 11.2%.

**The paper reported one season's improvement as the four-season aggregate,
twice.** "31.2 percent in MAE and 46.1 percent in rank correlation" are both the
2025/26 figures. The means are **29.8%** and **28.5%**, and the rank-correlation
margin is unstable across seasons (11.5% to 46.1%), which the paper now says.
The paper's Table 1 was also silently restricted to players who appeared — a
population that doubles every MAE — and now states so.

**A fourth defect was structural rather than numerical.** `rank.py` described
γ = −0.05 as the shipped default while the live pipeline still optimised plain
expected points; the result existed only in the backtest. The live optimiser now
uses it. Reporting was wrong too — the exported "squad xPts" was the LP objective
including the variance term, overstating the projection by 8 points.

## What this says

Three of the four defects flattered the result, which is the direction bias
runs when nobody checks. None changed a conclusion. The habit worth keeping is
that every number in a summary should be recomputed from source before it is
published, because the errors were all in *summaries* — the per-season tables
were right in every case, and the sentence underneath them was wrong.

---

# Chips — Build 3, +139 over four seasons

Four chip types, two half-season windows, valued on the projection alone.

| season | no chips | with chips | gain |
|---|---|---|---|
| 2022-23 | 1422 | 1488 | **+66** |
| 2023-24 | 1525 | 1570 | **+45** |
| 2024-25 | 1502 | 1536 | **+34** |
| 2025-26 | 1596 | 1590 | −6 |

**+139 total, mean +35 a season, 3 of 4 improved.**

| chip | plays | total | per play |
|---|---|---|---|
| Wildcard | 4 | +108 | **+27.0** |
| Triple Captain | 8 | +56 | +7.0 |
| Bench Boost | 8 | +51 | +6.4 |

## The rule, and the mistake in the first version

A chip must clear a bar set by what a typical gameweek in its window is worth.
Patience is barely rewarded — swept, the bar scored +116 at 1.15 and +115 at
1.00, falling away above 1.3 — because a chip held for a better week is a chip
at risk of expiring.

The first version collapsed the bar to zero on the window's final gameweek, on
the reasoning that any positive value beats expiry. **That is true only for
chips that cannot lose points.** Bench Boost and Triple Captain add a
multiplier to players already owned. Wildcard and Free Hit *replace the squad*,
and a squad re-solved on noisy projections can be worse than the one held.

Forcing a wildcard at the 2025-26 deadline cost that season 44 points — and the
immediate gain that gameweek was only −2, with the rest arriving over the
following weeks. It is the optimiser's curse again, the same mechanism that
made transfer hits unprofitable: re-solving selects the largest *estimate*, and
the largest estimate in a noisy set is biased upward.

Restricting the deadline collapse to additive chips cut the worst season from
−44 to −6. It also cost the good seasons — the unrestricted rule scored +192
against this rule's +139, because wildcards now play four times across four
seasons rather than eight, and a window's wildcard often expires unused.

**Shipped the safer rule.** +35 a season with a −6 worst case beats +48 a season
with a −44 worst case, on a project whose recurring finding is that seasonal
variance swamps structural edges. Expiring a wildcard unused is the correct
outcome when nothing clears the bar.

One caveat worth stating: the threshold was swept on the same four seasons it is
reported on. The effect is large enough that the sign is not in doubt, but the
magnitude is fitted.
