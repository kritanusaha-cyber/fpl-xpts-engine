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
