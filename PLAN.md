# FPL xPts Engine — Build Plan (revised against live API recon)

**Date:** 2026-08-18 · **Source doc:** `~/Downloads/FPL.pdf` · **Target repo:** `~/FPL`

---

## 0. What recon changed

I pulled `bootstrap-static/`, `fixtures/`, and checked the upstream data sources before
planning. Five findings change the build materially. The original doc is directionally
right; these are the deltas.

### 0.1 The season has NOT started — you have 3 days, not a cold start

The doc's framing premise is wrong, in your favour:

| | Doc assumed | Reality (API) |
|---|---|---|
| Season state | "already underway" | **0 of 38 events finished** |
| Data available | "near-zero current-season" | **zero** current-season |
| GW1 deadline | — | **2026-08-21 17:30 UTC** (Friday, ~3 days) |

Consequences:

- **The cold-start blend is not a week-1 concern.** For GW1–3 the blend weight is
  ~1.0 on priors by construction. Build the *mechanism* (the doc is right about that)
  but do not spend week 1 tuning a decay schedule you cannot fit for a month.
- **The snapshotter is urgent and everything else is not.** Price and ownership move
  continuously and the API exposes no history. Every day you do not snapshot
  `bootstrap-static` is a day of price/ownership dynamics permanently lost. The API
  publishes `price_change_deadlines` — the next three are 2026-08-18/19/20 at 23:00 UTC.
  **This is the only thing that is time-critical today.**
- You get a clean season boundary for the walk-forward backtest, which is a genuinely
  nice position to start from.

### 0.2 `opta_code` is populated for all 592 players — entity resolution mostly dissolves

Every element carries an Opta player ID (`p154561`, …), plus `birth_date` on 575/592.
The doc budgets a fuzzy matcher on (name, club, position, birth year) plus
"40–80 rows" of manual overrides, refreshed every window. If Opta codes join cleanly to
FBref (Opta-derived) this collapses to a key join with a small residual.

**This is the single largest de-risking item in the plan and it is cheap to test.**
Spike it on day 1 — one afternoon, ~50 players — before committing to the fuzzy-matcher
architecture. Keep `manual_overrides.csv` and the CI assertion regardless; just expect
the override table to be ~5 rows, not 80. Understat has no Opta ID, so the fuzzy path
still gets built — it just serves one source instead of three.

### 0.3 The full scoring table is in the API — generate the YAML, don't transcribe it

`game_config.scoring` ships the complete 2026/27 point values:

```
long_play 2 · short_play 1 · assists 3 · saves 1 · bonus 1
goals_scored      GKP 10 · DEF 6 · MID 5 · FWD 4
clean_sheets      GKP  4 · DEF 4 · MID 1 · FWD 0
goals_conceded    GKP -1 · DEF -1 · MID 0 · FWD 0   (per 2)
defensive_contribution  GKP 0 · DEF 2 · MID 2 · FWD 2
yellow -1 · red -3 · own_goal -2 · pen_missed -2 · pen_saved 5
```

So `config/scoring_2026_27.yaml` should be **generated from the API and diffed in CI**,
not hand-written. That removes a whole class of silent transcription bugs and makes the
2027/28 rule change a no-op rather than a config edit.

Two things are *not* in the API and remain genuine open items:
- **DefCon thresholds** (doc says 10 for DEF, 12 for MID/FWD). The API gives the *value*
  (2 pts) but not the threshold. Must be sourced from the rules page and verified against
  realized 2025/26 data. Note FWD now scores DefCon at 2 — confirm the threshold applied.
- **The BPS weight table.** `bps` scores 0 (it is an input to bonus, not scored directly)
  and the weight table is not exposed. The doc's 2026/27 changes (CBI 1-per-3, tackled
  penalty removed, saves flattened to 2, pen save 7) need independent verification.
  The doc's own suggestion — reconstruct BPS for completed matches and check against
  realized bonus — is the right validation and is the *only* trustworthy one.

### 0.4 DefCon inputs come free from the FPL API

`element_stats` confirms the API natively serves `tackles`, `clearances_blocks_interceptions`,
`recoveries`, and `defensive_contribution` per player per GW, alongside `expected_goals`,
`expected_assists`, `expected_goals_conceded`, and `starts`.

Phase 4's target variable and its composition split therefore come straight from FPL —
no FBref dependency for the *target*. FBref is still needed for covariates (progressive
passes, aerials by third, per-90 role context) and for backfilling seasons before FPL
began publishing DefCon (2025/26). **That backfill is the real Phase 4 constraint:
you have one season of native DefCon history, not four.** Plan the hierarchical NB
around a one-season-deep target with FBref-reconstructed history as the prior, and expect
this to be the weakest-identified component. The doc predicts it will be the
worst-calibrated; this is why.

### 0.5 Chips and transfer mechanics are richer than the doc assumes

- **8 chips, not 4.** Wildcard, Free Hit, Bench Boost, and Triple Captain each appear
  **twice** — one set for GW1–19, one for GW20–38 (Free Hit/Wildcard second set opens
  GW20; BB/TC first set is live from GW1). Chip scheduling is a real combinatorial
  sub-problem, not a bolt-on.
- `max_extra_free_transfers: 4` → 5 banked FTs max. Confirms the doc.
- `element_sell_at_purchase_price: false` with `transfers_sell_on_fee: 0.5` — **sell price
  is not purchase price and not current price.** Profit is taxed 50%, rounded. Over a
  5–8 GW MILP horizon this makes the budget constraint path-dependent on purchase price.
  The doc does not mention it and it will silently corrupt multi-GW budget feasibility.
- `price_change_projections` / `price_change_hourly_rate` are now official API fields
  (currently zeroed pre-season). Free forward-looking price signal once live — worth
  wiring into the optimizer's team-value term rather than modelling price yourself.
- `scout_risks` / `scout_news_link` — new official risk annotations, empty pre-season.
  Watch these; they may supersede `chance_of_playing_next_round` as the injury signal.

---

## 1. Revised strategy: ship a usable tool for GW1, then add depth

The doc's 8-week sequencing produces nothing usable until Phase 6 in week 6. Starting
3 days before GW1, that means flying blind for two months of live football and throwing
away the current season as training data.

The doc itself points at the fix in §13 — "a cut-down version of Phases 0, 1, 2, and 6
with naive DefCon and bonus estimates will already be a usable tool." **Promote that from
a footnote to the primary plan.** Two tracks:

### Track A — "Thin engine", days 1–3 (before Friday's deadline)

Goal: a defensible GW1 squad and a logging pipeline that starts accruing data immediately.
Explicitly *not* a good model — a scaffold with a correct spine.

1. **Snapshotter first (today).** Cron `bootstrap-static` + `fixtures` daily at 22:45 UTC
   (ahead of the 23:00 price deadlines), write versioned Parquet to `data/raw/`. Ten
   lines. Ship it before anything else.
2. Generate `scoring_2026_27.yaml` from `game_config.scoring`; add the CI diff.
3. Load vaastav 2016/17–2025/26 into DuckDB → `player_gw` fact table (both endpoints
   verified live and returning 200).
4. Opta-code join spike vs FBref. Decide the resolve/ architecture on the result.
5. **Naive xPts:** last-season per-90 → shrunk to position × price-tier prior →
   × naive minutes prior (start rate) → × fixture multiplier from bookmaker odds.
   No Poisson, no simulation, no DefCon model.
6. MILP squad selection with the *correct* constraint set (this part is exact and
   worth doing properly on day 1 — see §4.7).

Track A's value is not its projections. It is that the point-in-time discipline, the
scoring config, the fact table, and the constraint set are all correct from GW1, so every
later model drops into a spine that already works and every GW from 1 onward is logged.

### Track B — the real models, weeks 1–8

The doc's phases, resequenced by leverage-under-uncertainty:

| Week | Phase | Notes vs. doc |
|---|---|---|
| 1 | 0 — data layer | Unchanged. Opta spike may cut it short. |
| 1 | 8 — backtest harness **stub** | Doc says stub in week 2; do it in week 1. You have zero current-season data, so the *only* signal available for eight weeks is historical walk-forward. Without the harness you are tuning blind. |
| 2 | 1 — minutes | Unchanged. Highest leverage, correctly placed first. |
| 3 | 2 — team model | Unchanged. Market blend is the highest value-per-line item in the project. |
| 4 | 3 — attacking returns | Unchanged. |
| 5 | 5 — bonus | **Moved ahead of DefCon.** BPS is deterministic given the action log, fully specifiable from history, and unblocked. DefCon has one season of target data and needs live GWs to calibrate. Do the tractable one while the data accrues. |
| 6 | 4 — DefCon | By now you have ~5 live GWs of DefCon outcomes to validate against. |
| 6 | 6 — assembly | Unchanged. |
| 7 | 7 — optimizer (full) | Upgrade Track A's MILP to the H-horizon transfer planner. |
| 8 | 8 — full walk-forward | Unchanged. |

The single change with real content: **swap Phases 4 and 5**, because DefCon is the one
component whose training data arrives during the build and bonus is the one that does not.

---

## 2. Decisions I need from you

Three forks that change the architecture. I have a recommendation on each.

**D1 — Are you playing this season, or building a research tool?**
If you are entering a team for GW1 on Friday, Track A is mandatory and starts now. If this
is a research project, skip Track A, build Track B properly, and go live at GW10 or in
2027/28. *Recommend: play it.* A live team is a forcing function for calibration honesty
and the snapshotter has to run either way.

**D2 — Max expected points, or max overall rank?**
The doc flags this correctly and says decide before Phase 7. **It needs deciding before
Phase 6**, because rank-optimization requires the assembly step to preserve the *joint*
distribution and to carry effective ownership — which means ingesting ownership
time series from day 1 (another reason the snapshotter is urgent). *Recommend: build the
simulator for rank, report both.* Max-EV is a special case you get for free; the reverse
is not true.

**D3 — Is beating the template the success criterion, or is beating your own last season?**
The doc sets the template benchmark. It is the right *research* bar. If your actual goal
is a good rank rather than a publishable edge, the template benchmark can push you toward
differentials that are correct on EV and wrong on rank — which is D2 restated.
*Recommend: keep the template benchmark as the model-quality gate, but make simulated
final-rank distribution the ship/no-ship criterion.*

---

## 3. Repo layout

Essentially the doc's, with the additions recon implies:

```
fpl/
  ingest/            # one module per source → raw Parquet
    fpl_api.py       # bootstrap, fixtures, element-summary, event/{gw}/live
    snapshot.py      # daily versioned bootstrap snapshot  ← BUILD FIRST
    vaastav.py  understat.py  fbref.py  odds.py  clubelo.py
  resolve/           # opta-key join + fuzzy fallback + overrides
  features/          # all point-in-time safe, @as_of decorator enforced
  models/
    minutes.py  team_goals.py  attacking.py  defcon.py  bonus.py
    assemble.py      # joint simulation → xPts distribution
  optimize/
    squad.py         # MILP, incl. sell-price mechanics
    chips.py         # 8-chip scheduling (2 half-season sets)
  backtest/          # walk-forward harness  ← stub in week 1
config/
  scoring_2026_27.yaml     # GENERATED from game_config.scoring, CI-diffed
  defcon_thresholds.yaml   # NOT in API — hand-sourced, validated vs 2025/26
  bps_2026_27.yaml         # NOT in API — hand-sourced, validated by reconstruction
  manual_overrides.csv
data/
  raw/ interim/ features/ fpl.duckdb
```

**Toolchain:** `uv` is already installed; system Python is 3.9 so pin 3.12 (`~/.local/bin/python3.12`)
via `uv venv --python 3.12`. DuckDB, statsmodels, scikit-learn, PuLP/OR-Tools, NumPyro
for the hierarchical pieces. Makefile + cron, per the doc. No Postgres, no Airflow.

**The one rule, unchanged and non-negotiable:** every table gets `as_of_gw`; every feature
function takes `as_of_gw` and reads only `gw < as_of_gw`; enforce with a decorator; assert
in CI. Point-in-time leakage is the thing that makes a backtest lie.

---

## 4. Phase notes — deltas only

Where the doc is right I am not restating it. These are the places recon or sequencing
changes the spec.

**4.0 Data layer.** Verified live: FPL API 200 (592 elements, 20 teams, 380 fixtures),
vaastav 2025-26 `merged_gw.csv` and `players_raw.csv` both 200. Rate-limit FBref to
1 req/3s. Budget The Odds API at one pull per GW (500 credits/month). Evaluate
`olbauday/FPL-Core-Insights` as the doc suggests — but the Opta-code finding may make it
redundant; check that first, it is cheaper.

**4.1 Minutes.** As specified. Note `chance_of_playing_next_round` is joined by new
`scout_risks`/`scout_news_link` fields — snapshot all three from day 1 even though they
are empty now, so you have their history when they populate.

**4.2 Team model.** As specified — this is the strongest section of the doc. Dixon-Coles
on xG with a precision-weighted market blend. Fixture difficulty ratings are in the
`fixtures/` payload (`team_h_difficulty`/`team_a_difficulty`) as a free weak baseline to
benchmark your attack/defence parameters against.

**4.3 Attacking.** As specified. `penalties_order` is populated for 65 players in the API —
that is official penalty duty, better than inferring it from Understat's penalty-tagged
shots. Use the API field as primary, Understat as validation, overrides as the residual.
Same for `corners_and_indirect_freekicks_order` and `direct_freekicks_order` — set-piece
duty is now an official field. This is a meaningful simplification to the doc's §5.

**4.4 DefCon.** Hierarchical NB as specified, with the covariate structure the doc
describes (opponent possession share from Phase 2, log-minutes offset, role-varying
overdispersion, joint modelling with clean sheets rather than two independent screens).
**Constraint recon adds:** only 2025/26 has native `defensive_contribution`, so the target
is one season deep. Backfill earlier seasons from FBref action counts and treat the
reconstruction as a prior, not as ground truth. Verify the thresholds — the API confirms
FWD now scores DefCon at 2 pts, which was not true in earlier seasons.

**4.5 Bonus.** As specified, moved to week 5. The reconstruct-and-check validation is
mandatory, not optional — it is the only way to confirm an unpublished BPS table.

**4.6 Assembly.** As specified — simulate the joint, do not sum marginals. See D2: if
rank-optimizing, this step also needs effective ownership.

**4.7 Optimizer.** Constraints per the doc (£100.0m, 15 players 2/5/5/3, max 3/club, valid
formation, transfer continuity, 5 banked FTs) — all confirmed against `game_settings`.
**Add three the doc omits:**
- **Sell-price mechanics.** `element_sell_at_purchase_price: false`, 50% sell-on fee.
  Budget feasibility depends on purchase price per holding, so the MILP needs a per-player
  purchase-price state variable carried across the horizon.
- **8-chip scheduling** across two half-season windows.
- **`transfers_cap: 20`** per event.
The doc's two headline arguments — budget shadow price from the LP dual, and correct
pricing of bench option value — are exactly right and are the reason to do this as a MILP
rather than a ranking table.

**4.8 Backtest.** As specified. Walk-forward only. Calibrate components *before* looking
at aggregate MAE.

---

## 5. Traps — the doc's five, plus three from recon

The doc's list is good and stands: entity resolution rot, position reclassification as a
per-season attribute, survivorship in training data, overfitting September, optimizing the
wrong objective.

Adding:

6. **Silently lost snapshot history.** No source gives you retroactive price/ownership.
   A snapshotter that dies quietly costs data you cannot buy back. Alert on gaps.
7. **Trusting the BPS and DefCon-threshold configs.** Both are hand-sourced and neither is
   in the API. Reconstruct-and-verify or treat every downstream bonus number as suspect.
8. **Opta-code overconfidence.** If the join works it is a large win — but verify against a
   sample before deleting the fuzzy path, and keep the CI assertion either way. New
   signings mid-season are still the failure mode.

---

## 6. Immediate next actions

1. **Today:** snapshotter live before 23:00 UTC price change. (~30 min)
2. **Today:** `uv` env on 3.12, DuckDB init, vaastav backfill. (~2 h)
3. **Tomorrow:** Opta-code join spike → decide `resolve/` architecture. (~3 h)
4. **Tomorrow:** generate scoring YAML from API + CI diff. (~1 h)
5. **Thursday:** naive xPts + MILP → GW1 squad before Friday 17:30 UTC.
6. **Then:** Track B, week 1 — data layer proper + backtest harness stub.

Answers to D1–D3 change items 5 and 6; items 1–4 are unconditional and I can start now.
