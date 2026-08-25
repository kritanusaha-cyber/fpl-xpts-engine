# What to build next

Written 2026-08-19, after the four-season backtest. The plan is dictated by one
result:

| | outcome |
|---|---|
| projection accuracy vs template | **wins 4 of 4 seasons** |
| season points vs template | **2 of 4, mean +2.8, p = 0.96** |

**Better projections are no longer the bottleneck.** They already win everywhere
and the points do not move. Everything below is therefore structural — how the
squad is chosen and managed, not how players are rated.

---

## The decision that isn't a decision

The build plan says to choose between maximising expected points and maximising
rank *before* building the optimiser, because the objectives diverge. That was
true when there was no way to test it. There is now: the four-season harness
scores any objective in about two minutes.

So it becomes an experiment rather than a judgement call. Build both, run both,
keep whichever wins. That is Build 1.

---

## Build 1 — Rank optimisation  ✅ **LANDED 2026-08-21**

**Result: +289 points over four seasons, winning 4 of 4.** Not via the linear
tilt, which lost at every setting, but via the exact margin-variance objective
— which turns out to be linear in the squad choice. Optimal γ = −0.05: reward
variance in template players, penalise it in differentials. See FINDINGS.md.

<details><summary>original plan</summary>


**Why this first.** Against a field that overwhelmingly owns the template, points
are not the currency — *rank* is. Owning a template player who hauls gains you
nothing relative to the field; owning a differential who hauls gains a lot. The
max-EV squad is provably not rank-optimal, and the four-season null is exactly
what you would expect from optimising the wrong objective.

**What it needs**
1. **Effective ownership.** `selected` exists for all four seasons (up to 9.5M).
   EO = ownership × captaincy rate. Already in the warehouse; needs deriving.
2. **A rank objective.** Instead of maximising `E[points]`, maximise
   `P(squad beats the field)` — computed from the simulated point distributions
   already produced, weighted against the template's distribution.
3. **Differential pressure.** Where the model most disagrees with ownership is
   where rank is won. This falls out of (2) rather than being bolted on.

**Success criterion.** Beats the template on season points in **≥3 of 4 seasons**,
or pooled p < 0.05 across 124 gameweeks. Anything less and it goes in FINDINGS.md
as another negative result.

</details>

---

## Build 2 — Real transfer planning  *(~half day)*

The current simulation re-optimises greedily each week with a crude change limit.
The rules it ignores are worth real points:

* up to **5 banked free transfers** (`max_extra_free_transfers` = 4)
* **−4 per extra transfer**, which should be taken only when the projected gain
  over the horizon clears it
* **transfer continuity** across an H-gameweek horizon, so a move is judged on
  the whole run rather than the next match
* **sell-price mechanics** — 50% sell-on fee, so budget is path-dependent on
  purchase price (already documented, never implemented)

**Success criterion.** Beats the current greedy manager on the same four seasons.

---

## Build 3 — Chip strategy  *(~half day)*

Eight chips, two half-season windows, currently unmodelled. Bench Boost and
Triple Captain are straightforward given the simulator already produces
distributions. Wildcard is a free re-solve; Free Hit is a one-week re-solve.

**Success criterion.** Positive points contribution across four seasons. Chips
are a bounded, low-risk win — this is the safest item on the list.

---

## Build 4 — DefCon possession covariate  ❌ **TESTED AND REJECTED 2026-08-21**

The covariate was never unavailable — it is in the FotMob team block, now
ingested for six seasons at 100% coverage. The football effect is real and
large (threshold rate more than doubles, 11.9% → 26.6%). It adds **−0.03%** to the model.
The player's own rolling counts already measure it. See FINDINGS.md.

<details><summary>original plan</summary>


DefCon is the weakest component: calibration collapses above p = 0.5 and the
tail is capped at 0.40 as a result. The build plan specifies **opponent
possession share** as the key covariate, which is not in the FPL feed. FBref has
it and `soccerdata` already reaches FBref.

Do this only if Builds 1–3 land, since DefCon is ~10% of a typical projection.

</details>

---

## Explicitly NOT worth building

* **More projection accuracy.** Winning 4/4 already; the points do not follow.
  Effort here is measurably wasted.
* **xGOT.** Tested and rejected — finishing r = −0.065, placement r = −0.007
  against a control persisting at r = +0.742.
* ~~**SofaScore heatmaps.**~~ Superseded: FotMob serves season touch heatmaps.
  Collected for 383 players, 426,596 touches. Zone shares predict xG/90 at
  r = 0.90 and separate goal threat from creation cleanly.
* **SofaScore specifically.** Blocked by their bot detection after my request burst.
  The collector is written and waits; territory already persists at r = 0.894 via
  FotMob, so the marginal gain is small.

---

## Sequencing

| order | build | why then |
|---|---|---|
| 1 | Rank optimisation | the only item the evidence points at |
| 2 | Transfer planning | compounds with 1; rank needs good transfers to hold a differential |
| 3 | Chips | independent, bounded upside |
| 4 | DefCon possession | smallest effect, and gated on the others working |

Every build is scored on the same four seasons with the same harness. If Build 1
fails its criterion, that is a finding worth as much as a success: it would say
the template is close to unbeatable under FPL's constraints, which is a real
answer to the question the project set out to ask.
