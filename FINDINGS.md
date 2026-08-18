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
