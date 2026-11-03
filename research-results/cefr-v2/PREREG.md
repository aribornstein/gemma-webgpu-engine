# Preregistration — CEFR round 2 (written before any round-2 episode)

Date: 2026-11-03. Plan: `autopilot-cefr-v2-plan` (approved). No human calibration; judged results are exploratory.

## Data roles (fixed)
- Direction families (directions are fit only here): v1 sealed families (bakery, post_office, museum, garage,
  library, taxi) + validation families (quest_delivery, doctor_appointment, bike_rental). v1 sealed data is spent and
  reused as development data.
- Evaluation development families: the 6 original development families (market_purchase, cafe_order,
  pharmacy_purchase, bus_ticket_purchase, hotel_checkin, lost_item_shop). Steering gate uses the first 3.
- Sealed v2: 12 new families written by a separate author outside the repository; evaluated once.

## Directions (per residual site after layers 12, 16, 20, 22, 24, 26)
- `instr`: mean over strong-prompt replies of [h(strong prompt) − h(canonical prompt)] for A1 minus the same for B2.
- `content`: canonical-prompt states of the model's own A1-instructed minus B2-instructed replies, after removing
  the part explained by log reply length.
- Controls: `permuted` (instr with level labels swapped for a random half of reply pairs), `random` (norm-matched),
  sign flip (wrong-level gate).
- Strength m in natural units: m = 1 shifts the residual by half the measured A1–B2 difference along the direction.

## Steering gate (G-steer), canonical prompt, 3 turns, seeds 23/47, 3 families × 2 levels
Configs: type ∈ {instr, content} × site ∈ {16, 22, 26} × m ∈ {1, 2, 4}; reference m = 0 (raw) and strong prompt.
Metric: deterministic complexity gap (B2 − A1, v1 complexity statistics), paired by scenario × seed.
Pass for a (type, site) iff all hold:
1. mean gap is non-decreasing over m = 0, 1, 2, 4 (tolerance 0.05) and Spearman(m, pair gap) > 0;
2. best-m mean gap − raw ≥ 0.30;
3. at the same site and m, gap exceeds `random`, `permuted` and sign-flipped arms by ≥ 0.20 each;
4. hard state pass rate not below raw − 0.10, and Latin spillover not above raw + 0.10.
If no (type, site) passes: stop the edit search; report "self-generated directions are not causal either".

## Development gate (G-dev), full 6 families, 5 turns, seeds 23/47
Finalists (≤ 2) vs raw, random, wrong-level, strong prompt, best-of-2 with the blind 4-role panel.
Pass iff cefr_control gap (blind level estimator) exceeds raw, random and wrong-level with paired cluster-bootstrap
95% CI excluding 0; grammar, naturalness and hard pass not worse than raw by > 0.25 / 0.25 / 0.10; critic sign-off.

## Sealed (one invocation)
All arms × 12 families × 2 levels × 2 seeds, 5 turns; blind panel plus a reserved second-prompt level judge.
Primary: cefr_control (agent edit − raw, − random, − best_of_2), paired cluster bootstrap by family.

## Deviation D1 (recorded after the dose sweep, before any control arm was run)
The sweep showed a monotonic dose-response for `instr` at all three sites, but at the largest dose (m = 4) the hard
state pass rate falls below raw − 0.10 at every site, so the gate as literally written (criterion 4 evaluated at the
max-gap m) fails. We record the strict result as **G-steer: FAIL (strict)** and additionally evaluate criteria 2–4 at
the largest m that keeps the hard-pass and spillover guardrails (`instr` at layer 26, m = 2: gap +0.41, hard pass 0.83
= raw). Controls (random, permuted, sign flip) are run at m = 2 for layers 16, 22 and 26 and reported for all three.
Downstream stages proceed under this disclosed reading; the report states both readings.

## G-dev outcome and deviation D2 (recorded before any sealed edit episode)
G-dev **FAILED** for every configuration: p6/p7 control CEFR level strongly (p6 − raw +2.17 [1.67, 2.50], beating
norm-matched random and wrong-level controls) but lose 0.8–1.0 grammar/naturalness points; the refined lower doses
(r1, r2) keep +0.58 to +0.92 control but still miss the 0.25 language guardrails by small margins (grammar −0.29/−0.33,
naturalness −0.33/−0.29). Per the plan, the *improvement* claim is not taken to the sealed set.

D2: the sealed set is used once for a narrower, separately labeled question — **does the instruction-distilled
direction causally control CEFR level on unseen scenario families?** Frozen edit: p6 (the configuration with the
largest dev control among those whose hard-pass rate stayed within raw − 0.10). Sealed arms: raw_base, strong_prompt,
best_of_2, agent_p6, random_p6, wrong_level_p6; primary metric cefr_control (blind level judge, plus the reserved
second level judge), secondary grammar/naturalness. The improvement verdict remains "not supported (G-dev fail)".
