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
