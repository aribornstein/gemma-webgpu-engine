# REPORT — Agentic CEFR Hebrew weight-space campaign (pilot v1)

**Status:** executed end to end on 2026-11-03 (one exploratory pilot; not a powered confirmation study).
**Verdicts:** edit claim **contradicted_exploratory** · "latent but unexpressed" diagnosis **inconclusive** ·
metadata-only control gap **supported_exploratory** · `causal_use` of probe directions: tested, **not supported for the
tested operator** · `recoverable_steering`: **not demonstrated**.

All Hebrew fixtures are AI-drafted and unreviewed (`human_validated: false`). All judges are LLM sub-agents of one model
family; their calibration used AI-authored contrasts only. Every judged number below is therefore **exploratory**.

## 1. Question and setup

Can agents use internal representation evidence plus task feedback to discover a small, reversible,
metadata-conditioned effective-weight edit that improves held-out multi-turn Hebrew NPC dialogue (CEFR A1 vs B2, from
English metadata) beyond strong prompting and best-of-2, without grammar, naturalness or world-state regressions?

| Item | Value |
|---|---|
| Model | `google/gemma-4-E2B-it-qat-mobile-transformers` rev `9fcec64…`, sha256 `efab4290…a9a4`, frozen |
| Runtime | owned WebGPU engine (this repo), Chrome headless on Apple M-series, one session, sequential prefill |
| Engine commits | `30a4b2f` (research hooks), `2673138` (harness) on `research/agentic-cefr-weightspace` |
| Episodes | 5 scripted learner turns, NPC replies sampled (temperature 1, top-k 64, top-p 0.95, ≤96 tokens/turn), seed 23 (best-of-2 adds 47) |
| Splits | development 6 families · validation 3 families (probe test only) · sealed 6 families (separate author, outside repo, sha256 `8bd6f549…2d40`, one invocation) |
| Arms | raw_base (canonical prompt) · strong_prompt (level-specific instructions) · best_of_2 (strong prompt, dev judge picks seed 23/47) · no_op_overlay · random_rank1 · wrong_level_gate · agent_rank1 (overlay arms use the canonical prompt) |
| Judges | 4 blind roles: language (grammar, naturalness), cefr (suitability, pedagogy; sees target), level estimator (blind to target; primary), conversation (register, coherence). Order-shuffled, arm-blind, hashed ids |
| Budget used | 16 judge calls (4 calibration, 8 development, 4 sealed) of 300+120; 246 episodes; 4 analytical candidates + 1 strength refinement round |

Primary outcome (R5): **cefr_control** = blind estimated level of B2 episodes minus A1 episodes, paired by scenario.
Statistics: paired percentile cluster bootstrap by scenario family (2000×); n = 6 families per split.

## 2. Engineering gates (all passed)

- Research hooks off: full suite 241 passed / 9 skipped (env-gated live specs); greedy goldens unchanged.
- `npm run test:research-live`: identity overlay (coefficient 0) and revert are token-identical at both sites; active
  overlays change output; teacher-forced scores equal generation log-probabilities (max |Δ| = 0); captures are
  prefix-consistent and repeatable; residual capture after layer 34 equals the final residual; `W_head b` projection is
  exactly linear (output SRQ disabled for the projection; head output scale is 0).
- Residual edit kernel matches a CPU reference (|Δh| < 1e-4, 0 SRQ mismatches) and is a no-op at coefficient 0.
- End to end: no_op_overlay episodes identical to raw_base in 12/12 development and 11/12 sealed episodes. The one
  sealed difference is expected: without an overlay plain top-k sampling uses the GPU candidate path, whose softmax tail
  term is an f32 GPU sum, so a draw that lands within rounding of a cumulative boundary can differ.

## 3. Judge calibration (AI-authored contrasts → exploratory)

| Check | Result |
|---|---|
| Level estimator orders A1 < B2 reference replies (9 families) | 9/9 |
| Language judge rates correct > incorrect agreement minimal pairs (36) | 36/36 |
| CEFR judge rates matched > swapped target suitability (18) | 18/18 (mean 3.94 vs 1.00) |
| Conversation judge | not calibrated (no ground truth) |

## 4. Observed behavior (E), development and sealed

| Arm | cefr_control dev | cefr_control sealed | grammar dev / sealed | naturalness dev / sealed | hard state pass dev / sealed |
|---|---|---|---|---|---|
| raw_base (metadata only) | 0.50 | 0.33 | 2.42 / 1.92 | 2.08 / 1.58 | 0.67 / 0.75 |
| strong_prompt | 2.33 | 2.83 | 1.67 / 1.58 | 1.33 / 1.33 | 0.58 / 0.58 |
| best_of_2 | 2.50 | 2.83 | 2.00 / 2.42 | 1.75 / 1.92 | 0.67 / 0.58 |
| agent_rank1 | 0.17 | 0.00 | 2.67 / 2.17 | 2.00 / 1.92 | 0.58 / 0.67 |
| random_rank1 | 1.00 | 0.17 | 2.50 / 2.17 | 2.00 / 1.83 | 0.42 / 0.75 |
| wrong_level_gate | 0.33 | 0.33 | 2.50 / 1.83 | 2.08 / 1.67 | 0.50 / 0.67 |

Sealed paired differences (cluster bootstrap 95% CI): strong_prompt − raw_base +2.50 [1.83, 3.00]; best_of_2 − raw_base
+2.50 [1.83, 3.00]; agent_rank1 − raw_base −0.33 [−0.67, 0.00]; agent_rank1 − random_rank1 −0.17 [−0.50, 0.00];
agent_rank1 − best_of_2 −2.83 [−3.00, −2.50]. Deterministic complexity gap agrees (agent − raw −0.33 [−0.59, 0.01];
strong − raw +2.64 [2.10, 3.27]). Joint pass rates are near floor for every arm (0–0.17), as anticipated in R5.

**Diagnosis (supported_exploratory):** with English metadata alone, the model barely adapts its level (0.33–0.5 CEFR
steps, positive in 2–3 of 6 families). Explicit level instructions steer strongly but lower grammar and naturalness;
best-of-2 recovers part of the quality loss. Hard-state failures are almost all unsupported numbers (wrong prices, totals or quantities: 39/39 failing development episodes, 30/33 sealed), with occasional Latin-script spillover; no payment claims were detected by the phrase check.

## 5. Internal evidence (L and I)

| Measurement | Result (validation families held out) |
|---|---|
| Requested level, pre-decision state (last prompt token) | mean-difference AUC 1.00 at layers 12/20/22, 0.93 at 24, 0.78 at final; shuffled-label 0.58–0.70; n = 60 train / 30 test (3 test families) |
| Requested level, end of metadata | 0.56–0.89 (only 6 unique test states; underpowered) |
| Crossed suitability: reply style | AUC 0.56–0.89 by site |
| Crossed suitability: level × style match (joint) | ≈ 0.5 at every site (not linearly decodable) |
| Teacher-forced likelihood: requested level shifts preference toward the matching reference style | 8/9 families, mean +0.116 nats/token; matched style preferred outright only 50% |
| Agreement minimal pairs (likelihood) | correct preferred 33/36 (numeral 10/12, adjective 12/12, verb 11/12) |

Interpretation: the explicit level instruction is represented at the decision point and weakly conditions
likelihoods; this is decodability of the metadata, not evidence of a CEFR competence the model fails to express
(required distinction 1). Likelihood prefers grammatical agreement far more often than generation produces it.

## 6. Agentic search (all attempts)

| Round | Candidates (site, gate, alpha) | Screening result (complexity gap B2−A1) |
|---|---|---|
| 1 (investigator, 3 families, 1 seed) | c1 residual22 self 0.12 · c2 residual20 external 0.10 · c3 residual22 external 0.06 · c4 final self 0.08 | −0.13, +0.17, +0.18, +0.06 vs raw_base +0.03 (within noise) |
| 2 (refine strength, 2 seeds) | c3 0.13 · c3 0.20 · c2 0.15 · c2 0.22 | +0.02, −0.05, +0.02, −0.22 vs raw_base +0.08; larger alpha shortened B2 turns and degraded morphology |
| Freeze | agent_rank1 = c3 (residual22, external gate, alpha 0.06) | see §4 |

Edit directions: `b` = mean residual state over teacher-forced A1-style minus B2-style reference replies (development
families only; per-family cosine with the pooled direction 0.45–0.70), `a` = mean state direction (so the shift is
≈ alpha × state norm). Self gates read the model's own end-of-metadata state and separated A1/B2 correctly (gate
≈ +0.74 for A1, −1 for B2) but did not help. Methods critic (agents/critic-review.md) recommended
contradicted_exploratory before sealed data were seen.

## 7. Conclusions

1. **Edit claim — contradicted_exploratory.** No probe-guided rank-1 edit improved CEFR control: the frozen edit was
   at or below raw_base, random and wrong-level controls on development and sealed data, and far below strong
   prompting and best-of-2 (−2.83 [−3.00, −2.50] on sealed). No strength produced a dose-response.
2. **Metadata-only control gap — supported_exploratory** on both splits: the demo is not steerable from metadata alone.
3. **Latent-but-unexpressed — inconclusive.** The requested level is decodable before the NPC turn and nudges
   likelihoods, but the directions tested were not causal for generated complexity.
4. Strongest baseline: best_of_2 (strong control and the best sealed grammar), consistent with selection beating a
   single fixed edit at this scale.

Confounders and limitations: AI-authored fixtures and judges of one model family; 6 families per split, one sampled
episode per cell; directions built from teacher-forced reference replies (may encode wording/length rather than level);
external gate uses the parsed metadata; canonical prompt for overlay arms means overlays compete with raw metadata
conditioning, not with instructions; hard-state checks use a numeric oracle and a payment phrase list.

## 8. Next experiments

1. Directions from generated trajectories (A1-prompted vs B2-prompted episodes of the model itself), cross-fitted by
   family, length/topic matched, with label-permutation and sign controls.
2. Factorial direction × gate × alpha × site with several seeds per cell and a preregistered monotonic dose-response;
   include multi-layer low-rank (rank ≤ 4) edits as allowed by the promotion thresholds.
3. Native-speaker calibration of grammar, naturalness and level judgments (the optional ~60-item sheet), then a fresh
   sealed set.

## 9. Artifacts

`campaign.json`/`task.json`/schemas/fixtures: `research/cefr-agentic/` · episodes: `baselines.dev.jsonl`,
`screen-round1.jsonl`, `screen-round2.jsonl`, `frozen.dev.jsonl`, `sealed.jsonl` · blind judgments:
`judging/*.collected.jsonl` (+ prompts, items, keys) · ledger: `judge-ledger.jsonl` · calibration: `calibration/` ·
probes: `probes.json` (raw activations regenerable, not committed) · panels: `panel-*.json` · paired statistics:
`comparisons.dev.json`, `comparisons.sealed.json` · agent briefs/proposals/critique: `agents/` · patch manifest:
`patches/agent_rank1.json` (hash `85c71447…`) · trace: `experiments.jsonl`.
