# REPORT — CEFR campaign round 2: instruction-distilled directions

**Status:** executed 2026-11-03 under the approved round-2 plan and `PREREG.md` (with two disclosed deviations).
All judges are blind LLM sub-agents of one model family; no human calibration (by decision), so every judged result is
**exploratory**. Fixtures are AI-drafted Hebrew (`human_validated: false`).

## Verdicts

| Claim | Verdict |
|---|---|
| (a) An instruction-distilled direction, installed as a gradient-free effective-weight edit, **causally controls** the judged CEFR level of metadata-only dialogue on unseen scenario families | **supported_exploratory** (held on the sealed set) |
| (b) The edit **improves** on strong prompting and best-of-2 **without language regressions** (the preregistered improvement claim) | **contradicted_exploratory** (G-dev failed; sealed language costs confirm) |
| (c) Round 1's "latent but unexpressed" diagnosis | **inconclusive** |
| Strict preregistered steering gate | **FAIL** (m = 4 broke the hard-state guardrail at every site); passed only under disclosed reading D1 |

## 1. What changed from round 1

Round 1 built edit directions from teacher-forced reference replies and found nothing causal. Round 2 built them from
the model's own behavior on 9 *direction* families (v1 sealed + validation), never on evaluation families:

- **instr** — residual difference h(strong level prompt) − h(canonical prompt) over the model's own replies, A1 minus
  B2. Very consistent across families (cosine to pooled 0.87–0.97 at every site; half-difference ≈ 13–19% of the
  residual norm).
- **content** — the model's own A1-instructed minus B2-instructed replies re-read under the canonical prompt, with log
  length partialled out.
- Edit: per site `h ← h + c·b·(a·h)` after layer L (effective `W + c·b aᵀ`; packed weights untouched; multi-site =
  rank ≤ 3), `c = ±m·magnitude/(a·h)/sites`, gate +1 for A1 and −1 for B2 from the metadata.

Engine additions: multiple simultaneous overlays (low-rank edits), verified bit-identical at coefficient 0 and on
revert (`npm run test:research-live`).

## 2. Steering gate (3 dev families, canonical prompt, 3 turns, 2 seeds; deterministic complexity gap)

| Direction @ site | m = 0 (raw) | m = 1 | m = 2 | m = 4 |
|---|---|---|---|---|
| instr @ L16 | −0.02 | +0.41 | +0.82 | +4.90 |
| instr @ L22 | −0.02 | +0.08 | +0.41 | +1.92 |
| instr @ L26 | −0.02 | +0.26 | +0.41 | +0.89 |
| content @ L16 / L22 / L26 | −0.02 | +0.08 / +0.15 / +0.11 | −0.03 / +0.16 / +0.12 | +0.03 / +0.19 / +0.02 |

Strong level prompt on the same cells: +0.98. **instr** shows a clean monotonic dose-response at every site;
**content** does not. Controls at m = 2 (instr minus control): L26 random +0.28, permuted +0.65, sign-flip +0.64;
L16 sign-flip +1.46 (flip reverses the effect: −0.65). At m = 4 hard-state pass dropped below raw − 0.10 everywhere, so
the strict gate failed; D1 evaluated the gate at the largest m keeping the guardrails (instr L26 m = 2: passes).

## 3. Agentic search and development panel (6 dev families, 5 turns, 2 seeds, blind 4-role panel)

The investigator proposed 8 configurations (all instr; rank 1–4 across layers 12–26), screened on the 3 *other* dev
families; finalists p6 (layers 20/24/26, m = 4.5) and p7 (22/24/26, m = 6). Dev panel (paired cluster bootstrap, 95% CI):

| vs raw_base | Δ CEFR control | Δ grammar | Δ naturalness | Δ hard pass |
|---|---|---|---|---|
| p6 | **+2.17** [1.67, 2.50] | −0.79 [−1.21, −0.25] | −1.04 [−1.46, −0.63] | −0.04 |
| p7 | +2.83 [2.25, 3.33] | −0.92 | −1.04 | −0.17 |
| strong prompt | +2.00 [1.33, 2.58] | −0.58 | −0.42 | −0.04 |
| best-of-2 | +1.92 [1.50, 2.33] | −0.21 | 0.00 | −0.04 |
| refine r1 (20/24/26, m = 2) | +0.92 [0.50, 1.33] | −0.29 | −0.33 | −0.13 |
| refine r2 (26, m = 2.5) | +0.58 [0.33, 0.83] | −0.33 | −0.29 | −0.08 |

p6 also beat its norm-matched random edit (+2.83 [2.42, 3.17]) and its wrong-level edit (+4.50). **G-dev failed** on
the language guardrails (≤ 0.25 loss) for every configuration, including the refined low doses (which miss by
0.04–0.08), so the improvement claim was not taken to sealed data.

## 4. Sealed set (12 new families by a separate author; one invocation; D2: causal replication)

| Arm | CEFR control (B2 − A1 levels) | grammar | naturalness | hard pass | A1 / B2 words per turn |
|---|---|---|---|---|---|
| raw_base | 0.08 | 2.23 | 1.83 | 0.58 | 6.7 / 6.8 |
| strong_prompt | 2.04 | 1.85 | 1.33 | 0.52 | 5.3 / 19.7 |
| best_of_2 | 2.13 | 2.33 | 1.77 | 0.54 | 5.0 / 19.6 |
| **agent_p6 (frozen edit)** | **2.42** | 1.75 | 1.08 | 0.54 | 4.1 / 17.1 |
| random_p6 | −0.58 | 1.38 | 1.04 | 0.38 | 10.2 / 5.1 |
| wrong_level_p6 | −2.00 | 1.88 | 1.31 | 0.50 | 14.3 / 4.6 |

Paired by family (95% CI): edit − raw **+2.33 [2.04, 2.63]**; − random +3.00 [2.67, 3.33]; − wrong-level +4.42
[4.08, 4.75]; − strong prompt +0.38 [0.04, 0.71]; − best-of-2 +0.29 [−0.04, 0.63]. Language: edit − raw grammar −0.48
[−0.73, −0.27], naturalness −0.75 [−1.06, −0.48]; edit − strong prompt −0.10 / −0.25 (CIs include 0); edit − best-of-2
−0.58 / −0.69. The reserved second-prompt level judge agrees closely (exact 84%, within one level 100%, r = 0.94) and
gives the same ordering (edit 2.62, best-of-2 2.38, strong 2.29, raw 0.00).

## 5. Length confound (critic's main concern)

Judged level is largely predicted by turn length on unedited arms (R² 0.85 dev, 0.77 sealed). On sealed data, the
level gap beyond what length predicts is +0.50 for the edit, +0.33 for the strong prompt, +0.32 for best-of-2 and +0.09
for raw. So most of the "CEFR control" — for prompts and edits alike — is length/elaboration control, with a modest
residual beyond length that is at least as large for the edit as for the prompt it distils.

## 6. Interpretation

- A rank-3, gradient-free, reversible edit along the **instruction-distilled** direction reproduces what the explicit
  level instruction does — on unseen families, without any instruction text in the prompt — and matches or slightly
  exceeds the strong prompt's level separation. The effect is direction-specific (norm-matched random edits do not
  produce it; the sign flip reverses it) and dose-dependent.
- It inherits — and slightly amplifies — the strong prompt's language cost; best-of-2 remains the better quality/control
  trade-off. Lower doses reduce the cost but not below the preregistered guardrail.
- The **content** direction (what A1 vs B2 replies look like) does not steer; the **instruction** direction (how the
  model's state differs when told the level) does. This distinguishes "representing the output style" from "the
  control signal that produces it".
- This does not show that metadata alone activates a latent controller the model fails to use (round-1 diagnosis stays
  inconclusive): the steering vector comes from the instruction, not from the metadata.

## 7. Deviations and limitations

D1 (steering-gate reading) and D2 (sealed used for causal replication after G-dev failed) were recorded before the
affected data (PREREG.md); sealed baseline arms were generated during dev judging but not inspected before freeze.
LLM judges of one family; length explains most judged level; 6 dev / 12 sealed families; 2 seeds; directions from 9
families; hard-state checks are heuristics.

## 8. Next experiments

1. Length-matched (fixed token budget) dose experiment to isolate level from elaboration.
2. Combine the distilled edit with best-of-2 selection (edit for control, selection for quality) and with a
   grammar-preserving component (project out directions that the language judge penalises).
3. Apply the same instruction-distillation recipe to a coding-agent constraint (allowed APIs) with test-based ground
   truth.

## 9. Artifacts

`PREREG.md`, `experiments.jsonl` (trace), `directions.summary.json`, `steering-gate.json`, `agents/` (briefs,
proposals, critic review), episodes (`steer`, `controls`, `search`, `dev`, `refine`, `sealed` `.jsonl`), blind judgments
and keys (`judging/`), `judge-ledger.jsonl` (25 judge calls: 16 development, 9 sealed), `comparisons.{dev,refine,sealed}.json`,
`length-check.{dev,sealed}.json`, `reserved-judge.json`, patch manifest `patches/agent_p6.json`; bulk activations and
job files are regenerable and not committed. 1,008 simulated episodes in total.
