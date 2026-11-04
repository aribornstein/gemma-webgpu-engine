# Mechanism-first investigation: learner-error copying and conditional noun recovery

**Outcome: inconclusive; no promotable repair.** The bounded investigation delivered a
working browser/agent diagnostic workbench and specific evidence about two Hebrew
conversation failures. It did **not** locate a transferable native weight defect,
achieve all-criteria 9/10 quality, or modify pretrained weights.

The approved gates were not relaxed. Gate C (a localized, specific, preservation-safe
repair mechanism) was not established; weight-repair and final-evaluation gates were
therefore not entered. See [gate-status.json](./gate-status.json) and the independent
[methods review](./METHODS_REVIEW.md).

## What changed in the methodology

Instead of sweeping a predetermined CEFR steering vector, an investigator examined
actual paired free responses and selected concrete failure hypotheses. One final
learner agreement form changed; authoritative facts, prior messages, and the intended
request remained fixed.

The target was the pinned Gemma 4 E2B QAT checkpoint on the owned WebGPU engine:
revision `9fcec64df66cb1e4d972fc5cdc142afb25b2362c`, SHA-256
`efab429012b97ab986c4d4838a46ff3ad95d618b42ce514771ca40fadc76a9a4`.
All matched arms used fresh sequential prefill and full-logit CPU selection.
Free continuations were greedy; forced scores used the model's final softcap.
Neither sampling-path differences nor penalties varied across experimental arms.

Twelve AI-authored pairs covered six lexical/scenario families with A1/B2 targets
and different learner profiles. These are discovery cases, not a held-out benchmark
or a fully autonomous adaptive-player evaluation. The system prompt explicitly
distinguished quantity, unit price, total price, currency and payment state.
Vocabulary was intentionally unrestricted for this grammar diagnostic, hence
vocabulary adherence was **unassessed**, not successful.

## 1. Context changes actual local decisions

Two paired cases reproduced different behavior after a learner's agreement mistake:

| Case | Clean learner context: free reply | Error learner context: free reply |
|---|---|---|
| Two shirts, total 110 ILS | `שתי חולצות יהיו בסך 110 ש"ח.` | `שני חולצות יהיו בסך 110 ש"ח.` |
| Two books, total 48 ILS | `שני ספרים. המחיר הכולל יהיה 48 שקלים.` | `שתי גילידים. המחיר הכולל יהיה 48 שקלים.` |

The book error changes the item wording as well as retaining the learner's numeral.
We do not assign grammatical gender to the corrupt noun or pretend it is a simple
numeral-only error.

Both contrasts share the generated first token `ש`. At the *next*, differing
token, log odds for the grammatically correct alternative were:

| Case | Clean context | Error context |
|---|---:|---:|
| Shirts | +7.12 | -0.19 |
| Books | +3.78 | -1.90 |

These are full-vocabulary conditional model log probabilities, not probabilities
renormalized over two labels. This confirms context sensitivity at the actual
decision. It does **not** show that a model internally represents a grammatical
answer and a single component then suppresses it.

Other pairs included both-correct, both-wrong and error-correcting replies; not
every learner mistake was copied. Full evidence is in
[diagnosis.json](./discovery/diagnosis.json) and the investigator's
[hypothesis cards](./discovery/hypotheses.json).

## 2. Donor-state transfer can rescue an output, but does not identify a repair

We captured post-layer residuals in clean and erroneous contexts immediately before
the differing numeral token. At layers 0, 4, 12, 20, 28 and 34 we tested:

- clean-to-error replacement;
- self-replacement (sham);
- additive random perturbation matched to the clean/error state-difference norm;
- reverse error-to-clean replacement.

Each condition measured both forced alternatives and a free continuation from the
common prefix. The intervention was active at one token only; no donor answer was
added to the prompt, but the clean donor context already contained the correct
numeral and can supply answer information through its activation.

| Case | Finding | Limitation |
|---|---|---|
| Shirts | Several donor replacements produced `שתי חולצות` while retaining total 110 | Random perturbations also often changed the weak-margin decision; specificity is not established |
| Books | Layer-34 donor restored `שני ספרים` and total 48; reverse transfer restored the wrong local preference | At the last layer this is close to transferring an answer state, not tracing its origin |
| Books, earlier layers | Some donors restored the word for books but retained the wrong numeral | Lexical identity and agreement need not share one failure mechanism |

All **12 corrected sham continuations** were token-identical to their recipient
baselines. The final-layer book margin moved from -1.90 to +3.78 with the clean
donor; reverse transfer moved it back to -1.90. No native head, neuron, or weight
column was implicated sufficiently to justify modifying it.

See [patch-effects.json](./causal-v2/patch-effects.json). Multi-layer residual
transformations are not described here as one equivalent low-rank pretrained
weight edit.

### Instrumentation failure retained, not counted as a scientific result

The first 144 patch jobs accidentally ran without a patch: the dispatch was nested
under the persistent-overlay guard. Identity checks alone had failed to detect
the inactive intervention.

We stopped interpretation, reproduced the bug with a nonzero activity sentinel,
fixed the guard, reran the live gate, and repeated matched baselines/captures and
diagnostics under a new source hash and run ID. The first run's patch records are
explicitly [invalidated](./discovery/INVALIDATION.json), retained, and excluded
from every scientific conclusion. No acceptance threshold was changed.

## 3. Availability tests do not demonstrate autonomous grammatical repair

The investigator requested a bounded, task-switched follow-up, rather than more
weight inspection without a mechanism:

| Diagnostic | Result |
|---|---|
| Recognition of correct agreement, three nouns, both A/B orders, Hebrew instructions | B preferred in all six orders: 3/6 nominal accuracy, **0/3 order-consistent pairs** |
| Same choices under an English translation of the instructions | Again B preferred in all six orders |
| No-option Hebrew editing: three erroneous and three correct inputs | All copied unchanged: **0/3 errors repaired**, 3/3 clean inputs preserved |
| English editing control on the two target errors | Both copied unchanged: **0/2 errors repaired** |

This fails to establish recognition or editing ability under these probes. It
does not prove an absence of Hebrew knowledge under all possible elicitation.
The A/B probabilities were concentrated on the requested labels; this was not
simply an output-format failure.

### The book noun remains conditionally available

We crossed the original learner context with a forced assistant numeral, holding
the rest of the history fixed:

| Learner context | Supplied assistant prefix | Free continuation, including prefix |
|---|---|---|
| Clean | `שני` | `שני ספרים. המחיר הכולל יהיה 48 שקלים.` |
| Clean | `שתי` | `שתי ספרים. המחיר הכולל יהיה 48 שקלים.` |
| Error | `שני` | `שני ספרים. המחיר הכולל יהיה 48 שקלים.` |
| Error | `שתי` | `שתי גילידים. המחיר הכולל יהיה 48 שקלים.` |

The noun and correct total can be generated even after the learner error **if
the correct numeral is supplied**. Switching the prefix changes the books-versus-
corrupt-noun score margin by approximately 12.24/14.88 nats in the two contexts.
Both learner context and the model's preceding output matter.

This is the clearest useful finding: a local wrong choice can lead to subsequent
lexical corruption, while a correct continuation remains conditionally possible.
Supplying that choice is not an autonomous repair, and no regression-free way to
recover it was discovered.

See [behavioral-summary.json](./causal-v2/behavioral-summary.json) and
[h4-summary.json](./causal-v2/h4-summary.json).

## 4. Quality gate remains unmet

Three blind specialist agents judged the 24 original discovery replies. They saw
the transcript and authoritative requirements, not intervention identities or
selection scores. Evidence spans, identity, score range and criterion coverage
were validated programmatically. These are **AI judgments of single turns**, not
human-validated proficiency or full-episode certification.

| Criterion | Mean / 10 | Replies scoring at least 9 |
|---|---:|---:|
| Grammar | 8.54 | 15/24 |
| Naturalness | 8.13 | 14/24 |
| Grounding | 9.71 | 23/24 |
| Goal | 9.17 | 20/24 |
| Level suitability | 8.17 | 14/24 |
| Coherence | 9.00 | 19/24 |
| Vocabulary-policy adherence | Unassessed | Not eligible to pass |

There are **zero joint passes**, 11 explicit failures and 13 unknown outcomes.
An unassessed vocabulary policy, missing semantic evidence or an abstention is
not silently counted as a pass. Native repair, adaptive-player robustness and
fresh final evaluation were not attempted because the required earlier gates
were not met.

## 5. Delivered workbench and evidence

Open `research.html` on the dev server or in the production build. It exposes the
same typed operations to humans and agents: replay, score, capture, bounded
weight inspection, token-local patch, compare, and assess. It includes a ledger,
per-token decision table, capture inspector, paired-effect map, cancellation,
queue pause/resume, and saved-evidence import/export.

The compact [review-example.json](./review-example.json) imports without a GPU
and contains original, hash-bearing **valid** artifacts only. Raw captures and
failed attempts remain locally available; [budget-and-evidence.json](./budget-and-evidence.json)
records their sizes, hashes and source fingerprints. The CLI writes each completed
job atomically and logs explicit errors; repeated job IDs cannot silently change
their request.

Across all attempts, including the invalidated batch: **132,583 evaluated token
positions, 3,063 generated tokens, 438 jobs and 138 free-continuation requests**.
Ninety of those free continuations are valid scientific observations. The invalid
batch remains charged to the budget. This is not a claim of 138 independent
multi-turn episodes. Three judge calls produced 72 criterion-role assessments.

Validation before delivery:
- Full suite: 246 passed, 10 skipped (env-gated live tests).
- Separate new live/contract/UI gate: 6 passed, including nonzero-patch activity,
  sham/zero/revert parity, weights, scoring, captures, cancel/recovery.
- Existing CPU suite: 95 passed.
- Typecheck and production demo/library build passed; final validation is recorded
  separately in [validation.json](./validation.json).
- Independent code review identified crash-accounting, paused-queue cancellation,
  and close-during-load bugs. All three were fixed with regression tests. After
  these fixes, the targeted live/contract/console selection passed **32 tests**,
  and the real CLI persisted running reservations before two golden smoke jobs.
  The earlier full-suite result is not described as a new full-suite run.

## 6. What remains unresolved

The experiments show a context-dependent decision error and conditional
availability, not a clean weight-space opportunity. The terminal donor rescue
does not tell us which earlier computation caused the error. The shirt result is
too sensitive to nonspecific perturbations to support a targeted repair.

Further work would need a new bounded investigation of context routing or
within-layer contributions, tested on new lexical families without answer-bearing
donors. It must not be justified by calling these current activation transfers a
weight fix. No production settings, checkpoint weights or old benchmark claims
were changed.
