# Paired comparison and component investigation

## Bottom line

The approved engineering work is delivered: recorded and live before/after
comparisons, paired uncertainty estimates, judge-order checks, and token-local
attention/MLP/final-boundary diagnostics.

The scientific result is **limited causal sensitivity, not a transferable model
repair**. No intervention meets the all-criteria 9/10 requirement. The conditional
native-weight repair and fresh final evaluation branches were therefore stopped,
not silently replaced by weaker acceptance criteria.

## Try the demo

Run `npm run dev`, then open `http://localhost:5174/research.html`.
The compiled equivalent is `build/research.html` on a static server.

- **Comparison studio:** 24 historical CEFR pairs across six families, two
  levels and two seeds; two earlier diagnostic examples. Choose arms and step
  through the actual conversation. Original 0-4 scores and cited evidence are
  preserved; unavailable token probabilities are not invented.
- **Statistics:** fixed/broken/unchanged/unknown counts, family-bootstrap
  intervals, exact family sign-flip tests and Holm-adjusted endpoints.
- **Judge checks:** counterbalanced candidate order and hidden repeated packets.
- **Causal map:** 18 clickable token/layer/component cells with baseline, donor
  and reverse-sign text, vector norms, random controls and transfer outputs.
- **Live replay:** load a local pinned checkpoint or an existing browser cache,
  then compare baseline and historical p6 on the same sequential/full-logit path.
  Same-prefix and own-history conversation protocols are explicitly different.

Recorded playback needs no GPU or weights. Live outputs are unassessed until
separately judged. Hiding arm identities is only a display convenience, not
rigorous blinding. The old p6 overlay remains labeled rejected, not recommended.

See [usage and reproduction](../../research/mechanistic-repair/COMPARISON.md).

## What we learned

### 1. Historical steering damages quality rather than repairing it

Retrospectively comparing p6 with metadata-only baseline on 24 matched episodes:

| Narrow outcome | Count |
|---|---:|
| Fixed | 0 |
| Broken | 6 |
| Still correct | 0 |
| Still wrong | 18 |
| Unknown | 0 |

This table uses **grammar AND naturalness >=3 on the original 0-4 rubric**,
not the complete 9/10 promotion gate. It does not imply 18 general model failures
or 24 statistically independent families.

| Endpoint | Equal-family mean delta | Family-bootstrap 95% interval | Exact sign-flip p | Holm p |
|---|---:|---|---:|---:|
| Grammar | -0.792 | [-1.167, -0.250] | 0.0625 | 0.1875 |
| Naturalness | -1.042 | [-1.458, -0.625] | 0.03125 | 0.125 |
| Level suitability | approximately 0 | [-0.292, 0.333] | 1 | 1 |
| Coherence | -0.708 | [-0.958, -0.417] | 0.0625 | 0.1875 |

There are only six family clusters. Intervals assume representative independent
families; the sign-flip calculation assumes symmetric family effects under its
null, not randomized treatment assignment. Ordinal score arithmetic is a
descriptive approximation. Holm covers these four endpoints only, not the
historical search. This is retrospective evidence, not a new confirmatory test.
Historical best-of-two also reused its selection judges for reporting and remains
explicitly marked optimistic.

### 2. Component writes can flip a known shirt error

The frozen screen examined layers 28, 32 and 34 at attention, MLP and layer-output
boundaries on two known learner-error cases. It used the real common assistant
prefix `ש` immediately before the numeral fork, not a supplied corrected numeral.

For each stage we added the clean-minus-error difference of **adjacent normalized
residual contributions**, rather than replacing an entire clean hidden state.
Layer-output minus MLP includes PLE and layer-scalar effects; it is not isolated
PLE or a native linear weight update.

- All **18 zero controls** reproduced baseline generated token IDs.
- Shirt L28 attention/MLP and L34 attention/MLP produced the correct target phrase.
- The predeclared selection rule chose **shirt L34 attention**: fork log-odds
  effect **+1.462559**, delta norm 4.8712, approximately **7.62%** of receiver norm.
- Its reply was `שתי חולצות יהיו בסך 110 שקלים.` rather than the baseline's
  `שני חולצות יהיו בסך 110 ש"ח.`
- No book site passed the narrow selection rule. Even L34 attention's positive
  margin change did not repair its generated numeral/noun.

This is finer localization than a terminal whole-state donor replacement, but
still uses information from a correct-context donor and a nominated token.
It has not established which head, feature, neuron or native weight causes the
failure, or an intervention the model can autonomously apply.

An important counter-result is retained: **reversing the layer-34 output-stage
contribution delta repaired both cases using their own donors**, with fork-margin
effects +19.0103 for shirts and +23.6058 for books. The book reply became
`שני ספרים. המחיר הכולל יהיה 48 שקלים.` These are not transferable tests using a
single shared mechanism. This boundary mixes PLE and layer scaling, and its
delta is 60-71% of receiver norm. The reverse-sign successes challenge a simple
"copy the clean component" interpretation and motivate separating scaling from
PLE before making any circuit claim. They were not post-hoc promoted into the
predeclared positive-donor candidate search.

### 3. Random controls limit the claim; transfer fails

For the selected shirt site, 15 deterministic coordinate-uniform random vectors
were matched to the donor delta's L2 norm:

| Readout | Result |
|---|---|
| Donor fork-margin effect | +1.462559 |
| Random mean / sample SD | +0.068006 / 0.382938 |
| Random range | [-0.570695, +0.568325] |
| Donor rank including controls | 1 of 16 |
| One-sided empirical rank p | 0.0625 (minimum possible here) |
| Random directions that also restored the target phrase | **8 of 15** |

The donor is stronger on this continuous readout, but binary rescue is common
under random perturbations. Controls are norm-matched, not matched for learned
feature geometry or guaranteed rotationally uniform. Selection and testing use
the same discovery case. This is not a p<0.05 result or generalization evidence.

Applying the exact shirt delta without a new donor:

- **Book:** unchanged, incorrect `שתי גילידים. המחיר הכולל יהיה 48 שקלים.`
- **Library preservation:** token-identical to its baseline and retained
  `העמדות פועלות כרגע...` with the recorded 20-shekel total.

Thus the tested cross-case transfer failed. One preserved continuation is not a
general no-regression guarantee. Phrase/number presence does not adjudicate
currency, factual association, conversation success, or full linguistic quality.

### 4. The judge audit is consistent, not validated

One AI judge assessed 14 blinded packets from six historical B2 scenarios:
both candidate orders, with hidden duplicates for two cases. It preserved
candidate preference in **6/6** counterbalanced cases and repeated **8/8**
grammar/naturalness criterion scores exactly. All 116 quoted evidence spans
were checked against the actual NPC texts.

This measures repeatability within one correlated judge invocation, not
independent evaluator validity. No human calibration was used. The target model's
earlier B-only recognition behavior is separately labeled; it is not conflated
with this judge audit.

## Gates and next action

| Gate | Disposition |
|---|---|
| Instrumentation activity, sham and revert | Passed targeted actual-model checks |
| Donor-specific margin sensitivity | Exploratory signal only |
| Transfer to second target | Failed |
| Broad preservation | Unknown; one unchanged control |
| Every required quality criterion >=9/10 | Not established; not promoted |
| Native weight repair | Stopped by failed mechanism/transfer prerequisite |
| Fresh final evaluation | Not run; no final set consumed |

The next cheapest justified investigation is a **new preregistered development
set of numeral-gender pairs**, separating target morphology from lexical content.
Test whether a context-conditioned feature predicts the needed sign before
trying to edit weights: the shirt and book cases need opposite numeral choices.
Separate PLE from the final layer scalar to explain the large reverse-sign
boundary result. Use sign/reverse controls, multiple geometry-aware nulls and
untouched preservation families. Only a transferable, input-conditioned mechanism would justify bounded
head/neuron attribution and a quantization-aware native weight proposal.
Do not broaden this completed screen until the direction-specific hypothesis and
budget are approved. See the [skeptical methods review](METHODS_REVIEW.md).

## Verification, provenance and corrections

- **237 completed evaluation jobs:** 24 baseline/capture, 162 screen, 51 follow-up.
  They comprise 154 scores, 6 capture jobs and 77 free continuations, not 237
  independent conversations. Total: 73,879 evaluated positions, 1,406 generated
  tokens, within the declared 400/200,000/20,000 bounds.
- One GPU session at a time; pinned checkpoint SHA-256
  `efab429012b97ab986c4d4838a46ff3ad95d618b42ce514771ca40fadc76a9a4`.
- All research evaluations use source fingerprint
  `e355e96b7e8ff6891a5be8e3a7cdbc72fff5dd54fe3d0acd80ea8dffa0a244aa`.
  The frozen source snapshot and raw captures/requests/events are preserved,
  separately from later UI and reporting corrections.
- Targeted validation covered 44 distinct tests, including nonzero stage
  activity, exact sham/revert, actual zero/historical paired replay, old overlay
  parity, persistence/cancellation, statistics and responsive UI. The first run
  passed 43 and exposed an overly broad test selector; after scoping it to the
  studio, all 8 comparison tests passed. No unresolved test failure remains.
- Typecheck and production demo/library build passed. Existing bundle-size and
  Vite native-config migration warnings remain; no dependencies changed.
- Compiled static demo verified in the integrated browser: 26 examples, 18 cells,
  zero displayed load errors, no loaded GPU/model.
- Review corrected numeric substring selection (`148` must not satisfy `48`),
  B2 exported metadata and a missing-intervention fallback. Strict numeric
  reanalysis left the selected-site file **byte-identical**, SHA-256
  `9b696fde180d69920e18bfc0be64e5745e0b2fa6f85bd1bda0ac3af23a367f64`.
  No inference was rerun under mixed source versions.

Compact evidence is in [screen effects](screen-effects.json),
[control effects](control-effects.json), [norms](component-norms.json),
[protocol snapshot](protocol.snapshot.json) and [judge audit](judge-audit/ratings.json).
Raw vectors/events are gitignored but retained locally, with a separate hashed
archive; see [archive provenance](archive-provenance.json).
