# Methods review: component localization

## Verdict and evidence boundary

**Stop the conditionally approved native weight-repair branch. Do not start fresh final judging.** The experiment supports a local, donor-dependent activation effect, not a transferable repair, identified circuit, or native weight defect.

Reviewed `research/mechanistic-repair/component-protocol.json` and the component-localization `protocol.snapshot.json`, `screen-effects.json`, `selected.json`, `control-effects.json`, and `component-norms.json`; checked the saved artifact manifest, analysis implementation, and judge-audit summary. The protocol and snapshot agree apart from the snapshot hash. No models were run for this review.

There are **237 evaluation continuations/jobs**, including 12 score, 6 replay, 6 capture, and 213 patch jobs—not 237 independent examples. Two known discovery failures and one preservation case supply the behavioral scope. The 18 screened case-sites are two targets × three layers × three boundaries. Repeated alternatives, seeds, and continuations do not increase the number of independent cases.

## What the intervention means

For adjacent boundaries, the injected vector is
`(donor_after − donor_before) − (recipient_after − recipient_before)`.
This is more specific than replacing an entire post-layer state, but it still imports information from a correct-context donor.

Captures are **cumulative residual boundaries containing normalized branch contributions**, not isolated raw sublayer outputs. Attention-minus-input includes output projection, output quantization, and post-attention RMS normalization; MLP-minus-attention includes down projection, output quantization, and post-FFN normalization. Final-minus-MLP combines **PLE and layer-scalar multiplication**, including rescaling of the existing residual; it cannot isolate PLE.

The selected attention delta has L2 norm 4.871231, approximately 7.62% of its recipient boundary residual norm. This is not a percentage of isolated attention activity. Final-boundary deltas reach approximately 60–71% of recipient residual norm, cautioning against interpreting their very large effects as clean component localization.

The recorded common assistant prefix avoids explicitly supplying the corrected numeral. Nevertheless, donor contributions can already encode answer, lexical, or contextual information. Their causal efficacy does not establish a grammar mechanism or autonomous recovery of latent knowledge. A weight update would have to generate a suitable correction without access to the donor, across contexts; that has not been demonstrated.

## Observed behavior and controls

All **18 zero shams** reproduce their recorded baseline continuation, with the same per-case fork margin. This supports local intervention-path parity, not exhaustive bitwise state equivalence.

Only **shirt, L34 attention** qualifies. Its donor-minus-zero fork-margin effect is **1.462559**; absolute fork margins are donor **1.269176**, zero **−0.193384**, and reversed delta **−2.413962**. Donor output is:

> שתי חולצות יהיו בסך 110 שקלים.

Zero and reversal retain:

> שני חולצות יהיו בסך 110 ש"ח.

These complete sentences support the narrow numeral-agreement change and preservation of two shirts costing 110 shekels. They do not establish all-criteria quality.

Signs are not uniformly monotonic across sites. Reversing shirt L28/L32/L34 final-boundary deltas yields the corrected shirt sentence; reversing book L34 final-boundary delta produces:

> שני ספרים. המחיר הכולל יהיה 48 שקלים.

Every book **donor-direction** continuation remains the malformed baseline despite some improved margins. The reverse-final observation is retained evidence, not permission to select another direction post hoc.

Against **15 L2-matched coordinate-uniform random controls**, donor effect 1.462559 exceeds random mean **0.068006** and maximum **0.568325**: rank **1/16**, empirical upper-tail **p = 0.0625**. The coordinates are sampled uniformly then rescaled; these are not uniformly distributed sphere directions or activation-manifold-matched controls. **Eight of 15** random directions produce the same corrected shirt sentence. Thus donor margin superiority does not make the discrete correction donor-exclusive.

Selection used these same cases and outputs. The rank is descriptive, at its 1/16 resolution floor, not significance at 0.05. Holm correction over the single survivor does not account for screening 18 case-sites. No independent family-level generalization follows.

Transferring the selected shirt vector to books leaves:

> שתי גילידים. המחיר הכולל יהיה 48 שקלים.

The malformed entity phrase remains; “48” does not rescue its meaning. Library transfer is token-identical to baseline:

> העמדות פועלות כרגע. אני יכול לאשר את ההזמנה עבור שעה אחת. המחיר הכולל יהיה 20 שקלים. האם תרצה להמשיך?

This preserves the observed operation statement, one-hour booking, and 20-shekel statement in this case—not general preservation.

## Scientific gates

| Gate | Assessment |
|---|---|
| Zero-sham behavioral parity | **PASS**, narrowly observed. |
| Selected-site signed local effect | **PASS**, donor-dependent activation diagnostic. |
| Specificity beyond sampled random margins | **PASS descriptively**; confirmatory specificity **UNKNOWN** after selection. |
| Donor-independent grammar mechanism | **UNKNOWN**; answer import remains plausible. |
| Cross-target rescue | **FAIL**: books remain wrong. |
| Preservation | **PASS** for one saved library continuation; broader safety **UNKNOWN**. |
| Full ≥9/10 quality and zero critical-fact failures | **NOT PASSED**; full quality **UNKNOWN**. |
| Native weight repair / final evaluation authorization | **FAIL** under the frozen prerequisite gates. |

## Audit qualifications and next step

The selector had a numerical-substring defect: number presence could match inside another number. The parent is correcting the helper and regenerating summaries. Direct inspection confirms selected **110 is a complete price token**, so this defect does not affect the selected site. Even corrected numeric and phrase matching is not semantic judging.

Historical p6 versus baseline fixed **0** and broke **6 of 24** on the narrow grammar-and-naturalness **≥3/4** endpoint—not the ≥9/10 promotion criterion. One judge's **14 counterbalanced/duplicate packets** measure descriptive repeatability, not independent linguistic validation.

The cheapest justified next investigation is an **offline, locked audit of existing donor/recipient contexts, prefixes, and complete outputs**, separating grammatical changes from answer/fact changes. If further diagnostics are separately authorized, predeclare crossed-donor controls at the fixed site on disjoint development cases to distinguish agreement transfer from answer import. Do not execute a weight sweep or consume a fresh final set. No human calibration is required.
