# Mechanism-first research workbench

This investigation starts with a conversational failure, not a prescribed steering
direction. It uses the owned Gemma WebGPU path and the immutable pinned checkpoint.
All Hebrew fixtures and language judgments are AI-authored, not human-validated.
See [protocol.json](./protocol.json) for the frozen gates and budgets.

## Open the workbench

```sh
GEMMA_RESEARCH_RUN=1 npm run dev -- --host 127.0.0.1 --port 5190
```

Open `http://127.0.0.1:5190/research.html`. The production build also includes
`research.html`. Saved evidence can be imported and viewed without loading a model.
To execute, choose the local pinned checkpoint or an existing browser cache.
The CLI verifies the full local checkpoint SHA-256; interactive loading declares
the pin but does not independently hash a multi-gigabyte file.

Local checkpoint location (gitignored):
`public/models/gemma-4-e2b/model.safetensors`.

## Same interface for the agent and the UI

The page exposes `window.researchWorkbench` with `load`, `run`, `pause`, `resume`,
`cancel`, `artifacts`, and `close`. Neither UI nor CLI performs a separate numerical
implementation of the experiments.

Supported `run` operations:

- `replay`: frozen text history, optional assistant prefix, seeded free continuation.
- `score`: teacher-forced continuation with the model's softcap, not sampling transforms.
- `capture`: aligned post-layer residual vectors at bounded token positions.
- `inspectWeights`: selected projection rows/columns, signed codes, scales and
  dequantized values. No complete tensor or packed-weight writes.
- `patch`: one request-local activation replacement or addition at one layer/token.
  Cite source artifact IDs. This is a causal diagnostic, **not** a weight repair.
- `compare`: first-decision and sequence log probabilities and outputs; marks unequal
  prefixes rather than silently treating them as paired intervention results.
- `assess`: evidence-backed role judgments and fact claims, with pass/fail/unknown.
  It does not generate judgments itself or treat incomplete evidence as a pass.

Example request:

```json
{
  "id": "pair-clean-score",
  "op": "score",
  "messages": [
    {"role": "system", "content": "You are a Hebrew NPC. Two apples cost 10 ILS in total. Payment has not occurred."},
    {"role": "user", "content": "אני רוצה שני תפוחים."}
  ],
  "continuation": "שני תפוחים עולים עשרה שקלים."
}
```

All research comparisons use fresh sequential prefill and full-logit CPU selection.
The tokenizer rejects unstable prompt/continuation boundaries. GPU work is serialized.
Cancel propagates through the evaluation loop, clears temporary patches/captures, and
invalidates the K/V cache. Subsequent generation must remain usable.

## Run a checkpointed investigation

```sh
node research/mechanistic-repair/campaign.mjs prepare requests.json
npm run research:workbench -- requests.json discovery
node research/mechanistic-repair/campaign.mjs summarize \
  research-results/mechanistic-repair/discovery
```

`RESEARCH_URL` selects a serving workbench. `RESEARCH_OUTPUT` selects the output root.
Each run persists `manifest.json`, `events.jsonl` and `artifacts.json`. The CLI
acknowledges atomic disk persistence of the running budget reservation **before**
model evaluation, and persists the completed result before the next job.
The browser also stores individual artifacts in IndexedDB. A crash leaves an
explicit incomplete record; on restart it becomes failed and retains the
conservative reservation. Use a new attempt ID rather than replaying failed work
as though it were free. Cancelling an active job does not resume a paused queue.
Closing while the checkpoint is loading destroys the late-arriving session.

The runner enforces per-run limits. The campaign orchestrator must also account
for aggregate use across run IDs; changing an ID is not a new experiment budget.

Resuming an identical request reuses the recorded result. Reusing an ID with changed
inputs or source revision is an error. Every artifact carries exact requests,
source/checkpoint provenance, counters and hashes. Do not edit the implementation
while a frozen run is executing.

Captured vectors include token position, site, f32 dtype, shape and semantics.
The raw arrays are retained locally and excluded from git by default; summaries and
manifests should be committed. Do not delete the only copy of raw evidence.

## Interpretation and limitations

- Correct candidate likelihood is not proof of correct free generation.
- An explicit level-label decoder is not a Hebrew capability probe.
- A donor patch may import information; test unrelated donors, sham replacements,
  random perturbations and transfer cases before identifying a native mechanism.
- Post-layer residual transformations are not generally identical to edits of an
  existing MLP matrix. A real weight sidecar needs the same input and quantization/
  normalization ordering as the operator it modifies.
- Fact extraction from open Hebrew text is not solved by a numeric whitelist.
  The verifier checks explicit assessor claims against the authoritative contract
  and records unadjudicated text as unknown. It is not a formal semantic oracle.
- Unknown criteria, missing judgments, unassessed vocabulary and human-calibration
  limits must be visible in the report. Nine out of ten is a promotion floor, not a
  promised result.

Validation:

```sh
npm run typecheck
npm run test:workbench
MECHANISTIC_LIVE=1 npx playwright test tests/research-workbench.live.spec.ts --workers=1
```

The live test covers real-model replay, scoring, bounded weights, captured state,
same-state sham replacement, zero patch, revert, and cancellation/recovery.
