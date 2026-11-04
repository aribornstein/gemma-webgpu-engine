import { createHash } from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "../..");
const output = resolve(root, "research-results/mechanistic-repair");
const read = (name) => JSON.parse(readFileSync(resolve(output, name), "utf8"));
const write = (name, value) => writeFileSync(resolve(output, name), JSON.stringify(value, null, 2) + "\n");
const original = read("discovery/artifacts.json");
const corrected = read("causal-v2/artifacts.json");
const proposal = read("discovery/instruction-language-control.json");
const prior = read("causal-v2/behavioral-summary.json");
function completed(id) {
  const record = corrected.find((a) => a.request.id === id);
  if (!record || record.status !== "completed") throw new Error(`Missing completed ${id}`);
  return record;
}
const logp = (id) => completed(id).result.tokenLogProbs.reduce((x, y) => x + y, 0);
const recognition = proposal.recognitionPairs.map((pair) => {
  const [a, b] = pair.scoreRequestIds.map(logp);
  const preferred = a > b ? "A" : b > a ? "B" : "tie";
  return {
    id: pair.id, expected: pair.expectedLabel, preferred, logA: a, logB: b,
    correct: preferred === pair.expectedLabel,
    nominatedLabelMass: Math.exp(a) + Math.exp(b),
  };
});
const edits = proposal.requests.filter((r) => r.op === "replay").map((request) => {
  const baseline = prior.repairs.find((r) => r.id === request.id.replace("H4_", ""));
  if (!baseline) throw new Error("Missing matched Hebrew-instruction edit control");
  const record = completed(request.id);
  return {
    id: request.id, original: request.messages.at(-1).content, reply: record.result.text,
    expected: baseline.expected, exactMinimalEdit: record.result.text === baseline.expected,
    copiedInput: record.result.text === request.messages.at(-1).content,
    stopReason: record.result.stopReason,
  };
});
const nouns = corrected.filter((a) => /^N_.*\.free$/.test(a.request.id)).map((a) => ({
  id: a.id, prefix: a.request.assistantPrefix,
  text: a.result.fullAssistantText, stopReason: a.result.stopReason,
}));
write("causal-v2/h4-summary.json", {
  hypothesis: "H4_INSTRUCTION_LANGUAGE",
  recognition, edits, nounContinuations: nouns,
  conclusion: "The single English translation did not change B-only recognition or copy-only editing. No order-stable recognition or no-option repair was demonstrated; this does not prove that the underlying capability is absent.",
  additionalFreeContinuations: edits.length,
});
const counters = (records) => records.reduce((sum, record) => {
  for (const key of Object.keys(sum)) sum[key] += record.usage[key];
  return sum;
}, { evaluatedPositions: 0, generatedTokens: 0, diagnostics: 0, episodes: 0 });
const invalid = original.filter((a) => a.request.op === "patch");
const valid = [...original.filter((a) => a.request.op !== "patch"), ...corrected];
const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
write("budget-and-evidence.json", {
  totalAttempted: counters([...original, ...corrected]),
  invalidatedDispatchAttempts: counters(invalid),
  validDiscovery: counters(valid),
  modelJobs: original.length + corrected.length,
  judgeRequests: 3, criterionAssessments: 72,
  fixtureAuthorCalls: 1,
  scientificScopes: ["discovery", "conditional activation diagnostic", "task-switched availability control"],
  withheldScopes: ["pretrained weight repair", "all-criteria quality promotion", "fresh final evaluation"],
  rawEvidence: ["discovery", "causal-v2"].map((run) => {
    const bytes = readFileSync(resolve(output, run, "artifacts.json"));
    return {
      run, file: `${run}/artifacts.json`, bytes: bytes.byteLength, sha256: hash(bytes),
      version: read(`${run}/manifest.json`).provenance.revision,
      gitPolicy: "Raw evidence retained locally, not committed; compact review bundle committed separately",
    };
  }),
  rawSupportingFiles: ["discovery", "causal-v2"].flatMap((run) =>
    ["patch-requests.json", "events.jsonl"].map((name) => {
      const bytes = readFileSync(resolve(output, run, name));
      return { file: `${run}/${name}`, bytes: bytes.byteLength, sha256: hash(bytes), gitPolicy: "Retained locally with raw evidence archive" };
    })),
  invalidation: "discovery/INVALIDATION.json; invalid artifacts are retained and excluded from scientific conclusions",
});
// Original records (including their exact hashes) only; never strip fields from a purported artifact.
const ids = [
  "discovery:book-order-numeral.clean.replay", "discovery:book-order-numeral.error.replay",
  "discovery:shirt-order-numeral.clean.replay", "discovery:shirt-order-numeral.error.replay",
  "causal-v2:book-order-numeral.error.correct", "causal-v2:book-order-numeral.clean.correct",
  "causal-v2:book-order-numeral.clean.capture",
  "causal-v2:book-order-numeral.L34.donor.free", "causal-v2:book-order-numeral.L34.sham.free",
  "causal-v2:H4_E_SHIRT_ERROR", "causal-v2:H4_E_BOOK_ERROR",
  "causal-v2:N_ERROR_SHNEI.free", "causal-v2:N_ERROR_SHTEI.free",
];
const bundle = ids.map((id) => {
  const record = valid.find((a) => a.id === id);
  if (!record) throw new Error(`Review sample references missing or invalid record ${id}`);
  return record;
});
write("review-example.json", bundle);
const gates = {
  M: { result: "passed_for_diagnostic_scope", evidence: "Typed price/fact, complete-score and explicit-unknown tests; unrestricted vocabulary cannot pass" },
  W: { result: "passed", evidence: "Shared CLI/UI API, live scoring/capture/weight slices, activity sentinel, sham/revert and cancellation recovery" },
  D: { result: "passed_for_investigation", evidence: "Paired clean/error contexts reproduce two failures and correct behavior in matched contexts" },
  C: { result: "not_established", evidence: "Terminal donor state imports a correct decision; earlier effects fail specificity or complete rescue. No transferable native obstruction localized." },
  E: { result: "not_attempted_gate_C_not_passed", evidence: "No packed weight mutation, native matrix update or claim of weight-equivalent repair" },
  Q: { result: "not_passed", evidence: "No candidate; discovery includes sub-9 judgments and unassessed vocabulary; no full-episode promotion" },
  FINAL: { result: "not_run", evidence: "No relaxation of C/E/Q; no new final data generated or evaluated" },
};
write("gate-status.json", {
  campaign: "mechanistic-repair-v1", outcome: "inconclusive",
  deliverable: "Validated diagnostic workbench and bounded failure evidence; no promotable repair",
  gates, methodsReviewAvailable: existsSync(resolve(output, "METHODS_REVIEW.md")),
});
console.log(JSON.stringify({ usage: counters([...original, ...corrected]), validFreeRuns: counters(valid).episodes, h4Choices: recognition.map((r) => r.preferred), editedErrors: edits.filter((r) => r.exactMinimalEdit).length, reviewArtifacts: bundle.length }, null, 2));
