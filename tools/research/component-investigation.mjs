import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { familyPairedEffect, holmAdjust, randomControlTest, summarizeOutcomes } from "../../src/research/workbench/statistics.ts";
import { hasNumericAmount } from "../../src/research/workbench/contracts.ts";

const root = resolve(import.meta.dirname, "../..");
const out = resolve(root, "research-results/component-localization");
mkdirSync(out, { recursive: true });
const read = (file) => JSON.parse(readFileSync(file, "utf8"));
const write = (name, value) => writeFileSync(resolve(out, name), JSON.stringify(value, null, 2));
const protocol = read(resolve(root, "research/mechanistic-repair/component-protocol.json"));
const cases = read(resolve(root, "research-results/mechanistic-repair/discovery/fork-cases.json"));
const [stage, runId = "components"] = process.argv.slice(2);
const artifactsPath = resolve(out, runId, "artifacts.json");
const artifacts = existsSync(artifactsPath) ? read(artifactsPath) : [];
const byId = new Map(artifacts.map((a) => [a.request.id, a]));
const messages = (item, variant) => [...item.history, { role: "user", content: item[`${variant}User`] }];
const sum = (values) => values.reduce((a, b) => a + b, 0);
const diff = (a, b) => a.map((v, i) => v - b[i]);
const norm = (v) => Math.hypot(...v);
function artifact(id) {
  const a = byId.get(id);
  if (!a || a.status !== "completed") throw new Error(`Missing completed ${id}`);
  return a;
}
function captures(item, variant, layer) {
  const a = artifact(`${item.id}.${variant}.components`);
  const at = (stage) => a.result.captures.find((c) => c.site.residual === layer && c.stage === stage)?.values;
  for (const stage of protocol.captureStages) if (!at(stage)) throw new Error("Missing component capture");
  return { artifactId: a.id, at };
}
function deltaFor(item, layer, stage) {
  const previous = { attention: "layer-input", mlp: "attention", "layer-output": "mlp" }[stage];
  const clean = captures(item, "clean", layer), error = captures(item, "error", layer);
  return {
    delta: diff(diff(clean.at(stage), clean.at(previous)), diff(error.at(stage), error.at(previous))),
    sourceArtifactIds: [clean.artifactId, error.artifactId],
    recipientNorm: norm(error.at(stage)),
  };
}
function addCondition(jobs, item, prefix, layer, stage, vector, sourceArtifactIds, amount = 1) {
  for (const choice of ["correct", "incorrect", "free"]) {
    jobs.push({
      id: `${prefix}.${choice}`, op: "patch", layer, stage, mode: "add", amount, vector, sourceArtifactIds,
      messages: messages(item, "error"), assistantPrefix: item.assistantPrefix,
      ...(choice === "free" ? { maxNewTokens: 64, temperature: 0, seed: 23 } : { continuation: item[`${choice}Continuation`] }),
    });
  }
}
function observation(prefix, item) {
  const correct = artifact(`${prefix}.correct`), incorrect = artifact(`${prefix}.incorrect`), free = artifact(`${prefix}.free`);
  const text = free.result.fullAssistantText ?? item.assistantPrefix + free.result.text;
  const baseline = artifact(`${item.id}.error.free`);
  return {
    id: prefix, family: item.family,
    logOdds: correct.result.tokenLogProbs[0] - incorrect.result.tokenLogProbs[0],
    sequenceLogOdds: sum(correct.result.tokenLogProbs) - sum(incorrect.result.tokenLogProbs),
    text, stopReason: free.result.stopReason,
    identicalToBaseline: JSON.stringify(free.result.tokenIds) === JSON.stringify(baseline.result.tokenIds),
    // Narrow, observable target phrase only; not a morphology parser or semantic-fidelity verdict.
    targetPhrasePresent: text.startsWith(item.id === "book-order-numeral" ? "שני ספרים" :
      item.id === "shirt-order-numeral" ? "שתי חולצות" : "העמדות פועלות"),
    statedTotalPresent: hasNumericAmount(text, item.metadata.totalPrice),
    caveat: "Phrase and numeral presence alone are not quality, semantic correctness or a 9/10 pass",
    freeArtifact: free.id,
  };
}
if (stage === "prepare") {
  const jobs = [];
  for (const item of cases) for (const variant of ["clean", "error"]) {
    const base = { messages: messages(item, variant), assistantPrefix: item.assistantPrefix };
    for (const choice of ["correct", "incorrect"]) jobs.push({
      ...base, id: `${item.id}.${variant}.${choice}`, op: "score", continuation: item[`${choice}Continuation`],
    });
    jobs.push({ ...base, id: `${item.id}.${variant}.free`, op: "replay", maxNewTokens: 64, temperature: 0, seed: 23 });
    jobs.push({ ...base, id: `${item.id}.${variant}.components`, op: "capture", layers: protocol.initialLayers, stages: protocol.captureStages });
  }
  write("baseline-requests.json", jobs);
  write("protocol.snapshot.json", { ...protocol, sha256: createHash("sha256").update(JSON.stringify(protocol)).digest("hex") });
  console.log(`${jobs.length} baseline/component capture jobs`);
} else if (stage === "screen") {
  const jobs = [], summaries = [];
  for (const item of cases.filter((c) => protocol.cases.targets.includes(c.id))) {
    for (const layer of protocol.initialLayers) for (const boundary of ["attention", "mlp", "layer-output"]) {
      const { delta, sourceArtifactIds, recipientNorm } = deltaFor(item, layer, boundary);
      summaries.push({ caseId: item.id, layer, stage: boundary, deltaNorm: norm(delta), recipientNorm, ratio: norm(delta) / recipientNorm });
      for (const [condition, amount] of [["donor", 1], ["zero", 0], ["sign-flip", -1]]) {
        addCondition(jobs, item, `${item.id}.L${layer}.${boundary}.${condition}`, layer, boundary, delta, sourceArtifactIds, amount);
      }
    }
  }
  write("screen-requests.json", jobs);
  write("component-norms.json", summaries);
  console.log(`${jobs.length} component contribution and control jobs`);
} else if (stage === "summarize-screen") {
  const rows = [];
  for (const item of cases.filter((c) => protocol.cases.targets.includes(c.id))) {
    for (const layer of protocol.initialLayers) for (const boundary of ["attention", "mlp", "layer-output"]) {
      const prefix = `${item.id}.L${layer}.${boundary}`;
      const donor = observation(`${prefix}.donor`, item), zero = observation(`${prefix}.zero`, item), flip = observation(`${prefix}.sign-flip`, item);
      rows.push({ caseId: item.id, layer, stage: boundary, effect: donor.logOdds - zero.logOdds, donor, zero, flip });
    }
  }
  write("screen-effects.json", rows);
  const selected = cases.filter((c) => protocol.cases.targets.includes(c.id)).flatMap((item) => {
    const candidates = rows.filter((r) => r.caseId === item.id && r.effect > 0 && r.donor.targetPhrasePresent && r.donor.statedTotalPresent && r.donor.stopReason === "end-token")
      .sort((a, b) => b.effect - a.effect);
    return candidates.length ? [{ caseId: item.id, layer: candidates[0].layer, stage: candidates[0].stage, effect: candidates[0].effect }] : [];
  });
  write("selected.json", selected);
  console.log(JSON.stringify({ rows: rows.map((r) => ({ id: r.caseId, layer: r.layer, stage: r.stage, effect: r.effect, target: r.donor.targetPhrasePresent, totalPresent: r.donor.statedTotalPresent, text: r.donor.text })), selected }, null, 2));
} else if (stage === "controls") {
  const selected = read(resolve(out, "selected.json"));
  const jobs = [];
  for (const choice of selected) {
    const item = cases.find((c) => c.id === choice.caseId);
    const { delta, sourceArtifactIds } = deltaFor(item, choice.layer, choice.stage);
    const prefix = `${item.id}.L${choice.layer}.${choice.stage}`;
    for (const seed of protocol.followup.randomControlSeeds) {
      let state = seed;
      const random = delta.map(() => {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        return state / 4294967296 - 0.5;
      });
      const scale = norm(delta) / norm(random);
      addCondition(jobs, item, `${prefix}.random${seed}`, choice.layer, choice.stage, random.map((v) => v * scale), sourceArtifactIds);
    }
    for (const receiver of cases.filter((c) => c.id !== item.id)) {
      addCondition(jobs, receiver, `${prefix}.transfer-${receiver.id}`, choice.layer, choice.stage, delta, sourceArtifactIds);
    }
  }
  write("control-requests.json", jobs);
  console.log(`${jobs.length} multiple random and transfer jobs`);
} else if (stage === "summarize-controls") {
  const selected = read(resolve(out, "selected.json"));
  const results = [];
  for (const choice of selected) {
    const item = cases.find((c) => c.id === choice.caseId);
    const prefix = `${item.id}.L${choice.layer}.${choice.stage}`;
    const donor = observation(`${prefix}.donor`, item), zero = observation(`${prefix}.zero`, item);
    const random = protocol.followup.randomControlSeeds.map((s) => observation(`${prefix}.random${s}`, item));
    const transfer = cases.filter((c) => c.id !== item.id).map((receiver) => ({
      recipient: receiver.id, baseline: artifact(`${receiver.id}.error.free`).result.fullAssistantText,
      ...observation(`${prefix}.transfer-${receiver.id}`, receiver),
    }));
    results.push({ ...choice, donor, random, transfer,
      randomControl: randomControlTest(donor.logOdds - zero.logOdds, random.map((r) => r.logOdds - zero.logOdds)),
      targetOnlyOutcomeCounts: summarizeOutcomes(random.map((r) => ({
        id: r.id, family: item.family, baseline: zero.targetPhrasePresent, intervention: r.targetPhrasePresent,
      }))),
    });
  }
  write("control-effects.json", {
    exploratory: true, selectedOnSameCases: true, results,
    holmAdjustedEmpiricalP: holmAdjust(results.map((r) => r.randomControl.pValue)),
    familyEffects: selected.length > 0 ? familyPairedEffect(selected.map((s) => ({
      id: s.caseId, family: cases.find((c) => c.id === s.caseId).family, baseline: 0, intervention: s.effect,
    }))) : null,
    note: "Selected sites and two discovery families do not supply confirmatory significance. No quality promotion from log-odds or phrase matches.",
  });
  console.log(JSON.stringify(results.map((r) => ({ caseId: r.caseId, site: [r.layer, r.stage], null: r.randomControl,
    donor: r.donor.text, transfer: r.transfer.map((t) => ({ recipient: t.recipient, text: t.text, target: t.targetPhrasePresent })) })), null, 2));
} else throw new Error("prepare | screen | summarize-screen | controls | summarize-controls");
