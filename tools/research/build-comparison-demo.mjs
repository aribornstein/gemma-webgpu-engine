import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "../..");
const out = resolve(root, "public/research");
mkdirSync(out, { recursive: true });
const files = new Map();
const read = (path) => {
  const bytes = readFileSync(resolve(root, path));
  files.set(path, createHash("sha256").update(bytes).digest("hex"));
  return path.endsWith(".jsonl") ? bytes.toString().trim().split("\n").map(JSON.parse) : JSON.parse(bytes);
};
const cefrRoot = "research-results/cefr-v2";
const episodes = read(`${cefrRoot}/dev.jsonl`);
const panel = read(`${cefrRoot}/panel-dev1+dev2+dev3.json`);
const scenarios = read("research/cefr-agentic/scenarios.dev.jsonl");
const judgings = [1, 2, 3].flatMap((batch) =>
  ["language", "cefr", "level", "conversation"].flatMap((role) =>
    read(`${cefrRoot}/judging/dev${batch}-${role}.collected.jsonl`)));
const examples = [];
for (const base of episodes.filter((e) => e.arm === "raw_base")) {
  const scenario = scenarios.find((s) => s.scenario_id === base.scenarioId);
  const selected = panel.episodes.find((e) => e.arm === "best_of_2" && e.scenarioId === base.scenarioId && e.level === base.level &&
    (e.seed === base.seed || e.seed === base.seed + 1000));
  const definitions = [
    { id: "raw_base", label: "Metadata only", kind: "baseline", seed: base.seed },
    { id: "strong_prompt", label: "Strong level instruction", kind: "prompt-baseline", seed: base.seed },
    { id: "agent_p6", label: "Instruction-derived overlay p6", kind: "residual-steering", seed: base.seed },
    ...(selected ? [{ id: selected.arm, sourceArm: selected.key.split("|")[0], label: "Historical best-of-2", kind: "selected-baseline", seed: selected.seed }] : []),
  ];
  const arms = definitions.map((definition) => {
    const ep = episodes.find((e) => e.arm === (definition.sourceArm ?? definition.id) && e.scenarioId === base.scenarioId &&
      e.level === base.level && e.seed === definition.seed);
    if (!ep) throw new Error("Missing historical comparison arm");
    const judgments = judgings.filter((j) => j.item_id === ep.jobKey);
    return {
      id: definition.id, label: definition.label, kind: definition.kind,
      sourceId: ep.jobKey, sourceFile: `${cefrRoot}/dev.jsonl`,
      policy: ep.policy, seed: ep.seed,
      turns: ep.npcTurns.map((text, i) => ({
        player: ep.playerTurns[i], text, tokenCount: ep.npcTokenCounts[i], stopReason: ep.stopReasons[i],
      })),
      ratings: judgments.flatMap((j) => (j.ratings ?? []).map((r) => ({
        criterion: r.criterion, score: r.score, scale: 4,
        evidence: r.evidence.map((e) => ({ span: e.span, finding: e.finding })),
        judge: j.judge_model_id ?? "unknown",
      }))),
      warnings: [
        "Historical AI judgments on a 0-4 rubric, not new 9/10 certification.",
        ...(definition.kind === "selected-baseline" ? ["Selection reused the reported judges: optimistic historical comparison, not independent final grading."] : []),
        ...(definition.kind === "residual-steering" ? ["Rejected for language regressions. Residual steering is not an established native weight repair."] : []),
      ],
    };
  });
  examples.push({
    id: `${base.scenarioId}:${base.level}:${base.seed}`, experiment: "cefr",
    label: `${base.scenarioFamily} / ${base.level} / seed ${base.seed}`,
    family: base.scenarioFamily, level: base.level, seed: base.seed,
    metadata: { ...scenario.metadata_en, cefr_level: base.level }, arms,
    live: { scenario, patch: "agent_p6.json" },
    caveat: "Paired player script; each arm has its own NPC history. Prompt baseline intentionally uses stronger instructions. Old stochastic execution is preserved, not reinterpreted as a new matched-path run.",
  });
}
const sample = read("research-results/mechanistic-repair/review-example.json");
for (const id of ["book-order-numeral", "shirt-order-numeral"]) {
  const defs = [
    { source: `discovery:${id}.clean.replay`, label: "Clean learner context", kind: "context-control" },
    { source: `discovery:${id}.error.replay`, label: "Erroneous learner context", kind: "baseline" },
    { source: `causal-v2:${id}.L34.sham.free`, label: "Same-state sham", kind: "sham" },
    { source: `causal-v2:${id}.L34.donor.free`, label: "Clean donor at numeral fork", kind: "donor-diagnostic" },
  ];
  const arms = defs.flatMap((d) => {
    const a = sample.find((a) => a.id === d.source);
    if (!a) return [];
    if (a.status !== "completed") throw new Error("Review demo cannot display an incomplete artifact as an outcome");
    return [{
      id: d.kind, label: d.label, kind: d.kind, sourceId: a.id,
      sourceFile: "research-results/mechanistic-repair/review-example.json",
      policy: "Recorded diagnostic context", seed: 23, ratings: [],
      turns: [{
        player: a.request.messages.at(-1).content,
        text: a.result.fullAssistantText ?? (a.request.assistantPrefix ?? "") + a.result.text,
        tokenIds: a.result.tokenIds, tokenLogProbs: a.result.tokenLogProbs,
        firstDecision: a.result.firstDecision, tokenCount: a.result.tokenIds.length,
        forcedPrefix: a.request.assistantPrefix ?? "", stopReason: a.result.stopReason,
      }],
      warnings: [
        ...(d.kind === "donor-diagnostic" ? ["Correct-context information supplied through a donor activation. This is diagnostic answer-state transfer, not a deployable repair."] : []),
        ...(a.request.assistantPrefix ? ["Common prefix held fixed at the actual differing token; not a new fully unprompted generation."] : []),
        "No full quality panel attached to this recorded diagnostic.",
      ],
    }];
  });
  examples.push({
    id, experiment: "mechanistic", label: id, family: id, level: "diagnostic", seed: 23,
    metadata: { scope: "Discovery only", route: "fresh sequential / full-logit CPU", invalidatedAttemptsExcluded: true,
      unavailableArms: defs.filter((d) => !sample.some((a) => a.id === d.source)).map((d) => d.label) },
    arms, caveat: "Clean vs error changes the learner context. Sham vs donor uses the identical recipient prefix. Only the latter isolates an intervention." +
      (arms.some((a) => a.kind === "donor-diagnostic") ? "" : " No intervention recording is available here; only context controls can be compared."),
  });
}
const patch = read(`${cefrRoot}/patches/agent_p6.json`);
writeFileSync(resolve(out, "agent_p6.json"), JSON.stringify(patch));
writeFileSync(resolve(out, "comparisons.json"), JSON.stringify({
  schemaVersion: 1, title: "Recorded research, including failures",
  qualityClaim: "No intervention has passed all-criteria 9/10 promotion",
  examples,
  sources: [...files].map(([path, sha256]) => ({ path, sha256 })),
}));
console.log(`Built ${examples.length} recorded comparisons; no new model evaluations or inferred scores.`);
