import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { familyPairedEffect, summarizeOutcomes, holmAdjust, counterbalancedPreference, judgeAgreement } from "../../src/research/workbench/statistics.ts";

const root = resolve(import.meta.dirname, "../..");
const out = resolve(root, "public/research");
mkdirSync(out, { recursive: true });
const rows = JSON.parse(readFileSync(resolve(root, "research-results/cefr-v2/panel-dev1+dev2+dev3.json"), "utf8")).episodes;
const paired = rows.filter((r) => r.arm === "raw_base").map((baseline) => {
  const intervention = rows.find((r) => r.arm === "agent_p6" && r.scenarioId === baseline.scenarioId && r.level === baseline.level && r.seed === baseline.seed);
  if (!intervention) throw new Error("Missing historical paired episode");
  return { id: baseline.key, family: baseline.family, baseline, intervention };
});
const endpoints = ["morphosyntactic_correctness", "natural_hebrew", "cefr_suitability", "multi_turn_coherence"];
const effects = endpoints.map((criterion) => {
  const input = paired.map((p) => ({
    id: p.id, family: p.family, baseline: p.baseline[criterion], intervention: p.intervention[criterion],
  }));
  return { criterion, scale: "historical 0-4 ordinal judgment, not a new 0-10 score", ...familyPairedEffect(input, { seed: 912 }) };
});
const adjusted = holmAdjust(effects.map((e) => e.pValue));
effects.forEach((effect, i) => { effect.holmAdjustedP = adjusted[i]; });
const outcomes = summarizeOutcomes(paired.map((p) => ({
  id: p.id, family: p.family,
  baseline: p.baseline.morphosyntactic_correctness >= 3 && p.baseline.natural_hebrew >= 3,
  intervention: p.intervention.morphosyntactic_correctness >= 3 && p.intervention.natural_hebrew >= 3,
})));
const baseline = JSON.parse(readFileSync(resolve(root, "research-results/mechanistic-repair/causal-v2/behavioral-summary.json"), "utf8"));
const recognition = ["SHIRT", "BOOK", "LIBRARY"].map((item) => ({
  id: item,
  originalWinner: baseline.recognition.find((r) => r.id === `R_${item}_AB`).preferred,
  reversedWinner: baseline.recognition.find((r) => r.id === `R_${item}_BA`).preferred,
}));
const reservedRows = JSON.parse(readFileSync(resolve(root, "research-results/cefr-v2/panel-seal1+seal2+seal3.json"), "utf8")).episodes.filter((r) => r.arm !== "best_of_2");
const reserved = judgeAgreement(reservedRows.map((r) => ({
  id: r.key, first: r.estimated_level_value ?? null, second: r.estimated_level_reserved_value ?? null,
})));
const report = {
  schemaVersion: 1,
  title: "Repair and damage, not just average steering gain",
  dataset: "Historical CEFR v2 development episodes; p6 vs raw, same scenario/level/seed; different arm histories",
  outcomeDefinition: "Grammar AND naturalness >=3 on the original 0-4 rubric. This is NOT the full 9/10 promotion requirement.",
  outcomes,
  effects,
  recognitionOrderControl: {
    ...counterbalancedPreference(recognition),
    interpretation: "This is the target model's A/B recognition, not a judge calibration: B-only selections flip candidate identity under reversed order.",
  },
  reservedJudgeAgreement: {
    ...reserved,
    interpretation: "Two prompts of the same model family rating ordinal CEFR level; descriptive agreement is not independent validity or human calibration.",
  },
  limitations: [
    "Retrospective analysis of already-examined data; not a new confirmatory test.",
    "Bootstrap resamples families, not tokens or repeated seeds. Six families yield coarse/low-power inference.",
    "Sign-flip inference assumes symmetry/exchangeability of family effects; no randomized arm assignment is claimed.",
    "Holm adjustment covers only the four displayed endpoints, not all prior search choices.",
    "CEFR levels are ordinal; no continuous proficiency or 9/10 success is inferred from level separation.",
    "New component random-controls will be recorded separately and not pooled with the historical single-direction controls.",
  ],
};
writeFileSync(resolve(out, "paired-statistics.json"), JSON.stringify(report));
console.log(JSON.stringify({ outcomes, effects: effects.map((e) => ({ criterion: e.criterion, delta: e.meanDelta, ci: e.confidenceInterval95, p: e.pValue, adjusted: e.holmAdjustedP })) }, null, 2));
