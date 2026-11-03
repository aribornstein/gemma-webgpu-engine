// Deterministic analysis for the CEFR campaign (no agent judgment). Usage:
//   node --experimental-strip-types research/cefr-agentic/analyze.mjs calibration
//   node --experimental-strip-types research/cefr-agentic/analyze.mjs probes
//   node --experimental-strip-types research/cefr-agentic/analyze.mjs episodes <name>...
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { checkState, complexityIndex, COMPLEXITY_KEYS, turnFeatures } from "../../src/research/cefr/metrics.ts";
import { auc, fitLogistic, fitMeanDifference, groupBootstrapAuc, probeScore, rawDirection, shuffledLabels } from "../../src/research/probes.ts";

const ROOT = resolve(import.meta.dirname, "../..");
const OUT = resolve(ROOT, process.env.RESEARCH_OUT ?? "research-results/cefr-agentic");
const readJsonl = (path) => existsSync(path) ? readFileSync(path, "utf8").split("\n").filter((line) => line.trim().startsWith("{")).map((line) => JSON.parse(line)) : [];
const LEVELS = { A1: 1, A2: 2, B1: 3, B2: 4, C1: 5, C2: 6 };
const mean = (values) => values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
const round = (value, digits = 3) => Number.isFinite(value) ? Number(value.toFixed(digits)) : value;

function collected(name) {
  return readJsonl(resolve(OUT, `judging/${name}.collected.jsonl`));
}
function score(row, criterion) {
  return row.ratings?.find((rating) => rating.criterion === criterion)?.score ?? null;
}
function pairwise(a, b) {
  if (a === null || b === null) return null;
  return a > b ? 1 : a === b ? 0.5 : 0;
}

const [command, ...args] = process.argv.slice(2);
const report = {};

if (command === "calibration") {
  const truth = Object.fromEntries(readJsonl(resolve(OUT, "calibration/truth.jsonl")).map((row) => [row.item_id, row]));
  const level = collected("calib-level");
  const byFamily = {};
  for (const row of level) {
    const t = truth[row.item_id];
    (byFamily[t.family] ??= {})[t.style] = LEVELS[row.estimated_level] ?? null;
  }
  const ordering = Object.values(byFamily).map((pair) => pairwise(pair.B2, pair.A1)).filter((value) => value !== null);
  report.level_estimator = { pairs: ordering.length, ordering_accuracy: round(mean(ordering)) };
  const language = collected("calib-language");
  const byPair = {};
  for (const row of language) {
    const t = truth[row.item_id];
    (byPair[t.pair] ??= {})[t.correct ? "ok" : "bad"] = score(row, "morphosyntactic_correctness");
  }
  const grammar = Object.values(byPair).map((pair) => pairwise(pair.ok, pair.bad)).filter((value) => value !== null);
  report.language_judge = { pairs: grammar.length, correct_over_incorrect: round(mean(grammar)) };
  for (const name of ["calib-cefr", "calib-cefr-swapped"]) {
    const rows = collected(name);
    if (!rows.length) continue;
    report[name] = { n: rows.length, mean_suitability: round(mean(rows.map((row) => score(row, "cefr_suitability")).filter((v) => v !== null))) };
  }
  const matched = collected("calib-cefr");
  const swapped = collected("calib-cefr-swapped");
  if (matched.length && swapped.length) {
    const key = (row) => row.item_id.replace(/:swap$/, "");
    const swappedByKey = Object.fromEntries(swapped.map((row) => [key(row), score(row, "cefr_suitability")]));
    const comparisons = matched.map((row) => pairwise(score(row, "cefr_suitability"), swappedByKey[key(row)] ?? null)).filter((v) => v !== null);
    report.cefr_judge = { pairs: comparisons.length, matched_over_mismatched: round(mean(comparisons)) };
  }
  report.gate_threshold = 0.7;
  writeFileSync(resolve(OUT, "calibration/report.json"), JSON.stringify(report, null, 1));
} else if (command === "probes") {
  const rows = JSON.parse(readFileSync(resolve(OUT, "activations/probes.json"), "utf8"));
  const sites = [...new Set(rows.filter((row) => row.kind === "level").map((row) => String(row.site)))];
  const split = (row) => row.split === "development" ? "train" : "test";
  const runProbe = (selected, label, name) => {
    const train = selected.filter((row) => split(row) === "train");
    const test = selected.filter((row) => split(row) === "test");
    const data = (set) => ({ x: set.map((row) => Float32Array.from(row.values)), y: set.map(label), group: set.map((row) => row.scenarioFamily) });
    const trainData = data(train);
    const testData = data(test);
    const out = { name, n_train: train.length, n_test: test.length };
    for (const [kind, fit] of [["mean_difference", fitMeanDifference], ["logistic", (d) => fitLogistic(d, 10)]]) {
      const probe = fit(trainData);
      const scores = testData.x.map((row) => probeScore(probe, row));
      out[kind] = { test_auc: round(auc(scores, testData.y)), ci: groupBootstrapAuc(scores, testData.y, testData.group, 500).map((v) => round(v)) };
    }
    const shuffled = fitLogistic({ ...trainData, y: shuffledLabels(trainData.y, 17) }, 10);
    out.shuffled_label_auc = round(auc(testData.x.map((row) => probeScore(shuffled, row)), testData.y));
    return out;
  };
  report.level = {};
  for (const position of ["end_of_metadata", "last_prompt_token"]) {
    for (const site of sites) {
      const selected = rows.filter((row) => row.kind === "level" && row.position === position && String(row.site) === site);
      report.level[`${position}@${site}`] = runProbe(selected, (row) => row.level === "B2" ? 1 : 0, `requested level ${position} ${site}`);
    }
  }
  report.crossed = {};
  for (const site of sites) {
    const selected = rows.filter((row) => row.kind === "crossed" && String(row.site) === site);
    report.crossed[site] = {
      style: runProbe(selected, (row) => row.style === "B2" ? 1 : 0, "response style"),
      requested_level: runProbe(selected, (row) => row.level === "B2" ? 1 : 0, "requested level at candidate boundary"),
      matched: runProbe(selected, (row) => row.matched ? 1 : 0, "level/style match (joint)"),
    };
  }
  const crossed = rows.filter((row) => row.kind === "crossed" && String(row.site) === sites[0]);
  const prefersMatched = [];
  for (const id of [...new Set(crossed.map((row) => row.id.split(":")[0]))]) {
    for (const level of ["A1", "B2"]) {
      const matched = crossed.find((row) => row.id === `${id}:${level}:${level}`);
      const other = crossed.find((row) => row.id === `${id}:${level}:${level === "A1" ? "B2" : "A1"}`);
      if (matched && other) prefersMatched.push({ id, level, delta_mean_logprob: matched.meanLogProb - other.meanLogProb });
    }
  }
  // Interaction: does requested level shift likelihood toward the matching style (difference in differences)?
  const did = [];
  for (const id of [...new Set(prefersMatched.map((row) => row.id))]) {
    const a1 = prefersMatched.find((row) => row.id === id && row.level === "A1");
    const b2 = prefersMatched.find((row) => row.id === id && row.level === "B2");
    if (a1 && b2) did.push(a1.delta_mean_logprob + b2.delta_mean_logprob);
  }
  report.likelihood = {
    matched_style_preferred_fraction: round(mean(prefersMatched.map((row) => row.delta_mean_logprob > 0 ? 1 : 0))),
    level_conditioning_interaction_mean: round(mean(did), 4),
    interaction_positive_fraction: round(mean(did.map((value) => value > 0 ? 1 : 0))),
    families: did.length,
  };
  const contrasts = rows.filter((row) => row.kind === "contrast");
  const byPhenomenon = {};
  for (const row of contrasts) (byPhenomenon[row.phenomenon] ??= []).push(row.prefersCorrect ? 1 : 0);
  report.agreement_likelihood = Object.fromEntries(Object.entries(byPhenomenon).map(([key, values]) => [key, { n: values.length, prefers_correct: round(mean(values)) }]));
  report.agreement_likelihood.all = { n: contrasts.length, prefers_correct: round(mean(contrasts.map((row) => row.prefersCorrect ? 1 : 0))) };
  writeFileSync(resolve(OUT, "probes.json"), JSON.stringify(report, null, 1));
} else if (command === "panel") {
  // args: <batch prefix> <episode analysis name> ; merges blind judgments with deterministic checks
  const [prefix, analysisName] = args;
  const analysis = JSON.parse(readFileSync(resolve(OUT, `episodes-${analysisName}.analysis.json`), "utf8")).episodes;
  const byKey = Object.fromEntries(analysis.map((row) => [row.key, { ...row }]));
  for (const role of ["language", "cefr", "level", "conversation", "level2"]) {
    for (const row of prefix.split(",").flatMap((p) => collected(`${p}-${role}`))) {
      const target = byKey[row.item_id];
      if (!target) continue;
      for (const rating of row.ratings ?? []) target[rating.criterion] = rating.score;
      if (role === "level") target.estimated_level = row.estimated_level;
      if (role === "level2") target.estimated_level_reserved = row.estimated_level;
      target.critical = [...(target.critical ?? []), ...(row.critical_violation_codes ?? [])];
    }
  }
  const rows = Object.values(byKey);
  for (const row of rows) {
    row.estimated_level_value = LEVELS[row.estimated_level] ?? null;
    row.estimated_level_reserved_value = LEVELS[row.estimated_level_reserved] ?? null;
    row.joint_pass = (row.cefr_suitability ?? 0) >= 3 && (row.morphosyntactic_correctness ?? 0) >= 3 &&
      (row.natural_hebrew ?? 0) >= 3 && row.hard_pass;
  }
  const summarize = (subset) => {
    const gaps = [];
    for (const scenarioId of [...new Set(subset.map((row) => row.scenarioId))]) {
      const a1 = subset.filter((row) => row.scenarioId === scenarioId && row.level === "A1" && row.estimated_level_value !== null);
      const b2 = subset.filter((row) => row.scenarioId === scenarioId && row.level === "B2" && row.estimated_level_value !== null);
      if (a1.length && b2.length) gaps.push({ scenarioId, family: a1[0].family, gap: mean(b2.map((r) => r.estimated_level_value)) - mean(a1.map((r) => r.estimated_level_value)), complexity_gap: mean(b2.map((r) => r.complexity)) - mean(a1.map((r) => r.complexity)) });
    }
    const avg = (key) => round(mean(subset.map((row) => row[key]).filter((v) => typeof v === "number")), 2);
    return {
      episodes: subset.length,
      cefr_control_gap: round(mean(gaps.map((g) => g.gap)), 2),
      cefr_control_positive_fraction: round(mean(gaps.map((g) => g.gap > 0 ? 1 : 0)), 2),
      complexity_gap: round(mean(gaps.map((g) => g.complexity_gap)), 2),
      a1_estimated: round(mean(subset.filter((r) => r.level === "A1").map((r) => r.estimated_level_value).filter((v) => v !== null)), 2),
      b2_estimated: round(mean(subset.filter((r) => r.level === "B2").map((r) => r.estimated_level_value).filter((v) => v !== null)), 2),
      joint_pass_rate: round(mean(subset.map((r) => r.joint_pass ? 1 : 0)), 2),
      hard_pass_rate: round(mean(subset.map((r) => r.hard_pass ? 1 : 0)), 2),
      morphosyntactic_correctness: avg("morphosyntactic_correctness"),
      natural_hebrew: avg("natural_hebrew"),
      cefr_suitability: avg("cefr_suitability"),
      pedagogical_appropriateness: avg("pedagogical_appropriateness"),
      character_register_fidelity: avg("character_register_fidelity"),
      multi_turn_coherence: avg("multi_turn_coherence"),
      gaps,
    };
  };
  // best_of_2: per scenario and level, the dev judge (cefr suitability, then language mean) selects
  // between the strong-prompt episodes with seeds 23 and 47.
  const selected = [];
  for (const row of rows.filter((r) => r.arm === "strong_prompt")) {
    const other = rows.find((r) => r.arm === "best_of_2_candidate" && r.scenarioId === row.scenarioId && r.level === row.level && r.seed === row.seed + 1000) ??
      rows.find((r) => r.arm === "best_of_2_candidate" && r.scenarioId === row.scenarioId && r.level === row.level);
    if (!other) continue;
    const value = (r) => (r.cefr_suitability ?? 0) * 10 + ((r.morphosyntactic_correctness ?? 0) + (r.natural_hebrew ?? 0)) / 2;
    selected.push({ ...(value(other) > value(row) ? other : row), arm: "best_of_2" });
  }
  const all = [...rows, ...selected];
  report.arms = Object.fromEntries([...new Set(all.map((row) => row.arm))].map((arm) => [arm, summarize(all.filter((row) => row.arm === arm))]));
  writeFileSync(resolve(OUT, `panel-${prefix.replace(/,/g, "+")}.json`), JSON.stringify({ report, episodes: all }, null, 1));
  for (const row of all) delete row.gaps;
} else if (command === "episodes") {
  const scenarios = Object.fromEntries([
    ...readJsonl(resolve(ROOT, "research/cefr-agentic/scenarios.dev.jsonl")),
    ...readJsonl(resolve(ROOT, "research/cefr-agentic/scenarios.validation.jsonl")),
    ...(process.env.SEALED_PATH ? process.env.SEALED_PATH.split(",").flatMap((path) => readJsonl(path)) : []),
  ].map((row) => [row.scenario_id, row]));
  const statsPath = resolve(ROOT, "research-results/cefr-agentic/complexity-stats.json");
  const episodes = args.flatMap((name) => readJsonl(resolve(OUT, `${name}.jsonl`)));
  if (!existsSync(statsPath)) {
    const features = episodes.filter((row) => row.arm === "raw_base").flatMap((row) => row.npcTurns.map(turnFeatures));
    const stats = Object.fromEntries(COMPLEXITY_KEYS.map((key) => {
      const values = features.map((f) => f[key]);
      const m = mean(values);
      return [key, { mean: m, sd: Math.sqrt(mean(values.map((v) => (v - m) ** 2))) }];
    }));
    writeFileSync(statsPath, JSON.stringify(stats, null, 1));
  }
  const stats = JSON.parse(readFileSync(statsPath, "utf8"));
  const enriched = episodes.map((row) => {
    const scenario = scenarios[row.scenarioId];
    const entityNumbers = (scenario.metadata_en.entities ?? []).flatMap((entity) => [entity.quantity, entity.price_ils].filter((v) => typeof v === "number"));
    const checks = row.npcTurns.map((text) => checkState(text, scenario.oracle_en.critical_facts, scenario.oracle_en.must_not_assume, entityNumbers));
    const features = row.npcTurns.map(turnFeatures);
    return {
      key: row.jobKey, arm: row.arm, scenarioId: row.scenarioId, family: row.scenarioFamily, level: row.level, seed: row.seed, editId: row.editId,
      complexity: mean(features.map((f) => complexityIndex(f, stats))),
      words_per_turn: mean(features.map((f) => f.words)),
      words_per_sentence: mean(features.map((f) => f.wordsPerSentence)),
      hard_pass: checks.every((check) => check.pass),
      unexpected_numbers: checks.flatMap((check) => check.unexpectedNumbers),
      claims_payment: checks.some((check) => check.claimsPayment),
      english_spillover: checks.some((check) => check.englishSpillover),
      length_stops: row.stopReasons.filter((reason) => reason === "length").length,
    };
  });
  const arms = [...new Set(enriched.map((row) => row.arm))];
  report.arms = {};
  for (const arm of arms) {
    const rows = enriched.filter((row) => row.arm === arm);
    const gaps = [];
    for (const scenarioId of [...new Set(rows.map((row) => row.scenarioId))]) {
      const a1 = rows.filter((row) => row.scenarioId === scenarioId && row.level === "A1");
      const b2 = rows.filter((row) => row.scenarioId === scenarioId && row.level === "B2");
      if (a1.length && b2.length) gaps.push(mean(b2.map((r) => r.complexity)) - mean(a1.map((r) => r.complexity)));
    }
    report.arms[arm] = {
      episodes: rows.length,
      complexity_gap_b2_minus_a1: round(mean(gaps)),
      gap_positive_fraction: round(mean(gaps.map((g) => g > 0 ? 1 : 0))),
      a1_complexity: round(mean(rows.filter((r) => r.level === "A1").map((r) => r.complexity))),
      b2_complexity: round(mean(rows.filter((r) => r.level === "B2").map((r) => r.complexity))),
      a1_words_per_turn: round(mean(rows.filter((r) => r.level === "A1").map((r) => r.words_per_turn)), 1),
      b2_words_per_turn: round(mean(rows.filter((r) => r.level === "B2").map((r) => r.words_per_turn)), 1),
      hard_pass_rate: round(mean(rows.map((r) => r.hard_pass ? 1 : 0))),
      payment_claims: rows.filter((r) => r.claims_payment).length,
      english_spillover: rows.filter((r) => r.english_spillover).length,
    };
  }
  writeFileSync(resolve(OUT, `episodes-${args.join("+")}.analysis.json`), JSON.stringify({ report, episodes: enriched }, null, 1));
}
console.log(JSON.stringify(report, null, 1));
