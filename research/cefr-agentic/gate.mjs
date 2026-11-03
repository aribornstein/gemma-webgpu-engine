// Steering-gate analysis (G-steer, PREREG.md). Usage:
//   node --experimental-strip-types research/cefr-agentic/gate.mjs <episodes name>...
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { checkState, complexityIndex, turnFeatures } from "../../src/research/cefr/metrics.ts";

const ROOT = resolve(import.meta.dirname, "../..");
const OUT = resolve(ROOT, "research-results/cefr-v2");
const stats = JSON.parse(readFileSync(resolve(ROOT, "research-results/cefr-agentic/complexity-stats.json"), "utf8"));
const readJsonl = (path) => readFileSync(path, "utf8").split("\n").filter(Boolean).map((line) => JSON.parse(line));
const scenarios = Object.fromEntries(readJsonl(resolve(ROOT, "research/cefr-agentic/scenarios.dev.jsonl")).map((s) => [s.scenario_id, s]));
const mean = (values) => values.reduce((a, b) => a + b, 0) / Math.max(1, values.length);
const episodes = process.argv.slice(2).flatMap((name) => readJsonl(resolve(OUT, `${name}.jsonl`)));

const rows = episodes.map((episode) => {
  const scenario = scenarios[episode.scenarioId];
  const entityNumbers = (scenario.metadata_en.entities ?? []).flatMap((e) => [e.quantity, e.price_ils].filter((v) => typeof v === "number"));
  const checks = episode.npcTurns.map((text) => checkState(text, scenario.oracle_en.critical_facts, scenario.oracle_en.must_not_assume, entityNumbers));
  return {
    arm: episode.arm, scenarioId: episode.scenarioId, level: episode.level, seed: episode.seed,
    complexity: mean(episode.npcTurns.map((text) => complexityIndex(turnFeatures(text), stats))),
    words: mean(episode.npcTurns.map((text) => turnFeatures(text).words)),
    hardPass: checks.every((c) => c.pass),
    spillover: checks.some((c) => c.englishSpillover),
  };
});

function armSummary(arm) {
  const subset = rows.filter((r) => r.arm === arm);
  const pairs = [];
  for (const key of new Set(subset.map((r) => `${r.scenarioId}|${r.seed}`))) {
    const [scenarioId, seed] = key.split("|");
    const a1 = subset.find((r) => r.scenarioId === scenarioId && String(r.seed) === seed && r.level === "A1");
    const b2 = subset.find((r) => r.scenarioId === scenarioId && String(r.seed) === seed && r.level === "B2");
    if (a1 && b2) pairs.push({ key, gap: b2.complexity - a1.complexity });
  }
  return {
    n: subset.length,
    gap: mean(pairs.map((p) => p.gap)),
    pairs,
    a1Words: mean(subset.filter((r) => r.level === "A1").map((r) => r.words)),
    b2Words: mean(subset.filter((r) => r.level === "B2").map((r) => r.words)),
    hardPass: mean(subset.map((r) => (r.hardPass ? 1 : 0))),
    spillover: mean(subset.map((r) => (r.spillover ? 1 : 0))),
  };
}

function spearman(x, y) {
  const rank = (v) => {
    const order = v.map((value, index) => [value, index]).sort((a, b) => a[0] - b[0]);
    const ranks = new Array(v.length);
    for (let i = 0; i < order.length;) {
      let j = i;
      while (j + 1 < order.length && order[j + 1][0] === order[i][0]) j += 1;
      for (let k = i; k <= j; k += 1) ranks[order[k][1]] = (i + j) / 2 + 1;
      i = j + 1;
    }
    return ranks;
  };
  const rx = rank(x); const ry = rank(y);
  const mx = mean(rx); const my = mean(ry);
  const cov = mean(rx.map((v, i) => (v - mx) * (ry[i] - my)));
  const sx = Math.sqrt(mean(rx.map((v) => (v - mx) ** 2)));
  const sy = Math.sqrt(mean(ry.map((v) => (v - my) ** 2)));
  return cov / (sx * sy);
}

const arms = [...new Set(rows.map((r) => r.arm))];
const summaries = Object.fromEntries(arms.map((arm) => [arm, armSummary(arm)]));
const raw = summaries.raw;
const report = { arms: {}, candidates: {} };
for (const [arm, s] of Object.entries(summaries)) {
  report.arms[arm] = { n: s.n, gap: +s.gap.toFixed(3), a1Words: +s.a1Words.toFixed(1), b2Words: +s.b2Words.toFixed(1), hardPass: +s.hardPass.toFixed(2), spillover: +s.spillover.toFixed(2) };
}
for (const type of ["instr", "content"]) {
  for (const site of [12, 16, 20, 22, 24, 26]) {
    const ms = [1, 2, 4].filter((m) => summaries[`${type}-L${site}-m${m}`]);
    if (!ms.length || !raw) continue;
    const series = [{ m: 0, ...raw }, ...ms.map((m) => ({ m, ...summaries[`${type}-L${site}-m${m}`] }))];
    const gaps = series.map((s) => s.gap);
    const monotonic = gaps.every((g, i) => i === 0 || g >= gaps[i - 1] - 0.05);
    const xs = series.flatMap((s) => s.pairs.map(() => s.m));
    const ys = series.flatMap((s) => s.pairs.map((p) => p.gap));
    const rho = spearman(xs, ys);
    const best = series.slice(1).reduce((a, b) => (b.gap > a.gap ? b : a));
    const controls = {};
    for (const control of ["random", "permuted", "flip"]) {
      const c = summaries[`${control}-${type}-L${site}-m${best.m}`];
      if (c) controls[control] = +(best.gap - c.gap).toFixed(3);
    }
    const criteria = {
      monotonic,
      spearman_positive: rho > 0,
      best_minus_raw_ge_0_30: best.gap - raw.gap >= 0.3,
      controls_ge_0_20: Object.keys(controls).length === 3 ? Object.values(controls).every((d) => d >= 0.2) : null,
      hard_pass_ok: best.hardPass >= raw.hardPass - 0.1,
      spillover_ok: best.spillover <= raw.spillover + 0.1,
    };
    report.candidates[`${type}-L${site}`] = {
      gaps_by_m: Object.fromEntries(series.map((s) => [s.m, +s.gap.toFixed(3)])),
      spearman: +rho.toFixed(3),
      best_m: best.m,
      best_minus_raw: +(best.gap - raw.gap).toFixed(3),
      controls_minus: controls,
      criteria,
      pass_without_controls: criteria.monotonic && criteria.spearman_positive && criteria.best_minus_raw_ge_0_30 && criteria.hard_pass_ok && criteria.spillover_ok,
      pass: criteria.monotonic && criteria.spearman_positive && criteria.best_minus_raw_ge_0_30 && criteria.controls_ge_0_20 === true && criteria.hard_pass_ok && criteria.spillover_ok,
    };
  }
}
writeFileSync(resolve(OUT, "steering-gate.json"), JSON.stringify(report, null, 1));
console.log(JSON.stringify(report, null, 1));
