// Round 2: directions from the model's own level-conditioned behavior, and episode jobs for the
// steering gate, operator search, development panel and sealed run.
//
//   node research/cefr-agentic/v2.mjs directions
//   node research/cefr-agentic/v2.mjs jobs <configs.json> <out.json> <scenarios.jsonl> [families] [seeds] [turns]
//
// Config: { arm, type: "instr"|"content"|"random"|"permuted"|"none", sites: [layer...], m, gate:
//           "external"|"wrong_level"|"self", policy?: "canonical"|"level_specific_strong" }
// m scales the edit in natural units: m = 1 shifts the residual by half the measured A1-B2 difference
// along the direction (the size of the instruction's own level effect for "instr").
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";

const ROOT = resolve(import.meta.dirname, "../..");
const OUT = resolve(ROOT, "research-results/cefr-v2");
mkdirSync(resolve(OUT, "patches"), { recursive: true });
const readJsonl = (path) => readFileSync(path, "utf8").split("\n").filter(Boolean).map((line) => JSON.parse(line));
const norm = (v) => Math.hypot(...v);
const unit = (v) => { const n = norm(v); return v.map((x) => x / n); };
const sub = (a, b) => a.map((x, i) => x - b[i]);
const dot = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);
const mean = (rows) => rows[0].map((_, i) => rows.reduce((sum, row) => sum + row[i], 0) / rows.length);
const cosine = (a, b) => dot(a, b) / (norm(a) * norm(b));
function rng(seed) {
  let state = seed >>> 0 || 1;
  return () => (state = (Math.imul(state, 1664525) + 1013904223) >>> 0) / 4294967296;
}

/** Removes the part of each state explained by log reply length (pooled OLS per dimension). */
function lengthResiduals(rows, site) {
  const x = rows.map((row) => Math.log(row.replyTokens));
  const mx = x.reduce((a, b) => a + b, 0) / x.length;
  const vx = x.reduce((a, v) => a + (v - mx) ** 2, 0) / x.length;
  const states = rows.map((row) => row.canonical[site]);
  const mean0 = mean(states);
  const slope = mean0.map((_, i) => states.reduce((a, s, r) => a + (s[i] - mean0[i]) * (x[r] - mx), 0) / x.length / vx);
  return states.map((s, r) => s.map((v, i) => v - slope[i] * (x[r] - mx)));
}

const [command, ...args] = process.argv.slice(2);
if (command === "directions") {
  const rows = JSON.parse(readFileSync(resolve(OUT, "activations/trajectories.json"), "utf8"));
  const sites = Object.keys(rows[0].canonical);
  const families = [...new Set(rows.map((row) => row.family))];
  const random = rng(4242);
  const out = {};
  const summary = {};
  for (const site of sites) {
    const instrDelta = (subset) => sub(
      mean(subset.filter((r) => r.level === "A1").map((r) => sub(r.strong[site], r.canonical[site]))),
      mean(subset.filter((r) => r.level === "B2").map((r) => sub(r.strong[site], r.canonical[site]))),
    );
    const instr = instrDelta(rows);
    const residuals = lengthResiduals(rows, site);
    const content = sub(
      mean(residuals.filter((_, i) => rows[i].level === "A1")),
      mean(residuals.filter((_, i) => rows[i].level === "B2")),
    );
    const contentRaw = sub(
      mean(rows.filter((r) => r.level === "A1").map((r) => r.canonical[site])),
      mean(rows.filter((r) => r.level === "B2").map((r) => r.canonical[site])),
    );
    // Permutation control: swap level labels for a random half of (scenario, seed, turn) pairs.
    const swap = new Map();
    const permuted = rows.map((row) => {
      const key = `${row.scenarioId}|${row.seed}|${row.turn}`;
      if (!swap.has(key)) swap.set(key, random() < 0.5);
      return swap.get(key) ? { ...row, level: row.level === "A1" ? "B2" : "A1" } : row;
    });
    const permutedInstr = instrDelta(permuted);
    const randomDirection = unit(instr.map(() => random() - 0.5));
    const canonicalStates = rows.map((r) => r.canonical[site]);
    const a = unit(mean(canonicalStates));
    const s = canonicalStates.reduce((sum, h) => sum + dot(a, h), 0) / canonicalStates.length;
    const familyCosine = (vectorOf) => families.map((family) => {
      const subset = rows.filter((r) => r.family === family);
      return Number(cosine(vectorOf(subset), vectorOf(rows)).toFixed(3));
    });
    out[site] = {
      a, s,
      instr: { b: unit(instr), magnitude: norm(instr) / 2 },
      content: { b: unit(content), magnitude: norm(content) / 2 },
      permuted: { b: unit(permutedInstr), magnitude: norm(instr) / 2 },
      random: { b: randomDirection, magnitude: norm(instr) / 2 },
    };
    summary[site] = {
      state_norm: Number(norm(mean(canonicalStates)).toFixed(2)),
      a_dot_h: Number(s.toFixed(2)),
      instr_half_diff_norm: Number((norm(instr) / 2).toFixed(3)),
      content_half_diff_norm: Number((norm(content) / 2).toFixed(3)),
      content_raw_half_diff_norm: Number((norm(contentRaw) / 2).toFixed(3)),
      permuted_instr_half_diff_norm: Number((norm(permutedInstr) / 2).toFixed(3)),
      cos_instr_content: Number(cosine(instr, content).toFixed(3)),
      cos_content_contentRaw: Number(cosine(content, contentRaw).toFixed(3)),
      instr_family_cosines: familyCosine((subset) => instrDelta(subset)),
    };
  }
  writeFileSync(resolve(OUT, "patches/directions.json"), JSON.stringify(out));
  writeFileSync(resolve(OUT, "directions.summary.json"), JSON.stringify({ families, rows: rows.length, sites: summary }, null, 1));
  console.log(JSON.stringify({ families, rows: rows.length, sites: summary }, null, 1));
} else if (command === "jobs") {
  const [configsPath, outPath, scenariosPath, familiesText, seedsText = "23,47", turnsText = "3"] = args;
  const directions = JSON.parse(readFileSync(resolve(OUT, "patches/directions.json"), "utf8"));
  const configs = JSON.parse(readFileSync(configsPath, "utf8"));
  let scenarios = scenariosPath.split(",").flatMap((path) => readJsonl(path));
  if (familiesText && familiesText !== "all") scenarios = scenarios.filter((s) => familiesText.split(",").includes(s.scenario_family));
  const seeds = seedsText.split(",").map(Number);
  const turns = Number(turnsText);
  const jobs = [];
  for (const config of configs) {
    const overlays = (level) => {
      if (config.type === "none") return null;
      const sign = config.gate === "wrong_level" ? -1 : 1;
      const gate = (level === "A1" ? 1 : -1) * sign;
      return config.sites.map((layer) => {
        const site = directions[`residual${layer}`];
        const direction = site[config.type];
        // Random controls borrow the magnitude of the direction type they control for.
        const magnitude = config.type === "random" || config.type === "permuted" ? site[config.controlFor ?? "instr"].magnitude : direction.magnitude;
        return {
          id: `${config.arm}@${layer}`,
          site: { layer },
          a: site.a,
          b: direction.b,
          coefficient: (gate * config.m * magnitude) / site.s / config.sites.length,
        };
      });
    };
    for (const scenario of scenarios) {
      for (const level of ["A1", "B2"]) {
        for (const seed of seeds) {
          const job = { scenario, level, arm: config.arm, policy: config.policy ?? "canonical", seed: seed + (config.seedOffset ?? 0), turns, overlays: overlays(level) };
          job.key = `${config.arm}|${scenario.scenario_id}|${level}|${seed}`;
          jobs.push(job);
        }
      }
    }
  }
  writeFileSync(outPath, JSON.stringify(jobs));
  console.log(`${jobs.length} jobs -> ${outPath}`);
}
