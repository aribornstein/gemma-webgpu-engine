// Builds candidate edit ingredients from development-only captures (activations/directions.json),
// and materializes agent-proposed candidates and controls into episode job files.
//
//   node research/cefr-agentic/candidates.mjs ingredients
//   node research/cefr-agentic/candidates.mjs jobs <proposals.json> <out-jobs.json> <split> [families]
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const ROOT = resolve(import.meta.dirname, "../..");
const OUT = resolve(ROOT, "research-results/cefr-agentic");
const readJsonl = (path) => readFileSync(path, "utf8").split("\n").filter(Boolean).map((line) => JSON.parse(line));
const norm = (v) => Math.hypot(...v);
const unit = (v) => { const n = norm(v); return v.map((x) => x / n); };
const add = (a, b) => a.map((x, i) => x + b[i]);
const scale = (a, s) => a.map((x) => x * s);
const dot = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);
const meanVec = (rows) => scale(rows.reduce((acc, row) => add(acc, row), new Array(rows[0].length).fill(0)), 1 / rows.length);
const siteKey = (site) => typeof site === "object" ? `residual${site.residual}` : String(site);

const [command, ...args] = process.argv.slice(2);
if (command === "ingredients") {
  const rows = JSON.parse(readFileSync(resolve(OUT, "activations/directions.json"), "utf8"));
  if (rows.some((row) => row.split !== "development")) throw new Error("Directions must come from development data only");
  const replies = rows.filter((row) => row.kind === "reply_mean");
  const sites = Object.keys(replies[0].states);
  const ingredients = {};
  for (const site of sites) {
    const all = replies.map((row) => row.states[site]);
    const m = meanVec(all);
    const a1 = meanVec(replies.filter((row) => row.style === "A1").map((row) => row.states[site]));
    const b2 = meanVec(replies.filter((row) => row.style === "B2").map((row) => row.states[site]));
    const simplify = unit(add(a1, scale(b2, -1)));
    // Per-family agreement of the simplification direction (cosine with the pooled direction).
    const families = [...new Set(replies.map((row) => row.scenarioFamily))];
    const familyCosines = families.map((family) => {
      const f = replies.filter((row) => row.scenarioFamily === family);
      const d = add(meanVec(f.filter((r) => r.style === "A1").map((r) => r.states[site])), scale(meanVec(f.filter((r) => r.style === "B2").map((r) => r.states[site])), -1));
      return dot(unit(d), simplify);
    });
    // Self gate from the end-of-metadata state: requested level A1 (positive) vs B2.
    const meta = rows.filter((row) => row.kind === "level" && row.position === "end_of_metadata" && String(siteKey(row.site)) === site);
    let selfGate = null;
    if (meta.length) {
      const pa = meanVec(meta.filter((r) => r.level === "A1").map((r) => r.values));
      const pb = meanVec(meta.filter((r) => r.level === "B2").map((r) => r.values));
      const direction = unit(add(pa, scale(pb, -1)));
      const projA = dot(direction, pa);
      const projB = dot(direction, pb);
      const projections = meta.map((r) => ({ level: r.level, value: dot(direction, r.values) }));
      selfGate = {
        site: site.startsWith("residual") ? { residual: Number(site.slice(8)) } : (site === "final" ? "final" : Number(site)),
        direction,
        threshold: (projA + projB) / 2,
        scale: Math.abs(projA - projB) / 2,
        train_separation: projections.every((p) => (p.level === "A1") === (p.value > (projA + projB) / 2)),
      };
    }
    ingredients[site] = {
      a: unit(m),
      mean_state_norm: norm(m),
      b_simplify: simplify,
      raw_style_gap_norm: norm(add(a1, scale(b2, -1))),
      style_gap_over_state_norm: norm(add(a1, scale(b2, -1))) / norm(m),
      family_cosines: familyCosines.map((c) => Number(c.toFixed(3))),
      selfGate,
    };
  }
  writeFileSync(resolve(OUT, "patches/ingredients.json"), JSON.stringify(ingredients));
  const summary = Object.fromEntries(Object.entries(ingredients).map(([site, value]) => [site, {
    mean_state_norm: Number(value.mean_state_norm.toFixed(2)),
    style_gap_over_state_norm: Number(value.style_gap_over_state_norm.toFixed(4)),
    family_cosines: value.family_cosines,
    self_gate_train_separation: value.selfGate?.train_separation ?? null,
    cos_a_b: Number(dot(value.a, value.b_simplify).toFixed(3)),
  }]));
  writeFileSync(resolve(OUT, "patches/ingredients.summary.json"), JSON.stringify(summary, null, 1));
  console.log(JSON.stringify(summary, null, 1));
} else if (command === "jobs") {
  // proposals: [{ id, site: "final"|"residual20"|..., gate: "external"|"self"|"wrong_level"|"none", alpha, b: "simplify"|"random", seed? }]
  const [proposalsPath, outPath, split, familiesText] = args;
  const ingredients = JSON.parse(readFileSync(resolve(OUT, "patches/ingredients.json"), "utf8"));
  const proposals = JSON.parse(readFileSync(proposalsPath, "utf8"));
  const scenarioFile = split === "sealed" ? process.env.SEALED_PATH : resolve(ROOT, `research/cefr-agentic/scenarios.${split === "dev" ? "dev" : "validation"}.jsonl`);
  let scenarios = readJsonl(scenarioFile);
  if (familiesText) scenarios = scenarios.filter((s) => familiesText.split(",").includes(s.scenario_family));
  const jobs = [];
  for (const proposal of proposals) {
    const ing = ingredients[proposal.site];
    if (!ing) throw new Error(`unknown site ${proposal.site}`);
    let b = ing.b_simplify;
    if (proposal.b === "random") {
      let state = proposal.seed ?? 99;
      const next = () => (state = (Math.imul(state, 1664525) + 1013904223) >>> 0) / 4294967296 - 0.5;
      b = unit(b.map(() => next()));
    }
    const site = proposal.site === "final" ? "final" : { layer: Number(proposal.site.replace("residual", "")) };
    const patch = { id: proposal.id, site, a: ing.a, b };
    for (const scenario of scenarios) {
      for (const level of ["A1", "B2"]) {
        const externalGate = level === "A1" ? 1 : -1;
        let overlay;
        if (proposal.gate === "none") overlay = { ...patch, coefficient: 0 };
        else if (proposal.gate === "external") overlay = { ...patch, coefficient: externalGate * proposal.alpha };
        else if (proposal.gate === "wrong_level") overlay = { ...patch, coefficient: -externalGate * proposal.alpha };
        else if (proposal.gate === "self") {
          if (!ing.selfGate) throw new Error(`no self gate at ${proposal.site}`);
          overlay = { ...patch, alpha: proposal.alpha, selfGate: { site: ing.selfGate.site, direction: ing.selfGate.direction, threshold: ing.selfGate.threshold, scale: ing.selfGate.scale } };
        } else throw new Error(`unknown gate ${proposal.gate}`);
        for (const seed of (process.env.SEEDS ?? "23").split(",").map(Number)) {
          const job = { scenario, level, arm: proposal.arm ?? proposal.id, policy: "canonical", seed, overlay };
          job.key = `${job.arm}|${scenario.scenario_id}|${level}|${seed}`;
          jobs.push(job);
        }
      }
    }
  }
  writeFileSync(outPath, JSON.stringify(jobs));
  console.log(`${jobs.length} jobs -> ${outPath}`);
}
