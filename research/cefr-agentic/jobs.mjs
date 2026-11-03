// Builds episode job lists. Usage: node research/cefr-agentic/jobs.mjs baselines <out.json> [split]
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
const ROOT = resolve(import.meta.dirname, "../..");
const readJsonl = (path) => readFileSync(path, "utf8").split("\n").filter(Boolean).map((line) => JSON.parse(line));
const [kind, out, split = "dev"] = process.argv.slice(2);
const scenarios = split === "sealed"
  ? readJsonl(process.env.SEALED_PATH)
  : readJsonl(resolve(ROOT, `research/cefr-agentic/scenarios.${split === "dev" ? "dev" : "validation"}.jsonl`));
const jobs = [];
if (kind === "baselines") {
  for (const scenario of scenarios) {
    for (const level of ["A1", "B2"]) {
      const base = { scenario, level };
      jobs.push({ ...base, arm: "raw_base", policy: "canonical", seed: 23 });
      jobs.push({ ...base, arm: "strong_prompt", policy: "level_specific_strong", seed: 23 });
      jobs.push({ ...base, arm: "best_of_2_candidate", policy: "level_specific_strong", seed: 47 });
    }
  }
}
for (const job of jobs) job.key = `${job.arm}|${job.scenario.scenario_id}|${job.level}|${job.seed}`;
writeFileSync(out, JSON.stringify(jobs));
console.log(`${jobs.length} jobs -> ${out}`);
