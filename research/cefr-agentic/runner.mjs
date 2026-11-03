// Research campaign driver: runs one stage on the owned WebGPU engine in headless Chrome and writes
// artifacts under research-results/cefr-agentic/. Requires the Vite dev server (RESEARCH_URL, default
// http://127.0.0.1:5190/) serving this repository with the checkpoint at
// public/models/gemma-4-e2b/model.safetensors.
//
//   node research/cefr-agentic/runner.mjs baselines|probes|episodes <args>
//
// Progress lines ("PROGRESS <pct> <message>") are printed for heartbeat wrappers.
import { chromium } from "@playwright/test";
import { createHash } from "node:crypto";
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const ROOT = resolve(import.meta.dirname, "../..");
const OUT = resolve(ROOT, "research-results/cefr-agentic");
const URL = process.env.RESEARCH_URL ?? "http://127.0.0.1:5190/";
mkdirSync(resolve(OUT, "activations"), { recursive: true });

const readJsonl = (path) => readFileSync(path, "utf8").split("\n").filter(Boolean).map((line) => JSON.parse(line));
const fixture = (name) => readJsonl(resolve(ROOT, "research/cefr-agentic", name));
const sha = (text) => createHash("sha256").update(text).digest("hex");

async function inBrowser(stage, payload) {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const page = await browser.newPage();
  page.on("console", (message) => {
    const text = message.text();
    if (text.startsWith("PROGRESS") || text.startsWith("INFO")) console.log(text);
  });
  page.on("pageerror", (error) => console.log("PAGEERROR", String(error)));
  await page.goto(URL);
  try {
    return await page.evaluate(async ({ stage, payload }) => {
      const sessionModule = "/src/runtime/gemma-session.ts";
      const experimentsModule = "/src/research/cefr/experiments.ts";
      const { loadGemmaGenerationSession } = await import(sessionModule);
      const experiments = await import(experimentsModule);
      const session = await loadGemmaGenerationSession({
        sourceUrl: "/models/gemma-4-e2b/model.safetensors",
        cacheCapacity: 2048,
        prefillStrategy: "sequential",
      });
      const progress = (done, total, message) =>
        console.log(`PROGRESS ${Math.floor((100 * done) / total)} ${message}`);
      const adapter = navigator.gpu ? await navigator.gpu.requestAdapter() : null;
      const provenance = {
        browserVersion: navigator.userAgent,
        gpuAdapter: adapter?.info ? `${adapter.info.vendor} ${adapter.info.architecture}` : "unknown",
      };
      const results = [];
      try {
        if (stage === "episodes") {
          const jobs = payload.jobs;
          for (let index = 0; index < jobs.length; index += 1) {
            const job = jobs[index];
            let gate = null;
            if (job.overlay?.selfGate) {
              // Gate from the model's own state at the end of the metadata (read with no overlay).
              await session.setResearchOverlay(null);
              const rows = await experiments.captureLevelStates(session, job.scenario, job.level, 0, [job.overlay.selfGate.site]);
              const state = rows.find((row) => row.position === "end_of_metadata").values;
              const { direction, threshold, scale } = job.overlay.selfGate;
              let projection = 0;
              for (let i = 0; i < state.length; i += 1) projection += direction[i] * state[i];
              gate = Math.max(-1, Math.min(1, (projection - threshold) / scale));
            }
            if (job.overlay) {
              const coefficient = gate === null ? job.overlay.coefficient : gate * job.overlay.alpha;
              job.overlay.appliedCoefficient = coefficient;
              job.overlay.gateValue = gate;
              await session.setResearchOverlay({
                site: job.overlay.site,
                a: Float32Array.from(job.overlay.a),
                b: Float32Array.from(job.overlay.b),
                coefficient,
              });
            } else {
              await session.setResearchOverlay(null);
            }
            const episode = await experiments.runEpisode(session, job.scenario, job.level, job.policy, job.seed);
            results.push({ ...episode, arm: job.arm, editId: job.overlay?.id ?? null, coefficient: job.overlay?.appliedCoefficient ?? 0, gateValue: job.overlay?.gateValue ?? null, jobKey: job.key });
            progress(index + 1, jobs.length, `${job.arm} ${job.scenario.scenario_id} ${job.level} seed ${job.seed}`);
          }
          await session.setResearchOverlay(null);
        } else if (stage === "directions") {
          const { scenarios, references, sites } = payload;
          const total = references.length * 4 + scenarios.length * 2;
          let done = 0;
          for (const reference of references) {
            const scenario = scenarios.find((item) => item.scenario_id === reference.scenario_id);
            for (const level of ["A1", "B2"]) {
              for (const style of ["A1", "B2"]) {
                const reply = style === "A1" ? reference.npc_reply_a1_he : reference.npc_reply_b2_he;
                const states = await experiments.meanReplyStates(session, scenario, level, reply, sites);
                results.push({ kind: "reply_mean", id: `${scenario.scenario_id}:${level}:${style}`, scenarioFamily: scenario.scenario_family, split: scenario.split, level, style, states });
                progress(++done, total, `reply means ${scenario.scenario_id} ${level}/${style}`);
              }
            }
          }
          for (const scenario of scenarios) {
            for (const level of ["A1", "B2"]) {
              results.push(...(await experiments.captureLevelStates(session, scenario, level, 0, sites)).map((row) => ({ kind: "level", ...row })));
              progress(++done, total, `metadata states ${scenario.scenario_id} ${level}`);
            }
          }
        } else if (stage === "probes") {
          const { scenarios, references, contrasts, sites } = payload;
          const total = scenarios.length * 10 + references.length * 4 + contrasts.length + references.length * 4;
          let done = 0;
          for (const scenario of scenarios) {
            for (const level of ["A1", "B2"]) {
              for (let turn = 0; turn < scenario.player_policy.player_turns_he.length; turn += 1) {
                results.push(...(await experiments.captureLevelStates(session, scenario, level, turn, sites)).map((row) => ({ kind: "level", ...row })));
                progress(++done, total, `level ${scenario.scenario_id} ${level} turn ${turn}`);
              }
            }
          }
          for (const reference of references) {
            const scenario = scenarios.find((item) => item.scenario_id === reference.scenario_id);
            for (const level of ["A1", "B2"]) {
              for (const style of ["A1", "B2"]) {
                const reply = style === "A1" ? reference.npc_reply_a1_he : reference.npc_reply_b2_he;
                results.push(...(await experiments.captureCrossed(session, scenario, level, style, reply, sites)).map((row) => ({ kind: "crossed", ...row })));
                progress(++done, total, `crossed ${scenario.scenario_id} ${level}/${style}`);
              }
            }
          }
          for (const pair of contrasts) {
            results.push({ kind: "contrast", ...(await experiments.scoreContrast(session, pair)) });
            progress(++done, total, `contrast ${pair.id}`);
          }
          for (const reference of references) {
            const scenario = scenarios.find((item) => item.scenario_id === reference.scenario_id);
            for (const level of ["A1", "B2"]) {
              for (const style of ["A1", "B2"]) {
                const reply = style === "A1" ? reference.npc_reply_a1_he : reference.npc_reply_b2_he;
                const states = await experiments.meanReplyStates(session, scenario, level, reply, sites);
                results.push({ kind: "reply_mean", id: `${scenario.scenario_id}:${level}:${style}`, scenarioFamily: scenario.scenario_family, split: scenario.split, level, style, states });
                progress(++done, total, `reply means ${scenario.scenario_id} ${level}/${style}`);
              }
            }
          }
        }
      } finally {
        session.destroy();
      }
      return { results, provenance };
    }, { stage, payload });
  } finally {
    await browser.close();
  }
}

const [stage, ...args] = process.argv.slice(2);
const startedAt = new Date().toISOString();
const manifestPath = resolve(OUT, "experiments.jsonl");

if (stage === "probes") {
  const scenarios = [...fixture("scenarios.dev.jsonl"), ...fixture("scenarios.validation.jsonl")];
  const references = fixture("references.dev.jsonl");
  const contrasts = fixture("contrasts.dev.jsonl");
  const sites = [12, 20, 22, 24, "final"];
  const { results, provenance } = await inBrowser("probes", { scenarios, references, contrasts, sites });
  writeFileSync(resolve(OUT, "activations/probes.json"), JSON.stringify(results));
  appendFileSync(manifestPath, JSON.stringify({ stage, startedAt, finishedAt: new Date().toISOString(), rows: results.length, sites, provenance, artifact: "activations/probes.json" }) + "\n");
  console.log(`wrote ${results.length} probe rows`);
} else if (stage === "directions") {
  // Development scenarios only: edit directions must never see validation or sealed data.
  const scenarios = fixture("scenarios.dev.jsonl");
  const references = fixture("references.dev.jsonl").filter((row) => row.split === "development");
  const sites = [{ residual: 20 }, { residual: 22 }, { residual: 24 }, "final"];
  const { results, provenance } = await inBrowser("directions", { scenarios, references, sites });
  writeFileSync(resolve(OUT, "activations/directions.json"), JSON.stringify(results));
  appendFileSync(manifestPath, JSON.stringify({ stage, startedAt, finishedAt: new Date().toISOString(), rows: results.length, sites, provenance, artifact: "activations/directions.json" }) + "\n");
  console.log(`wrote ${results.length} direction rows`);
} else if (stage === "episodes") {
  // args: <jobs.json> <output name>
  const [jobsPath, outputName] = args;
  const jobs = JSON.parse(readFileSync(jobsPath, "utf8"));
  const outputPath = resolve(OUT, `${outputName}.jsonl`);
  const done = new Set(existsSync(outputPath) ? readJsonl(outputPath).map((row) => row.jobKey) : []);
  const pending = jobs.filter((job) => !done.has(job.key));
  console.log(`INFO ${pending.length}/${jobs.length} episode jobs pending`);
  if (pending.length) {
    const { results, provenance } = await inBrowser("episodes", { jobs: pending });
    for (const row of results) appendFileSync(outputPath, JSON.stringify({ ...row, provenance }) + "\n");
    appendFileSync(manifestPath, JSON.stringify({ stage, outputName, startedAt, finishedAt: new Date().toISOString(), episodes: results.length, jobsHash: sha(JSON.stringify(jobs)), provenance }) + "\n");
  }
  console.log("PROGRESS 100 episodes complete");
} else {
  console.error("usage: runner.mjs probes | episodes <jobs.json> <output>");
  process.exit(2);
}
