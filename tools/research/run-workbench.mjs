import { chromium } from "@playwright/test";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { appendFileSync, existsSync, mkdirSync, readFileSync, readdirSync, renameSync, writeFileSync } from "node:fs";
import { resolve, relative } from "node:path";

// Use the browser workbench's exact API. Completed artifacts are fsynced by file replacement before
// the next job, and an append-only event ledger records success/failure. No arbitrary browser code
// from request files is evaluated.
const [requestsPath, runId = "mechanistic-v1"] = process.argv.slice(2);
if (!requestsPath || !/^[a-zA-Z0-9._-]+$/.test(runId)) throw new Error("Usage: run-workbench.mjs <requests.json> [run-id]");
const root = resolve(import.meta.dirname, "../..");
const output = resolve(process.env.RESEARCH_OUTPUT ?? resolve(root, "research-results/mechanistic-repair"), runId);
const modelPath = resolve(root, "public/models/gemma-4-e2b/model.safetensors");
const checkpointSha256 = "efab429012b97ab986c4d4838a46ff3ad95d618b42ce514771ca40fadc76a9a4";
if (!existsSync(modelPath)) throw new Error("Provide the pinned local checkpoint before running research");
const requests = JSON.parse(readFileSync(resolve(requestsPath), "utf8"));
if (!Array.isArray(requests) || !requests.length) throw new Error("Expected a nonempty experiment request array");
mkdirSync(output, { recursive: true });

function sourceFingerprint() {
  const hash = createHash("sha256");
  function walk(path) {
    for (const entry of readdirSync(path, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
      const file = resolve(path, entry.name);
      if (entry.isDirectory()) walk(file);
      else if (/\.(ts|css|html|mjs)$/.test(entry.name)) hash.update(relative(root, file)).update(readFileSync(file));
    }
  }
  walk(resolve(root, "src"));
  walk(resolve(root, "tools/research"));
  for (const name of ["package.json", "vite.config.ts", "research.html", "research/mechanistic-repair/protocol.json"]) hash.update(name).update(readFileSync(resolve(root, name)));
  return hash.digest("hex");
}
const revision = sourceFingerprint();
const manifestPath = resolve(output, "manifest.json");
const artifactPath = resolve(output, "artifacts.json");
const started = Date.now();
let progress = { completed: 0, total: null, phase: "verifying checkpoint", id: "" };
let completed = 0;
const timer = setInterval(() => {
  const fill = Math.floor(20 * completed / requests.length);
  console.log(`HEARTBEAT [${"#".repeat(fill)}${".".repeat(20 - fill)}] jobs ${completed}/${requests.length} elapsed ${Math.floor((Date.now() - started) / 1000)}s ${progress.phase} ${progress.completed}/${progress.total ?? "?"} ${progress.id}`);
}, 15000);
let browser;
const saved = existsSync(artifactPath) ? JSON.parse(readFileSync(artifactPath, "utf8")) : [];
function atomicJson(path, data) {
  writeFileSync(`${path}.part`, JSON.stringify(data), { flush: true });
  renameSync(`${path}.part`, path);
}
const event = (data) => appendFileSync(resolve(output, "events.jsonl"), JSON.stringify({ at: new Date().toISOString(), ...data }) + "\n", { flush: true });
try {
  const sha = await import("node:fs").then(async ({ createReadStream }) => {
    const hash = createHash("sha256");
    for await (const chunk of createReadStream(modelPath)) hash.update(chunk);
    return hash.digest("hex");
  });
  if (sha !== checkpointSha256) throw new Error("Checkpoint hash mismatch");
  const provenance = {
    revision, checkpointRevision: "9fcec64df66cb1e4d972fc5cdc142afb25b2362c",
    checkpointSha256, environment: "Chrome/WebGPU; actual browser version recorded in manifest", runId,
  };
  if (existsSync(manifestPath)) {
    const previous = JSON.parse(readFileSync(manifestPath, "utf8"));
    if (previous.provenance.revision !== revision) throw new Error("Source changed: use a new run-id instead of mixing implementations");
  }
  event({ type: "start", provenance, requestsHash: createHash("sha256").update(JSON.stringify(requests)).digest("hex") });
  browser = await chromium.launch({ channel: "chrome", headless: true });
  const page = await browser.newPage();
  page.on("pageerror", (error) => event({ type: "page-error", error: error.message }));
  await page.exposeFunction("recordResearchProgress", (value) => { progress = value; });
  await page.exposeFunction("persistResearchArtifact", (artifact) => {
    if (artifact.provenance.runId !== runId || artifact.provenance.revision !== revision) {
      throw new Error("Artifact persistence provenance mismatch");
    }
    const index = saved.findIndex((a) => a.id === artifact.id);
    if (index >= 0) saved[index] = artifact; else saved.push(artifact);
    atomicJson(artifactPath, saved);
    event({ type: "artifact-checkpoint", id: artifact.id, status: artifact.status, usage: artifact.usage });
  });
  await page.goto(process.env.RESEARCH_URL ?? "http://127.0.0.1:5190/research.html");
  await page.waitForFunction(() => Boolean(window.researchWorkbench), null, { timeout: 30000 });
  await page.evaluate(() => window.addEventListener("research-progress", (e) => window.recordResearchProgress(e.detail)));
  atomicJson(manifestPath, {
    schemaVersion: 1, startedAt: new Date().toISOString(), provenance,
    gitCommit: execFileSync("git", ["rev-parse", "HEAD"], { cwd: root, encoding: "utf8" }).trim(),
    dirty: execFileSync("git", ["status", "--short"], { cwd: root, encoding: "utf8" }),
    browser: browser.version(), requestFile: resolve(requestsPath), status: "running",
  });
  await page.evaluate(async ({ provenance, saved }) => window.researchWorkbench.load(provenance, saved, true), { provenance, saved });
  for (const request of requests) {
    if (sourceFingerprint() !== revision) throw new Error("Source changed during a frozen experiment");
    progress = { completed: 0, total: null, phase: "starting", id: request.id };
    event({ type: "job-start", request });
    try {
      const artifact = await page.evaluate((request) => window.researchWorkbench.run(request), request);
      const index = saved.findIndex((a) => a.id === artifact.id);
      if (index >= 0) saved[index] = artifact; else saved.push(artifact);
      atomicJson(artifactPath, saved);
      event({ type: "job-completed", id: artifact.id, resultHash: artifact.resultHash, usage: artifact.usage });
    } catch (error) {
      const records = await page.evaluate(() => window.researchWorkbench.artifacts());
      atomicJson(artifactPath, records);
      event({ type: "job-failed", id: request.id, error: String(error) });
      throw error;
    }
    completed += 1;
    console.log(`PROGRESS ${completed}/${requests.length} ${request.id}`);
  }
  if (sourceFingerprint() !== revision) throw new Error("Source changed before final persistence");
  await page.evaluate(() => window.researchWorkbench.close());
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  atomicJson(manifestPath, { ...manifest, status: "completed", completedAt: new Date().toISOString(), artifacts: saved.length });
  console.log(`Evidence persisted: ${artifactPath}`);
} catch (error) {
  event({ type: "run-failed", error: String(error) });
  if (existsSync(manifestPath)) {
    const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
    atomicJson(manifestPath, { ...manifest, status: "failed", error: String(error) });
  }
  throw error;
} finally {
  clearInterval(timer);
  await browser?.close();
}
